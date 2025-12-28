"""
Benchmark Framework

Provides infrastructure for evaluating uncertainty-guided retrieval:
- Standard metrics (accuracy, token efficiency, calibration)
- Dataset loading (synthetic and real)
- Comparison with baselines

Key metrics unique to our approach:
- Early stop rate: How often we stop before reaching leaves
- Calibration error: How well confidence matches accuracy
- Uncertainty correlation: Does high uncertainty predict errors?
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import time
import json
from pathlib import Path
import numpy as np

from src.fractal_tree import FractalTree
from src.zoom_agent import UncertaintyZoomAgent, SearchResult, RelevanceOnlyAgent
from src.llm_interface import SLMInterface


@dataclass
class BenchmarkSample:
    """A single benchmark sample (query + ground truth)."""
    query: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""
    sample: BenchmarkSample
    search_result: SearchResult
    is_correct: bool
    similarity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.sample.query,
            "ground_truth": self.sample.ground_truth,
            "predicted": self.search_result.answer,
            "is_correct": self.is_correct,
            "similarity_score": self.similarity_score,
            "confidence": self.search_result.confidence,
            "tokens_used": self.search_result.tokens_used,
            "num_expansions": self.search_result.num_expansions,
            "depth_reached": self.search_result.depth_reached,
            "status": self.search_result.status.value,
        }


@dataclass
class BenchmarkMetrics:
    """
    Comprehensive metrics for uncertainty-guided retrieval.
    
    Contains both standard metrics and our novel metrics:
    - Standard: accuracy, precision, recall, F1
    - Efficiency: tokens used, early stop rate
    - Calibration: ECE, uncertainty-error correlation
    """
    
    # Standard metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Efficiency metrics
    avg_tokens_used: float = 0.0
    tokens_saved_vs_full: float = 0.0
    avg_expansions: float = 0.0
    avg_depth: float = 0.0
    
    # Novel metrics - key contribution
    early_stop_rate: float = 0.0  # % stopped before max depth
    avg_confidence: float = 0.0
    calibration_error: float = 0.0  # ECE
    uncertainty_error_correlation: float = 0.0  # Higher = uncertainty predicts errors
    
    # Timing
    avg_wall_time: float = 0.0
    total_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_tokens_used": round(self.avg_tokens_used, 1),
            "tokens_saved_vs_full": round(self.tokens_saved_vs_full, 2),
            "avg_expansions": round(self.avg_expansions, 2),
            "avg_depth": round(self.avg_depth, 2),
            "early_stop_rate": round(self.early_stop_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "calibration_error": round(self.calibration_error, 4),
            "uncertainty_error_correlation": round(self.uncertainty_error_correlation, 4),
            "avg_wall_time": round(self.avg_wall_time, 3),
            "total_samples": self.total_samples,
        }


class AnswerMatcher:
    """Methods for matching predicted answers to ground truth."""
    
    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> bool:
        """Exact string match (ignoring case and whitespace)."""
        pred_clean = predicted.strip().lower()
        truth_clean = ground_truth.strip().lower()
        return pred_clean == truth_clean
    
    @staticmethod
    def contains_match(predicted: str, ground_truth: str) -> bool:
        """Check if ground truth is contained in prediction."""
        pred_clean = predicted.strip().lower()
        truth_clean = ground_truth.strip().lower()
        return truth_clean in pred_clean
    
    @staticmethod
    def fuzzy_match(predicted: str, ground_truth: str, threshold: float = 0.8) -> bool:
        """Fuzzy string matching using character overlap."""
        pred_chars = set(predicted.lower())
        truth_chars = set(ground_truth.lower())
        
        if not truth_chars:
            return not pred_chars
        
        overlap = len(pred_chars & truth_chars)
        similarity = overlap / len(truth_chars)
        return similarity >= threshold
    
    @staticmethod
    def semantic_match(
        predicted: str,
        ground_truth: str,
        embedder: Callable[[str], np.ndarray],
        threshold: float = 0.85,
    ) -> Tuple[bool, float]:
        """Semantic similarity matching using embeddings."""
        pred_emb = embedder(predicted)
        truth_emb = embedder(ground_truth)
        
        similarity = np.dot(pred_emb, truth_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(truth_emb) + 1e-8
        )
        
        return similarity >= threshold, float(similarity)


class BenchmarkRunner:
    """
    Main class for running benchmarks.
    
    Evaluates uncertainty-guided retrieval against baselines
    and computes comprehensive metrics.
    """
    
    def __init__(
        self,
        slm: SLMInterface,
        match_strategy: str = "contains",
        semantic_threshold: float = 0.85,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            slm: SLM interface for generation and embedding
            match_strategy: How to match answers ("exact", "contains", "fuzzy", "semantic")
            semantic_threshold: Threshold for semantic matching
        """
        self.slm = slm
        self.match_strategy = match_strategy
        self.semantic_threshold = semantic_threshold
    
    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        search_result: SearchResult,
    ) -> EvaluationResult:
        """Evaluate a single sample."""
        predicted = search_result.answer
        ground_truth = sample.ground_truth
        
        if self.match_strategy == "exact":
            is_correct = AnswerMatcher.exact_match(predicted, ground_truth)
            similarity = 1.0 if is_correct else 0.0
        elif self.match_strategy == "contains":
            is_correct = AnswerMatcher.contains_match(predicted, ground_truth)
            similarity = 1.0 if is_correct else 0.0
        elif self.match_strategy == "fuzzy":
            is_correct = AnswerMatcher.fuzzy_match(predicted, ground_truth)
            similarity = 0.8 if is_correct else 0.0
        elif self.match_strategy == "semantic":
            is_correct, similarity = AnswerMatcher.semantic_match(
                predicted, ground_truth, self.slm.embed, self.semantic_threshold
            )
        else:
            raise ValueError(f"Unknown match strategy: {self.match_strategy}")
        
        return EvaluationResult(
            sample=sample,
            search_result=search_result,
            is_correct=is_correct,
            similarity_score=similarity,
        )
    
    def run_benchmark(
        self,
        agent: UncertaintyZoomAgent,
        tree: FractalTree,
        samples: List[BenchmarkSample],
        budget: int = 2048,
    ) -> Tuple[List[EvaluationResult], BenchmarkMetrics]:
        """
        Run full benchmark.
        
        Args:
            agent: The uncertainty zoom agent
            tree: The fractal tree to search
            samples: List of benchmark samples
            budget: Token budget per query
        
        Returns:
            Tuple of (evaluation results, metrics)
        """
        results = []
        
        for sample in samples:
            search_result = agent.search(sample.query, tree, budget)
            eval_result = self.evaluate_sample(sample, search_result)
            results.append(eval_result)
        
        metrics = self._compute_metrics(results, tree)
        return results, metrics
    
    def _compute_metrics(
        self,
        results: List[EvaluationResult],
        tree: FractalTree,
    ) -> BenchmarkMetrics:
        """Compute comprehensive metrics from results."""
        if not results:
            return BenchmarkMetrics()
        
        n = len(results)
        correct = sum(1 for r in results if r.is_correct)
        
        # Basic metrics
        accuracy = correct / n
        
        # Efficiency metrics
        total_tokens = sum(r.search_result.tokens_used for r in results)
        avg_tokens = total_tokens / n
        avg_expansions = sum(r.search_result.num_expansions for r in results) / n
        avg_depth = sum(r.search_result.depth_reached for r in results) / n
        
        # Token savings vs full context
        full_context_tokens = tree.total_tokens
        tokens_saved = (full_context_tokens - avg_tokens) / full_context_tokens
        
        # Early stopping rate
        early_stops = sum(
            1 for r in results 
            if r.search_result.depth_reached < tree.max_depth
        )
        early_stop_rate = early_stops / n
        
        # Confidence metrics
        avg_confidence = sum(r.search_result.confidence for r in results) / n
        
        # Calibration error (ECE)
        calibration_error = self._compute_ece(results)
        
        # Uncertainty-error correlation
        uncertainty_correlation = self._compute_uncertainty_correlation(results)
        
        # Timing
        avg_time = sum(r.search_result.wall_time_seconds for r in results) / n
        
        return BenchmarkMetrics(
            accuracy=accuracy,
            precision=accuracy,  # For single-answer QA, precision ≈ accuracy
            recall=accuracy,
            f1=accuracy,
            avg_tokens_used=avg_tokens,
            tokens_saved_vs_full=tokens_saved,
            avg_expansions=avg_expansions,
            avg_depth=avg_depth,
            early_stop_rate=early_stop_rate,
            avg_confidence=avg_confidence,
            calibration_error=calibration_error,
            uncertainty_error_correlation=uncertainty_correlation,
            avg_wall_time=avg_time,
            total_samples=n,
        )
    
    def _compute_ece(self, results: List[EvaluationResult]) -> float:
        """
        Compute Expected Calibration Error.
        
        This measures how well confidence predicts accuracy.
        Lower is better (perfect calibration = 0).
        """
        if not results:
            return 0.0
        
        # Bin by confidence
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        n = len(results)
        ece = 0.0
        
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            bin_results = [
                r for r in results
                if low <= r.search_result.confidence < high
            ]
            
            if bin_results:
                avg_confidence = sum(r.search_result.confidence for r in bin_results) / len(bin_results)
                accuracy = sum(1 for r in bin_results if r.is_correct) / len(bin_results)
                bin_weight = len(bin_results) / n
                ece += bin_weight * abs(accuracy - avg_confidence)
        
        return ece
    
    def _compute_uncertainty_correlation(self, results: List[EvaluationResult]) -> float:
        """
        Compute correlation between uncertainty and errors.
        
        Higher value means uncertainty is a good predictor of errors.
        """
        if len(results) < 2:
            return 0.0
        
        # Get final uncertainty for each result
        uncertainties = []
        errors = []
        
        for r in results:
            if r.search_result.uncertainty_trace:
                final_uncertainty = r.search_result.uncertainty_trace[-1]
            else:
                final_uncertainty = 1.0 - r.search_result.confidence
            
            uncertainties.append(final_uncertainty)
            errors.append(0.0 if r.is_correct else 1.0)
        
        # Compute Pearson correlation
        u_mean = sum(uncertainties) / len(uncertainties)
        e_mean = sum(errors) / len(errors)
        
        numerator = sum((u - u_mean) * (e - e_mean) for u, e in zip(uncertainties, errors))
        u_var = sum((u - u_mean) ** 2 for u in uncertainties)
        e_var = sum((e - e_mean) ** 2 for e in errors)
        
        denominator = (u_var ** 0.5) * (e_var ** 0.5)
        
        if denominator < 1e-8:
            return 0.0
        
        return numerator / denominator
    
    def compare_with_baseline(
        self,
        tree: FractalTree,
        samples: List[BenchmarkSample],
        budget: int = 2048,
        confidence_threshold: float = 0.75,
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Compare uncertainty-guided approach with baselines.
        
        Returns metrics for:
        - uncertainty_guided: Our approach
        - relevance_only: RAPTOR-style collapsed retrieval
        """
        # Our approach
        uncertainty_agent = UncertaintyZoomAgent(
            self.slm,
            confidence_threshold=confidence_threshold,
        )
        _, uncertainty_metrics = self.run_benchmark(
            uncertainty_agent, tree, samples, budget
        )
        
        # Baseline
        baseline_agent = RelevanceOnlyAgent(self.slm)
        baseline_results = []
        
        for sample in samples:
            search_result = baseline_agent.search(sample.query, tree, budget)
            eval_result = self.evaluate_sample(sample, search_result)
            baseline_results.append(eval_result)
        
        baseline_metrics = self._compute_metrics(baseline_results, tree)
        
        return {
            "uncertainty_guided": uncertainty_metrics,
            "relevance_only": baseline_metrics,
        }


def create_synthetic_benchmark(
    tree: FractalTree,
    num_samples: int = 20,
) -> List[BenchmarkSample]:
    """
    Create synthetic benchmark samples from a tree.
    
    Generates questions about specific content in leaf nodes.
    """
    leaves = tree.root.get_all_leaves()
    
    if not leaves:
        return []
    
    samples = []
    
    for i, leaf in enumerate(leaves[:num_samples]):
        # Create a question about the content
        content_preview = leaf.content[:100]
        
        sample = BenchmarkSample(
            query=f"What information is contained in section {i+1}?",
            ground_truth=content_preview,
            metadata={"leaf_id": leaf.id, "depth": leaf.depth},
        )
        samples.append(sample)
    
    return samples
