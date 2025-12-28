"""
Enhanced Evaluation Pipeline with Semantic Similarity Matching

This module provides:
1. Semantic similarity-based answer matching (not just substring)
2. Tuned parameters for better early stopping
3. Multi-metric evaluation (F1, ROUGE-L, Semantic Similarity)
4. Proper QA dataset generation with varied difficulty
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import re

from src.fractal_tree import FractalTree, FractalNode
from src.zoom_agent import UncertaintyZoomAgent, SearchResult, SearchStatus
from src.llm_interface import SLMInterface


@dataclass
class QAPair:
    """A question-answer pair for evaluation."""
    question: str
    ground_truth: str
    difficulty: str  # "easy", "medium", "hard"
    source_node_id: str = ""
    category: str = "factual"  # "factual", "reasoning", "summary"


@dataclass 
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Accuracy metrics
    exact_match: float
    semantic_similarity: float
    f1_score: float
    rouge_l: float
    
    # Efficiency metrics
    avg_tokens_used: int
    token_savings_pct: float
    
    # Our novel metrics
    early_stop_rate: float
    avg_depth_reached: float
    calibration_error: float  # ECE
    uncertainty_correlation: float  # correlation between uncertainty and errors
    
    # Timing
    avg_wall_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact_match": f"{self.exact_match:.1%}",
            "semantic_similarity": f"{self.semantic_similarity:.1%}",
            "f1_score": f"{self.f1_score:.1%}",
            "rouge_l": f"{self.rouge_l:.1%}",
            "avg_tokens": self.avg_tokens_used,
            "token_savings": f"{self.token_savings_pct:.1%}",
            "early_stop_rate": f"{self.early_stop_rate:.1%}",
            "avg_depth": f"{self.avg_depth_reached:.1f}",
            "calibration_error": f"{self.calibration_error:.3f}",
            "uncertainty_correlation": f"{self.uncertainty_correlation:.3f}",
            "avg_time": f"{self.avg_wall_time:.2f}s",
        }


class SemanticMatcher:
    """
    Semantic similarity-based answer matching.
    
    Much better than substring matching for evaluating LLM responses.
    """
    
    def __init__(self, slm: SLMInterface, threshold: float = 0.7):
        self.slm = slm
        self.threshold = threshold
    
    def compute_similarity(self, pred: str, gold: str) -> float:
        """Compute semantic similarity between predicted and gold answer."""
        if not pred or not gold:
            return 0.0
        
        pred_emb = self.slm.embed(pred.lower().strip())
        gold_emb = self.slm.embed(gold.lower().strip())
        
        # Cosine similarity
        similarity = np.dot(pred_emb, gold_emb) / (
            np.linalg.norm(pred_emb) * np.linalg.norm(gold_emb) + 1e-8
        )
        
        # Normalize to [0, 1]
        return max(0.0, (similarity + 1) / 2)
    
    def is_match(self, pred: str, gold: str) -> bool:
        """Check if prediction matches ground truth semantically."""
        return self.compute_similarity(pred, gold) >= self.threshold
    
    def compute_f1(self, pred: str, gold: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = pred_tokens & gold_tokens
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def compute_rouge_l(self, pred: str, gold: str) -> float:
        """Compute ROUGE-L (longest common subsequence) score."""
        pred_words = pred.lower().split()
        gold_words = gold.lower().split()
        
        if not pred_words or not gold_words:
            return 0.0
        
        # LCS using dynamic programming
        m, n = len(pred_words), len(gold_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i-1] == gold_words[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        precision = lcs_length / m if m > 0 else 0
        recall = lcs_length / n if n > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


def create_enhanced_qa_dataset(
    tree: FractalTree,
    num_pairs: int = 20,
) -> List[QAPair]:
    """
    Create QA pairs with varied difficulty levels.
    
    Easy: Direct facts from leaves (early stop should work)
    Medium: Requires combining 2-3 nodes
    Hard: Requires deep traversal or reasoning
    """
    leaves = tree.root.get_all_leaves()
    qa_pairs = []
    
    # Easy questions - direct facts (should enable early stopping)
    for i, leaf in enumerate(leaves[:num_pairs // 3]):
        content = leaf.content.strip()
        if len(content) < 20:
            continue
        
        # Extract first significant phrase
        sentences = content.split('.')
        if sentences:
            fact = sentences[0].strip()
            if len(fact) > 10:
                # Generate question about this fact
                words = fact.split()[:4]
                subject = " ".join(words)
                
                qa_pairs.append(QAPair(
                    question=f"What does the document say about {subject}?",
                    ground_truth=fact,
                    difficulty="easy",
                    source_node_id=leaf.id,
                    category="factual",
                ))
    
    # Medium questions - require some context
    for i, leaf in enumerate(leaves[num_pairs // 3: 2 * num_pairs // 3]):
        content = leaf.content.strip()
        
        # Find key terms
        words = [w for w in content.split() if len(w) > 5][:3]
        if words:
            key_term = words[0] if words else "this topic"
            
            qa_pairs.append(QAPair(
                question=f"Explain the concept of {key_term} based on the document.",
                ground_truth=content[:200],
                difficulty="medium",
                source_node_id=leaf.id,
                category="summary",
            ))
    
    # Hard questions - require traversal
    if tree.root.children:
        for i, section in enumerate(tree.root.children[:num_pairs // 3]):
            all_content = " ".join(
                child.content for child in section.get_all_leaves()
            )[:300]
            
            if all_content:
                qa_pairs.append(QAPair(
                    question=f"Summarize section {i+1} of the document.",
                    ground_truth=all_content[:200],
                    difficulty="hard",
                    source_node_id=section.id,
                    category="reasoning",
                ))
    
    return qa_pairs[:num_pairs]


# Tuned parameters for better early stopping
TUNED_PARAMS = {
    # Lower confidence threshold to enable more early stops
    "confidence_threshold": 0.5,  # Was 0.75
    
    # Higher entropy threshold for stopping
    "entropy_threshold": 2.5,  # Was 1.5
    
    # Lower minimum depth for easy questions
    "min_depth": 0,  # Was 1
    
    # Faster patience for confident answers
    "patience": 1,  # Was 2
    
    # Limit expansions to save tokens
    "max_expansions": 8,  # Was 10
}


class TunedZoomAgent(UncertaintyZoomAgent):
    """
    Uncertainty Zoom Agent with tuned parameters for better early stopping.
    """
    
    def __init__(self, slm: SLMInterface):
        super().__init__(
            slm,
            confidence_threshold=TUNED_PARAMS["confidence_threshold"],
            entropy_threshold=TUNED_PARAMS["entropy_threshold"],
            min_depth=TUNED_PARAMS["min_depth"],
            patience=TUNED_PARAMS["patience"],
            max_expansions=TUNED_PARAMS["max_expansions"],
        )


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline with semantic matching.
    """
    
    def __init__(self, slm: SLMInterface, similarity_threshold: float = 0.6):
        self.slm = slm
        self.matcher = SemanticMatcher(slm, similarity_threshold)
    
    def evaluate(
        self,
        agent: UncertaintyZoomAgent,
        tree: FractalTree,
        qa_pairs: List[QAPair],
        budget: int = 2048,
        verbose: bool = True,
    ) -> Tuple[List[Dict], EvaluationMetrics]:
        """
        Run comprehensive evaluation.
        
        Returns:
            Tuple of (detailed results, summary metrics)
        """
        results = []
        
        for i, qa in enumerate(qa_pairs):
            if verbose:
                print(f"  [{i+1}/{len(qa_pairs)}] {qa.difficulty}: {qa.question[:40]}...")
            
            start_time = time.time()
            search_result = agent.search(qa.question, tree, budget)
            wall_time = time.time() - start_time
            
            # Compute all similarity metrics
            semantic_sim = self.matcher.compute_similarity(
                search_result.answer, qa.ground_truth
            )
            f1 = self.matcher.compute_f1(search_result.answer, qa.ground_truth)
            rouge_l = self.matcher.compute_rouge_l(search_result.answer, qa.ground_truth)
            exact_match = qa.ground_truth.lower() in search_result.answer.lower()
            
            # Check for early stop
            early_stopped = (
                search_result.status == SearchStatus.CONFIDENT and
                search_result.depth_reached < search_result.max_depth
            )
            
            results.append({
                "question": qa.question,
                "ground_truth": qa.ground_truth[:100],
                "predicted": search_result.answer[:100],
                "difficulty": qa.difficulty,
                "semantic_sim": semantic_sim,
                "f1": f1,
                "rouge_l": rouge_l,
                "exact_match": exact_match,
                "confidence": search_result.confidence,
                "tokens_used": search_result.tokens_used,
                "depth_reached": search_result.depth_reached,
                "max_depth": search_result.max_depth,
                "early_stopped": early_stopped,
                "wall_time": wall_time,
            })
            
            if verbose:
                status = "✓ Early!" if early_stopped else ("✓" if semantic_sim > 0.6 else "✗")
                print(f"      {status} sim={semantic_sim:.2f}, depth={search_result.depth_reached}/{search_result.max_depth}")
        
        # Compute summary metrics
        metrics = self._compute_metrics(results, tree.total_tokens, budget)
        
        return results, metrics
    
    def _compute_metrics(
        self,
        results: List[Dict],
        total_doc_tokens: int,
        budget: int,
    ) -> EvaluationMetrics:
        """Compute summary metrics from results."""
        n = len(results)
        if n == 0:
            return EvaluationMetrics(
                exact_match=0, semantic_similarity=0, f1_score=0, rouge_l=0,
                avg_tokens_used=0, token_savings_pct=0,
                early_stop_rate=0, avg_depth_reached=0,
                calibration_error=0, uncertainty_correlation=0,
                avg_wall_time=0,
            )
        
        # Accuracy metrics
        exact_match = sum(1 for r in results if r["exact_match"]) / n
        semantic_sim = sum(r["semantic_sim"] for r in results) / n
        f1 = sum(r["f1"] for r in results) / n
        rouge_l = sum(r["rouge_l"] for r in results) / n
        
        # Efficiency
        avg_tokens = sum(r["tokens_used"] for r in results) / n
        token_savings = (budget - avg_tokens) / budget
        
        # Novel metrics
        early_stops = sum(1 for r in results if r["early_stopped"])
        early_stop_rate = early_stops / n
        avg_depth = sum(r["depth_reached"] for r in results) / n
        
        # Calibration error (ECE)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ece = 0.0
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            bin_results = [r for r in results if low <= r["confidence"] < high]
            if bin_results:
                avg_conf = sum(r["confidence"] for r in bin_results) / len(bin_results)
                accuracy = sum(1 for r in bin_results if r["semantic_sim"] > 0.6) / len(bin_results)
                ece += (len(bin_results) / n) * abs(accuracy - avg_conf)
        
        # Uncertainty-error correlation
        confidences = [r["confidence"] for r in results]
        correct = [1 if r["semantic_sim"] > 0.6 else 0 for r in results]
        
        if len(set(confidences)) > 1 and len(set(correct)) > 1:
            correlation = np.corrcoef(confidences, correct)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        avg_time = sum(r["wall_time"] for r in results) / n
        
        return EvaluationMetrics(
            exact_match=exact_match,
            semantic_similarity=semantic_sim,
            f1_score=f1,
            rouge_l=rouge_l,
            avg_tokens_used=int(avg_tokens),
            token_savings_pct=token_savings,
            early_stop_rate=early_stop_rate,
            avg_depth_reached=avg_depth,
            calibration_error=ece,
            uncertainty_correlation=correlation,
            avg_wall_time=avg_time,
        )
    
    def evaluate_by_difficulty(
        self,
        results: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """Break down metrics by question difficulty."""
        by_difficulty = {}
        
        for difficulty in ["easy", "medium", "hard"]:
            subset = [r for r in results if r["difficulty"] == difficulty]
            if subset:
                by_difficulty[difficulty] = {
                    "count": len(subset),
                    "semantic_sim": sum(r["semantic_sim"] for r in subset) / len(subset),
                    "early_stop_rate": sum(1 for r in subset if r["early_stopped"]) / len(subset),
                    "avg_depth": sum(r["depth_reached"] for r in subset) / len(subset),
                    "avg_tokens": sum(r["tokens_used"] for r in subset) / len(subset),
                }
        
        return by_difficulty


def run_tuned_evaluation(
    tree: FractalTree,
    slm: SLMInterface,
    num_pairs: int = 10,
    budget: int = 2048,
) -> Tuple[EvaluationMetrics, Dict[str, Dict]]:
    """
    Run evaluation with tuned parameters.
    
    Returns metrics and breakdown by difficulty.
    """
    print("\n" + "="*60)
    print("TUNED EVALUATION WITH SEMANTIC MATCHING")
    print("="*60)
    
    # Create QA dataset
    qa_pairs = create_enhanced_qa_dataset(tree, num_pairs)
    print(f"\nGenerated {len(qa_pairs)} QA pairs:")
    for difficulty in ["easy", "medium", "hard"]:
        count = sum(1 for q in qa_pairs if q.difficulty == difficulty)
        print(f"  - {difficulty}: {count}")
    
    # Create tuned agent
    agent = TunedZoomAgent(slm)
    
    print(f"\nTuned Parameters:")
    for key, value in TUNED_PARAMS.items():
        print(f"  - {key}: {value}")
    
    # Run evaluation
    print("\nRunning evaluation...")
    pipeline = EvaluationPipeline(slm, similarity_threshold=0.6)
    results, metrics = pipeline.evaluate(agent, tree, qa_pairs, budget)
    
    # Get breakdown by difficulty
    by_difficulty = pipeline.evaluate_by_difficulty(results)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nOverall Metrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\nBy Difficulty:")
    for diff, stats in by_difficulty.items():
        print(f"\n  {diff.upper()}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
    
    return metrics, by_difficulty


if __name__ == "__main__":
    from src.llm_interface import MockSLMInterface
    from src.fractal_tree import build_tree_from_chunks
    
    print("Testing enhanced evaluation pipeline...")
    
    # Create sample tree
    chunks = [
        "Machine learning uses algorithms to learn patterns from data.",
        "The learning rate of 0.001 controls how fast weights are updated.",
        "Batch size determines how many samples are processed together.",
        "Adam optimizer combines momentum and RMSprop for better convergence.",
        "Early stopping prevents overfitting by monitoring validation loss.",
        "Dropout randomly disables neurons during training for regularization.",
    ]
    
    def mock_summarizer(texts):
        return f"Summary of {len(texts)} ML concepts."
    
    def mock_embedder(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(10)
    
    tree = build_tree_from_chunks(
        chunks=chunks,
        summarizer=mock_summarizer,
        embedder=mock_embedder,
        token_counter=lambda x: len(x) // 4,
        max_children=2,
        document_name="ML Tutorial",
    )
    
    slm = MockSLMInterface(embedding_dim=10)
    metrics, by_difficulty = run_tuned_evaluation(tree, slm, num_pairs=6)
    
    print("\nDone!")
