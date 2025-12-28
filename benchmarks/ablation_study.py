"""
Ablation Study: Comparing Uncertainty-Guided Retrieval with Existing Methodologies

This module implements a comprehensive ablation study comparing:
1. Our approach: Uncertainty-Guided Hierarchical Retrieval
2. Baseline 1: Flat RAG (standard vector similarity search)
3. Baseline 2: RAPTOR-style (collapsed tree retrieval)
4. Baseline 3: Relevance-only tree traversal (no uncertainty)

Ablations:
- With vs without uncertainty guidance
- With vs without information gain scoring
- With vs without early stopping
- Different confidence thresholds
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from src.fractal_tree import FractalTree, FractalNode, build_tree_from_chunks
from src.zoom_agent import UncertaintyZoomAgent, RelevanceOnlyAgent, SearchResult, SearchStatus
from src.uncertainty import UncertaintyEstimator, EntropyBasedStopping
from src.llm_interface import SLMInterface, MockSLMInterface


@dataclass
class AblationConfig:
    """Configuration for an ablation variant."""
    name: str
    description: str
    use_uncertainty: bool = True
    use_information_gain: bool = True
    use_early_stopping: bool = True
    confidence_threshold: float = 0.75
    use_tree_structure: bool = True  # False = flat RAG


@dataclass
class AblationResult:
    """Result from a single ablation run."""
    config_name: str
    query: str
    answer: str
    is_correct: bool
    confidence: float
    tokens_used: int
    tokens_budget: int
    depth_reached: int
    max_depth: int
    num_expansions: int
    wall_time: float
    early_stopped: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config_name,
            "query": self.query,
            "answer": self.answer,
            "correct": self.is_correct,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "token_efficiency": self.tokens_used / self.tokens_budget,
            "depth_reached": self.depth_reached,
            "expansions": self.num_expansions,
            "wall_time": self.wall_time,
            "early_stopped": self.early_stopped,
        }


@dataclass
class AblationSummary:
    """Summary statistics for an ablation configuration."""
    config_name: str
    num_samples: int
    accuracy: float
    avg_tokens_used: float
    token_savings: float  # vs full context
    avg_depth: float
    avg_expansions: float
    avg_wall_time: float
    early_stop_rate: float
    avg_confidence: float
    calibration_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config_name,
            "samples": self.num_samples,
            "accuracy": f"{self.accuracy:.1%}",
            "avg_tokens": f"{self.avg_tokens_used:.0f}",
            "token_savings": f"{self.token_savings:.1%}",
            "avg_depth": f"{self.avg_depth:.1f}",
            "avg_expansions": f"{self.avg_expansions:.1f}",
            "avg_time": f"{self.avg_wall_time:.2f}s",
            "early_stop_rate": f"{self.early_stop_rate:.1%}",
            "calibration_error": f"{self.calibration_error:.3f}",
        }


class FlatRAGAgent:
    """
    Baseline 1: Standard flat RAG without tree structure.
    
    Simply retrieves top-k chunks by similarity and generates answer.
    This is how most basic RAG systems work.
    """
    
    def __init__(self, slm: SLMInterface, top_k: int = 5):
        self.slm = slm
        self.top_k = top_k
    
    def search(
        self,
        query: str,
        tree: FractalTree,
        budget: int = 2048,
    ) -> SearchResult:
        """Search by retrieving top-k chunks (ignoring tree structure)."""
        start_time = time.time()
        
        # Get all leaf nodes (actual content chunks)
        leaves = tree.root.get_all_leaves()
        
        # Embed query
        query_emb = self.slm.embed(query)
        
        # Score all leaves by similarity
        scored_leaves = []
        for leaf in leaves:
            if leaf.embedding is not None:
                similarity = np.dot(query_emb, leaf.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(leaf.embedding) + 1e-8
                )
                scored_leaves.append((similarity, leaf))
        
        # Sort by similarity and take top-k
        scored_leaves.sort(reverse=True, key=lambda x: x[0])
        top_leaves = scored_leaves[:self.top_k]
        
        # Build context from top leaves
        context_parts = [leaf.content for _, leaf in top_leaves]
        context = "\n\n".join(context_parts)
        
        # Count tokens
        tokens_used = self.slm.count_tokens(context) + self.slm.count_tokens(query)
        
        # Generate answer
        prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        answer = self.slm.generate(prompt, max_tokens=100)
        
        wall_time = time.time() - start_time
        
        return SearchResult(
            answer=answer,
            confidence=0.5,  # No confidence estimation in flat RAG
            is_confident=False,
            status=SearchStatus.MAX_DEPTH_REACHED,
            tokens_used=min(tokens_used, budget),
            tokens_budget=budget,
            depth_reached=1,  # Flat, no depth
            max_depth=1,
            num_expansions=0,  # No expansion in flat RAG
            wall_time_seconds=wall_time,
            expansion_path=[],
            expansion_trace=[],
            uncertainty_trace=[],
            final_visible_nodes=[l.id for _, l in top_leaves],
        )


class NoEarlyStopAgent:
    """
    Ablation: Uncertainty-guided but without early stopping.
    
    Always traverses to maximum depth regardless of confidence.
    """
    
    def __init__(self, slm: SLMInterface, confidence_threshold: float = 0.75):
        self.slm = slm
        self.uncertainty_estimator = UncertaintyEstimator(
            confidence_threshold=confidence_threshold,
        )
        # Stopping criterion that never stops early
        self.stopping_criterion = EntropyBasedStopping(
            confidence_threshold=1.0,  # Never confident enough
            min_depth=999,  # Always go deeper
            patience=999,
        )
        self.max_expansions = 20
    
    def search(
        self,
        query: str,
        tree: FractalTree,
        budget: int = 2048,
    ) -> SearchResult:
        """Search without early stopping."""
        # Use the regular agent but with disabled stopping
        agent = UncertaintyZoomAgent(
            self.slm,
            confidence_threshold=1.0,  # Never confident
            min_depth=tree.max_depth,  # Must reach max
            max_expansions=self.max_expansions,
        )
        return agent.search(query, tree, budget)


class NoInfoGainAgent:
    """
    Ablation: Tree traversal without information gain scoring.
    
    Uses only relevance (similarity) to decide which nodes to expand.
    Similar to RAPTOR's approach but with tree navigation.
    """
    
    def __init__(self, slm: SLMInterface):
        self.slm = slm
        self.max_expansions = 10
    
    def search(
        self,
        query: str,
        tree: FractalTree,
        budget: int = 2048,
    ) -> SearchResult:
        """Search using only relevance, not information gain."""
        start_time = time.time()
        
        visible_nodes = [tree.root]
        expansion_path = []
        tokens_used = 0
        
        # Embed query once
        query_emb = self.slm.embed(query)
        
        for step in range(self.max_expansions):
            # Find expandable nodes
            expandable = [n for n in visible_nodes if not n.is_leaf and n.children]
            
            if not expandable:
                break
            
            # Score by RELEVANCE ONLY (no information gain)
            scored = []
            for node in expandable:
                if node.embedding is not None:
                    similarity = np.dot(query_emb, node.embedding) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-8
                    )
                    scored.append((similarity, node))
            
            if not scored:
                break
            
            # Expand most relevant node
            scored.sort(reverse=True, key=lambda x: x[0])
            best_node = scored[0][1]
            
            # Expand
            visible_nodes = [n for n in visible_nodes if n.id != best_node.id]
            visible_nodes.extend(best_node.children)
            expansion_path.append(best_node.id)
            
            # Check budget
            context_tokens = sum(self.slm.count_tokens(n.summary) for n in visible_nodes)
            if context_tokens > budget:
                break
            tokens_used = context_tokens
        
        # Generate answer
        context = "\n\n".join(n.summary for n in visible_nodes)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        answer = self.slm.generate(prompt, max_tokens=100)
        
        wall_time = time.time() - start_time
        
        return SearchResult(
            answer=answer,
            confidence=0.5,  # No uncertainty estimation
            is_confident=False,
            status=SearchStatus.MAX_DEPTH_REACHED,
            tokens_used=tokens_used,
            tokens_budget=budget,
            depth_reached=max((n.depth for n in visible_nodes), default=0),
            max_depth=tree.max_depth,
            num_expansions=len(expansion_path),
            wall_time_seconds=wall_time,
            expansion_path=expansion_path,
            expansion_trace=[],
            uncertainty_trace=[],
            final_visible_nodes=[n.id for n in visible_nodes],
        )


class AblationStudy:
    """
    Main class for running ablation studies.
    
    Compares different configurations systematically to isolate
    the contribution of each component.
    """
    
    # Default configurations for ablation
    DEFAULT_CONFIGS = [
        AblationConfig(
            name="Ours (Full)",
            description="Full uncertainty-guided with IG and early stopping",
            use_uncertainty=True,
            use_information_gain=True,
            use_early_stopping=True,
            use_tree_structure=True,
        ),
        AblationConfig(
            name="No Early Stop",
            description="Uncertainty-guided but must reach max depth",
            use_uncertainty=True,
            use_information_gain=True,
            use_early_stopping=False,
            use_tree_structure=True,
        ),
        AblationConfig(
            name="No Info Gain",
            description="Tree traversal with relevance-only scoring",
            use_uncertainty=False,
            use_information_gain=False,
            use_early_stopping=False,
            use_tree_structure=True,
        ),
        AblationConfig(
            name="Flat RAG",
            description="Standard RAG without tree structure",
            use_uncertainty=False,
            use_information_gain=False,
            use_early_stopping=False,
            use_tree_structure=False,
        ),
        AblationConfig(
            name="Relevance Only",
            description="RAPTOR-style collapsed tree retrieval",
            use_uncertainty=False,
            use_information_gain=False,
            use_early_stopping=False,
            use_tree_structure=True,
        ),
    ]
    
    def __init__(
        self,
        slm: SLMInterface,
        configs: Optional[List[AblationConfig]] = None,
    ):
        self.slm = slm
        self.configs = configs or self.DEFAULT_CONFIGS
        self.results: Dict[str, List[AblationResult]] = {c.name: [] for c in self.configs}
    
    def _create_agent(self, config: AblationConfig):
        """Create agent based on configuration."""
        if not config.use_tree_structure:
            return FlatRAGAgent(self.slm)
        
        if not config.use_uncertainty and not config.use_information_gain:
            if config.name == "Relevance Only":
                return RelevanceOnlyAgent(self.slm)
            return NoInfoGainAgent(self.slm)
        
        if not config.use_early_stopping:
            return NoEarlyStopAgent(self.slm, config.confidence_threshold)
        
        # Full approach
        return UncertaintyZoomAgent(
            self.slm,
            confidence_threshold=config.confidence_threshold,
        )
    
    def run_single_query(
        self,
        config: AblationConfig,
        query: str,
        ground_truth: str,
        tree: FractalTree,
        budget: int = 2048,
    ) -> AblationResult:
        """Run a single query with a specific configuration."""
        agent = self._create_agent(config)
        result = agent.search(query, tree, budget)
        
        # Check correctness (simple substring match)
        is_correct = ground_truth.lower() in result.answer.lower()
        
        # Determine if early stopped
        early_stopped = (
            result.status == SearchStatus.CONFIDENT and 
            result.depth_reached < result.max_depth
        )
        
        return AblationResult(
            config_name=config.name,
            query=query,
            answer=result.answer,
            is_correct=is_correct,
            confidence=result.confidence,
            tokens_used=result.tokens_used,
            tokens_budget=result.tokens_budget,
            depth_reached=result.depth_reached,
            max_depth=result.max_depth,
            num_expansions=result.num_expansions,
            wall_time=result.wall_time_seconds,
            early_stopped=early_stopped,
        )
    
    def run_study(
        self,
        queries: List[Tuple[str, str]],  # (query, ground_truth)
        tree: FractalTree,
        budget: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, AblationSummary]:
        """
        Run full ablation study across all configurations.
        
        Args:
            queries: List of (query, ground_truth) tuples
            tree: The document tree to search
            budget: Token budget per query
            verbose: Print progress
        
        Returns:
            Dictionary mapping config names to summary statistics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ABLATION STUDY: {len(self.configs)} configs × {len(queries)} queries")
            print(f"{'='*60}\n")
        
        for config in self.configs:
            if verbose:
                print(f"\n[{config.name}] {config.description}")
                print("-" * 40)
            
            for i, (query, ground_truth) in enumerate(queries):
                if verbose:
                    print(f"  Query {i+1}/{len(queries)}: {query[:50]}...")
                
                result = self.run_single_query(
                    config, query, ground_truth, tree, budget
                )
                self.results[config.name].append(result)
                
                if verbose:
                    status = "✓" if result.is_correct else "✗"
                    print(f"    {status} Tokens: {result.tokens_used}, "
                          f"Depth: {result.depth_reached}/{result.max_depth}, "
                          f"Time: {result.wall_time:.2f}s")
        
        # Compute summaries
        summaries = {}
        for config in self.configs:
            summaries[config.name] = self._compute_summary(
                config.name, self.results[config.name], tree.total_tokens
            )
        
        return summaries
    
    def _compute_summary(
        self,
        config_name: str,
        results: List[AblationResult],
        total_doc_tokens: int,
    ) -> AblationSummary:
        """Compute summary statistics for a configuration."""
        if not results:
            return AblationSummary(
                config_name=config_name,
                num_samples=0,
                accuracy=0.0,
                avg_tokens_used=0.0,
                token_savings=0.0,
                avg_depth=0.0,
                avg_expansions=0.0,
                avg_wall_time=0.0,
                early_stop_rate=0.0,
                avg_confidence=0.0,
                calibration_error=0.0,
            )
        
        n = len(results)
        correct = sum(1 for r in results if r.is_correct)
        early_stops = sum(1 for r in results if r.early_stopped)
        
        avg_tokens = sum(r.tokens_used for r in results) / n
        token_savings = (total_doc_tokens - avg_tokens) / total_doc_tokens
        
        # Compute calibration error (ECE)
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ece = 0.0
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            bin_results = [r for r in results if low <= r.confidence < high]
            if bin_results:
                avg_conf = sum(r.confidence for r in bin_results) / len(bin_results)
                accuracy = sum(1 for r in bin_results if r.is_correct) / len(bin_results)
                ece += (len(bin_results) / n) * abs(accuracy - avg_conf)
        
        return AblationSummary(
            config_name=config_name,
            num_samples=n,
            accuracy=correct / n,
            avg_tokens_used=avg_tokens,
            token_savings=token_savings,
            avg_depth=sum(r.depth_reached for r in results) / n,
            avg_expansions=sum(r.num_expansions for r in results) / n,
            avg_wall_time=sum(r.wall_time for r in results) / n,
            early_stop_rate=early_stops / n,
            avg_confidence=sum(r.confidence for r in results) / n,
            calibration_error=ece,
        )
    
    def print_comparison_table(self, summaries: Dict[str, AblationSummary]) -> str:
        """Generate a formatted comparison table."""
        headers = [
            "Method", "Accuracy", "Tokens", "Savings", 
            "Depth", "Early Stop", "ECE", "Time"
        ]
        
        rows = []
        for name, summary in summaries.items():
            rows.append([
                name,
                f"{summary.accuracy:.1%}",
                f"{summary.avg_tokens_used:.0f}",
                f"{summary.token_savings:.1%}",
                f"{summary.avg_depth:.1f}",
                f"{summary.early_stop_rate:.1%}",
                f"{summary.calibration_error:.3f}",
                f"{summary.avg_wall_time:.2f}s",
            ])
        
        # Calculate column widths
        col_widths = [max(len(h), max(len(r[i]) for r in rows)) + 2 
                      for i, h in enumerate(headers)]
        
        # Build table
        lines = []
        header_line = "│".join(h.center(w) for h, w in zip(headers, col_widths))
        separator = "┼".join("─" * w for w in col_widths)
        
        lines.append("┌" + "┬".join("─" * w for w in col_widths) + "┐")
        lines.append("│" + header_line + "│")
        lines.append("├" + separator + "┤")
        
        for row in rows:
            row_line = "│".join(cell.center(w) for cell, w in zip(row, col_widths))
            lines.append("│" + row_line + "│")
        
        lines.append("└" + "┴".join("─" * w for w in col_widths) + "┘")
        
        return "\n".join(lines)
    
    def save_results(self, path: str) -> None:
        """Save detailed results to JSON."""
        data = {
            "configs": [
                {"name": c.name, "description": c.description}
                for c in self.configs
            ],
            "results": {
                name: [r.to_dict() for r in results]
                for name, results in self.results.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def create_synthetic_qa_pairs(tree: FractalTree, num_pairs: int = 10) -> List[Tuple[str, str]]:
    """
    Create synthetic QA pairs from tree content for ablation study.
    
    Extracts key facts from leaves and generates questions about them.
    """
    leaves = tree.root.get_all_leaves()
    qa_pairs = []
    
    question_templates = [
        ("What is {}?", "describes"),
        ("How does {} work?", "method"),
        ("What are the benefits of {}?", "advantage"),
        ("What is the purpose of {}?", "purpose"),
    ]
    
    for i, leaf in enumerate(leaves[:num_pairs]):
        # Extract a fact from the content
        content = leaf.content[:200]
        
        # Simple heuristic: use first sentence as ground truth
        sentences = content.split('.')
        if sentences:
            ground_truth = sentences[0].strip()
            
            # Generate a generic question
            template_idx = i % len(question_templates)
            template, _ = question_templates[template_idx]
            
            # Extract subject (first few words)
            words = ground_truth.split()[:3]
            subject = " ".join(words)
            
            question = f"What information is in section {i+1} about {subject}?"
            qa_pairs.append((question, ground_truth))
    
    return qa_pairs


def run_quick_ablation(
    tree: FractalTree,
    slm: SLMInterface,
    num_queries: int = 5,
) -> Dict[str, AblationSummary]:
    """
    Run a quick ablation study for demonstration.
    
    Args:
        tree: Document tree to search
        slm: Language model interface
        num_queries: Number of test queries
    
    Returns:
        Summary statistics for each configuration
    """
    # Create synthetic QA pairs
    qa_pairs = create_synthetic_qa_pairs(tree, num_queries)
    
    if not qa_pairs:
        print("Warning: No QA pairs generated. Check tree structure.")
        return {}
    
    # Run study
    study = AblationStudy(slm)
    summaries = study.run_study(qa_pairs, tree, verbose=True)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(study.print_comparison_table(summaries))
    
    return summaries


if __name__ == "__main__":
    # Demo with mock SLM
    print("Running ablation study demo with mock SLM...")
    
    # Create sample tree
    chunks = [
        "The learning rate is 0.001 for neural network training.",
        "Batch size of 32 is commonly used in deep learning.",
        "Adam optimizer provides adaptive learning rates.",
        "Early stopping prevents overfitting during training.",
        "Dropout regularization randomly disables neurons.",
        "Cross-validation helps evaluate model performance.",
    ]
    
    def mock_summarizer(texts):
        return f"Summary of {len(texts)} sections about machine learning."
    
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
    
    # Run ablation
    slm = MockSLMInterface(embedding_dim=10)
    summaries = run_quick_ablation(tree, slm, num_queries=4)
    
    print("\nDone!")
