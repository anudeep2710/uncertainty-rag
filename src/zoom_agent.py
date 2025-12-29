"""
Uncertainty-Guided Zoom Agent

This is the MAIN ALGORITHM implementing uncertainty-guided hierarchical retrieval.

Key differences from RAPTOR/MemTree:
1. Uses INFORMATION GAIN (not relevance) for node selection
2. Uses UNCERTAINTY-BASED early stopping (not fixed depth)
3. Produces INTERPRETABLE uncertainty traces
4. Operates under STRICT TOKEN BUDGET constraints

The algorithm:
1. Start with root node visible
2. Score each expandable node by information gain
3. Expand highest IG node (replace with children)
4. Prune low-relevance nodes to respect budget
5. Estimate uncertainty; if confident, attempt answer
6. Repeat until confident or budget exhausted
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import time

from .fractal_tree import FractalNode, FractalTree, NodeType
from .uncertainty import (
    UncertaintyEstimator, 
    UncertaintyEstimate, 
    InformationGainEstimate,
    EntropyBasedStopping,
)
from .llm_interface import SLMInterface


class SearchStatus(Enum):
    """Status of the search."""
    IN_PROGRESS = "in_progress"
    CONFIDENT = "confident"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_DEPTH_REACHED = "max_depth_reached"


@dataclass
class ExpansionStep:
    """Record of a single expansion step."""
    step_number: int
    expanded_node_id: str
    expanded_node_summary: str
    information_gain: float
    uncertainty_before: float
    uncertainty_after: float
    tokens_used: int
    visible_nodes_count: int
    surprise_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step_number,
            "expanded": self.expanded_node_id,
            "summary": self.expanded_node_summary[:100],
            "info_gain": round(self.information_gain, 4),
            "surprise": round(self.surprise_score, 4),
            "uncertainty_before": round(self.uncertainty_before, 4),
            "uncertainty_after": round(self.uncertainty_after, 4),
            "tokens": self.tokens_used,
            "visible_nodes": self.visible_nodes_count,
        }


@dataclass
class SearchResult:
    """
    Result of an uncertainty-guided search.
    
    Contains the answer, confidence, and full trace of the search process.
    """
    # Core result
    answer: str
    confidence: float
    is_confident: bool
    
    # Search statistics
    status: SearchStatus
    tokens_used: int
    tokens_budget: int
    depth_reached: int
    max_depth: int
    num_expansions: int
    wall_time_seconds: float
    
    # Interpretability - the key novelty
    expansion_path: List[str]  # Node IDs in order of expansion
    expansion_trace: List[ExpansionStep]  # Detailed trace
    uncertainty_trace: List[float]  # Uncertainty at each step
    final_visible_nodes: List[str]  # What's visible at the end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "is_confident": self.is_confident,
            "status": self.status.value,
            "tokens_used": self.tokens_used,
            "tokens_budget": self.tokens_budget,
            "depth_reached": self.depth_reached,
            "max_depth": self.max_depth,
            "num_expansions": self.num_expansions,
            "wall_time_seconds": round(self.wall_time_seconds, 3),
            "expansion_path": self.expansion_path,
            "expansion_trace": [e.to_dict() for e in self.expansion_trace],
            "uncertainty_trace": [round(u, 4) for u in self.uncertainty_trace],
            "final_visible_nodes": self.final_visible_nodes,
        }
    
    def format_expansion_path(self) -> str:
        """Format expansion path for display."""
        if not self.expansion_trace:
            return "No expansions made"
        
        parts = []
        for step in self.expansion_trace:
            parts.append(f"{step.step_number}. {step.expanded_node_summary[:50]}... (IG={step.information_gain:.3f})")
        return "\n".join(parts)


class UncertaintyZoomAgent:
    """
    The main agent implementing uncertainty-guided hierarchical retrieval.
    
    This is the core contribution of the research. Instead of:
    - RAPTOR: Collapse tree → similarity search → answer
    - MemTree: Semantic matching → expand → answer
    
    We do:
    - Measure uncertainty → compute information gain → expand highest IG → 
    - check if confident → repeat or stop early
    
    Key innovations:
    1. Information-theoretic node selection
    2. Calibration-aware confidence thresholds
    3. Interpretable uncertainty traces
    4. Provable efficiency bounds (under assumptions)
    """
    
    def __init__(
        self,
        slm: SLMInterface,
        confidence_threshold: float = 0.75,
        entropy_threshold: float = 1.5,
        min_depth: int = 1,
        patience: int = 2,
        max_expansions: int = 20,
    ):
        """
        Initialize the zoom agent.
        
        Args:
            slm: The small language model interface
            confidence_threshold: Stop if confidence exceeds this
            entropy_threshold: Stop if entropy below this
            min_depth: Minimum depth before allowing early stop
            patience: Consecutive confident answers before stopping
            max_expansions: Maximum number of expansion steps
        """
        self.slm = slm
        self.uncertainty_estimator = UncertaintyEstimator(
            confidence_threshold=confidence_threshold,
            max_entropy_threshold=entropy_threshold,
        )
        self.stopping_criterion = EntropyBasedStopping(
            confidence_threshold=confidence_threshold,
            entropy_threshold=entropy_threshold,
            min_depth=min_depth,
            patience=patience,
        )
        self.max_expansions = max_expansions
    
    def search(
        self,
        query: str,
        tree: FractalTree,
        budget: int = 2048,
        confidence_threshold: Optional[float] = None,
    ) -> SearchResult:
        """
        Perform uncertainty-guided search.
        
        Args:
            query: The user's question
            tree: The fractal tree to search
            budget: Maximum token budget
            confidence_threshold: Override default confidence threshold
        
        Returns:
            SearchResult with answer, confidence, and full trace
        """
        start_time = time.time()
        
        # Initialize state
        visible_nodes: List[FractalNode] = [tree.root]
        expansion_path: List[str] = []
        expansion_trace: List[ExpansionStep] = []
        uncertainty_trace: List[float] = []
        
        tokens_used = self.slm.count_tokens(tree.root.content)
        step = 0
        current_depth = 0
        status = SearchStatus.IN_PROGRESS
        last_estimate: Optional[UncertaintyEstimate] = None
        
        self.stopping_criterion.reset()
        
        # Main search loop
        while step < self.max_expansions:
            # Build context from visible nodes
            context = self._build_context(visible_nodes)
            context_tokens = self.slm.count_tokens(context)
            
            # Check budget
            if context_tokens > budget:
                # Prune to fit budget
                visible_nodes = self._prune_to_budget(query, visible_nodes, budget)
                context = self._build_context(visible_nodes)
            
            # Estimate uncertainty
            estimate = self.uncertainty_estimator.estimate_uncertainty(
                self.slm, query, context
            )
            uncertainty_trace.append(estimate.entropy)
            last_estimate = estimate
            
            # Check stopping criteria
            should_stop, stop_reason = self.stopping_criterion.should_stop(
                estimate, current_depth, budget - context_tokens
            )
            
            if should_stop:
                if "Confident" in stop_reason or "entropy" in stop_reason.lower():
                    status = SearchStatus.CONFIDENT
                else:
                    status = SearchStatus.BUDGET_EXHAUSTED
                break
            
            # Find expandable nodes (non-leaf nodes in visible set)
            expandable = [n for n in visible_nodes if not n.is_leaf]
            
            if not expandable:
                status = SearchStatus.MAX_DEPTH_REACHED
                break
            
            # Score nodes by information gain
            ig_scores = []
            for node in expandable:
                ig = self.uncertainty_estimator.estimate_information_gain(
                    llm=self.slm,
                    query=query,
                    current_context=context,
                    node_summary=node.summary,
                    node_id=node.id,
                    child_diversity=node.metadata.child_diversity,
                    current_uncertainty=estimate.entropy,
                )
                ig_scores.append((node, ig))
            
            # Select node with highest information gain
            best_node, best_ig = max(ig_scores, key=lambda x: x[1].combined_score)
            
            # Record pre-expansion state
            pre_expansion_uncertainty = estimate.entropy
            
            # Expand: replace node with its children
            visible_nodes = self._expand_node(visible_nodes, best_node)
            current_depth = max(n.depth for n in visible_nodes)
            
            # Update expansion tracking
            expansion_path.append(best_node.id)
            best_node.metadata.expansion_count += 1
            
            # Estimate post-expansion uncertainty for trace
            new_context = self._build_context(visible_nodes)
            post_estimate = self.uncertainty_estimator.estimate_uncertainty(
                self.slm, query, new_context
            )
            
            # Record expansion step
            tokens_this_step = self.slm.count_tokens(new_context) - context_tokens
            tokens_used += max(0, tokens_this_step)
            
            expansion_trace.append(ExpansionStep(
                step_number=step + 1,
                expanded_node_id=best_node.id,
                expanded_node_summary=best_node.summary,
                information_gain=best_ig.expected_ig,
                uncertainty_before=pre_expansion_uncertainty,
                uncertainty_after=post_estimate.entropy,
                tokens_used=tokens_used,
                visible_nodes_count=len(visible_nodes),
                surprise_score=best_ig.surprise_score,
            ))
            
            step += 1
        
        # Final answer generation
        final_context = self._build_context(visible_nodes)
        if last_estimate is None:
            last_estimate = self.uncertainty_estimator.estimate_uncertainty(
                self.slm, query, final_context
            )
        
        wall_time = time.time() - start_time
        
        return SearchResult(
            answer=last_estimate.answer,
            confidence=last_estimate.confidence,
            is_confident=last_estimate.is_confident,
            status=status,
            tokens_used=tokens_used,
            tokens_budget=budget,
            depth_reached=current_depth,
            max_depth=tree.max_depth,
            num_expansions=step,
            wall_time_seconds=wall_time,
            expansion_path=expansion_path,
            expansion_trace=expansion_trace,
            uncertainty_trace=uncertainty_trace,
            final_visible_nodes=[n.id for n in visible_nodes],
        )
    
    def _build_context(self, nodes: List[FractalNode]) -> str:
        """Build context string from visible nodes."""
        parts = []
        
        # Sort by depth (higher level context first)
        sorted_nodes = sorted(nodes, key=lambda n: (n.depth, n.id))
        
        for node in sorted_nodes:
            prefix = "  " * node.depth
            if node.is_leaf:
                parts.append(f"{prefix}[Content] {node.content}")
            else:
                parts.append(f"{prefix}[Summary] {node.summary}")
        
        return "\n".join(parts)
    
    def _expand_node(
        self, 
        visible: List[FractalNode], 
        node: FractalNode
    ) -> List[FractalNode]:
        """
        Expand a node by replacing it with its children.
        
        Args:
            visible: Current visible nodes
            node: Node to expand
        
        Returns:
            Updated visible nodes list
        """
        result = []
        for n in visible:
            if n.id == node.id:
                # Replace with children
                result.extend(node.children)
            else:
                result.append(n)
        return result
    
    def _prune_to_budget(
        self,
        query: str,
        nodes: List[FractalNode],
        budget: int,
    ) -> List[FractalNode]:
        """
        Prune nodes to fit within token budget.
        
        Strategy: Remove lowest relevance nodes first, but keep at least root context.
        """
        if not nodes:
            return nodes
        
        # Compute relevance for each node
        query_emb = self.slm.embed(query)
        scored_nodes = []
        
        for node in nodes:
            if node.embedding is not None:
                import numpy as np
                similarity = np.dot(query_emb, node.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-8
                )
            else:
                similarity = 0.0
            
            # Boost score for root/high-level nodes
            depth_boost = 1.0 / (1.0 + node.depth)
            scored_nodes.append((node, similarity + depth_boost))
        
        # Sort by score descending
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Add nodes until budget exceeded
        result = []
        total_tokens = 0
        
        for node, score in scored_nodes:
            node_tokens = node.token_count or self.slm.count_tokens(node.content)
            if total_tokens + node_tokens <= budget:
                result.append(node)
                total_tokens += node_tokens
            elif not result:
                # Always include at least one node
                result.append(node)
                break
        
        return result
    
    def score_multiple_queries(
        self,
        queries: List[str],
        tree: FractalTree,
        budget: int = 2048,
    ) -> List[SearchResult]:
        """
        Process multiple queries, learning from each for calibration.
        
        Args:
            queries: List of queries
            tree: The fractal tree
            budget: Token budget per query
        
        Returns:
            List of SearchResults
        """
        results = []
        
        for query in queries:
            result = self.search(query, tree, budget)
            results.append(result)
        
        return results


class RelevanceOnlyAgent:
    """
    Baseline agent using only relevance (like RAPTOR collapsed tree).
    
    Used for comparison to show the benefit of uncertainty-guided approach.
    """
    
    def __init__(self, slm: SLMInterface, top_k: int = 5):
        """Initialize baseline agent."""
        self.slm = slm
        self.top_k = top_k
    
    def search(
        self,
        query: str,
        tree: FractalTree,
        budget: int = 2048,
    ) -> SearchResult:
        """
        Search using relevance-only approach (RAPTOR-style collapsed retrieval).
        """
        start_time = time.time()
        
        # Get all nodes (collapsed tree)
        all_nodes = list(tree.node_index.values())
        
        # Score by relevance
        query_emb = self.slm.embed(query)
        scored = []
        
        for node in all_nodes:
            if node.embedding is not None:
                import numpy as np
                sim = np.dot(query_emb, node.embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-8
                )
                scored.append((node, sim))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Take top-k that fit budget
        selected = []
        total_tokens = 0
        
        for node, score in scored[:self.top_k * 2]:
            tokens = node.token_count or self.slm.count_tokens(node.content)
            if total_tokens + tokens <= budget:
                selected.append(node)
                total_tokens += tokens
        
        # Build context and generate answer
        context = "\n".join(n.content for n in selected)
        
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.slm.generate(prompt)
        
        return SearchResult(
            answer=answer,
            confidence=0.5,  # No confidence estimation
            is_confident=False,
            status=SearchStatus.BUDGET_EXHAUSTED,
            tokens_used=total_tokens,
            tokens_budget=budget,
            depth_reached=0,
            max_depth=tree.max_depth,
            num_expansions=0,
            wall_time_seconds=time.time() - start_time,
            expansion_path=[],
            expansion_trace=[],
            uncertainty_trace=[],
            final_visible_nodes=[n.id for n in selected],
        )
