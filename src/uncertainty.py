"""
Uncertainty Estimation Module

This is the KEY INNOVATION of our approach. Unlike RAPTOR/MemTree that use
relevance scores, we estimate the SLM's epistemic uncertainty about the answer
given the current context.

Key Components:
1. Entropy Estimation: Token-level entropy of the answer distribution
2. Confidence Scoring: Probability of the top answer
3. Information Gain: Expected reduction in uncertainty from expanding a node
4. Calibration Metrics: How well confidence matches actual accuracy

The uncertainty estimator enables:
- Early stopping when confident
- Prioritizing expansion of high-information-gain nodes
- Interpretable uncertainty traces
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Protocol
import numpy as np
import math


class LLMInterface(Protocol):
    """Protocol for LLM interfaces that support uncertainty estimation."""
    
    def generate_with_logprobs(
        self, 
        prompt: str, 
        max_tokens: int = 100
    ) -> Tuple[str, List[float]]:
        """Generate response with token log probabilities."""
        ...
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        ...


@dataclass
class UncertaintyEstimate:
    """
    Container for uncertainty estimates about an answer.
    
    Attributes:
        entropy: Average token-level entropy (higher = more uncertain)
        confidence: Probability of the generated answer (lower entropy = higher confidence)
        answer: The generated answer text
        token_entropies: Per-token entropy values
        is_confident: Whether confidence exceeds threshold
    """
    entropy: float
    confidence: float
    answer: str
    token_entropies: List[float]
    is_confident: bool = False
    
    @property
    def normalized_entropy(self) -> float:
        """Entropy normalized to [0, 1] range (assuming max entropy ~ 10 nats)."""
        return min(1.0, self.entropy / 10.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entropy": self.entropy,
            "confidence": self.confidence,
            "answer": self.answer,
            "token_entropies": self.token_entropies,
            "is_confident": self.is_confident,
        }


@dataclass
class InformationGainEstimate:
    """
    Estimate of information gain from expanding a node.
    
    Attributes:
        node_id: ID of the node being evaluated
        expected_ig: Expected information gain
        current_uncertainty: Uncertainty before expansion
        predicted_uncertainty: Predicted uncertainty after expansion
        child_diversity: Semantic diversity of children
        relevance_score: Base relevance to the query
        surprise_score: Novelty relative to current context (Titans logic)
    """
    node_id: str
    expected_ig: float
    current_uncertainty: float
    predicted_uncertainty: float
    child_diversity: float
    relevance_score: float
    surprise_score: float = 0.0
    
    @property
    def combined_score(self) -> float:
        """Combined score for ranking nodes."""
        # Weighted combination: Relevance + Info Gain + Surprise (Titans logic)
        # Surprise acts as a "curiosity" bonus for novel information
        return (
            0.5 * self.expected_ig + 
            0.3 * self.relevance_score + 
            0.2 * self.surprise_score
        )


class UncertaintyEstimator:
    """
    Estimates epistemic uncertainty of an SLM's answers.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_entropy_threshold: float = 2.0,
        use_calibration: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_entropy_threshold = max_entropy_threshold
        self.use_calibration = use_calibration
        self.calibration_history: List[Tuple[float, bool]] = []
    
    def estimate_uncertainty(
        self,
        llm: LLMInterface,
        query: str,
        context: str,
        max_answer_tokens: int = 100,
    ) -> UncertaintyEstimate:
        # Construct prompt
        prompt = self._build_qa_prompt(query, context)
        
        # Generate with log probabilities
        answer, log_probs = llm.generate_with_logprobs(prompt, max_answer_tokens)
        
        if not log_probs:
            return UncertaintyEstimate(
                entropy=5.0,
                confidence=0.5,
                answer=answer,
                token_entropies=[],
                is_confident=False,
            )
        
        token_entropies = [-lp for lp in log_probs]
        avg_entropy = sum(token_entropies) / len(token_entropies)
        
        avg_log_prob = sum(log_probs) / len(log_probs)
        confidence = math.exp(avg_log_prob)
        
        if self.use_calibration and len(self.calibration_history) >= 10:
            confidence = self._apply_calibration(confidence)
        
        is_confident = (
            confidence >= self.confidence_threshold and 
            avg_entropy <= self.max_entropy_threshold
        )
        
        return UncertaintyEstimate(
            entropy=avg_entropy,
            confidence=confidence,
            answer=answer,
            token_entropies=token_entropies,
            is_confident=is_confident,
        )
    
    def estimate_information_gain(
        self,
        llm: LLMInterface,
        query: str,
        current_context: str,
        node_summary: str,
        node_id: str,
        child_diversity: float,
        current_uncertainty: float,
    ) -> InformationGainEstimate:
        """
        Estimate information gain with Titans-inspired 'Surprise' metric.
        """
        # 1. Relevance: How well does this match the query?
        relevance = self._compute_relevance(llm, query, node_summary)
        
        # 2. Surprise (Titans logic): How different is this from what we already see?
        # High surprise = High potential for new information
        surprise = self._compute_surprise(llm, current_context, node_summary)
        
        # 3. Reduction Factor: Unifies Relevance, Diversity, and Surprise
        # We want nodes that are Relevant AND (Diverse OR Surprising)
        reduction_factor = relevance * (0.4 + 0.3 * child_diversity + 0.3 * surprise)
        
        predicted_uncertainty = current_uncertainty * (1.0 - reduction_factor * 0.4)
        expected_ig = max(0.0, current_uncertainty - predicted_uncertainty)
        
        return InformationGainEstimate(
            node_id=node_id,
            expected_ig=expected_ig,
            current_uncertainty=current_uncertainty,
            predicted_uncertainty=predicted_uncertainty,
            child_diversity=child_diversity,
            relevance_score=relevance,
            surprise_score=surprise,
        )
    
    def _compute_surprise(
        self,
        llm: LLMInterface,
        context: str,
        node_summary: str,
    ) -> float:
        """
        Compute 'Surprise' metric based on semantic dissimilarity to context.
        
        Inspired by Titans/Neural Memory: We attend to what is surprising (novel).
        If a node is very similar to current context, it has low surprise (redundant).
        If it is very different, it has high surprise (novel).
        """
        if not context.strip():
            return 1.0  # Everything is surprising if context is empty
            
        context_emb = llm.embed(context[:500])  # Embed partial context
        node_emb = llm.embed(node_summary)
        
        dot_product = np.dot(context_emb, node_emb)
        norm = np.linalg.norm(context_emb) * np.linalg.norm(node_emb)
        cosine_sim = dot_product / (norm + 1e-8)
        
        # Surprise is the inverse of similarity
        # Map similarity [-1, 1] to surprise [0, 1]
        # Sim 1.0 -> Surprise 0.0
        # Sim 0.0 -> Surprise 0.5
        # Sim -1.0 -> Surprise 1.0
        return 1.0 - (cosine_sim + 1.0) / 2.0
    
    def _compute_relevance(
        self,
        llm: LLMInterface,
        query: str,
        text: str,
    ) -> float:
        """
        Compute relevance of text to query using embedding similarity.
        
        Returns value in [0, 1].
        """
        query_emb = llm.embed(query)
        text_emb = llm.embed(text)
        
        # Cosine similarity
        similarity = np.dot(query_emb, text_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(text_emb) + 1e-8
        )
        
        # Convert to [0, 1] range
        return (similarity + 1.0) / 2.0
    
    def _build_qa_prompt(self, query: str, context: str) -> str:
        """Build the QA prompt."""
        return f"""Given the following context, answer the question. If you're not sure, say "I don't know".

Context:
{context}

Question: {query}

Answer:"""
    
    def _apply_calibration(self, raw_confidence: float) -> float:
        """
        Apply calibration correction based on historical accuracy.
        
        If the model tends to be overconfident, we reduce confidence.
        If underconfident, we increase it.
        """
        if not self.calibration_history:
            return raw_confidence
        
        # Bin confidences and compute actual accuracy per bin
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            if low <= raw_confidence < high:
                # Find historical samples in this bin
                bin_samples = [
                    (conf, correct) 
                    for conf, correct in self.calibration_history
                    if low <= conf < high
                ]
                
                if len(bin_samples) >= 5:
                    actual_accuracy = sum(c for _, c in bin_samples) / len(bin_samples)
                    expected_confidence = (low + high) / 2
                    
                    # Adjust towards actual accuracy
                    calibration_factor = actual_accuracy / (expected_confidence + 1e-8)
                    return min(1.0, raw_confidence * calibration_factor)
        
        return raw_confidence
    
    def update_calibration(self, confidence: float, was_correct: bool) -> None:
        """
        Update calibration history with a new sample.
        
        Args:
            confidence: The confidence that was predicted
            was_correct: Whether the answer was actually correct
        """
        self.calibration_history.append((confidence, was_correct))
        
        # Keep only recent history
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-500:]
    
    def get_calibration_error(self) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures how well confidence matches actual accuracy.
        Lower is better.
        """
        if len(self.calibration_history) < 10:
            return float('nan')
        
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        total_samples = len(self.calibration_history)
        ece = 0.0
        
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            bin_samples = [
                (conf, correct)
                for conf, correct in self.calibration_history
                if low <= conf < high
            ]
            
            if bin_samples:
                avg_confidence = sum(c for c, _ in bin_samples) / len(bin_samples)
                accuracy = sum(c for _, c in bin_samples) / len(bin_samples)
                bin_weight = len(bin_samples) / total_samples
                ece += bin_weight * abs(accuracy - avg_confidence)
        
        return ece


class EntropyBasedStopping:
    """
    Determines when to stop the search based on uncertainty.
    
    Unlike fixed-depth stopping, this enables early termination
    when the model is confident, saving tokens.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.75,
        entropy_threshold: float = 1.5,
        min_depth: int = 1,
        patience: int = 2,
    ):
        """
        Initialize stopping criteria.
        
        Args:
            confidence_threshold: Stop if confidence exceeds this
            entropy_threshold: Stop if entropy is below this
            min_depth: Minimum depth before allowing early stop
            patience: Number of consecutive confident answers before stopping
        """
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.min_depth = min_depth
        self.patience = patience
        
        self._consecutive_confident = 0
    
    def should_stop(
        self,
        estimate: UncertaintyEstimate,
        current_depth: int,
        budget_remaining: int,
    ) -> Tuple[bool, str]:
        """
        Determine if search should stop.
        
        Args:
            estimate: Current uncertainty estimate
            current_depth: Current depth in tree
            budget_remaining: Remaining token budget
        
        Returns:
            Tuple of (should_stop, reason)
        """
        # Never stop before minimum depth
        if current_depth < self.min_depth:
            self._consecutive_confident = 0
            return False, "Below minimum depth"
        
        # Stop if budget exhausted
        if budget_remaining <= 0:
            return True, "Budget exhausted"
        
        # Check confidence
        if estimate.is_confident:
            self._consecutive_confident += 1
            
            if self._consecutive_confident >= self.patience:
                return True, f"Confident for {self.patience} consecutive checks"
        else:
            self._consecutive_confident = 0
        
        # Additional stopping conditions
        if estimate.entropy < self.entropy_threshold * 0.5:
            return True, "Very low entropy"
        
        if estimate.confidence > 0.95:
            return True, "Very high confidence"
        
        return False, "Continue search"
    
    def reset(self) -> None:
        """Reset state for new search."""
        self._consecutive_confident = 0
