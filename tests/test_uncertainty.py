"""
Tests for uncertainty estimation module.
"""

import pytest
import numpy as np
from src.uncertainty import (
    UncertaintyEstimator,
    UncertaintyEstimate,
    InformationGainEstimate,
    EntropyBasedStopping,
)
from src.llm_interface import MockSLMInterface


class TestUncertaintyEstimate:
    """Tests for UncertaintyEstimate dataclass."""
    
    def test_creation(self):
        """Test estimate creation."""
        estimate = UncertaintyEstimate(
            entropy=1.5,
            confidence=0.7,
            answer="Test answer",
            token_entropies=[1.0, 1.5, 2.0],
            is_confident=True,
        )
        
        assert estimate.entropy == 1.5
        assert estimate.confidence == 0.7
        assert estimate.is_confident
    
    def test_normalized_entropy(self):
        """Test entropy normalization."""
        low = UncertaintyEstimate(entropy=0.5, confidence=0.9, answer="", token_entropies=[])
        high = UncertaintyEstimate(entropy=15.0, confidence=0.1, answer="", token_entropies=[])
        
        assert low.normalized_entropy < 0.1
        assert high.normalized_entropy == 1.0  # Capped at 1
    
    def test_to_dict(self):
        """Test serialization."""
        estimate = UncertaintyEstimate(
            entropy=1.5,
            confidence=0.7,
            answer="Answer",
            token_entropies=[1.0, 2.0],
        )
        
        data = estimate.to_dict()
        assert "entropy" in data
        assert "confidence" in data
        assert data["answer"] == "Answer"


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator class."""
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = UncertaintyEstimator(
            confidence_threshold=0.8,
            max_entropy_threshold=1.5,
        )
        
        assert estimator.confidence_threshold == 0.8
        assert estimator.max_entropy_threshold == 1.5
    
    def test_estimate_uncertainty(self):
        """Test uncertainty estimation with mock LLM."""
        slm = MockSLMInterface()
        estimator = UncertaintyEstimator()
        
        estimate = estimator.estimate_uncertainty(
            llm=slm,
            query="What is X?",
            context="X is a test value.",
            max_answer_tokens=50,
        )
        
        assert isinstance(estimate, UncertaintyEstimate)
        assert estimate.answer  # Should have some answer
        assert 0 <= estimate.confidence <= 1
        assert estimate.entropy >= 0
    
    def test_estimate_information_gain(self):
        """Test information gain estimation."""
        slm = MockSLMInterface()
        estimator = UncertaintyEstimator()
        
        ig = estimator.estimate_information_gain(
            llm=slm,
            query="What is the value?",
            current_context="Some context",
            node_summary="Node about values",
            node_id="node_1",
            child_diversity=0.5,
            current_uncertainty=2.0,
        )
        
        assert isinstance(ig, InformationGainEstimate)
        assert ig.node_id == "node_1"
        assert ig.expected_ig >= 0
        assert ig.current_uncertainty == 2.0
        assert ig.predicted_uncertainty <= ig.current_uncertainty
    
    def test_calibration_update(self):
        """Test calibration history updating."""
        estimator = UncertaintyEstimator()
        
        # Add some calibration samples
        estimator.update_calibration(0.9, True)
        estimator.update_calibration(0.9, True)
        estimator.update_calibration(0.3, False)
        
        assert len(estimator.calibration_history) == 3
    
    def test_calibration_error(self):
        """Test calibration error computation."""
        estimator = UncertaintyEstimator()
        
        # Not enough samples
        assert np.isnan(estimator.get_calibration_error())
        
        # Add sufficient samples
        for _ in range(10):
            estimator.update_calibration(0.9, True)
            estimator.update_calibration(0.1, False)
        
        ece = estimator.get_calibration_error()
        assert not np.isnan(ece)
        assert 0 <= ece <= 1


class TestInformationGainEstimate:
    """Tests for InformationGainEstimate."""
    
    def test_combined_score(self):
        """Test combined score computation."""
        ig = InformationGainEstimate(
            node_id="test",
            expected_ig=0.5,
            current_uncertainty=2.0,
            predicted_uncertainty=1.5,
            child_diversity=0.3,
            relevance_score=0.8,
        )
        
        # Combined = 0.5 * IG + 0.3 * relevance + 0.2 * surprise (default 0)
        expected = 0.5 * 0.5 + 0.3 * 0.8 + 0.2 * 0.0
        assert abs(ig.combined_score - expected) < 0.01


class TestEntropyBasedStopping:
    """Tests for EntropyBasedStopping."""
    
    def test_initialization(self):
        """Test stopping criterion initialization."""
        stopping = EntropyBasedStopping(
            confidence_threshold=0.8,
            entropy_threshold=1.0,
            min_depth=2,
            patience=3,
        )
        
        assert stopping.confidence_threshold == 0.8
        assert stopping.min_depth == 2
        assert stopping.patience == 3
    
    def test_min_depth_enforcement(self):
        """Test that stopping respects minimum depth."""
        stopping = EntropyBasedStopping(min_depth=2)
        
        confident_estimate = UncertaintyEstimate(
            entropy=0.1,
            confidence=0.99,
            answer="Very confident",
            token_entropies=[],
            is_confident=True,
        )
        
        # Should not stop at depth 0 or 1
        should_stop, _ = stopping.should_stop(confident_estimate, current_depth=0, budget_remaining=1000)
        assert not should_stop
        
        should_stop, _ = stopping.should_stop(confident_estimate, current_depth=1, budget_remaining=1000)
        assert not should_stop
    
    def test_budget_exhaustion(self):
        """Test stopping when budget exhausted."""
        stopping = EntropyBasedStopping()
        
        estimate = UncertaintyEstimate(entropy=5.0, confidence=0.3, answer="", token_entropies=[])
        
        should_stop, reason = stopping.should_stop(estimate, current_depth=5, budget_remaining=0)
        assert should_stop
        assert "budget" in reason.lower()
    
    def test_confidence_patience(self):
        """Test patience for confident answers."""
        stopping = EntropyBasedStopping(
            confidence_threshold=0.7,
            entropy_threshold=1.5,  # Threshold is 1.5
            min_depth=0,
            patience=3,
        )
        
        # Use entropy=1.0 which is above 0.5*1.5=0.75 (won't trigger very-low-entropy)
        confident = UncertaintyEstimate(
            entropy=1.0,  # Above 0.75 so won't trigger immediate stop
            confidence=0.8,  # Above threshold but below 0.95
            answer="Confident",
            token_entropies=[],
            is_confident=True,
        )
        
        # First two confident checks - should not stop yet (need patience=3)
        stopping.should_stop(confident, current_depth=1, budget_remaining=1000)
        should_stop, _ = stopping.should_stop(confident, current_depth=2, budget_remaining=1000)
        assert not should_stop
        
        # Third confident check - should stop now
        should_stop, reason = stopping.should_stop(confident, current_depth=3, budget_remaining=1000)
        assert should_stop
        assert "confident" in reason.lower()
    
    def test_reset(self):
        """Test state reset."""
        stopping = EntropyBasedStopping(patience=2, min_depth=0, entropy_threshold=1.5)
        
        confident = UncertaintyEstimate(
            entropy=1.0,  # Above very-low-entropy threshold
            confidence=0.8,  # Above confidence_threshold but below 0.95
            answer="", token_entropies=[], is_confident=True
        )
        
        # Build up confidence count
        stopping.should_stop(confident, 1, 1000)  # Count = 1
        
        # Reset
        stopping.reset()
        
        # After reset, count should be 0
        # First check after reset - should NOT stop (need to build patience again)
        should_stop, _ = stopping.should_stop(confident, 1, 1000)
        assert not should_stop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
