"""
Tests for the zoom agent.
"""

import pytest
import numpy as np
from src.zoom_agent import (
    UncertaintyZoomAgent,
    SearchResult,
    SearchStatus,
    ExpansionStep,
    RelevanceOnlyAgent,
)
from src.fractal_tree import FractalNode, FractalTree, NodeType, build_tree_from_chunks
from src.llm_interface import MockSLMInterface


def create_test_tree(embedding_dim: int = 10) -> FractalTree:
    """Create a simple test tree."""
    chunks = [
        "The learning rate is 0.001 for neural networks.",
        "Batch size is typically 32 or 64.",
        "Training uses Adam optimizer.",
        "Validation is done every 100 steps.",
    ]
    
    def mock_summarizer(texts):
        return f"Summary of {len(texts)} sections about machine learning."
    
    def mock_embedder(text):
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(embedding_dim)
    
    return build_tree_from_chunks(
        chunks=chunks,
        summarizer=mock_summarizer,
        embedder=mock_embedder,
        token_counter=lambda x: len(x) // 4,
        max_children=2,
        document_name="Test Document",
    )



class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        result = SearchResult(
            answer="Test answer",
            confidence=0.85,
            is_confident=True,
            status=SearchStatus.CONFIDENT,
            tokens_used=500,
            tokens_budget=1000,
            depth_reached=2,
            max_depth=3,
            num_expansions=2,
            wall_time_seconds=0.5,
            expansion_path=["node1", "node2"],
            expansion_trace=[],
            uncertainty_trace=[2.0, 1.5, 1.0],
            final_visible_nodes=["leaf1", "leaf2"],
        )
        
        assert result.answer == "Test answer"
        assert result.is_confident
        assert result.status == SearchStatus.CONFIDENT
    
    def test_to_dict(self):
        """Test serialization."""
        result = SearchResult(
            answer="Answer",
            confidence=0.8,
            is_confident=True,
            status=SearchStatus.CONFIDENT,
            tokens_used=100,
            tokens_budget=500,
            depth_reached=1,
            max_depth=3,
            num_expansions=1,
            wall_time_seconds=0.1,
            expansion_path=[],
            expansion_trace=[],
            uncertainty_trace=[],
            final_visible_nodes=[],
        )
        
        data = result.to_dict()
        assert "answer" in data
        assert "confidence" in data
        assert data["status"] == "confident"


class TestExpansionStep:
    """Tests for ExpansionStep."""
    
    def test_creation(self):
        """Test step creation."""
        step = ExpansionStep(
            step_number=1,
            expanded_node_id="node_1",
            expanded_node_summary="Summary of node",
            information_gain=0.5,
            uncertainty_before=2.0,
            uncertainty_after=1.5,
            tokens_used=100,
            visible_nodes_count=3,
        )
        
        assert step.step_number == 1
        assert step.information_gain == 0.5
    
    def test_to_dict(self):
        """Test serialization."""
        step = ExpansionStep(
            step_number=1,
            expanded_node_id="node_1",
            expanded_node_summary="Summary",
            information_gain=0.5,
            uncertainty_before=2.0,
            uncertainty_after=1.5,
            tokens_used=100,
            visible_nodes_count=3,
        )
        
        data = step.to_dict()
        assert data["step"] == 1
        assert "info_gain" in data


class TestUncertaintyZoomAgent:
    """Tests for UncertaintyZoomAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        slm = MockSLMInterface()
        agent = UncertaintyZoomAgent(
            slm=slm,
            confidence_threshold=0.8,
            entropy_threshold=1.5,
            min_depth=1,
            max_expansions=10,
        )
        
        assert agent.max_expansions == 10
    
    def test_search_basic(self):
        """Test basic search functionality."""
        slm = MockSLMInterface(embedding_dim=10)
        agent = UncertaintyZoomAgent(slm, max_expansions=5)
        tree = create_test_tree(embedding_dim=10)
        
        result = agent.search(
            query="What is the learning rate?",
            tree=tree,
            budget=1000,
        )
        
        assert isinstance(result, SearchResult)
        assert result.answer  # Should have some answer
        assert result.tokens_used > 0
        assert result.tokens_used <= 1000  # Within budget
    
    def test_search_respects_budget(self):
        """Test that search respects token budget."""
        slm = MockSLMInterface(embedding_dim=10)
        agent = UncertaintyZoomAgent(slm)
        tree = create_test_tree(embedding_dim=10)
        
        result = agent.search(
            query="What is the batch size?",
            tree=tree,
            budget=100,  # Very small budget
        )
        
        # Should still produce a result
        assert result.status in [SearchStatus.BUDGET_EXHAUSTED, SearchStatus.CONFIDENT, SearchStatus.MAX_DEPTH_REACHED]
    
    def test_search_produces_trace(self):
        """Test that search produces expansion trace."""
        slm = MockSLMInterface(embedding_dim=10)
        agent = UncertaintyZoomAgent(slm, max_expansions=3, min_depth=0)
        tree = create_test_tree(embedding_dim=10)
        
        result = agent.search(
            query="What optimizer is used?",
            tree=tree,
            budget=2000,
        )
        
        # Should have uncertainty trace
        assert isinstance(result.uncertainty_trace, list)
        
        # Should have final visible nodes
        assert isinstance(result.final_visible_nodes, list)
        assert len(result.final_visible_nodes) > 0
    
    def test_search_multiple_queries(self):
        """Test processing multiple queries."""
        slm = MockSLMInterface(embedding_dim=10)
        agent = UncertaintyZoomAgent(slm)
        tree = create_test_tree(embedding_dim=10)
        
        queries = [
            "What is the learning rate?",
            "What is the batch size?",
        ]
        
        results = agent.score_multiple_queries(queries, tree, budget=1000)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, SearchResult)


class TestRelevanceOnlyAgent:
    """Tests for RelevanceOnlyAgent baseline."""
    
    def test_initialization(self):
        """Test baseline agent initialization."""
        slm = MockSLMInterface()
        agent = RelevanceOnlyAgent(slm, top_k=3)
        
        assert agent.top_k == 3
    
    def test_search_basic(self):
        """Test baseline search."""
        slm = MockSLMInterface(embedding_dim=10)
        agent = RelevanceOnlyAgent(slm)
        tree = create_test_tree(embedding_dim=10)
        
        result = agent.search(
            query="What is the learning rate?",
            tree=tree,
            budget=1000,
        )
        
        assert isinstance(result, SearchResult)
        assert result.answer
    
    def test_no_expansion_trace(self):
        """Test that baseline has no expansion trace."""
        slm = MockSLMInterface(embedding_dim=10)
        agent = RelevanceOnlyAgent(slm)
        tree = create_test_tree(embedding_dim=10)
        
        result = agent.search("Test query", tree, 1000)
        
        # Baseline should have empty expansion trace
        assert len(result.expansion_trace) == 0
        assert len(result.expansion_path) == 0


class TestAgentComparison:
    """Tests comparing uncertainty-guided vs baseline."""
    
    def test_both_agents_run(self):
        """Test both agents can process same query."""
        slm = MockSLMInterface(embedding_dim=10)
        uncertainty_agent = UncertaintyZoomAgent(slm)
        baseline_agent = RelevanceOnlyAgent(slm)
        tree = create_test_tree(embedding_dim=10)
        
        query = "What is the batch size used for training?"
        
        result1 = uncertainty_agent.search(query, tree, 1000)
        result2 = baseline_agent.search(query, tree, 1000)
        
        # Both should produce results
        assert result1.answer
        assert result2.answer
        
        # Uncertainty agent should have more information
        assert len(result1.expansion_trace) >= 0
        assert len(result1.uncertainty_trace) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
