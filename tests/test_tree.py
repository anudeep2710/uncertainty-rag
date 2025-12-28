"""
Tests for fractal tree data structure.
"""

import pytest
import numpy as np
from src.fractal_tree import (
    FractalNode,
    FractalTree,
    NodeType,
    UncertaintyMetadata,
    build_tree_from_chunks,
    compute_child_diversity,
)


class TestFractalNode:
    """Tests for FractalNode class."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = FractalNode(
            id="test_node",
            node_type=NodeType.LEAF,
            content="Test content here",
            parent_id="some_parent",  # Give it a parent so it's not root
        )
        
        assert node.id == "test_node"
        assert node.node_type == NodeType.LEAF
        assert node.content == "Test content here"
        assert node.is_leaf
        assert not node.is_root  # Now correctly not root
    
    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        node = FractalNode(
            id="",
            node_type=NodeType.LEAF,
            content="Some content",
        )
        
        assert node.id != ""
        assert "leaf" in node.id
    
    def test_is_leaf_property(self):
        """Test is_leaf correctly identifies leaf nodes."""
        leaf = FractalNode(id="leaf", node_type=NodeType.LEAF, content="Leaf")
        parent = FractalNode(
            id="parent",
            node_type=NodeType.SECTION,
            content="Parent",
            children=[leaf],
        )
        
        assert leaf.is_leaf
        assert not parent.is_leaf
    
    def test_get_all_leaves(self):
        """Test recursive leaf collection."""
        leaf1 = FractalNode(id="l1", node_type=NodeType.LEAF, content="Leaf 1")
        leaf2 = FractalNode(id="l2", node_type=NodeType.LEAF, content="Leaf 2")
        leaf3 = FractalNode(id="l3", node_type=NodeType.LEAF, content="Leaf 3")
        
        subsection = FractalNode(
            id="sub",
            node_type=NodeType.SUBSECTION,
            content="Subsection",
            children=[leaf2, leaf3],
        )
        
        root = FractalNode(
            id="root",
            node_type=NodeType.ROOT,
            content="Root",
            children=[leaf1, subsection],
        )
        
        leaves = root.get_all_leaves()
        assert len(leaves) == 3
        assert all(l.is_leaf for l in leaves)
    
    def test_serialization_roundtrip(self):
        """Test to_dict and from_dict preserve data."""
        original = FractalNode(
            id="test",
            node_type=NodeType.SECTION,
            content="Test content",
            summary="Test summary",
            embedding=np.array([1.0, 2.0, 3.0]),
            depth=2,
            token_count=10,
            metadata=UncertaintyMetadata(
                child_diversity=0.5,
                expansion_count=3,
            ),
        )
        
        data = original.to_dict()
        restored = FractalNode.from_dict(data)
        
        assert restored.id == original.id
        assert restored.node_type == original.node_type
        assert restored.content == original.content
        assert restored.summary == original.summary
        assert np.allclose(restored.embedding, original.embedding)
        assert restored.metadata.child_diversity == original.metadata.child_diversity


class TestFractalTree:
    """Tests for FractalTree class."""
    
    def test_tree_creation(self):
        """Test basic tree creation."""
        root = FractalNode(id="root", node_type=NodeType.ROOT, content="Root")
        tree = FractalTree(root=root, document_name="Test Doc")
        
        assert tree.root == root
        assert tree.document_name == "Test Doc"
        assert "root" in tree.node_index
    
    def test_node_index(self):
        """Test node index is built correctly."""
        leaf1 = FractalNode(id="l1", node_type=NodeType.LEAF, content="L1")
        leaf2 = FractalNode(id="l2", node_type=NodeType.LEAF, content="L2")
        root = FractalNode(
            id="root",
            node_type=NodeType.ROOT,
            content="Root",
            children=[leaf1, leaf2],
        )
        
        tree = FractalTree(root=root)
        
        assert len(tree.node_index) == 3
        assert tree.get_node("l1") == leaf1
        assert tree.get_node("l2") == leaf2
        assert tree.get_node("root") == root
    
    def test_get_leaf_count(self):
        """Test leaf counting."""
        leaf1 = FractalNode(id="l1", node_type=NodeType.LEAF, content="L1")
        leaf2 = FractalNode(id="l2", node_type=NodeType.LEAF, content="L2")
        section = FractalNode(
            id="sec",
            node_type=NodeType.SECTION,
            content="Section",
            children=[leaf1, leaf2],
        )
        root = FractalNode(
            id="root",
            node_type=NodeType.ROOT,
            content="Root",
            children=[section],
        )
        
        tree = FractalTree(root=root)
        assert tree.get_leaf_count() == 2
    
    def test_tree_visualization(self):
        """Test tree visualization generates output."""
        leaf = FractalNode(id="l", node_type=NodeType.LEAF, content="Leaf")
        root = FractalNode(
            id="root",
            node_type=NodeType.ROOT,
            content="Root",
            children=[leaf],
        )
        
        tree = FractalTree(root=root)
        viz = tree.visualize()
        
        assert "root" in viz.lower()
        assert "leaf" in viz.lower()


class TestTreeBuilding:
    """Tests for tree building functions."""
    
    def test_compute_child_diversity(self):
        """Test child diversity computation."""
        # Identical embeddings = 0 diversity
        same = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        assert compute_child_diversity(same) < 0.1
        
        # Orthogonal embeddings = 1 diversity
        different = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        assert compute_child_diversity(different) > 0.9
        
        # Single embedding = 0 diversity
        single = [np.array([1.0, 0.0])]
        assert compute_child_diversity(single) == 0.0
    
    def test_build_tree_from_chunks(self):
        """Test tree building from chunks."""
        chunks = [
            "First chunk of content.",
            "Second chunk of content.",
            "Third chunk of content.",
            "Fourth chunk of content.",
        ]
        
        def mock_summarizer(texts):
            return f"Summary of {len(texts)} texts"
        
        def mock_embedder(text):
            # Deterministic mock embedding
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(10)
        
        def mock_counter(text):
            return len(text) // 4
        
        tree = build_tree_from_chunks(
            chunks=chunks,
            summarizer=mock_summarizer,
            embedder=mock_embedder,
            token_counter=mock_counter,
            max_children=2,
        )
        
        assert tree.get_leaf_count() == 4
        assert tree.max_depth >= 2
        assert tree.root.node_type == NodeType.ROOT
    
    def test_build_tree_single_chunk(self):
        """Test tree building with single chunk."""
        chunks = ["Single chunk"]
        
        tree = build_tree_from_chunks(
            chunks=chunks,
            summarizer=lambda x: "Summary",
            embedder=lambda x: np.zeros(10),
            token_counter=lambda x: len(x),
            max_children=4,
        )
        
        # Single chunk should still create a valid tree
        assert tree.root is not None
    
    def test_build_tree_empty_raises(self):
        """Test that empty chunks raise error."""
        with pytest.raises(ValueError):
            build_tree_from_chunks(
                chunks=[],
                summarizer=lambda x: "Summary",
                embedder=lambda x: np.zeros(10),
                token_counter=lambda x: len(x),
            )


class TestUncertaintyMetadata:
    """Tests for UncertaintyMetadata."""
    
    def test_update_success(self):
        """Test success rate updating."""
        meta = UncertaintyMetadata()
        
        assert meta.success_rate == 0.0
        assert meta.visit_count == 0
        
        # Update with success
        meta.update_success(True)
        assert meta.visit_count == 1
        assert meta.success_rate > 0.0
        
        # Update with failure
        meta.update_success(False)
        assert meta.visit_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
