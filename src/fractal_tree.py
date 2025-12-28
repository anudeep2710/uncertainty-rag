"""
Fractal Tree Data Structure

Implements the hierarchical document representation with:
- FractalNode: Individual tree nodes with content, summary, and metadata
- FractalTree: Container for the full tree structure
- Build utilities for constructing trees from document chunks

The key innovation here is tracking uncertainty-related metadata at each node:
- Child diversity (semantic spread of children)
- Expansion history
- Information density estimates
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import hashlib
import json
import numpy as np


class NodeType(Enum):
    """Type of node in the fractal tree."""
    ROOT = "root"           # Global summary of entire document
    SECTION = "section"     # Summary of a document section
    SUBSECTION = "subsection"  # Summary of a subsection
    LEAF = "leaf"           # Actual text chunk


@dataclass
class UncertaintyMetadata:
    """Metadata for uncertainty-guided traversal."""
    
    # Semantic diversity of children (higher = more varied content)
    child_diversity: float = 0.0
    
    # Estimated information density (higher = more specific content)
    information_density: float = 0.0
    
    # Number of times this node was expanded during searches
    expansion_count: int = 0
    
    # Average relevance score when this node was visited
    avg_relevance_when_visited: float = 0.0
    
    # Number of times visiting this node led to correct answer
    success_rate: float = 0.0
    visit_count: int = 0
    
    def update_success(self, was_successful: bool) -> None:
        """Update success rate after a search."""
        self.visit_count += 1
        # Exponential moving average
        alpha = 0.1
        self.success_rate = alpha * float(was_successful) + (1 - alpha) * self.success_rate


@dataclass
class FractalNode:
    """
    A node in the fractal tree representing a document segment at some level of abstraction.
    
    Attributes:
        id: Unique identifier for the node
        node_type: Type of node (root, section, subsection, leaf)
        content: The actual text content (summary for non-leaf, original text for leaf)
        summary: Condensed summary (same as content for non-leaf nodes)
        embedding: Vector representation for similarity computations
        children: List of child nodes (empty for leaves)
        parent_id: ID of parent node (None for root)
        depth: Depth in tree (0 for root)
        token_count: Number of tokens in content
        metadata: Uncertainty-related metadata
        source_info: Optional source information (page numbers, section titles, etc.)
    """
    
    id: str
    node_type: NodeType
    content: str
    summary: str = ""
    embedding: Optional[np.ndarray] = None
    children: List[FractalNode] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    token_count: int = 0
    metadata: UncertaintyMetadata = field(default_factory=UncertaintyMetadata)
    source_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()
        if not self.summary:
            self.summary = self.content[:200] + "..." if len(self.content) > 200 else self.content
    
    def _generate_id(self) -> str:
        """Generate unique ID based on content hash."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.node_type.value}_{self.depth}_{content_hash}"
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    @property
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent_id is None
    
    @property
    def child_count(self) -> int:
        """Number of direct children."""
        return len(self.children)
    
    def get_descendant_count(self) -> int:
        """Get total number of descendants (recursive)."""
        count = len(self.children)
        for child in self.children:
            count += child.get_descendant_count()
        return count
    
    def get_all_leaves(self) -> List[FractalNode]:
        """Get all leaf nodes under this node."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        return leaves
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "summary": self.summary,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "children": [c.to_dict() for c in self.children],
            "parent_id": self.parent_id,
            "depth": self.depth,
            "token_count": self.token_count,
            "metadata": {
                "child_diversity": self.metadata.child_diversity,
                "information_density": self.metadata.information_density,
                "expansion_count": self.metadata.expansion_count,
                "avg_relevance_when_visited": self.metadata.avg_relevance_when_visited,
                "success_rate": self.metadata.success_rate,
                "visit_count": self.metadata.visit_count,
            },
            "source_info": self.source_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FractalNode:
        """Create node from dictionary."""
        metadata = UncertaintyMetadata(
            child_diversity=data.get("metadata", {}).get("child_diversity", 0.0),
            information_density=data.get("metadata", {}).get("information_density", 0.0),
            expansion_count=data.get("metadata", {}).get("expansion_count", 0),
            avg_relevance_when_visited=data.get("metadata", {}).get("avg_relevance_when_visited", 0.0),
            success_rate=data.get("metadata", {}).get("success_rate", 0.0),
            visit_count=data.get("metadata", {}).get("visit_count", 0),
        )
        
        node = cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            summary=data.get("summary", ""),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            children=[],  # Will be populated below
            parent_id=data.get("parent_id"),
            depth=data.get("depth", 0),
            token_count=data.get("token_count", 0),
            metadata=metadata,
            source_info=data.get("source_info", {}),
        )
        
        # Recursively create children
        for child_data in data.get("children", []):
            child = cls.from_dict(child_data)
            child.parent_id = node.id
            node.children.append(child)
        
        return node


@dataclass
class FractalTree:
    """
    Container for the complete fractal tree representation of a document.
    
    Attributes:
        root: The root node of the tree
        document_name: Name/title of the source document
        total_tokens: Total tokens across all leaf nodes
        max_depth: Maximum depth of the tree
        node_index: Index mapping node IDs to nodes for fast lookup
    """
    
    root: FractalNode
    document_name: str = ""
    total_tokens: int = 0
    max_depth: int = 0
    node_index: Dict[str, FractalNode] = field(default_factory=dict)
    
    def __post_init__(self):
        """Build the node index after initialization."""
        self._build_index()
        self._compute_stats()
    
    def _build_index(self) -> None:
        """Build index mapping node IDs to nodes."""
        self.node_index = {}
        self._index_node(self.root)
    
    def _index_node(self, node: FractalNode) -> None:
        """Recursively add node to index."""
        self.node_index[node.id] = node
        for child in node.children:
            self._index_node(child)
    
    def _compute_stats(self) -> None:
        """Compute tree statistics."""
        self.max_depth = self._get_max_depth(self.root)
        self.total_tokens = sum(leaf.token_count for leaf in self.root.get_all_leaves())
    
    def _get_max_depth(self, node: FractalNode) -> int:
        """Get maximum depth from a node."""
        if node.is_leaf:
            return node.depth
        return max(self._get_max_depth(child) for child in node.children)
    
    def get_node(self, node_id: str) -> Optional[FractalNode]:
        """Get node by ID."""
        return self.node_index.get(node_id)
    
    def get_nodes_at_depth(self, depth: int) -> List[FractalNode]:
        """Get all nodes at a specific depth."""
        return [node for node in self.node_index.values() if node.depth == depth]
    
    def get_leaf_count(self) -> int:
        """Get number of leaf nodes."""
        return len(self.root.get_all_leaves())
    
    def get_siblings(self, node: FractalNode) -> List[FractalNode]:
        """Get sibling nodes (nodes with same parent)."""
        if node.parent_id is None:
            return []
        parent = self.get_node(node.parent_id)
        if parent is None:
            return []
        return [child for child in parent.children if child.id != node.id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary for serialization."""
        return {
            "root": self.root.to_dict(),
            "document_name": self.document_name,
            "total_tokens": self.total_tokens,
            "max_depth": self.max_depth,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FractalTree:
        """Create tree from dictionary."""
        root = FractalNode.from_dict(data["root"])
        return cls(
            root=root,
            document_name=data.get("document_name", ""),
        )
    
    def save(self, path: str) -> None:
        """Save tree to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> FractalTree:
        """Load tree from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def visualize(self, max_content_length: int = 50) -> str:
        """Generate ASCII visualization of tree structure."""
        lines = []
        self._visualize_node(self.root, "", True, lines, max_content_length)
        return "\n".join(lines)
    
    def _visualize_node(
        self, 
        node: FractalNode, 
        prefix: str, 
        is_last: bool, 
        lines: List[str],
        max_content_length: int
    ) -> None:
        """Recursively build visualization."""
        connector = "└── " if is_last else "├── "
        content_preview = node.summary[:max_content_length]
        if len(node.summary) > max_content_length:
            content_preview += "..."
        
        lines.append(f"{prefix}{connector}[{node.node_type.value}] {content_preview}")
        
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            self._visualize_node(
                child, 
                child_prefix, 
                i == len(node.children) - 1,
                lines,
                max_content_length
            )


def compute_child_diversity(embeddings: List[np.ndarray]) -> float:
    """
    Compute semantic diversity of a set of embeddings.
    
    Higher diversity means children cover more varied topics,
    which is useful for information gain estimation.
    
    Uses average pairwise cosine distance as the metric.
    """
    if len(embeddings) < 2:
        return 0.0
    
    # Normalize embeddings
    normalized = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]
    
    # Compute pairwise cosine similarities
    n = len(normalized)
    total_distance = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = np.dot(normalized[i], normalized[j])
            distance = 1.0 - similarity  # Convert to distance
            total_distance += distance
            count += 1
    
    return total_distance / count if count > 0 else 0.0


def build_tree_from_chunks(
    chunks: List[str],
    summarizer: Callable[[List[str]], str],
    embedder: Callable[[str], np.ndarray],
    token_counter: Callable[[str], int],
    max_children: int = 4,
    document_name: str = "",
) -> FractalTree:
    """
    Build a fractal tree from document chunks.
    
    Algorithm:
    1. Create leaf nodes from chunks
    2. Group leaves into clusters of max_children
    3. Summarize each cluster to create parent nodes
    4. Repeat until we have a single root
    
    Args:
        chunks: List of text chunks (leaves)
        summarizer: Function to summarize a list of texts
        embedder: Function to compute embeddings
        token_counter: Function to count tokens
        max_children: Maximum children per node
        document_name: Name of source document
    
    Returns:
        FractalTree with hierarchical structure
    """
    if not chunks:
        raise ValueError("Cannot build tree from empty chunks")
    
    # Step 1: Create leaf nodes
    current_level: List[FractalNode] = []
    for i, chunk in enumerate(chunks):
        embedding = embedder(chunk)
        node = FractalNode(
            id="",  # Will be auto-generated
            node_type=NodeType.LEAF,
            content=chunk,
            summary=chunk[:200] + "..." if len(chunk) > 200 else chunk,
            embedding=embedding,
            depth=0,  # Will be updated later
            token_count=token_counter(chunk),
            source_info={"chunk_index": i},
        )
        current_level.append(node)
    
    depth = 0
    
    # Step 2: Build tree bottom-up
    while len(current_level) > 1:
        next_level: List[FractalNode] = []
        depth += 1
        
        # Group nodes into clusters
        for i in range(0, len(current_level), max_children):
            group = current_level[i:i + max_children]
            
            # Summarize the group
            group_contents = [node.content for node in group]
            summary = summarizer(group_contents)
            
            # Compute embedding for summary
            embedding = embedder(summary)
            
            # Compute child diversity
            child_embeddings = [n.embedding for n in group if n.embedding is not None]
            diversity = compute_child_diversity(child_embeddings) if child_embeddings else 0.0
            
            # Determine node type
            if len(current_level) <= max_children:
                node_type = NodeType.ROOT
            elif depth == 1:
                node_type = NodeType.SUBSECTION
            else:
                node_type = NodeType.SECTION
            
            # Create parent node
            parent = FractalNode(
                id="",
                node_type=node_type,
                content=summary,
                summary=summary,
                embedding=embedding,
                children=group,
                depth=0,  # Will be updated
                token_count=token_counter(summary),
                metadata=UncertaintyMetadata(
                    child_diversity=diversity,
                    information_density=len(summary) / (sum(len(c) for c in group_contents) + 1),
                ),
            )
            
            # Set parent references
            for child in group:
                child.parent_id = parent.id
            
            next_level.append(parent)
        
        current_level = next_level
    
    # The last remaining node is the root
    root = current_level[0]
    root.node_type = NodeType.ROOT
    
    # Update depths (root is 0, goes down)
    _update_depths(root, 0)
    
    return FractalTree(root=root, document_name=document_name)


def _update_depths(node: FractalNode, depth: int) -> None:
    """Recursively update node depths."""
    node.depth = depth
    for child in node.children:
        _update_depths(child, depth + 1)
