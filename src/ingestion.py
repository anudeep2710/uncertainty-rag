"""
Document Ingestion Module

Handles the conversion of raw documents into fractal trees:
1. Load documents (PDF, TXT, MD)
2. Chunk into semantic sections
3. Build hierarchical tree with bottom-up summarization
4. Compute embeddings and metadata

Supports multiple chunking strategies:
- Fixed size: Equal token chunks
- Semantic: Based on headers/sections
- Sentence: At sentence boundaries
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
from enum import Enum
import re

from .fractal_tree import FractalTree, FractalNode, NodeType, build_tree_from_chunks
from .llm_interface import SLMInterface


class ChunkingStrategy(Enum):
    """Strategies for splitting documents into chunks."""
    FIXED_SIZE = "fixed_size"      # Fixed token count
    SEMANTIC = "semantic"          # Based on headers/sections
    SENTENCE = "sentence"          # At sentence boundaries
    PARAGRAPH = "paragraph"        # At paragraph boundaries


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    target_chunk_size: int = 500   # Target tokens per chunk
    chunk_overlap: int = 50        # Overlap between chunks
    min_chunk_size: int = 100      # Minimum chunk size
    max_chunk_size: int = 1000     # Maximum chunk size
    respect_headers: bool = True   # Try to keep headers with content


@dataclass 
class IngestionResult:
    """Result of document ingestion."""
    tree: FractalTree
    stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tree": self.tree.to_dict(),
            "stats": self.stats,
        }


def load_document(path: str) -> str:
    """
    Load document from file.
    
    Supports: TXT, MD, PDF
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix in [".txt", ".md", ".markdown"]:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif suffix == ".pdf":
        return _load_pdf(path)
    
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> str:
    """Load text from PDF file."""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(str(path))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    except ImportError:
        raise ImportError("Please install pypdf: pip install pypdf")


def chunk_document(
    text: str,
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> List[str]:
    """
    Split document into chunks based on configuration.
    
    Args:
        text: Raw document text
        config: Chunking configuration
        token_counter: Function to count tokens
    
    Returns:
        List of text chunks
    """
    if config.strategy == ChunkingStrategy.FIXED_SIZE:
        return _chunk_fixed_size(text, config, token_counter)
    elif config.strategy == ChunkingStrategy.SEMANTIC:
        return _chunk_semantic(text, config, token_counter)
    elif config.strategy == ChunkingStrategy.SENTENCE:
        return _chunk_sentence(text, config, token_counter)
    elif config.strategy == ChunkingStrategy.PARAGRAPH:
        return _chunk_paragraph(text, config, token_counter)
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")


def _chunk_fixed_size(
    text: str,
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> List[str]:
    """Split into fixed-size token chunks with overlap."""
    words = text.split()
    chunks = []
    
    # Estimate tokens per word (rough: 1.3 tokens per word)
    words_per_chunk = int(config.target_chunk_size / 1.3)
    overlap_words = int(config.chunk_overlap / 1.3)
    
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        
        # Verify size
        while token_counter(chunk) > config.max_chunk_size and end > start + 1:
            end -= 1
            chunk = " ".join(words[start:end])
        
        chunks.append(chunk)
        start = end - overlap_words
        
        if start <= 0 and chunks:
            start = end
    
    return chunks


def _chunk_semantic(
    text: str,
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> List[str]:
    """Split based on semantic boundaries (headers, sections)."""
    # Patterns for semantic boundaries
    header_pattern = re.compile(
        r'^(?:#{1,6}\s+.+|[A-Z][^.!?]*?(?::|$)|(?:Chapter|Section|Part)\s+\d+)',
        re.MULTILINE
    )
    
    # Find all headers
    headers = list(header_pattern.finditer(text))
    
    if not headers:
        # No headers found, fall back to paragraph chunking
        return _chunk_paragraph(text, config, token_counter)
    
    chunks = []
    prev_end = 0
    
    for i, match in enumerate(headers):
        # Content before this header
        if match.start() > prev_end:
            pre_content = text[prev_end:match.start()].strip()
            if pre_content and token_counter(pre_content) >= config.min_chunk_size:
                chunks.append(pre_content)
        
        # Find end of this section
        if i + 1 < len(headers):
            section_end = headers[i + 1].start()
        else:
            section_end = len(text)
        
        section = text[match.start():section_end].strip()
        section_tokens = token_counter(section)
        
        if section_tokens <= config.max_chunk_size:
            chunks.append(section)
        else:
            # Section too large, split it further
            sub_chunks = _chunk_paragraph(section, config, token_counter)
            chunks.extend(sub_chunks)
        
        prev_end = section_end
    
    # Any remaining text
    if prev_end < len(text):
        remaining = text[prev_end:].strip()
        if remaining and token_counter(remaining) >= config.min_chunk_size:
            chunks.append(remaining)
    
    return chunks


def _chunk_sentence(
    text: str,
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> List[str]:
    """Split at sentence boundaries."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = token_counter(sentence)
        
        if current_tokens + sentence_tokens > config.target_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            if sentence_tokens > config.max_chunk_size:
                # Split long sentence
                words = sentence.split()
                while words:
                    part_words = words[:int(config.target_chunk_size / 1.3)]
                    part = " ".join(part_words)
                    chunks.append(part)
                    words = words[len(part_words):]
            else:
                current_chunk.append(sentence)
                current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def _chunk_paragraph(
    text: str,
    config: ChunkingConfig,
    token_counter: Callable[[str], int],
) -> List[str]:
    """Split at paragraph boundaries."""
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_tokens = token_counter(para)
        
        if current_tokens + para_tokens > config.target_chunk_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            if para_tokens > config.max_chunk_size:
                # Split large paragraph by sentences
                sub_chunks = _chunk_sentence(para, config, token_counter)
                chunks.extend(sub_chunks)
            else:
                current_chunk.append(para)
                current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def ingest_document(
    source: str,
    slm: SLMInterface,
    config: Optional[ChunkingConfig] = None,
    document_name: Optional[str] = None,
    max_children: int = 4,
) -> IngestionResult:
    """
    Ingest a document into a fractal tree.
    
    This is the main entry point for document processing.
    
    Args:
        source: Either file path or raw text
        slm: The SLM interface for summarization and embedding
        config: Chunking configuration
        document_name: Name for the document
        max_children: Maximum children per tree node
    
    Returns:
        IngestionResult with tree and statistics
    """
    config = config or ChunkingConfig()
    
    # Load document if path
    if Path(source).exists():
        text = load_document(source)
        document_name = document_name or Path(source).name
    else:
        text = source
        document_name = document_name or "Untitled"
    
    # Chunk document
    chunks = chunk_document(text, config, slm.count_tokens)
    
    if not chunks:
        raise ValueError("No chunks produced from document")
    
    # Define summarizer function
    def summarizer(texts: List[str]) -> str:
        return slm.summarize(texts)
    
    # Define embedder function
    def embedder(text: str) -> any:
        return slm.embed(text)
    
    # Build tree
    tree = build_tree_from_chunks(
        chunks=chunks,
        summarizer=summarizer,
        embedder=embedder,
        token_counter=slm.count_tokens,
        max_children=max_children,
        document_name=document_name,
    )
    
    # Compute statistics
    stats = {
        "document_name": document_name,
        "total_chunks": len(chunks),
        "total_tokens": tree.total_tokens,
        "tree_depth": tree.max_depth,
        "total_nodes": len(tree.node_index),
        "leaf_nodes": tree.get_leaf_count(),
        "chunking_strategy": config.strategy.value,
        "avg_chunk_tokens": tree.total_tokens / len(chunks) if chunks else 0,
    }
    
    return IngestionResult(tree=tree, stats=stats)


def ingest_from_text(
    text: str,
    slm: SLMInterface,
    document_name: str = "Document",
    chunk_size: int = 500,
) -> FractalTree:
    """
    Convenience function to ingest raw text.
    
    Args:
        text: Raw document text
        slm: SLM interface
        document_name: Name for the document
        chunk_size: Target chunk size in tokens
    
    Returns:
        FractalTree
    """
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        target_chunk_size=chunk_size,
    )
    
    result = ingest_document(
        source=text,
        slm=slm,
        config=config,
        document_name=document_name,
    )
    
    return result.tree
