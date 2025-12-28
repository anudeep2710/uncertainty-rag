"""
Utility functions for the SLM research project.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import hashlib
from pathlib import Path


def compute_hash(text: str) -> str:
    """Compute MD5 hash of text."""
    return hashlib.md5(text.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_tokens(count: int) -> str:
    """Format token count for display."""
    if count >= 1000000:
        return f"{count / 1000000:.1f}M"
    elif count >= 1000:
        return f"{count / 1000:.1f}K"
    return str(count)


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(v1, v2) / (norm1 * norm2))


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


class ProgressTracker:
    """Simple progress tracking for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """Initialize tracker."""
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, delta: int = 1) -> None:
        """Update progress."""
        self.current = min(self.current + delta, self.total)
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print progress bar."""
        pct = self.current / self.total if self.total > 0 else 0
        bar_width = 40
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r{self.description}: [{bar}] {pct:.1%} ({self.current}/{self.total})", end="")
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def complete(self) -> None:
        """Mark as complete."""
        self.current = self.total
        self._print_progress()
