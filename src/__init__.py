"""
Uncertainty-Guided Hierarchical Retrieval for Token-Constrained Language Models

A novel approach to document retrieval using epistemic uncertainty and 
information gain for navigation decisions.
"""

__version__ = "0.1.0"
__author__ = "SLM Research"

from .fractal_tree import FractalNode, FractalTree
from .uncertainty import UncertaintyEstimator
from .zoom_agent import UncertaintyZoomAgent
from .ingestion import ingest_document
from .llm_interface import SLMInterface, OllamaInterface

__all__ = [
    "FractalNode",
    "FractalTree", 
    "UncertaintyEstimator",
    "UncertaintyZoomAgent",
    "ingest_document",
    "SLMInterface",
    "OllamaInterface",
]
