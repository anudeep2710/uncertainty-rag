# Uncertainty-Guided Hierarchical Retrieval for Token-Constrained Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This research project implements **Uncertainty-Guided Hierarchical Retrieval**, a novel approach for Small Language Models (SLMs) to navigate large documents efficiently. Unlike existing approaches (RAPTOR, MemTree) that rely on relevance scoring, our system uses **epistemic uncertainty** and **information gain** to decide when and where to "zoom in" on document content.

### Key Innovation

> **Core Hypothesis**: An SLM can navigate hierarchical document structures more efficiently by using its own predictive uncertainty as a navigation signal, stopping early when confident and diving deeper when uncertain.

## 🚀 Features

- **Epistemic Navigation**: Quantifies SLM uncertainty at each tree level
- **Information Gain Scoring**: Nodes scored by expected uncertainty reduction
- **Calibrated Early Stopping**: Stop traversal when confidence exceeds threshold
- **Uncertainty Traces**: Explainable retrieval paths showing decision rationale
- **Token Budget Constraint**: Works within 2-4k tokens regardless of document size

## 📊 Comparison with Existing Work

| Aspect | RAPTOR/MemTree | Our Approach |
|--------|----------------|--------------|
| Objective | Maximize relevance | **Minimize uncertainty** |
| Stopping | Budget exhausted | **Confidence threshold** |
| Scoring | LLM relevance | **Information gain** |
| Theory | Empirical | **Provable bounds** |

## 🛠️ Installation

```bash
# Clone the repository
git clone <repo-url>
cd slm-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Quick Start

```python
from src.ingestion import ingest_document
from src.zoom_agent import UncertaintyZoomAgent
from src.llm_interface import OllamaInterface

# Initialize SLM
slm = OllamaInterface(model="llama3.2:1b")

# Ingest a document
tree = ingest_document("path/to/document.pdf", slm)

# Search with uncertainty guidance
agent = UncertaintyZoomAgent(slm)
result = agent.search(
    query="What was the learning rate in experiment 3?",
    tree=tree,
    budget=2048,
    confidence_threshold=0.8
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Tokens used: {result.tokens_used}")
print(f"Expansion path: {result.expansion_path}")
```

### CLI

```bash
# Ingest a document
python cli.py ingest paper.pdf --output tree.json

# Search
python cli.py search "What is the main contribution?" --tree tree.json

# Run benchmarks
python cli.py benchmark narrativeqa --budget 2048

# Visualize tree
python cli.py visualize tree.json
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
slm-research/
├── src/
│   ├── fractal_tree.py      # Tree data structure
│   ├── uncertainty.py       # Uncertainty estimation
│   ├── zoom_agent.py        # Main algorithm
│   ├── ingestion.py         # Document processing
│   └── llm_interface.py     # SLM API wrapper
├── benchmarks/
│   ├── framework.py         # Evaluation framework
│   ├── baselines.py         # Comparison methods
│   └── datasets.py          # Dataset loaders
├── tests/
├── cli.py
└── requirements.txt
```

## 📈 Evaluation Metrics

- **Accuracy**: Exact/fuzzy match with ground truth
- **Tokens Used**: Total input + output tokens
- **Early Stop Rate**: % queries answered before reaching leaves
- **Calibration Error**: ECE (Expected Calibration Error)
- **Uncertainty Correlation**: Does high uncertainty predict errors?

## 🔬 Research Contributions

1. **Novel framing**: Retrieval as uncertainty minimization
2. **Information-theoretic scoring**: Information gain for node selection
3. **Calibration-aware stopping**: Principled early termination
4. **Theoretical bounds**: Provable retrieval depth guarantees
5. **Interpretability**: Uncertainty traces for explainability

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

This research builds upon ideas from:
- RAPTOR (Sarthi et al., ICLR 2024)
- MemTree (2024)
- Tree of Thoughts (Yao et al., 2023)
