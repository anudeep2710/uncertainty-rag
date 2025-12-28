# Uncertainty-Guided Hierarchical Retrieval
## Complete Implementation Documentation

---

# Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Use Case & Motivation](#2-use-case--motivation)
3. [Core Concepts](#3-core-concepts)
4. [Architecture Overview](#4-architecture-overview)
5. [Implementation Deep Dive](#5-implementation-deep-dive)
6. [Algorithm Walkthrough](#6-algorithm-walkthrough)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Usage Guide](#8-usage-guide)
9. [Research Novelty](#9-research-novelty)
10. [Future Directions](#10-future-directions)

---

# 1. Problem Statement

## The Challenge

**How can a Small Language Model (SLM) find specific information in a very large document without ever reading the entire document?**

Consider this scenario:
- You have a 100-page technical paper (≈50,000 tokens)
- You want to ask: "What learning rate was used in experiment 3?"
- Your SLM can only process 2,000-4,000 tokens at once
- You cannot feed the entire document to the model

### The Needle-in-Haystack Problem

```
┌─────────────────────────────────────────────────────────────────┐
│                     100-PAGE DOCUMENT                           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │     │ │     │ │     │ │ 🎯  │ │     │ │     │ │     │ ...   │
│  │     │ │     │ │     │ │     │ │     │ │     │ │     │       │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
│   Sec 1   Sec 2   Sec 3   Sec 4   Sec 5   Sec 6   Sec 7        │
│                             ↑                                   │
│                      The answer is                              │
│                      hidden here!                               │
└─────────────────────────────────────────────────────────────────┘

SLM Context Window: [________] (only 2-4K tokens)
Document Size:      [████████████████████████████] (50K tokens)
```

### Constraints

1. **Fixed Token Budget**: The SLM cannot process more than B tokens (e.g., 2048)
2. **No Full Context**: We cannot fit the entire document in memory
3. **Efficiency Required**: Minimize the number of tokens processed
4. **Accuracy Required**: Still find the correct answer

---

# 2. Use Case & Motivation

## Real-World Applications

### 1. Technical Documentation Q&A
```
User: "What is the maximum batch size supported by this API?"
Document: 200-page API documentation
Challenge: Answer is in one paragraph, somewhere in the middle
```

### 2. Legal Document Analysis
```
User: "What is the termination clause in this contract?"
Document: 50-page legal contract
Challenge: Specific clause buried among boilerplate text
```

### 3. Research Paper Analysis
```
User: "What dataset did they use for the ablation study?"
Document: 30-page research paper with appendices
Challenge: Specific detail in methods or appendix section
```

### 4. Customer Support (Edge Devices)
```
User: "How do I reset my device?"
Document: Product manual
Challenge: Running on edge device with limited compute
```

## Why Small Language Models?

| Aspect | Large LLM (GPT-4) | Small LLM (Llama 3.2 1B) |
|--------|-------------------|--------------------------|
| Context Window | 128K+ tokens | 2-8K tokens |
| Cost per Query | $0.01-0.10 | $0.0001 or free (local) |
| Latency | 500ms-2s | 50-200ms |
| Deployment | Cloud only | Edge devices, local |
| Privacy | Data leaves device | Data stays local |

**Our Goal**: Make SLMs as effective as large LLMs for document Q&A by being *smarter* about what they read.

---

# 3. Core Concepts

## 3.1 Fractal Memory Structure

We organize the document as a **hierarchical tree** where:
- **Root**: Global summary of the entire document
- **Branches**: Summaries of major sections
- **Leaves**: Actual text chunks

```
                    ┌─────────────────┐
                    │  ROOT SUMMARY   │
                    │ "This paper is  │
                    │ about ML..."    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
    │ Section │        │ Section │        │ Section │
    │ Summary │        │ Summary │        │ Summary │
    │   1     │        │   2     │        │   3     │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                  │                  │
    ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
    │         │        │         │        │         │
   ┌▼┐       ┌▼┐      ┌▼┐       ┌▼┐      ┌▼┐       ┌▼┐
   │ │       │ │      │ │       │ │      │ │       │ │
   │L│       │L│      │L│       │L│      │L│       │L│
   │1│       │2│      │3│       │4│      │5│       │6│
   └─┘       └─┘      └─┘       └─┘      └─┘       └─┘
   
   L = Leaf (actual text chunk)
```

**Property**: `Summary(Parent) ≈ Synthesize(Children)`

This means reading a parent gives you a compressed understanding of all its children.

## 3.2 The "Zoom" Metaphor

Imagine using Google Maps:
1. Start zoomed out (see whole country)
2. Recognize the city you want
3. Zoom in on that city
4. Recognize the neighborhood
5. Zoom in on that neighborhood
6. Find the exact street

We do the same with documents:
1. Start with document summary
2. Identify promising sections
3. Zoom into those sections
4. Find the exact paragraph with the answer

## 3.3 Uncertainty as a Guide

**Key Insight**: The SLM *knows* when it's uncertain.

When you ask the SLM a question, it produces:
- An answer
- Confidence (how sure it is)
- Entropy (how "spread out" its probability distribution is)

```
High Confidence (Low Entropy):          Low Confidence (High Entropy):
┌─────────────────────┐                 ┌─────────────────────┐
│ █                   │                 │ ▄  ▄  ▃  ▄  ▃      │
│ █                   │                 │ █  █  █  █  █      │
│ █      ▁            │                 │ █  █  █  █  █  ▂   │
│ █  ▁   █   ▁        │                 │ █  █  █  █  █  █   │
├─┴──┴───┴───┴────────┤                 ├─┴──┴──┴──┴──┴──┴───┤
│ A  B   C   D        │                 │ A  B  C  D  E  F   │
└─────────────────────┘                 └─────────────────────┘
"I'm pretty sure it's A"               "Could be any of these..."
```

**Our Strategy**: 
- If confident → Stop and answer
- If uncertain → Zoom into promising areas to gather more information

## 3.4 Information Gain

When deciding which node to expand, we ask:
> "Which expansion will reduce my uncertainty the most?"

**Information Gain** = Current Uncertainty - Expected Uncertainty After Expansion

```
Current State: Uncertainty = 2.5 nats

Option A: Expand Node X          Option B: Expand Node Y
├── Predicted Uncertainty: 2.0   ├── Predicted Uncertainty: 1.5
├── Information Gain: 0.5        ├── Information Gain: 1.0
└── Not great                    └── Better choice! ✓
```

We pick the node with the **highest information gain**.

---

# 4. Architecture Overview

## System Components

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                 │
│                "What learning rate was used?"                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    UNCERTAINTY ZOOM AGENT                         │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Uncertainty   │  │   Information   │  │  Early Stopping  │  │
│  │  Estimator     │──│   Gain Scorer   │──│  Controller      │  │
│  └────────────────┘  └─────────────────┘  └──────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FRACTAL TREE                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Document Hierarchy                        │ │
│  │    Root → Sections → Subsections → Paragraphs → Sentences   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SLM INTERFACE                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    Ollama    │  │   OpenAI     │  │    Mock      │           │
│  │   (Local)    │  │   (Cloud)    │  │   (Testing)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
1. INGESTION PHASE (Offline, One-time)
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Document │ ─► │  Chunk   │ ─► │Summarize │ ─► │  Build   │
   │  (PDF)   │    │ (Semantic│    │(Bottom-  │    │  Tree    │
   │          │    │  splits) │    │  up)     │    │          │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘

2. SEARCH PHASE (Online, Per-query)
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Query   │ ─► │ Estimate │ ─► │  Score   │ ─► │  Expand  │
   │          │    │Uncertainty│   │   IG     │    │  Best    │
   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                         ▲                              │
                         │         ┌──────────┐         │
                         └─────────│ Confident?│◄───────┘
                                   │  Yes: Stop│
                                   │  No: Loop │
                                   └──────────┘
```

---

# 5. Implementation Deep Dive

## 5.1 Fractal Tree (`src/fractal_tree.py`)

### FractalNode Class

Each node in our tree contains:

```python
@dataclass
class FractalNode:
    id: str                          # Unique identifier
    node_type: NodeType              # ROOT, SECTION, SUBSECTION, LEAF
    content: str                     # The actual text (or summary)
    summary: str                     # Short summary for display
    embedding: np.ndarray            # Vector representation
    children: List[FractalNode]      # Child nodes
    parent_id: str                   # Reference to parent
    depth: int                       # How deep in tree (0 = root)
    token_count: int                 # How many tokens in content
    metadata: UncertaintyMetadata    # Stats for navigation
```

### Uncertainty Metadata

This is **key to our innovation**:

```python
@dataclass
class UncertaintyMetadata:
    child_diversity: float       # How semantically different are children
    information_density: float   # How much info per token
    expansion_count: int         # How often this node was expanded
    avg_relevance_when_visited: float
    success_rate: float          # Did expanding here lead to correct answer?
    visit_count: int
```

The `child_diversity` is crucial for information gain:
- High diversity = children cover different topics = more potential information
- Low diversity = children are similar = less likely to help

### Tree Building

```python
def build_tree_from_chunks(chunks, summarizer, embedder, token_counter, max_children=4):
    """
    Build tree bottom-up:
    1. Create leaf nodes from chunks
    2. Group leaves into clusters of max_children
    3. Summarize each cluster → parent node
    4. Repeat until single root
    """
```

Example:
```
Input: ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5", "Chunk 6"]
       max_children = 2

Step 1: Create leaves
        L1, L2, L3, L4, L5, L6

Step 2: Group and summarize (level 1)
        [L1, L2] → S1 "Summary of chunks 1-2"
        [L3, L4] → S2 "Summary of chunks 3-4"
        [L5, L6] → S3 "Summary of chunks 5-6"

Step 3: Group and summarize (level 2)
        [S1, S2] → S4 "Summary of sections 1-2"
        [S3]     → S5 (same as S3)

Step 4: Create root
        [S4, S5] → ROOT "Summary of entire document"

Result:
              ROOT
             /    \
           S4      S5
          /  \      |
        S1    S2   S3
       / \   / \  / \
      L1 L2 L3 L4 L5 L6
```

## 5.2 Uncertainty Estimation (`src/uncertainty.py`)

### Core Insight: Entropy from Log Probabilities

When an LLM generates text, it produces **log probabilities** for each token:

```
Token: "The"    log_prob: -0.1  (very likely)
Token: "learning" log_prob: -0.5  (likely)
Token: "rate"   log_prob: -0.3  (likely)
Token: "was"    log_prob: -0.2  (likely)
Token: "0.001"  log_prob: -2.5  (uncertain!)
```

**Entropy** = negative average log probability:
```python
entropy = -mean(log_probs)  # Higher = more uncertain
```

### UncertaintyEstimate Class

```python
@dataclass
class UncertaintyEstimate:
    entropy: float          # Average token entropy (higher = more uncertain)
    confidence: float       # exp(mean(log_probs)) - probability of this exact answer
    answer: str             # The generated answer
    token_entropies: List   # Per-token breakdown
    is_confident: bool      # Does confidence exceed threshold?
```

### Information Gain Estimation

```python
def estimate_information_gain(self, llm, query, current_context, node_summary, 
                               node_id, child_diversity, current_uncertainty):
    """
    Estimate how much expanding this node will reduce uncertainty.
    
    Formula:
    IG = current_uncertainty - predicted_uncertainty_after_expansion
    
    Key insight: 
    - Higher relevance to query = more likely to reduce uncertainty
    - Higher child diversity = more different paths to explore
    """
    
    # Compute relevance via embedding similarity
    relevance = cosine_similarity(embed(query), embed(node_summary))
    
    # Heuristic: reduction is proportional to relevance × diversity
    reduction_factor = relevance * (0.5 + 0.5 * child_diversity)
    predicted_uncertainty = current_uncertainty * (1 - reduction_factor * 0.3)
    
    return InformationGainEstimate(
        expected_ig = current_uncertainty - predicted_uncertainty,
        # ... other fields
    )
```

### Early Stopping Controller

```python
class EntropyBasedStopping:
    """
    Decides when to stop searching and return an answer.
    
    Unlike fixed-depth stopping (RAPTOR, MemTree), we stop when CONFIDENT.
    """
    
    def should_stop(self, estimate, current_depth, budget_remaining):
        # Never stop before minimum depth
        if current_depth < self.min_depth:
            return False, "Below minimum depth"
        
        # Stop if budget exhausted
        if budget_remaining <= 0:
            return True, "Budget exhausted"
        
        # NOVEL: Stop if consistently confident
        if estimate.is_confident:
            self._consecutive_confident += 1
            if self._consecutive_confident >= self.patience:
                return True, "Confident for multiple checks"
        
        # Very confident? Stop immediately
        if estimate.confidence > 0.95:
            return True, "Very high confidence"
        
        return False, "Continue search"
```

## 5.3 Zoom Agent (`src/zoom_agent.py`)

### Main Search Algorithm

```python
def search(self, query, tree, budget=2048, confidence_threshold=None):
    """
    The core algorithm that makes this research novel.
    
    Instead of:
    - RAPTOR: similarity search on collapsed tree → answer
    - MemTree: semantic matching → expand → answer
    
    We do:
    - Estimate uncertainty → compute information gain → expand highest IG →
    - Check if confident → repeat or stop early
    """
    
    # Initialize: start with just the root visible
    visible_nodes = [tree.root]
    expansion_path = []
    uncertainty_trace = []
    
    for step in range(self.max_expansions):
        # 1. Build context from visible nodes
        context = self._build_context(visible_nodes)
        
        # 2. Estimate uncertainty about the answer
        estimate = self.uncertainty_estimator.estimate_uncertainty(
            self.slm, query, context
        )
        uncertainty_trace.append(estimate.entropy)
        
        # 3. Check if we should stop (NOVEL: uncertainty-based)
        should_stop, reason = self.stopping_criterion.should_stop(
            estimate, current_depth, budget - tokens_used
        )
        if should_stop:
            break
        
        # 4. Find expandable nodes (non-leaves in visible set)
        expandable = [n for n in visible_nodes if not n.is_leaf]
        if not expandable:
            break  # Reached max depth
        
        # 5. Score each by INFORMATION GAIN (NOVEL)
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
        
        # 6. Expand node with highest information gain
        best_node, best_ig = max(ig_scores, key=lambda x: x[1].combined_score)
        visible_nodes = self._expand_node(visible_nodes, best_node)
        expansion_path.append(best_node.id)
    
    # Return answer with full trace for interpretability
    return SearchResult(
        answer=estimate.answer,
        confidence=estimate.confidence,
        expansion_path=expansion_path,
        uncertainty_trace=uncertainty_trace,
        # ... other fields
    )
```

### Visual Example of Search

```
Query: "What learning rate was used?"

Step 0: Visible = [ROOT]
        ┌─────────────────────────────────────────┐
        │ ROOT: "This paper presents a new ML    │
        │        framework for image..."          │
        └─────────────────────────────────────────┘
        Uncertainty: 3.2 (HIGH - don't know yet)
        → Expand ROOT

Step 1: Visible = [Intro, Methods, Results]
        ┌────────────┐ ┌──────────────┐ ┌────────────┐
        │   Intro    │ │   Methods    │ │  Results   │
        │"We propose"│ │"Training..." │ │"Table 1..."│
        └────────────┘ └──────────────┘ └────────────┘
        Uncertainty: 2.5 (still high)
        IG scores: Intro=0.3, Methods=0.9, Results=0.4
        → Expand METHODS (highest IG!)

Step 2: Visible = [Intro, Training, Evaluation, Results]
        ┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────┐
        │   Intro    │ │  Training    │ │ Evaluation │ │  Results   │
        │            │ │"LR=0.001..." │ │            │ │            │
        └────────────┘ └──────────────┘ └────────────┘ └────────────┘
        Uncertainty: 0.8 (much lower!)
        Confidence: 0.85 > threshold
        → STOP! Found the answer.

Answer: "The learning rate was 0.001"
Tokens used: 1,247 (out of 2,048 budget)
Expansions: 2 (out of possible 4 depth levels)
```

## 5.4 Document Ingestion (`src/ingestion.py`)

### Chunking Strategies

```python
class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"   # Equal token chunks
    SEMANTIC = "semantic"        # Based on headers/sections
    SENTENCE = "sentence"        # At sentence boundaries
    PARAGRAPH = "paragraph"      # At paragraph boundaries
```

**Semantic chunking** is preferred because it:
1. Preserves document structure
2. Keeps related content together
3. Produces more meaningful summaries

### Ingestion Pipeline

```python
def ingest_document(source, slm, config=None, document_name=None, max_children=4):
    """
    Full pipeline: File → Chunks → Tree
    
    1. Load document (PDF, TXT, MD)
    2. Split into semantic chunks
    3. Create embeddings for each chunk
    4. Build tree bottom-up with summarization
    5. Compute child diversity for each node
    """
```

## 5.5 LLM Interface (`src/llm_interface.py`)

### Abstraction Layer

```python
class SLMInterface(ABC):
    """Abstract interface for any SLM backend."""
    
    @abstractmethod
    def generate(self, prompt, max_tokens, temperature) -> str:
        """Generate text."""
        pass
    
    @abstractmethod
    def generate_with_logprobs(self, prompt, max_tokens, temperature) -> Tuple[str, List[float]]:
        """Generate with log probabilities for uncertainty estimation."""
        pass
    
    @abstractmethod
    def embed(self, text) -> np.ndarray:
        """Get embedding for similarity computation."""
        pass
    
    @abstractmethod
    def count_tokens(self, text) -> int:
        """Count tokens for budget management."""
        pass
    
    @abstractmethod
    def summarize(self, texts, max_length) -> str:
        """Summarize texts for tree building."""
        pass
```

### Implementations

1. **OllamaInterface**: Local models via Ollama (llama3.2, phi3, mistral)
2. **OpenAIInterface**: Cloud models via API (gpt-3.5-turbo, gpt-4)
3. **MockSLMInterface**: Testing without actual LLM calls

---

# 6. Algorithm Walkthrough

## Step-by-Step Example

Let's trace through a complete search:

### Setup
```
Document: 50-page research paper
Tree depth: 4 levels
Budget: 2048 tokens
Confidence threshold: 0.75
Query: "What optimizer was used for training?"
```

### Initial State
```
visible_nodes = [ROOT]
expansion_path = []
tokens_used = 450 (root summary)
```

### Iteration 1

```
1. Build context:
   "[Summary] This paper presents a novel approach to image classification
    using deep neural networks. We introduce a new architecture called
    ResNet-X that achieves state-of-the-art results..."

2. Estimate uncertainty:
   Answer: "The paper uses neural networks for training." 
   Entropy: 3.1
   Confidence: 0.32
   → NOT confident (0.32 < 0.75)

3. Expandable nodes: [ROOT]

4. Score information gain for ROOT:
   - Relevance to "optimizer": 0.4 (mentioned training)
   - Child diversity: 0.7 (Intro, Methods, Results, Conclusion)
   - Expected IG: 0.8

5. Expand ROOT:
   visible_nodes = [Intro, Methods, Results, Conclusion]
   tokens_used = 1,100
```

### Iteration 2

```
1. Build context:
   "[Summary] Intro: We propose ResNet-X for image classification..."
   "[Summary] Methods: We train using SGD with momentum. The model..."
   "[Summary] Results: Table 1 shows accuracy comparisons..."
   "[Summary] Conclusion: ResNet-X achieves new SOTA..."

2. Estimate uncertainty:
   Answer: "SGD with momentum" 
   Entropy: 1.8
   Confidence: 0.58
   → NOT confident (0.58 < 0.75)

3. Expandable nodes: [Intro, Methods, Results, Conclusion]

4. Score information gain:
   - Intro: IG = 0.2 (low relevance to optimizer)
   - Methods: IG = 0.9 (high! mentions training)
   - Results: IG = 0.3 (tables, numbers)
   - Conclusion: IG = 0.1 (summary)

5. Expand Methods:
   visible_nodes = [Intro, Training, Architecture, Evaluation, Results, Conclusion]
   tokens_used = 1,600
```

### Iteration 3

```
1. Build context:
   "[Summary] Intro: ..."
   "[Summary] Training: We use Adam optimizer with learning rate 0.001.
    Batch size is 32. Training runs for 100 epochs..."
   "[Summary] Architecture: ..."
   "[Summary] Evaluation: ..."
   "[Summary] Results: ..."
   "[Summary] Conclusion: ..."

2. Estimate uncertainty:
   Answer: "Adam optimizer with learning rate 0.001" 
   Entropy: 0.6
   Confidence: 0.87
   → CONFIDENT! (0.87 > 0.75)

3. STOP! Return answer.
```

### Final Result

```python
SearchResult(
    answer="Adam optimizer with learning rate 0.001",
    confidence=0.87,
    is_confident=True,
    status=SearchStatus.CONFIDENT,
    tokens_used=1600,
    tokens_budget=2048,
    depth_reached=2,
    max_depth=4,
    num_expansions=2,
    expansion_path=["root_abc123", "methods_def456"],
    uncertainty_trace=[3.1, 1.8, 0.6],
)
```

### What We Saved

```
Without our approach (read everything): 50,000 tokens
With our approach: 1,600 tokens
Savings: 96.8%!

Depth reached: 2 out of 4 (early stopped!)
```

---

# 7. Evaluation Framework

## 7.1 Metrics

### Standard Metrics
- **Accuracy**: Did we get the right answer?
- **F1 Score**: Precision and recall balance

### Efficiency Metrics
- **Tokens Used**: Total input + output tokens
- **Token Savings**: vs. reading entire document
- **Average Expansions**: How many nodes did we expand?
- **Average Depth**: How deep did we go?

### Novel Metrics (Our Contribution)

#### Early Stop Rate
```
early_stop_rate = (queries_stopped_before_max_depth) / (total_queries)
```
Higher is better - means we're confidently answering without exhaustive search.

#### Calibration Error (ECE)
```
ECE = Σ (bin_weight × |accuracy_in_bin - avg_confidence_in_bin|)
```
Lower is better - means our confidence matches actual accuracy.

Example:
```
Bin [0.8, 1.0]:
  - 10 predictions with avg confidence 0.9
  - 9 were correct (accuracy 0.9)
  - |0.9 - 0.9| = 0 → Perfect calibration!

Bin [0.6, 0.8]:
  - 10 predictions with avg confidence 0.7
  - 4 were correct (accuracy 0.4)
  - |0.4 - 0.7| = 0.3 → Poor calibration (overconfident)
```

#### Uncertainty-Error Correlation
```
correlation = pearson(final_uncertainty, was_wrong)
```
Higher is better - means uncertainty predicts when we'll be wrong.

## 7.2 Baselines

### Flat RAG (Standard Retrieval)
```
1. Chunk document
2. Embed all chunks
3. Find top-k chunks similar to query
4. Feed to LLM
5. Answer
```
- **Problem**: Ignores document structure
- **Problem**: No early stopping

### RAPTOR-Style (Collapsed Tree)
```
1. Build tree with summaries
2. Flatten tree (all nodes in one list)
3. Similarity search on all nodes
4. Take top-k
5. Answer
```
- **Problem**: Doesn't use tree structure for navigation
- **Problem**: No uncertainty guidance

### Our Approach (Uncertainty-Guided)
```
1. Build tree with summaries
2. Start at root
3. Estimate uncertainty
4. If confident: stop
5. Else: expand highest IG node
6. Repeat
```
- **Advantage**: Uses tree structure
- **Advantage**: Uncertainty-guided navigation
- **Advantage**: Early stopping when confident

---

# 8. Usage Guide

## 8.1 Installation

```bash
# Clone the repository
cd c:\Users\anude\Documents\slm-research

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 8.2 Quick Demo (No Ollama Required)

```bash
python cli.py demo
```

This runs with a mock SLM to demonstrate the algorithm.

## 8.3 Full Usage with Ollama

### Start Ollama
```bash
# In a separate terminal
ollama serve
ollama pull llama3.2:1b
```

### Ingest a Document
```bash
python cli.py ingest paper.pdf -o paper_tree.json -m llama3.2:1b
```

### Search
```bash
python cli.py search "What is the main contribution?" -t paper_tree.json
```

### Search with Trace
```bash
python cli.py search "What learning rate was used?" -t paper_tree.json --trace
```

Output:
```
Answer: The learning rate was 0.001

Search Statistics:
├── Confidence: 87.3%
├── Tokens Used: 1,247 / 2,048
├── Depth Reached: 2 / 4
└── Expansions: 2

Expansion Trace:
  1. Methods: Training and optimization... (IG=0.92, Δuncertainty=-1.3)
  2. Training Setup: We use Adam optimizer... (IG=0.78, Δuncertainty=-1.2)
```

### Run Benchmark
```bash
python cli.py benchmark -t paper_tree.json --compare
```

Output:
```
┌─────────────────────────────────────────────────────────────┐
│                  Benchmark Comparison                        │
├──────────────────────┬──────────────────┬───────────────────┤
│ Metric               │ Uncertainty-Guided│ Relevance-Only   │
├──────────────────────┼──────────────────┼───────────────────┤
│ Accuracy             │ 78.5%            │ 76.2%            │
│ Avg Tokens           │ 1,124            │ 1,856            │
│ Token Savings        │ 42.3%            │ 18.1%            │
│ Early Stop Rate      │ 35.2%            │ N/A              │
│ Calibration Error    │ 0.089            │ N/A              │
└──────────────────────┴──────────────────┴───────────────────┘
```

## 8.4 Python API

```python
from src.ingestion import ingest_document
from src.zoom_agent import UncertaintyZoomAgent
from src.llm_interface import OllamaInterface

# Initialize SLM
slm = OllamaInterface(model="llama3.2:1b")

# Ingest document
result = ingest_document("paper.pdf", slm)
tree = result.tree

# Create agent
agent = UncertaintyZoomAgent(
    slm,
    confidence_threshold=0.75,
    entropy_threshold=1.5,
    max_expansions=10,
)

# Search
result = agent.search(
    query="What is the main finding?",
    tree=tree,
    budget=2048,
)

# Access results
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Tokens used: {result.tokens_used}")
print(f"Expansion path: {result.expansion_path}")
print(f"Uncertainty trace: {result.uncertainty_trace}")
```

---

# 9. Research Novelty

## What Existing Work Does

| System | How it Navigates | How it Stops |
|--------|------------------|--------------|
| RAPTOR | Similarity to query | Fixed k retrievals |
| MemTree | Semantic matching | Budget/depth |
| LATTICE | Calibrated relevance | Budget exhausted |
| SUGAR | (Flat) Semantic entropy | Query classification |

## What We Do Differently

| Aspect | Existing Work | Our Approach |
|--------|---------------|--------------|
| **Primary Signal** | Relevance to query | Uncertainty about answer |
| **Node Selection** | Most similar | Highest information gain |
| **Stopping Criterion** | Budget/depth exhausted | Confidence threshold |
| **Optimization Goal** | Find relevant text | Minimize uncertainty |

## Our Unique Contributions

1. **Epistemic Navigation**: Use the SLM's own uncertainty as a compass
2. **Information-Gain Scoring**: Select nodes by expected uncertainty reduction
3. **Confidence-Based Stopping**: Stop when confident, not when budget exhausted
4. **Uncertainty Traces**: Interpretable record of why we navigated where we did
5. **Calibration Metrics**: Measure whether confidence is trustworthy

## Why This Matters

```
Traditional: "Let me find text relevant to your question"
Ours: "Let me figure out what I don't know and learn it efficiently"
```

It's the difference between:
- **Keyword matching** (find relevant terms)
- **Active learning** (identify and fill knowledge gaps)

---

# 10. Future Directions

## Immediate Next Steps

1. **Empirical Validation**
   - Test on NarrativeQA, QuALITY, and other long-document QA datasets
   - Compare with RAPTOR, MemTree, and LATTICE

2. **Ablation Studies**
   - Uncertainty vs. relevance scoring
   - Early stopping impact
   - Information gain components

3. **Theoretical Analysis**
   - Prove bounds on retrieval depth
   - Analyze convergence properties

## Potential Extensions

1. **Multi-Document Retrieval**
   - Navigate across multiple documents
   - Forest instead of tree

2. **Streaming Updates**
   - Update tree as document changes
   - Incremental ingestion

3. **Query-Adaptive Confidence**
   - Different thresholds for different query types
   - Learn optimal thresholds

4. **Reinforcement Learning**
   - Learn navigation policy from feedback
   - Optimize for user satisfaction

## Publication Targets

- **ACL/EMNLP**: Main NLP venues
- **NeurIPS/ICML**: Machine learning venues
- **SIGIR**: Information retrieval venue

---

# Appendix: File Reference

| File | Purpose |
|------|---------|
| `src/fractal_tree.py` | FractalNode, FractalTree, tree building |
| `src/uncertainty.py` | UncertaintyEstimator, EntropyBasedStopping |
| `src/zoom_agent.py` | UncertaintyZoomAgent, SearchResult |
| `src/ingestion.py` | Document loading, chunking, ingestion |
| `src/llm_interface.py` | OllamaInterface, OpenAIInterface, MockSLMInterface |
| `src/utils.py` | Helper functions |
| `benchmarks/framework.py` | BenchmarkRunner, metrics computation |
| `tests/test_*.py` | Unit tests |
| `cli.py` | Command-line interface |
| `requirements.txt` | Dependencies |
| `README.md` | Quick start guide |

---

*Documentation generated for the Uncertainty-Guided Hierarchical Retrieval research project.*
