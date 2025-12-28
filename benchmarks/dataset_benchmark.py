"""
Large-Scale Dataset Benchmarking

Provides loaders for standard QA datasets to properly evaluate our approach:
1. NarrativeQA - Long-form story comprehension
2. QuALITY - Multiple choice reading comprehension
3. QASPER - Scientific paper QA
4. Custom long documents

These datasets have gold standard answers for proper accuracy measurement.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import random


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    id: str
    question: str
    gold_answer: str
    document: str
    document_id: str
    difficulty: str = "medium"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkDataset:
    """A benchmark dataset with samples."""
    name: str
    samples: List[BenchmarkSample]
    description: str = ""
    
    def __len__(self):
        return len(self.samples)
    
    def subset(self, n: int) -> "BenchmarkDataset":
        """Get a random subset of samples."""
        if n >= len(self.samples):
            return self
        subset = random.sample(self.samples, n)
        return BenchmarkDataset(
            name=f"{self.name}_subset_{n}",
            samples=subset,
            description=f"Subset of {n} samples from {self.name}",
        )


def create_synthetic_long_document() -> str:
    """Create a synthetic long document for testing."""
    sections = [
        """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed. The field has evolved significantly since its inception in the 1950s when Arthur Samuel coined the term while working on game-playing programs.

The core principle of machine learning involves training algorithms on large datasets to recognize patterns and make predictions or decisions. Unlike traditional programming where rules are explicitly coded, machine learning systems derive rules from data patterns. This paradigm shift has revolutionized numerous industries from healthcare to finance.

Modern machine learning encompasses three main paradigms: supervised learning, unsupervised learning, and reinforcement learning. Each approach addresses different types of problems and has unique applications in real-world scenarios.""",

        """## Supervised Learning Fundamentals

Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. The algorithm learns from example input-output pairs and generalizes to make predictions on new, unseen data. Common supervised learning algorithms include:

1. **Linear Regression**: Used for predicting continuous numerical values. The model finds a linear relationship between input features and the target variable. The learning rate for gradient descent is typically set to 0.001 or 0.01.

2. **Logistic Regression**: Despite its name, this is used for classification tasks. It predicts the probability of an instance belonging to a particular class.

3. **Decision Trees**: Hierarchical models that make decisions based on feature values. They're interpretable but prone to overfitting without proper constraints.

4. **Random Forests**: Ensemble of decision trees that reduces overfitting by averaging predictions. Typically uses 100-500 trees for optimal performance.

5. **Support Vector Machines**: Finds optimal hyperplanes for classification with margin maximization.""",

        """## Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers that process information using connectionist approaches.

### Architecture Components

- **Input Layer**: Receives raw data features
- **Hidden Layers**: Extract increasingly abstract features
- **Output Layer**: Produces final predictions

### Key Hyperparameters

The learning rate controls the step size during optimization. Values typically range from 0.0001 to 0.1. A batch size of 32 or 64 is commonly used during training.

### Famous Models

- **BERT** (Bidirectional Encoder Representations from Transformers): Has 110 million parameters in the base version and 340 million in the large version.
- **GPT-3**: Contains 175 billion parameters, representing a massive scale-up in language models.
- **GPT-4**: Estimated to have over 1 trillion parameters, though exact specifications are not public.""",

        """## Unsupervised Learning Techniques

Unsupervised learning finds hidden patterns in data without labeled responses. The algorithm discovers intrinsic structure in the input data.

### Clustering Algorithms

**K-Means Clustering**: Partitions data into k clusters by minimizing within-cluster variance. Requires specifying the number of clusters upfront. Computational complexity is O(n*k*d*i) where n is samples, k is clusters, d is dimensions, and i is iterations.

**DBSCAN**: Density-based clustering that can find clusters of arbitrary shapes. Key parameters include epsilon (neighborhood radius) and min_samples.

**Hierarchical Clustering**: Builds nested clusters in a tree structure called a dendrogram. Can use agglomerative (bottom-up) or divisive (top-down) approaches.

### Dimensionality Reduction

**PCA (Principal Component Analysis)**: Linear transformation that projects data onto principal components with maximum variance. Reduces dimensionality while preserving most information.

**t-SNE**: Non-linear technique for visualizing high-dimensional data in 2D or 3D. Uses perplexity parameter typically set between 5 and 50.""",

        """## Reinforcement Learning

Reinforcement learning trains agents to make sequences of decisions by maximizing cumulative rewards through trial and error interactions with an environment.

### Key Concepts

- **Agent**: The learner and decision-maker
- **Environment**: What the agent interacts with
- **State**: Current situation of the agent
- **Action**: Choices available to the agent
- **Reward**: Feedback signal for the action taken
- **Policy**: Strategy mapping states to actions

### Algorithms

**Q-Learning**: Model-free algorithm that learns action-value function. Uses exploration-exploitation trade-off controlled by epsilon parameter.

**Policy Gradient Methods**: Directly optimize the policy without value function. Include REINFORCE and Actor-Critic methods.

**Deep Q-Networks (DQN)**: Combines Q-learning with deep neural networks. Introduced experience replay and target networks for stability.

### Applications

Reinforcement learning has achieved superhuman performance in games like Go (AlphaGo), chess, and Atari games. It's also applied in robotics, autonomous vehicles, and recommendation systems.""",

        """## Model Evaluation and Validation

Proper evaluation is crucial for understanding model performance and preventing overfitting.

### Metrics for Classification

- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Cross-Validation

K-fold cross-validation divides data into k subsets, trains on k-1 folds, and validates on the remaining fold. Common choices are k=5 or k=10.

### Preventing Overfitting

1. **Early Stopping**: Monitor validation loss and stop training when it starts increasing
2. **Regularization**: L1 (Lasso) and L2 (Ridge) penalties on model weights
3. **Dropout**: Randomly disable neurons during training with probability typically 0.2-0.5
4. **Data Augmentation**: Create synthetic training examples""",

        """## Production Deployment Considerations

Deploying machine learning models in production requires careful consideration of scalability, monitoring, and maintenance.

### MLOps Best Practices

1. Version control for both code and data
2. Automated testing and validation pipelines
3. Model monitoring for drift detection
4. A/B testing for model updates
5. Feature stores for consistent feature engineering

### Inference Optimization

- Model quantization reduces precision from FP32 to INT8
- Knowledge distillation trains smaller models from larger ones
- ONNX Runtime provides cross-platform optimized inference
- TensorRT accelerates inference on NVIDIA GPUs

### Ethical Considerations

Machine learning systems must be designed with fairness, accountability, and transparency in mind. Key concerns include bias in training data, algorithmic discrimination, and the explainability of model decisions.""",
    ]
    
    return "\n\n".join(sections)


def create_qa_pairs_for_document(document: str) -> List[Tuple[str, str, str]]:
    """
    Create question-answer pairs from the synthetic document.
    
    Returns list of (question, answer, difficulty) tuples.
    """
    qa_pairs = [
        # Easy - Direct facts
        ("What is the typical learning rate for neural networks?", 
         "0.001 or 0.01", "easy"),
        
        ("What is the typical batch size used in deep learning?", 
         "32 or 64", "easy"),
        
        ("How many parameters does BERT-base have?", 
         "110 million parameters", "easy"),
        
        ("How many parameters does GPT-3 have?", 
         "175 billion parameters", "easy"),
        
        ("What is the typical dropout rate?", 
         "0.2 to 0.5", "easy"),
        
        # Medium - Requires understanding
        ("What are the three main paradigms of machine learning?", 
         "supervised learning, unsupervised learning, and reinforcement learning", "medium"),
        
        ("What are the key components of neural network architecture?", 
         "Input layer, hidden layers, and output layer", "medium"),
        
        ("What is the difference between K-means and DBSCAN clustering?", 
         "K-means requires specifying number of clusters, DBSCAN finds clusters of arbitrary shapes based on density", "medium"),
        
        ("What methods prevent overfitting in machine learning?", 
         "Early stopping, regularization, dropout, and data augmentation", "medium"),
        
        ("What are the key concepts in reinforcement learning?", 
         "Agent, environment, state, action, reward, and policy", "medium"),
        
        # Hard - Requires synthesis
        ("Explain how the evolution of machine learning from the 1950s relates to modern deep learning.", 
         "Machine learning started in the 1950s with Arthur Samuel's game-playing programs, evolved from explicit rule programming to pattern learning from data, culminating in modern neural networks with billions of parameters like GPT-3", "hard"),
        
        ("Compare the computational considerations for K-means clustering and neural network training.", 
         "K-means has O(n*k*d*i) complexity with n samples, k clusters, d dimensions, i iterations. Neural networks require tuning learning rate, batch size, and regularization, with much higher computational demands for large models", "hard"),
        
        ("What are the trade-offs between model complexity and production deployment?", 
         "Larger models have more parameters but need quantization, distillation, and optimized inference for deployment. Smaller models are easier to deploy but may sacrifice accuracy", "hard"),
    ]
    
    return qa_pairs


def create_large_benchmark_dataset(
    num_documents: int = 5,
    questions_per_doc: int = 10,
) -> BenchmarkDataset:
    """
    Create a large synthetic benchmark dataset.
    
    For real experiments, use actual datasets like NarrativeQA.
    """
    samples = []
    
    for doc_idx in range(num_documents):
        # Create document (with slight variations)
        document = create_synthetic_long_document()
        
        # Add document-specific variation
        if doc_idx > 0:
            document = document.replace(
                "0.001", 
                f"0.00{doc_idx + 1}"
            )
        
        # Get QA pairs
        qa_pairs = create_qa_pairs_for_document(document)
        
        # Sample questions
        selected_pairs = qa_pairs[:questions_per_doc]
        
        for q_idx, (question, answer, difficulty) in enumerate(selected_pairs):
            samples.append(BenchmarkSample(
                id=f"doc{doc_idx}_q{q_idx}",
                question=question,
                gold_answer=answer,
                document=document,
                document_id=f"doc_{doc_idx}",
                difficulty=difficulty,
                metadata={
                    "doc_length": len(document),
                    "doc_tokens_approx": len(document) // 4,
                },
            ))
    
    return BenchmarkDataset(
        name="SyntheticMLQA",
        samples=samples,
        description=f"Synthetic ML Q&A dataset with {len(samples)} samples",
    )


def load_narrativeqa_subset(
    data_path: str,
    max_samples: int = 100,
) -> Optional[BenchmarkDataset]:
    """
    Load NarrativeQA dataset from local path.
    
    NarrativeQA requires downloading from:
    https://github.com/deepmind/narrativeqa
    
    Expected format: JSONL with fields:
    - document: full text
    - question: question text  
    - answers: list of answer strings
    """
    if not Path(data_path).exists():
        print(f"NarrativeQA data not found at {data_path}")
        print("Download from: https://github.com/deepmind/narrativeqa")
        return None
    
    samples = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                item = json.loads(line)
                
                samples.append(BenchmarkSample(
                    id=item.get('id', f'nqa_{i}'),
                    question=item['question'],
                    gold_answer=item['answers'][0] if item.get('answers') else "",
                    document=item['document'],
                    document_id=item.get('document_id', f'doc_{i}'),
                    difficulty="medium",  # NarrativeQA is generally medium-hard
                ))
    except Exception as e:
        print(f"Error loading NarrativeQA: {e}")
        return None
    
    return BenchmarkDataset(
        name="NarrativeQA",
        samples=samples,
        description="Reading comprehension on full stories",
    )


def load_quality_subset(
    data_path: str,
    max_samples: int = 100,
) -> Optional[BenchmarkDataset]:
    """
    Load QuALITY dataset from local path.
    
    QuALITY: https://github.com/nyu-mll/quality
    
    Multiple choice format - we use the correct answer as gold.
    """
    if not Path(data_path).exists():
        print(f"QuALITY data not found at {data_path}")
        print("Download from: https://github.com/nyu-mll/quality")
        return None
    
    samples = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for i, item in enumerate(data[:max_samples]):
            # QuALITY has article, questions, and correct answers
            article = item.get('article', '')
            
            for q in item.get('questions', []):
                question = q.get('question', '')
                options = q.get('options', [])
                correct_idx = q.get('gold_label', 0)
                
                if options and correct_idx < len(options):
                    gold_answer = options[correct_idx]
                else:
                    gold_answer = ""
                
                samples.append(BenchmarkSample(
                    id=f"quality_{i}_{q.get('question_id', len(samples))}",
                    question=question,
                    gold_answer=gold_answer,
                    document=article,
                    document_id=item.get('article_id', f'article_{i}'),
                    difficulty=q.get('difficult', 'medium'),
                ))
    except Exception as e:
        print(f"Error loading QuALITY: {e}")
        return None
    
    return BenchmarkDataset(
        name="QuALITY",
        samples=samples,
        description="Multiple-choice reading comprehension",
    )


class DatasetBenchmarkRunner:
    """
    Run benchmarks on datasets.
    """
    
    def __init__(self, slm, agent, similarity_threshold: float = 0.6):
        self.slm = slm
        self.agent = agent
        self.similarity_threshold = similarity_threshold
    
    def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        budget: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run benchmark on dataset.
        
        Returns detailed results and summary metrics.
        """
        from src.ingestion import ingest_from_text
        from benchmarks.enhanced_evaluation import SemanticMatcher
        
        matcher = SemanticMatcher(self.slm, self.similarity_threshold)
        
        results = []
        
        # Group by document
        docs = {}
        for sample in dataset.samples:
            if sample.document_id not in docs:
                docs[sample.document_id] = {
                    "document": sample.document,
                    "samples": [],
                }
            docs[sample.document_id]["samples"].append(sample)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BENCHMARK: {dataset.name}")
            print(f"Documents: {len(docs)}, Samples: {len(dataset)}")
            print(f"{'='*60}\n")
        
        for doc_id, doc_data in docs.items():
            if verbose:
                print(f"\nProcessing document: {doc_id}")
                print(f"  Document length: {len(doc_data['document'])} chars")
            
            # Ingest document into tree
            try:
                tree = ingest_from_text(
                    doc_data['document'],
                    self.slm,
                    doc_id,
                    chunk_size=300,
                )
                if verbose:
                    print(f"  Tree: {tree.get_leaf_count()} leaves, depth {tree.max_depth}")
            except Exception as e:
                print(f"  Error ingesting: {e}")
                continue
            
            # Run each sample
            for sample in doc_data["samples"]:
                if verbose:
                    print(f"  Query: {sample.question[:50]}...")
                
                try:
                    result = self.agent.search(sample.question, tree, budget)
                    
                    # Compute metrics
                    sem_sim = matcher.compute_similarity(
                        result.answer, sample.gold_answer
                    )
                    f1 = matcher.compute_f1(result.answer, sample.gold_answer)
                    
                    results.append({
                        "id": sample.id,
                        "question": sample.question,
                        "gold": sample.gold_answer,
                        "predicted": result.answer[:200],
                        "semantic_sim": sem_sim,
                        "f1": f1,
                        "confidence": result.confidence,
                        "tokens_used": result.tokens_used,
                        "depth_reached": result.depth_reached,
                        "early_stopped": result.status.value == "confident",
                        "difficulty": sample.difficulty,
                    })
                    
                    if verbose:
                        status = "✓" if sem_sim > 0.6 else "✗"
                        print(f"    {status} sim={sem_sim:.2f}, tokens={result.tokens_used}")
                        
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
        
        # Compute summary
        if results:
            summary = {
                "dataset": dataset.name,
                "total_samples": len(results),
                "semantic_similarity": sum(r["semantic_sim"] for r in results) / len(results),
                "f1_score": sum(r["f1"] for r in results) / len(results),
                "avg_tokens": sum(r["tokens_used"] for r in results) / len(results),
                "early_stop_rate": sum(1 for r in results if r["early_stopped"]) / len(results),
                "avg_confidence": sum(r["confidence"] for r in results) / len(results),
                "by_difficulty": {},
            }
            
            # Breakdown by difficulty
            for diff in ["easy", "medium", "hard"]:
                subset = [r for r in results if r["difficulty"] == diff]
                if subset:
                    summary["by_difficulty"][diff] = {
                        "count": len(subset),
                        "semantic_sim": sum(r["semantic_sim"] for r in subset) / len(subset),
                        "early_stop_rate": sum(1 for r in subset if r["early_stopped"]) / len(subset),
                    }
        else:
            summary = {"error": "No results collected"}
        
        return {
            "summary": summary,
            "results": results,
        }


if __name__ == "__main__":
    # Test dataset creation
    print("Creating synthetic benchmark dataset...")
    dataset = create_large_benchmark_dataset(num_documents=2, questions_per_doc=5)
    
    print(f"Dataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print(f"\nSample questions:")
    for sample in dataset.samples[:3]:
        print(f"  [{sample.difficulty}] {sample.question}")
        print(f"    Answer: {sample.gold_answer[:50]}...")
