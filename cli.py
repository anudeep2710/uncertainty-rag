"""
CLI Application for Uncertainty-Guided Hierarchical Retrieval

Commands:
  ingest   - Build fractal tree from document
  search   - Run uncertainty-guided search
  benchmark - Run evaluation benchmark
  visualize - Display tree structure
  demo     - Interactive demo with sample document
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree as RichTree
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
import json
from typing import Optional

app = typer.Typer(
    name="slm-research",
    help="Uncertainty-Guided Hierarchical Retrieval for SLMs",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    file_path: str = typer.Argument(..., help="Path to document file (PDF, TXT, MD)"),
    output: str = typer.Option("tree.json", "-o", "--output", help="Output path for tree JSON"),
    model: str = typer.Option("llama3.2:1b", "-m", "--model", help="Ollama model name"),
    chunk_size: int = typer.Option(500, "-c", "--chunk-size", help="Target chunk size in tokens"),
    max_children: int = typer.Option(4, "--max-children", help="Max children per tree node"),
):
    """
    Ingest a document and build a fractal tree.
    
    Example:
        python cli.py ingest paper.pdf -o paper_tree.json
    """
    from src.ingestion import ingest_document, ChunkingConfig, ChunkingStrategy
    from src.llm_interface import OllamaInterface
    
    console.print(f"[bold blue]Ingesting document:[/] {file_path}")
    
    # Check file exists
    if not Path(file_path).exists():
        console.print(f"[red]Error: File not found: {file_path}[/]")
        raise typer.Exit(1)
    
    # Initialize SLM
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing model...", total=None)
        
        try:
            slm = OllamaInterface(model=model)
            # Test connection
            slm.count_tokens("test")
        except Exception as e:
            console.print(f"[red]Error connecting to Ollama: {e}[/]")
            console.print("[yellow]Make sure Ollama is running: ollama serve[/]")
            raise typer.Exit(1)
        
        progress.update(task, description="Loading document...")
        
        config = ChunkingConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            target_chunk_size=chunk_size,
        )
        
        progress.update(task, description="Building fractal tree...")
        
        result = ingest_document(
            source=file_path,
            slm=slm,
            config=config,
            max_children=max_children,
        )
    
    # Save tree
    result.tree.save(output)
    
    # Display stats
    stats_table = Table(title="Ingestion Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    for key, value in result.stats.items():
        if isinstance(value, float):
            value = f"{value:.2f}"
        stats_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(stats_table)
    console.print(f"\n[green]✓ Tree saved to:[/] {output}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Question to answer"),
    tree_path: str = typer.Option("tree.json", "-t", "--tree", help="Path to tree JSON"),
    model: str = typer.Option("llama3.2:1b", "-m", "--model", help="Ollama model name"),
    budget: int = typer.Option(2048, "-b", "--budget", help="Token budget"),
    confidence: float = typer.Option(0.75, "-c", "--confidence", help="Confidence threshold"),
    show_trace: bool = typer.Option(False, "--trace", help="Show expansion trace"),
):
    """
    Search the fractal tree with uncertainty-guided navigation.
    
    Example:
        python cli.py search "What is the main contribution?" -t paper_tree.json
    """
    from src.fractal_tree import FractalTree
    from src.zoom_agent import UncertaintyZoomAgent
    from src.llm_interface import OllamaInterface
    
    # Load tree
    if not Path(tree_path).exists():
        console.print(f"[red]Error: Tree not found: {tree_path}[/]")
        console.print("[yellow]Run 'ingest' command first to create a tree.[/]")
        raise typer.Exit(1)
    
    tree = FractalTree.load(tree_path)
    
    # Initialize
    console.print(f"[bold blue]Query:[/] {query}")
    console.print(f"[dim]Tree: {tree.document_name} ({tree.get_leaf_count()} leaves, depth {tree.max_depth})[/]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        
        slm = OllamaInterface(model=model)
        agent = UncertaintyZoomAgent(
            slm,
            confidence_threshold=confidence,
        )
        
        result = agent.search(query, tree, budget)
    
    # Display result
    console.print()
    console.print(Panel(
        result.answer,
        title="[bold green]Answer[/]",
        border_style="green",
    ))
    
    # Stats table
    stats_table = Table(title="Search Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")
    
    stats_table.add_row("Confidence", f"{result.confidence:.1%}")
    stats_table.add_row("Status", result.status.value)
    stats_table.add_row("Tokens Used", f"{result.tokens_used} / {result.tokens_budget}")
    stats_table.add_row("Depth Reached", f"{result.depth_reached} / {result.max_depth}")
    stats_table.add_row("Expansions", str(result.num_expansions))
    stats_table.add_row("Wall Time", f"{result.wall_time_seconds:.2f}s")
    
    console.print(stats_table)
    
    # Show trace if requested
    if show_trace and result.expansion_trace:
        console.print("\n[bold]Expansion Trace:[/]")
        for step in result.expansion_trace:
            console.print(
                f"  {step.step_number}. [cyan]{step.expanded_node_summary[:60]}...[/] "
                f"(IG={step.information_gain:.3f}, Δuncertainty={step.uncertainty_before - step.uncertainty_after:+.3f})"
            )


@app.command()
def benchmark(
    tree_path: str = typer.Option("tree.json", "-t", "--tree", help="Path to tree JSON"),
    model: str = typer.Option("llama3.2:1b", "-m", "--model", help="Ollama model name"),
    budget: int = typer.Option(2048, "-b", "--budget", help="Token budget"),
    num_samples: int = typer.Option(10, "-n", "--samples", help="Number of samples"),
    compare: bool = typer.Option(True, "--compare/--no-compare", help="Compare with baseline"),
):
    """
    Run benchmark evaluation.
    
    Example:
        python cli.py benchmark -t paper_tree.json --compare
    """
    from src.fractal_tree import FractalTree
    from src.llm_interface import OllamaInterface
    from benchmarks.framework import BenchmarkRunner, create_synthetic_benchmark
    
    # Load tree
    if not Path(tree_path).exists():
        console.print(f"[red]Error: Tree not found: {tree_path}[/]")
        raise typer.Exit(1)
    
    tree = FractalTree.load(tree_path)
    
    # Create samples
    samples = create_synthetic_benchmark(tree, num_samples)
    
    if not samples:
        console.print("[red]Error: Could not generate benchmark samples[/]")
        raise typer.Exit(1)
    
    console.print(f"[bold blue]Running benchmark with {len(samples)} samples...[/]")
    
    slm = OllamaInterface(model=model)
    runner = BenchmarkRunner(slm, match_strategy="contains")
    
    if compare:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running comparison...", total=None)
            results = runner.compare_with_baseline(tree, samples, budget)
        
        # Display comparison
        comparison_table = Table(title="Benchmark Comparison")
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Uncertainty-Guided", style="green")
        comparison_table.add_column("Relevance-Only", style="yellow")
        
        ug = results["uncertainty_guided"]
        ro = results["relevance_only"]
        
        comparison_table.add_row("Accuracy", f"{ug.accuracy:.1%}", f"{ro.accuracy:.1%}")
        comparison_table.add_row("Avg Tokens", f"{ug.avg_tokens_used:.0f}", f"{ro.avg_tokens_used:.0f}")
        comparison_table.add_row("Token Savings", f"{ug.tokens_saved_vs_full:.1%}", f"{ro.tokens_saved_vs_full:.1%}")
        comparison_table.add_row("Early Stop Rate", f"{ug.early_stop_rate:.1%}", "N/A")
        comparison_table.add_row("Calibration Error", f"{ug.calibration_error:.3f}", "N/A")
        comparison_table.add_row("Avg Wall Time", f"{ug.avg_wall_time:.2f}s", f"{ro.avg_wall_time:.2f}s")
        
        console.print(comparison_table)
    else:
        from src.zoom_agent import UncertaintyZoomAgent
        
        agent = UncertaintyZoomAgent(slm)
        _, metrics = runner.run_benchmark(agent, tree, samples, budget)
        
        metrics_table = Table(title="Benchmark Results")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        
        for key, value in metrics.to_dict().items():
            metrics_table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(metrics_table)


@app.command()
def visualize(
    tree_path: str = typer.Argument(..., help="Path to tree JSON"),
    max_depth: int = typer.Option(3, "-d", "--depth", help="Max depth to display"),
    content_length: int = typer.Option(50, "-l", "--length", help="Max content length"),
):
    """
    Visualize tree structure.
    
    Example:
        python cli.py visualize tree.json -d 2
    """
    from src.fractal_tree import FractalTree
    
    if not Path(tree_path).exists():
        console.print(f"[red]Error: Tree not found: {tree_path}[/]")
        raise typer.Exit(1)
    
    tree = FractalTree.load(tree_path)
    
    console.print(f"[bold blue]Tree: {tree.document_name}[/]")
    console.print(f"[dim]Nodes: {len(tree.node_index)}, Leaves: {tree.get_leaf_count()}, Depth: {tree.max_depth}[/]")
    console.print()
    
    # Build rich tree
    def add_node(rich_node, fractal_node, depth=0):
        if depth >= max_depth:
            if fractal_node.children:
                rich_node.add(f"[dim]... ({len(fractal_node.children)} more)[/]")
            return
        
        for child in fractal_node.children:
            preview = child.summary[:content_length]
            if len(child.summary) > content_length:
                preview += "..."
            
            style = "green" if child.is_leaf else "cyan"
            type_label = "📄" if child.is_leaf else "📁"
            
            child_node = rich_node.add(f"{type_label} [{style}]{preview}[/]")
            add_node(child_node, child, depth + 1)
    
    root_preview = tree.root.summary[:content_length]
    rich_tree = RichTree(f"🌳 [bold]{root_preview}[/]")
    add_node(rich_tree, tree.root)
    
    console.print(rich_tree)


@app.command()
def demo():
    """
    Run interactive demo with a sample document.
    
    Creates a sample document, ingests it, and allows interactive queries.
    """
    from src.ingestion import ingest_from_text
    from src.zoom_agent import UncertaintyZoomAgent
    from src.llm_interface import MockSLMInterface
    
    console.print("[bold blue]🚀 Uncertainty-Guided Retrieval Demo[/]")
    console.print()
    
    # Create sample document
    sample_doc = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn from data.
The field has grown significantly in recent years, with applications in healthcare, finance, and more.

## Supervised Learning

Supervised learning uses labeled data to train models. Common algorithms include:
- Linear Regression: Used for predicting continuous values
- Logistic Regression: Used for classification tasks
- Decision Trees: Hierarchical models for classification and regression
- Random Forests: Ensemble of decision trees
- Neural Networks: Deep learning models inspired by the brain

The learning rate is typically set to 0.001 for neural networks. Batch size of 32 is common.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.
- Clustering: K-means, DBSCAN, hierarchical clustering
- Dimensionality Reduction: PCA, t-SNE, UMAP

## Deep Learning

Deep learning uses neural networks with many layers.
- CNNs: For image processing and computer vision
- RNNs: For sequential data like text and time series
- Transformers: State-of-the-art for NLP tasks

The transformer architecture uses attention mechanisms. BERT and GPT are famous examples.
GPT-3 has 175 billion parameters. BERT-base has 110 million parameters.
"""
    
    console.print("[dim]Creating sample document about machine learning...[/]")
    
    # Use mock SLM for demo (no Ollama required)
    slm = MockSLMInterface()
    
    # Ingest
    tree = ingest_from_text(sample_doc, slm, "ML Tutorial", chunk_size=200)
    
    console.print(f"[green]✓ Created tree with {tree.get_leaf_count()} leaves, depth {tree.max_depth}[/]")
    console.print()
    
    # Demo queries
    demo_queries = [
        "What is the learning rate for neural networks?",
        "What is supervised learning?",
        "How many parameters does GPT-3 have?",
    ]
    
    agent = UncertaintyZoomAgent(slm, confidence_threshold=0.7)
    
    for query in demo_queries:
        console.print(f"[bold cyan]Q: {query}[/]")
        result = agent.search(query, tree, budget=1024)
        console.print(f"[green]A: {result.answer}[/]")
        console.print(f"[dim]   (confidence: {result.confidence:.1%}, tokens: {result.tokens_used}, expansions: {result.num_expansions})[/]")
        console.print()
    
    console.print("[bold green]Demo complete![/]")
    console.print("[dim]For real documents, use 'ingest' with Ollama running.[/]")


@app.command()
def ablation(
    tree_path: str = typer.Option("tree.json", "-t", "--tree", help="Path to tree JSON"),
    model: str = typer.Option("llama3.2:1b", "-m", "--model", help="Ollama model name"),
    budget: int = typer.Option(2048, "-b", "--budget", help="Token budget"),
    num_queries: int = typer.Option(5, "-n", "--queries", help="Number of test queries"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output JSON for detailed results"),
    use_mock: bool = typer.Option(False, "--mock", help="Use mock SLM for quick testing"),
):
    """
    Run ablation study comparing different retrieval methodologies.
    
    Compares:
    1. Ours (Full) - Uncertainty + Info Gain + Early Stop
    2. No Early Stop - Must reach max depth
    3. No Info Gain - Relevance-only scoring
    4. Flat RAG - Standard vector similarity
    5. Relevance Only - RAPTOR-style baseline
    
    Example:
        python cli.py ablation -t paper_tree.json -n 5
    """
    from src.fractal_tree import FractalTree
    from benchmarks.ablation_study import (
        AblationStudy, 
        create_synthetic_qa_pairs,
        run_quick_ablation,
    )
    
    # Load tree
    if not Path(tree_path).exists():
        console.print(f"[red]Error: Tree not found: {tree_path}[/]")
        console.print("[yellow]Run 'ingest' command first to create a tree.[/]")
        raise typer.Exit(1)
    
    tree = FractalTree.load(tree_path)
    
    console.print(f"[bold blue]Running Ablation Study[/]")
    console.print(f"[dim]Tree: {tree.document_name} ({tree.get_leaf_count()} leaves, depth {tree.max_depth})[/]")
    console.print(f"[dim]Queries: {num_queries}, Budget: {budget}[/]")
    console.print()
    
    # Initialize SLM
    if use_mock:
        from src.llm_interface import MockSLMInterface
        # Detect embedding dimension from tree
        embedding_dim = 384  # Default
        if tree.root.embedding is not None:
            embedding_dim = len(tree.root.embedding)
        slm = MockSLMInterface(embedding_dim=embedding_dim)
        console.print(f"[yellow]Using mock SLM for quick testing (embedding dim: {embedding_dim})[/]")
    else:
        from src.llm_interface import OllamaInterface
        try:
            slm = OllamaInterface(model=model)
            slm.count_tokens("test")
        except Exception as e:
            console.print(f"[red]Error connecting to Ollama: {e}[/]")
            console.print("[yellow]Use --mock for testing without Ollama[/]")
            raise typer.Exit(1)
    
    # Generate QA pairs
    qa_pairs = create_synthetic_qa_pairs(tree, num_queries)
    
    if not qa_pairs:
        console.print("[red]Error: Could not generate QA pairs from tree[/]")
        raise typer.Exit(1)
    
    console.print(f"[green]Generated {len(qa_pairs)} test queries[/]")
    console.print()
    
    # Run ablation study
    study = AblationStudy(slm)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running ablation study...", total=None)
        summaries = study.run_study(qa_pairs, tree, budget, verbose=False)
        progress.update(task, description="Generating results...")
    
    # Display results
    console.print()
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold green]                    ABLATION STUDY RESULTS                      [/]")
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/]")
    console.print()
    
    # Create comparison table
    results_table = Table(title="Methodology Comparison")
    results_table.add_column("Method", style="cyan", width=15)
    results_table.add_column("Accuracy", style="green", justify="center")
    results_table.add_column("Tokens", style="yellow", justify="center")
    results_table.add_column("Savings", style="blue", justify="center")
    results_table.add_column("Depth", justify="center")
    results_table.add_column("Early Stop", style="magenta", justify="center")
    results_table.add_column("ECE", justify="center")
    results_table.add_column("Time", justify="center")
    
    for name, summary in summaries.items():
        early_stop = f"{summary.early_stop_rate:.0%}" if summary.early_stop_rate > 0 else "—"
        ece = f"{summary.calibration_error:.3f}" if summary.calibration_error > 0 else "—"
        
        results_table.add_row(
            name[:15],
            f"{summary.accuracy:.0%}",
            f"{summary.avg_tokens_used:.0f}",
            f"{summary.token_savings:.0%}",
            f"{summary.avg_depth:.1f}",
            early_stop,
            ece,
            f"{summary.avg_wall_time:.1f}s",
        )
    
    console.print(results_table)
    
    # Print insights
    console.print()
    console.print("[bold]Key Insights:[/]")
    
    # Find best by accuracy
    best_acc = max(summaries.values(), key=lambda x: x.accuracy)
    console.print(f"  • [green]Best Accuracy:[/] {best_acc.config_name} ({best_acc.accuracy:.0%})")
    
    # Find best by efficiency
    best_eff = min(summaries.values(), key=lambda x: x.avg_tokens_used)
    console.print(f"  • [yellow]Most Efficient:[/] {best_eff.config_name} ({best_eff.avg_tokens_used:.0f} tokens)")
    
    # Our method's early stop rate
    if "Ours (Full)" in summaries:
        ours = summaries["Ours (Full)"]
        console.print(f"  • [magenta]Early Stop Rate:[/] {ours.early_stop_rate:.0%} of queries answered before max depth")
    
    # Save detailed results if requested
    if output:
        study.save_results(output)
        console.print(f"\n[green]✓ Detailed results saved to: {output}[/]")


@app.command("tuned-eval")
def tuned_evaluation(
    tree_path: str = typer.Option("tree.json", "-t", "--tree", help="Path to tree JSON"),
    model: str = typer.Option("llama3.2:1b", "-m", "--model", help="Ollama model name"),
    budget: int = typer.Option(2048, "-b", "--budget", help="Token budget"),
    num_queries: int = typer.Option(10, "-n", "--queries", help="Number of test queries"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output JSON for results"),
):
    """
    Run evaluation with tuned parameters for better early stopping.
    
    Uses:
    - Lower confidence threshold (0.5 vs 0.75)
    - Higher entropy threshold (2.5 vs 1.5)
    - Single patience (1 vs 2)
    - Semantic similarity matching
    
    Example:
        python cli.py tuned-eval -t paper_tree.json -n 10
    """
    from src.fractal_tree import FractalTree
    from benchmarks.enhanced_evaluation import (
        TunedZoomAgent,
        EvaluationPipeline,
        create_enhanced_qa_dataset,
        TUNED_PARAMS,
    )
    from src.llm_interface import OllamaInterface
    import json
    
    # Load tree
    if not Path(tree_path).exists():
        console.print(f"[red]Error: Tree not found: {tree_path}[/]")
        raise typer.Exit(1)
    
    tree = FractalTree.load(tree_path)
    
    console.print("[bold blue]Tuned Evaluation with Semantic Matching[/]")
    console.print(f"[dim]Tree: {tree.document_name} ({tree.get_leaf_count()} leaves, depth {tree.max_depth})[/]")
    console.print()
    
    # Show tuned parameters
    console.print("[bold]Tuned Parameters:[/]")
    for key, value in TUNED_PARAMS.items():
        console.print(f"  • {key}: {value}")
    console.print()
    
    # Initialize
    try:
        slm = OllamaInterface(model=model)
        slm.count_tokens("test")
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {e}[/]")
        raise typer.Exit(1)
    
    # Generate QA dataset
    qa_pairs = create_enhanced_qa_dataset(tree, num_queries)
    
    if not qa_pairs:
        console.print("[red]Could not generate QA pairs[/]")
        raise typer.Exit(1)
    
    console.print(f"[green]Generated {len(qa_pairs)} QA pairs:[/]")
    for difficulty in ["easy", "medium", "hard"]:
        count = sum(1 for q in qa_pairs if q.difficulty == difficulty)
        console.print(f"  • {difficulty}: {count}")
    console.print()
    
    # Run evaluation
    agent = TunedZoomAgent(slm)
    pipeline = EvaluationPipeline(slm, similarity_threshold=0.6)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running tuned evaluation...", total=None)
        results, metrics = pipeline.evaluate(agent, tree, qa_pairs, budget, verbose=False)
        progress.update(task, description="Complete!")
    
    # Display results
    console.print()
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold green]                    TUNED EVALUATION RESULTS                    [/]")
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/]")
    console.print()
    
    # Results table
    results_table = Table(title="Performance Metrics")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    for key, value in metrics.to_dict().items():
        results_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(results_table)
    
    # Breakdown by difficulty
    by_difficulty = pipeline.evaluate_by_difficulty(results)
    
    if by_difficulty:
        console.print()
        diff_table = Table(title="Breakdown by Difficulty")
        diff_table.add_column("Difficulty", style="cyan")
        diff_table.add_column("Sem-Sim", justify="center")
        diff_table.add_column("Early Stop", justify="center", style="magenta")
        diff_table.add_column("Avg Depth", justify="center")
        diff_table.add_column("Tokens", justify="center")
        
        for diff, stats in by_difficulty.items():
            diff_table.add_row(
                diff.upper(),
                f"{stats.get('semantic_sim', 0):.0%}",
                f"{stats.get('early_stop_rate', 0):.0%}",
                f"{stats.get('avg_depth', 0):.1f}",
                f"{stats.get('avg_tokens', 0):.0f}",
            )
        
        console.print(diff_table)
    
    # Key insights
    console.print()
    console.print("[bold]Key Insights:[/]")
    console.print(f"  • [green]Semantic Similarity:[/] {metrics.semantic_similarity:.0%}")
    console.print(f"  • [magenta]Early Stop Rate:[/] {metrics.early_stop_rate:.0%}")
    console.print(f"  • [yellow]Token Savings:[/] {metrics.token_savings_pct:.0%}")
    console.print(f"  • [cyan]Calibration Error (ECE):[/] {metrics.calibration_error:.3f}")
    
    # Save if requested
    if output:
        with open(output, "w") as f:
            json.dump({
                "metrics": metrics.to_dict(),
                "by_difficulty": by_difficulty,
                "results": results,
            }, f, indent=2, default=str)
        console.print(f"\n[green]✓ Results saved to: {output}[/]")


@app.command("dataset-benchmark")
def dataset_benchmark(
    num_docs: int = typer.Option(3, "-d", "--docs", help="Number of documents"),
    questions_per_doc: int = typer.Option(5, "-q", "--questions", help="Questions per document"),
    model: str = typer.Option("llama3.2:1b", "-m", "--model", help="Ollama model name"),
    budget: int = typer.Option(2048, "-b", "--budget", help="Token budget"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output JSON path"),
    tuned: bool = typer.Option(True, "--tuned/--default", help="Use tuned parameters"),
):
    """
    Run benchmark on large synthetic dataset.
    
    Creates multi-document dataset with varied difficulty questions
    and runs comprehensive evaluation.
    
    Example:
        python cli.py dataset-benchmark -d 3 -q 5 --tuned
    """
    from benchmarks.dataset_benchmark import (
        create_large_benchmark_dataset,
        DatasetBenchmarkRunner,
    )
    from src.llm_interface import OllamaInterface
    import json
    
    console.print("[bold blue]Large-Scale Dataset Benchmark[/]")
    console.print(f"[dim]Documents: {num_docs}, Questions/doc: {questions_per_doc}[/]")
    console.print()
    
    # Initialize
    try:
        slm = OllamaInterface(model=model)
        slm.count_tokens("test")
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {e}[/]")
        raise typer.Exit(1)
    
    # Create dataset
    console.print("[cyan]Creating synthetic dataset...[/]")
    dataset = create_large_benchmark_dataset(num_docs, questions_per_doc)
    console.print(f"[green]✓ Created {len(dataset)} samples[/]")
    
    # Show difficulty breakdown
    for diff in ["easy", "medium", "hard"]:
        count = sum(1 for s in dataset.samples if s.difficulty == diff)
        console.print(f"  • {diff}: {count}")
    console.print()
    
    # Create agent
    if tuned:
        from benchmarks.enhanced_evaluation import TunedZoomAgent
        agent = TunedZoomAgent(slm)
        console.print("[yellow]Using tuned parameters[/]")
    else:
        from src.zoom_agent import UncertaintyZoomAgent
        agent = UncertaintyZoomAgent(slm)
        console.print("[dim]Using default parameters[/]")
    
    # Run benchmark
    console.print("\n[bold]Running benchmark...[/]")
    runner = DatasetBenchmarkRunner(slm, agent)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        benchmark_results = runner.run_benchmark(dataset, budget, verbose=False)
    
    # Display results
    summary = benchmark_results["summary"]
    
    console.print()
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/]")
    console.print("[bold green]                  DATASET BENCHMARK RESULTS                     [/]")
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/]")
    console.print()
    
    # Summary table
    results_table = Table(title=f"Results: {summary.get('dataset', 'Unknown')}")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Total Samples", str(summary.get("total_samples", 0)))
    results_table.add_row("Semantic Similarity", f"{summary.get('semantic_similarity', 0):.1%}")
    results_table.add_row("F1 Score", f"{summary.get('f1_score', 0):.1%}")
    results_table.add_row("Avg Tokens", f"{summary.get('avg_tokens', 0):.0f}")
    results_table.add_row("Early Stop Rate", f"{summary.get('early_stop_rate', 0):.1%}")
    results_table.add_row("Avg Confidence", f"{summary.get('avg_confidence', 0):.1%}")
    
    console.print(results_table)
    
    # By difficulty
    by_diff = summary.get("by_difficulty", {})
    if by_diff:
        console.print()
        diff_table = Table(title="Breakdown by Difficulty")
        diff_table.add_column("Difficulty", style="cyan")
        diff_table.add_column("Count")
        diff_table.add_column("Sem-Sim", justify="center")
        diff_table.add_column("Early Stop", justify="center", style="magenta")
        
        for diff, stats in by_diff.items():
            diff_table.add_row(
                diff.upper(),
                str(stats.get("count", 0)),
                f"{stats.get('semantic_sim', 0):.0%}",
                f"{stats.get('early_stop_rate', 0):.0%}",
            )
        
        console.print(diff_table)
    
    # Save if requested
    if output:
        with open(output, "w") as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        console.print(f"\n[green]✓ Results saved to: {output}[/]")


if __name__ == "__main__":
    app()
