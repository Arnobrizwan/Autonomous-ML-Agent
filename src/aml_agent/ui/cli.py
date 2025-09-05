"""
Command-line interface for the Autonomous ML Agent.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

from ..config import load_config, create_default_config
# Import moved to avoid circular import
from ..ui.leaderboard import Leaderboard, display_leaderboard_from_file
from ..export.model_card import ModelCardGenerator
from ..interpret.explain import explain_model
from ..service.app import create_app
from ..logging import setup_logging, get_logger
from ..utils import create_sample_data

# Initialize Typer app
app = typer.Typer(
    name="aml",
    help="Autonomous ML Agent - Intelligent machine learning pipeline",
    no_args_is_help=True
)

console = Console()
logger = get_logger()


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="Data file path"),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column name"),
    time_budget: Optional[int] = typer.Option(None, "--time-budget", help="Time budget in seconds"),
    metric: Optional[str] = typer.Option(None, "--metric", help="Evaluation metric"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run autonomous ML pipeline."""
    try:
        # Setup logging
        log_level = "DEBUG" if verbose else "INFO"
        setup_logging(level=log_level)
        
        # Load configuration
        if config:
            config_obj = load_config(config)
        else:
            config_obj = create_default_config()
        
        # Override config with CLI arguments
        if data:
            config_obj.data_path = str(data)
        if target:
            config_obj.target = target
        if time_budget:
            config_obj.time_budget_seconds = time_budget
        if metric:
            config_obj.metric = metric
        
        # Load data
        console.print(f"[bold blue]Loading data from {config_obj.data_path}[/bold blue]")
        
        if not Path(config_obj.data_path).exists():
            console.print(f"[yellow]Data file not found, creating sample data[/yellow]")
            create_sample_data()
        
        import pandas as pd
        data = pd.read_csv(config_obj.data_path)
        
        console.print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Prepare features and target
        if config_obj.target and config_obj.target in data.columns:
            X = data.drop(columns=[config_obj.target])
            y = data[config_obj.target]
        else:
            # Auto-detect target (last column)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
        # Run pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running autonomous ML pipeline...", total=None)
            
            # Import here to avoid circular import
            from ..agent.loop import run_autonomous_ml
            results = run_autonomous_ml(config_obj, X, y)
            
            progress.update(task, description="Pipeline completed!")
        
        # Display results
        console.print(f"\n[bold green]Pipeline completed successfully![/bold green]")
        console.print(f"Run ID: {results['run_id']}")
        console.print(f"Best Score: {results['best_score']:.4f}")
        console.print(f"Best Model: {results['best_model']}")
        console.print(f"Total Trials: {results['total_trials']}")
        console.print(f"Artifacts: {results['artifacts_dir']}")
        
        # Show leaderboard
        leaderboard_file = Path(results['artifacts_dir']) / "leaderboard.csv"
        if leaderboard_file.exists():
            console.print("\n[bold]Top Models:[/bold]")
            display_leaderboard_from_file(leaderboard_file, n=5)
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def leaderboard(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run ID"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Leaderboard file path"),
    n: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    show_params: bool = typer.Option(False, "--params", help="Show parameters")
):
    """Display model performance leaderboard."""
    try:
        if file:
            display_leaderboard_from_file(file, n)
        elif run_id:
            leaderboard_file = Path(f"artifacts/{run_id}/leaderboard.csv")
            if leaderboard_file.exists():
                display_leaderboard_from_file(leaderboard_file, n)
            else:
                console.print(f"[red]Leaderboard file not found for run {run_id}[/red]")
        else:
            # Find latest run
            artifacts_dir = Path("artifacts")
            if not artifacts_dir.exists():
                console.print("[yellow]No artifacts directory found[/yellow]")
                return
            
            # Find latest run directory
            run_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
            if not run_dirs:
                console.print("[yellow]No runs found[/yellow]")
                return
            
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            leaderboard_file = latest_run / "leaderboard.csv"
            
            if leaderboard_file.exists():
                console.print(f"[blue]Showing leaderboard for latest run: {latest_run.name}[/blue]")
                display_leaderboard_from_file(leaderboard_file, n)
            else:
                console.print(f"[red]Leaderboard file not found in {latest_run}[/red]")
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@app.command()
def explain(
    run_id: str = typer.Argument(..., help="Run ID"),
    feature: Optional[str] = typer.Option(None, "--feature", help="Specific feature to explain"),
    instance: Optional[int] = typer.Option(None, "--instance", help="Instance index to explain")
):
    """Generate model explanations and feature importance."""
    try:
        artifacts_dir = Path(f"artifacts/{run_id}")
        if not artifacts_dir.exists():
            console.print(f"[red]Run {run_id} not found[/red]")
            return
        
        # Load model
        model_file = artifacts_dir / "model.joblib"
        if not model_file.exists():
            console.print(f"[red]Model file not found for run {run_id}[/red]")
            return
        
        import joblib
        model = joblib.load(model_file)
        
        # Load metadata
        metadata_file = artifacts_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load feature names
        feature_names_file = artifacts_dir / "feature_names.json"
        if feature_names_file.exists():
            with open(feature_names_file, 'r') as f:
                feature_names = json.load(f)
        else:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
        
        console.print(f"[bold blue]Generating explanations for run {run_id}[/bold blue]")
        
        # Generate model card
        card_generator = ModelCardGenerator()
        
        # Load trial results if available
        trial_results_file = artifacts_dir / "trial_results.csv"
        trial_results = []
        if trial_results_file.exists():
            import pandas as pd
            df = pd.read_csv(trial_results_file)
            # Convert to TrialResult objects (simplified)
            for _, row in df.iterrows():
                from ..types import TrialResult, ModelType, MetricType
                from datetime import datetime
                trial_results.append(TrialResult(
                    trial_id=int(row['trial_id']),
                    model_type=ModelType(row['model_type']),
                    params=json.loads(row['params']) if isinstance(row['params'], str) else row['params'],
                    score=float(row['score']),
                    metric=MetricType(row['metric']),
                    cv_scores=[],
                    fit_time=float(row['fit_time']),
                    predict_time=float(row['predict_time']),
                    timestamp=datetime.now()
                ))
        
        # Generate model card
        model_card = card_generator.generate_card(
            trial_results=trial_results,
            task_type=metadata.get('task_type', 'classification'),
            model=model
        )
        
        # Display model card
        console.print(f"\n[bold]Model Card for {model_card.model_name}[/bold]")
        console.print(f"Task Type: {model_card.task_type}")
        console.print(f"Best Score: {model_card.performance_metrics.get('best_score', 0):.4f}")
        
        if model_card.top_features:
            console.print(f"\n[bold]Top Features:[/bold]")
            for i, feature in enumerate(model_card.top_features, 1):
                importance = model_card.feature_importance.get(feature, 0)
                console.print(f"{i}. {feature}: {importance:.4f}")
        
        if model_card.limitations:
            console.print(f"\n[bold]Limitations:[/bold]")
            for limitation in model_card.limitations:
                console.print(f"- {limitation}")
        
        if model_card.recommendations:
            console.print(f"\n[bold]Recommendations:[/bold]")
            for recommendation in model_card.recommendations:
                console.print(f"- {recommendation}")
        
        # Save model card
        card_file = artifacts_dir / "model_card.md"
        card_generator.save_card(model_card, card_file)
        console.print(f"\n[green]Model card saved to {card_file}[/green]")
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@app.command()
def serve(
    run_id: str = typer.Argument(..., help="Run ID"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    workers: int = typer.Option(1, "--workers", help="Number of workers")
):
    """Start FastAPI prediction service."""
    try:
        artifacts_dir = Path(f"artifacts/{run_id}")
        if not artifacts_dir.exists():
            console.print(f"[red]Run {run_id} not found[/red]")
            return
        
        console.print(f"[bold blue]Starting prediction service for run {run_id}[/bold blue]")
        console.print(f"Host: {host}, Port: {port}")
        
        # Create FastAPI app
        app_instance = create_app(artifacts_dir)
        
        # Start server
        import uvicorn
        uvicorn.run(app_instance, host=host, port=port, workers=workers)
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@app.command()
def predict(
    run_id: str = typer.Argument(..., help="Run ID"),
    row: Optional[str] = typer.Option(None, "--row", help="JSON row to predict"),
    file: Optional[Path] = typer.Option(None, "--file", help="CSV file to predict"),
    stdin: bool = typer.Option(False, "--stdin", help="Read from stdin")
):
    """Make predictions using trained model."""
    try:
        artifacts_dir = Path(f"artifacts/{run_id}")
        if not artifacts_dir.exists():
            console.print(f"[red]Run {run_id} not found[/red]")
            return
        
        # Load model and preprocessor
        import joblib
        model_file = artifacts_dir / "model.joblib"
        preprocessor_file = artifacts_dir / "preprocessor.joblib"
        
        if not model_file.exists():
            console.print(f"[red]Model file not found for run {run_id}[/red]")
            return
        
        model = joblib.load(model_file)
        preprocessor = joblib.load(preprocessor_file) if preprocessor_file.exists() else None
        
        # Load feature names
        feature_names_file = artifacts_dir / "feature_names.json"
        if feature_names_file.exists():
            with open(feature_names_file, 'r') as f:
                feature_names = json.load(f)
        else:
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
        
        # Get input data
        if stdin:
            import sys
            row = sys.stdin.read().strip()
        
        if row:
            # Single prediction
            data = json.loads(row)
            df = pd.DataFrame([data])
            
            # Preprocess if available
            if preprocessor:
                df = preprocessor.transform(df)
            
            # Make prediction
            prediction = model.predict(df)[0]
            
            console.print(f"[bold green]Prediction: {prediction}[/bold green]")
            
            # Show probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(df)[0]
                console.print(f"Probabilities: {probabilities}")
        
        elif file:
            # Batch prediction
            import pandas as pd
            data = pd.read_csv(file)
            
            # Preprocess if available
            if preprocessor:
                data = preprocessor.transform(data)
            
            # Make predictions
            predictions = model.predict(data)
            
            # Save predictions
            output_file = file.parent / f"{file.stem}_predictions.csv"
            data['prediction'] = predictions
            data.to_csv(output_file, index=False)
            
            console.print(f"[bold green]Predictions saved to {output_file}[/bold green]")
            console.print(f"Predicted {len(predictions)} samples")
        
        else:
            console.print("[red]Please provide either --row, --file, or --stdin[/red]")
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID"),
    format: str = typer.Option("joblib", "--format", help="Export format"),
    output_dir: Optional[Path] = typer.Option(None, "--output", help="Output directory")
):
    """Export trained model and artifacts."""
    try:
        artifacts_dir = Path(f"artifacts/{run_id}")
        if not artifacts_dir.exists():
            console.print(f"[red]Run {run_id} not found[/red]")
            return
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy artifacts
            import shutil
            for item in artifacts_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, output_dir)
            
            console.print(f"[green]Artifacts exported to {output_dir}[/green]")
        else:
            console.print(f"[blue]Artifacts available in {artifacts_dir}[/blue]")
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


@app.command()
def init():
    """Initialize project with sample data and configuration."""
    try:
        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("artifacts").mkdir(exist_ok=True)
        Path("configs").mkdir(exist_ok=True)
        
        # Create sample data
        console.print("[blue]Creating sample data...[/blue]")
        create_sample_data()
        
        # Create default config if not exists
        config_file = Path("configs/default.yaml")
        if not config_file.exists():
            console.print("[blue]Creating default configuration...[/blue]")
            # Copy default config
            import shutil
            shutil.copy2("configs/default.yaml", config_file)
        
        console.print("[green]Project initialized successfully![/green]")
        console.print("Run 'aml run' to start the pipeline")
    
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
