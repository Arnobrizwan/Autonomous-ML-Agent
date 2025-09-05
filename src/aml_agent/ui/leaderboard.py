"""
Leaderboard display and management for the Autonomous ML Agent.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ..types import LeaderboardEntry, ModelType, MetricType, TrialResult
from ..logging import get_logger

logger = get_logger()
console = Console()


class Leaderboard:
    """Leaderboard for tracking model performance."""
    
    def __init__(self):
        self.entries = []
        self.next_rank = 1
    
    def add_entry(self, entry: LeaderboardEntry) -> None:
        """Add entry to leaderboard."""
        # Set rank
        entry.rank = self.next_rank
        self.next_rank += 1
        
        # Add to entries
        self.entries.append(entry)
        
        # Sort by score
        self.entries.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, entry in enumerate(self.entries):
            entry.rank = i + 1
    
    def add_trial_result(self, result: TrialResult) -> None:
        """Add trial result to leaderboard."""
        if result.status != "completed":
            return
        
        # Calculate CV statistics
        cv_mean = np.mean(result.cv_scores) if result.cv_scores else 0
        cv_std = np.std(result.cv_scores) if result.cv_scores else 0
        
        # Create leaderboard entry
        entry = LeaderboardEntry(
            rank=0,  # Will be set by add_entry
            model_type=result.model_type,
            score=result.score,
            metric=result.metric,
            params=result.params,
            cv_mean=cv_mean,
            cv_std=cv_std,
            fit_time=result.fit_time,
            predict_time=result.predict_time,
            trial_id=result.trial_id
        )
        
        self.add_entry(entry)
    
    def get_top_entries(self, n: int = 10) -> List[LeaderboardEntry]:
        """Get top N entries."""
        return self.entries[:n]
    
    def get_best_entry(self) -> Optional[LeaderboardEntry]:
        """Get best performing entry."""
        return self.entries[0] if self.entries else None
    
    def get_entries_by_model(self, model_type: str) -> List[LeaderboardEntry]:
        """Get entries for specific model type."""
        return [entry for entry in self.entries if entry.model_type.value == model_type]
    
    def display(self, n: int = 10, show_params: bool = False) -> None:
        """Display leaderboard."""
        if not self.entries:
            console.print("[yellow]No entries in leaderboard[/yellow]")
            return
        
        # Create table
        table = Table(title="Model Performance Leaderboard")
        
        # Add columns
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Score", style="green")
        table.add_column("Metric", style="blue")
        table.add_column("CV Mean", style="green")
        table.add_column("CV Std", style="yellow")
        table.add_column("Fit Time", style="red")
        table.add_column("Predict Time", style="red")
        
        if show_params:
            table.add_column("Parameters", style="dim")
        
        # Add rows
        for entry in self.entries[:n]:
            row = [
                str(entry.rank),
                entry.model_type.value,
                f"{entry.score:.4f}",
                entry.metric.value,
                f"{entry.cv_mean:.4f}",
                f"{entry.cv_std:.4f}",
                f"{entry.fit_time:.2f}s",
                f"{entry.predict_time:.4f}s"
            ]
            
            if show_params:
                # Truncate parameters for display
                params_str = str(entry.params)[:50] + "..." if len(str(entry.params)) > 50 else str(entry.params)
                row.append(params_str)
            
            table.add_row(*row)
        
        console.print(table)
    
    def save_csv(self, file_path: Path) -> None:
        """Save leaderboard to CSV file."""
        if not self.entries:
            logger.warning("No entries to save")
            return
        
        # Convert to DataFrame
        data = []
        for entry in self.entries:
            data.append({
                "rank": entry.rank,
                "model_type": entry.model_type.value,
                "score": entry.score,
                "metric": entry.metric.value,
                "cv_mean": entry.cv_mean,
                "cv_std": entry.cv_std,
                "fit_time": entry.fit_time,
                "predict_time": entry.predict_time,
                "trial_id": entry.trial_id,
                "params": str(entry.params)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        
        logger.info(f"Leaderboard saved to {file_path}")
    
    def load_csv(self, file_path: Path) -> None:
        """Load leaderboard from CSV file."""
        if not file_path.exists():
            logger.warning(f"Leaderboard file not found: {file_path}")
            return
        
        try:
            df = pd.read_csv(file_path)
            
            # Clear existing entries
            self.entries = []
            self.next_rank = 1
            
            # Load entries
            for _, row in df.iterrows():
                # Convert strings back to enums
                model_type = ModelType(row["model_type"]) if isinstance(row["model_type"], str) else row["model_type"]
                metric = MetricType(row["metric"]) if isinstance(row["metric"], str) else row["metric"]
                
                entry = LeaderboardEntry(
                    rank=int(row["rank"]),
                    model_type=model_type,
                    score=float(row["score"]),
                    metric=metric,
                    params=eval(row["params"]) if isinstance(row["params"], str) else row["params"],
                    cv_mean=float(row["cv_mean"]),
                    cv_std=float(row["cv_std"]),
                    fit_time=float(row["fit_time"]),
                    predict_time=float(row["predict_time"]),
                    trial_id=int(row["trial_id"])
                )
                self.entries.append(entry)
            
            # Sort by score
            self.entries.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Loaded {len(self.entries)} entries from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load leaderboard: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get leaderboard statistics."""
        if not self.entries:
            return {"total_entries": 0}
        
        scores = [entry.score for entry in self.entries]
        fit_times = [entry.fit_time for entry in self.entries]
        predict_times = [entry.predict_time for entry in self.entries]
        
        # Group by model type
        model_stats = {}
        for entry in self.entries:
            model_type = entry.model_type.value
            if model_type not in model_stats:
                model_stats[model_type] = []
            model_stats[model_type].append(entry.score)
        
        # Calculate statistics
        stats = {
            "total_entries": len(self.entries),
            "best_score": max(scores),
            "worst_score": min(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "mean_fit_time": np.mean(fit_times),
            "mean_predict_time": np.mean(predict_times),
            "model_types": list(model_stats.keys()),
            "model_performance": {
                model_type: {
                    "count": len(scores),
                    "best_score": max(scores),
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores)
                }
                for model_type, scores in model_stats.items()
            }
        }
        
        return stats
    
    def display_statistics(self) -> None:
        """Display leaderboard statistics."""
        stats = self.get_statistics()
        
        if stats["total_entries"] == 0:
            console.print("[yellow]No entries in leaderboard[/yellow]")
            return
        
        # Create statistics panel
        stats_text = f"""
Total Entries: {stats['total_entries']}
Best Score: {stats['best_score']:.4f}
Worst Score: {stats['worst_score']:.4f}
Mean Score: {stats['mean_score']:.4f} ± {stats['std_score']:.4f}
Mean Fit Time: {stats['mean_fit_time']:.2f}s
Mean Predict Time: {stats['mean_predict_time']:.4f}s
Model Types: {', '.join(stats['model_types'])}
"""
        
        panel = Panel(stats_text, title="Leaderboard Statistics", border_style="blue")
        console.print(panel)
        
        # Display model performance
        if stats["model_performance"]:
            console.print("\n[bold]Model Performance Summary:[/bold]")
            for model_type, perf in stats["model_performance"].items():
                console.print(f"  {model_type}: {perf['count']} trials, "
                            f"best={perf['best_score']:.4f}, "
                            f"mean={perf['mean_score']:.4f}±{perf['std_score']:.4f}")


def create_leaderboard() -> Leaderboard:
    """Create new leaderboard instance."""
    return Leaderboard()


def display_leaderboard_from_file(file_path: Path, n: int = 10) -> None:
    """Display leaderboard from CSV file."""
    leaderboard = Leaderboard()
    leaderboard.load_csv(file_path)
    leaderboard.display(n)


def compare_models(leaderboard: Leaderboard, model_types: List[str]) -> None:
    """Compare specific model types."""
    if not model_types:
        console.print("[yellow]No model types specified[/yellow]")
        return
    
    # Filter entries by model types
    filtered_entries = [
        entry for entry in leaderboard.entries
        if entry.model_type.value in model_types
    ]
    
    if not filtered_entries:
        console.print(f"[yellow]No entries found for model types: {', '.join(model_types)}[/yellow]")
        return
    
    # Create comparison table
    table = Table(title=f"Model Comparison: {', '.join(model_types)}")
    
    table.add_column("Model", style="magenta")
    table.add_column("Best Score", style="green")
    table.add_column("Mean Score", style="green")
    table.add_column("Std Score", style="yellow")
    table.add_column("Count", style="cyan")
    table.add_column("Mean Fit Time", style="red")
    
    # Group by model type
    model_groups = {}
    for entry in filtered_entries:
        model_type = entry.model_type.value
        if model_type not in model_groups:
            model_groups[model_type] = []
        model_groups[model_type].append(entry)
    
    # Add rows
    for model_type, entries in model_groups.items():
        scores = [entry.score for entry in entries]
        fit_times = [entry.fit_time for entry in entries]
        
        row = [
            model_type,
            f"{max(scores):.4f}",
            f"{np.mean(scores):.4f}",
            f"{np.std(scores):.4f}",
            str(len(entries)),
            f"{np.mean(fit_times):.2f}s"
        ]
        
        table.add_row(*row)
    
    console.print(table)
