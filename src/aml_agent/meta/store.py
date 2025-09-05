"""
Meta-learning store for storing and retrieving historical run data.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from ..types import DatasetProfile, TrialResult, ModelType
from ..logging import get_logger

logger = get_logger()


class MetaStore:
    """Store for meta-learning data and historical run information."""
    
    def __init__(self, store_path: str = "artifacts/meta"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.store_path / "meta_store.json"
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load meta data from file."""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load meta store: {e}")
                return {"runs": [], "fingerprints": {}}
        return {"runs": [], "fingerprints": {}}
    
    def _save_data(self):
        """Save meta data to file."""
        try:
            with open(self.meta_file, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save meta store: {e}")
    
    def store_run(self, 
                  run_id: str,
                  dataset_profile: DatasetProfile,
                  trial_results: List[TrialResult],
                  best_params: Dict[str, Any],
                  performance_metrics: Dict[str, float]) -> None:
        """
        Store run data for meta-learning.
        
        Args:
            run_id: Unique run identifier
            dataset_profile: Dataset characteristics
            trial_results: Trial results
            best_params: Best hyperparameters found
            performance_metrics: Performance metrics
        """
        # Create dataset fingerprint
        fingerprint = self._create_fingerprint(dataset_profile)
        
        # Create run record
        run_record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_profile": {
                "n_rows": dataset_profile.n_rows,
                "n_cols": dataset_profile.n_cols,
                "n_numeric": dataset_profile.n_numeric,
                "n_categorical": dataset_profile.n_categorical,
                "n_datetime": dataset_profile.n_datetime,
                "n_text": dataset_profile.n_text,
                "missing_ratio": dataset_profile.missing_ratio,
                "class_balance": dataset_profile.class_balance,
                "task_type": dataset_profile.task_type.value if dataset_profile.task_type else None,
                "data_hash": dataset_profile.data_hash
            },
            "fingerprint": fingerprint,
            "best_params": best_params,
            "performance_metrics": performance_metrics,
            "trial_summary": self._summarize_trials(trial_results)
        }
        
        # Store run
        self.data["runs"].append(run_record)
        
        # Update fingerprint index
        if fingerprint not in self.data["fingerprints"]:
            self.data["fingerprints"][fingerprint] = []
        self.data["fingerprints"][fingerprint].append(run_id)
        
        # Keep only last 1000 runs to prevent file from growing too large
        if len(self.data["runs"]) > 1000:
            self.data["runs"] = self.data["runs"][-1000:]
        
        self._save_data()
        logger.info(f"Stored run {run_id} with fingerprint {fingerprint}")
    
    def _create_fingerprint(self, dataset_profile: DatasetProfile) -> str:
        """Create dataset fingerprint for similarity matching."""
        # Create fingerprint based on dataset characteristics
        fingerprint_data = {
            "n_rows_bin": self._bin_value(dataset_profile.n_rows, [100, 1000, 10000, 100000]),
            "n_cols_bin": self._bin_value(dataset_profile.n_cols, [5, 20, 50, 100]),
            "numeric_ratio": dataset_profile.n_numeric / dataset_profile.n_cols,
            "categorical_ratio": dataset_profile.n_categorical / dataset_profile.n_cols,
            "missing_ratio_bin": self._bin_value(dataset_profile.missing_ratio, [0.01, 0.05, 0.1, 0.2]),
            "task_type": dataset_profile.task_type.value if dataset_profile.task_type else "unknown",
            "class_balance_bin": self._bin_value(dataset_profile.class_balance, [0.1, 0.3, 0.5]) if dataset_profile.class_balance else "unknown"
        }
        
        # Create hash of fingerprint
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def _bin_value(self, value: float, bins: List[float]) -> str:
        """Bin a value into categorical ranges."""
        if value is None:
            return "unknown"
        
        for i, bin_val in enumerate(bins):
            if value <= bin_val:
                return f"bin_{i}"
        return f"bin_{len(bins)}"
    
    def _summarize_trials(self, trial_results: List[TrialResult]) -> Dict[str, Any]:
        """Summarize trial results for storage."""
        if not trial_results:
            return {}
        
        successful_trials = [t for t in trial_results if t.status == "completed"]
        
        if not successful_trials:
            return {"n_trials": len(trial_results), "success_rate": 0.0}
        
        # Group by model type
        model_performance = {}
        for trial in successful_trials:
            model_type = trial.model_type.value
            if model_type not in model_performance:
                model_performance[model_type] = []
            model_performance[model_type].append(trial.score)
        
        # Calculate statistics
        summary = {
            "n_trials": len(trial_results),
            "success_rate": len(successful_trials) / len(trial_results),
            "best_score": max(t.score for t in successful_trials),
            "model_performance": {
                model_type: {
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "n_trials": len(scores)
                }
                for model_type, scores in model_performance.items()
            }
        }
        
        return summary
    
    def find_similar_datasets(self, 
                            dataset_profile: DatasetProfile,
                            max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar datasets based on fingerprint.
        
        Args:
            dataset_profile: Current dataset profile
            max_results: Maximum number of results to return
            
        Returns:
            List of similar run records
        """
        current_fingerprint = self._create_fingerprint(dataset_profile)
        
        # Find exact matches first
        exact_matches = []
        if current_fingerprint in self.data["fingerprints"]:
            for run_id in self.data["fingerprints"][current_fingerprint]:
                run_record = self._get_run_by_id(run_id)
                if run_record:
                    exact_matches.append(run_record)
        
        if exact_matches:
            return exact_matches[:max_results]
        
        # Find similar fingerprints
        similar_runs = []
        for fingerprint, run_ids in self.data["fingerprints"].items():
            similarity = self._calculate_fingerprint_similarity(current_fingerprint, fingerprint)
            if similarity > 0.7:  # Threshold for similarity
                for run_id in run_ids:
                    run_record = self._get_run_by_id(run_id)
                    if run_record:
                        run_record["similarity"] = similarity
                        similar_runs.append(run_record)
        
        # Sort by similarity and return top results
        similar_runs.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return similar_runs[:max_results]
    
    def _calculate_fingerprint_similarity(self, fp1: str, fp2: str) -> float:
        """Calculate similarity between two fingerprints."""
        # For now, use exact match (can be improved with more sophisticated similarity)
        return 1.0 if fp1 == fp2 else 0.0
    
    def _get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run record by ID."""
        for run in self.data["runs"]:
            if run["run_id"] == run_id:
                return run
        return None
    
    def get_best_params_for_model(self, 
                                model_type: ModelType,
                                dataset_profile: DatasetProfile) -> Optional[Dict[str, Any]]:
        """
        Get best parameters for a model type based on similar datasets.
        
        Args:
            model_type: Model type
            dataset_profile: Current dataset profile
            
        Returns:
            Best parameters or None
        """
        similar_runs = self.find_similar_datasets(dataset_profile)
        
        if not similar_runs:
            return None
        
        # Find best parameters for this model type
        best_params = None
        best_score = -np.inf
        
        for run in similar_runs:
            if "best_params" in run and model_type.value in run["best_params"]:
                params = run["best_params"][model_type.value]
                score = run.get("performance_metrics", {}).get("best_score", 0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        return best_params
    
    def get_model_recommendations(self, dataset_profile: DatasetProfile) -> List[ModelType]:
        """
        Get model recommendations based on similar datasets.
        
        Args:
            dataset_profile: Current dataset profile
            
        Returns:
            List of recommended model types
        """
        similar_runs = self.find_similar_datasets(dataset_profile)
        
        if not similar_runs:
            return []
        
        # Count model performance across similar runs
        model_scores = {}
        
        for run in similar_runs:
            trial_summary = run.get("trial_summary", {})
            model_performance = trial_summary.get("model_performance", {})
            
            for model_type, perf in model_performance.items():
                if model_type not in model_scores:
                    model_scores[model_type] = []
                model_scores[model_type].append(perf["mean_score"])
        
        # Calculate average scores and sort
        model_avg_scores = {
            model_type: np.mean(scores)
            for model_type, scores in model_scores.items()
        }
        
        # Sort by average score
        sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert to ModelType enum
        recommendations = []
        for model_type_str, _ in sorted_models:
            try:
                recommendations.append(ModelType(model_type_str))
            except ValueError:
                continue
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get meta store statistics."""
        runs = self.data["runs"]
        
        if not runs:
            return {"total_runs": 0}
        
        # Calculate statistics
        task_types = [run["dataset_profile"]["task_type"] for run in runs if run["dataset_profile"]["task_type"]]
        model_types = set()
        
        for run in runs:
            trial_summary = run.get("trial_summary", {})
            model_performance = trial_summary.get("model_performance", {})
            model_types.update(model_performance.keys())
        
        return {
            "total_runs": len(runs),
            "unique_fingerprints": len(self.data["fingerprints"]),
            "task_types": list(set(task_types)),
            "model_types": list(model_types),
            "date_range": {
                "earliest": min(run["timestamp"] for run in runs),
                "latest": max(run["timestamp"] for run in runs)
            }
        }


def create_meta_store(store_path: str = "artifacts/meta") -> MetaStore:
    """Create meta store instance."""
    return MetaStore(store_path)
