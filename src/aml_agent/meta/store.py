"""
Meta-learning store for storing and retrieving dataset fingerprints and best hyperparameters.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging import get_logger
from ..types import DatasetProfile, RunMetadata

logger = get_logger()


class MetaStore:
    """Store for meta-learning data."""

    def __init__(self, store_path: str = "artifacts/meta"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.meta_file = self.store_path / "meta_store.json"
        self._load_store()

    def _load_store(self) -> None:
        """Load existing meta store."""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r") as f:
                    self.store = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.store = {"runs": [], "fingerprints": []}
        else:
            self.store = {"runs": [], "fingerprints": []}

    def _save_store(self) -> None:
        """Save meta store to disk."""
        with open(self.meta_file, "w") as f:
            json.dump(self.store, f, indent=2)

    def upsert_run(self, run_metadata: RunMetadata) -> None:
        """Add or update run metadata."""
        # Check if run already exists
        existing_idx = None
        for i, run in enumerate(self.store["runs"]):
            if run["run_id"] == run_metadata.run_id:
                existing_idx = i
                break

        # Convert to dict for JSON serialization
        run_dict = {
            "run_id": run_metadata.run_id,
            "dataset_hash": run_metadata.dataset_hash,
            "task_type": run_metadata.task_type.value,
            "n_rows": run_metadata.n_rows,
            "n_features": run_metadata.n_features,
            "n_numeric": run_metadata.n_numeric,
            "n_categorical": run_metadata.n_categorical,
            "missing_ratio": run_metadata.missing_ratio,
            "class_balance": run_metadata.class_balance,
            "best_model": run_metadata.best_model,
            "best_score": run_metadata.best_score,
            "best_params": run_metadata.best_params,
            "timestamp": run_metadata.timestamp.isoformat(),
        }

        if existing_idx is not None:
            self.store["runs"][existing_idx] = run_dict
        else:
            self.store["runs"].append(run_dict)

        self._save_store()
        logger.info(f"Upserted run metadata for {run_metadata.run_id}")

    def query_similar(
        self, profile: DatasetProfile, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query similar datasets based on profile."""
        if not self.store["runs"]:
            return []

        similarities = []
        for run in self.store["runs"]:
            similarity = self._calculate_similarity(profile, run)
            similarities.append((similarity, run))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [run for _, run in similarities[:top_k]]

    def _calculate_similarity(
        self, profile: DatasetProfile, run: Dict[str, Any]
    ) -> float:
        """Calculate similarity between dataset profile and stored run."""
        # Weighted similarity based on key characteristics
        weights = {
            "task_type": 0.3,
            "n_features": 0.2,
            "n_rows": 0.1,
            "n_numeric": 0.1,
            "n_categorical": 0.1,
            "missing_ratio": 0.1,
            "class_balance": 0.1,
        }

        similarity = 0.0
        total_weight = 0.0

        # Task type match
        if profile.task_type and run["task_type"]:
            if profile.task_type.value == run["task_type"]:
                similarity += weights["task_type"]
            total_weight += weights["task_type"]

        # Feature count similarity (normalized)
        if profile.n_cols and run["n_features"]:
            feature_sim = 1.0 - abs(profile.n_cols - run["n_features"]) / max(
                profile.n_cols, run["n_features"]
            )
            similarity += weights["n_features"] * feature_sim
            total_weight += weights["n_features"]

        # Row count similarity (normalized)
        if profile.n_rows and run["n_rows"]:
            row_sim = 1.0 - abs(profile.n_rows - run["n_rows"]) / max(
                profile.n_rows, run["n_rows"]
            )
            similarity += weights["n_rows"] * row_sim
            total_weight += weights["n_rows"]

        # Numeric features similarity
        if profile.n_numeric and run["n_numeric"]:
            numeric_sim = 1.0 - abs(profile.n_numeric - run["n_numeric"]) / max(
                profile.n_numeric, run["n_numeric"]
            )
            similarity += weights["n_numeric"] * numeric_sim
            total_weight += weights["n_numeric"]

        # Categorical features similarity
        if profile.n_categorical and run["n_categorical"]:
            cat_sim = 1.0 - abs(profile.n_categorical - run["n_categorical"]) / max(
                profile.n_categorical, run["n_categorical"]
            )
            similarity += weights["n_categorical"] * cat_sim
            total_weight += weights["n_categorical"]

        # Missing ratio similarity
        if profile.missing_ratio is not None and run["missing_ratio"] is not None:
            missing_sim = 1.0 - abs(profile.missing_ratio - run["missing_ratio"])
            similarity += weights["missing_ratio"] * missing_sim
            total_weight += weights["missing_ratio"]

        # Class balance similarity
        if profile.class_balance is not None and run["class_balance"] is not None:
            balance_sim = 1.0 - abs(profile.class_balance - run["class_balance"])
            similarity += weights["class_balance"] * balance_sim
            total_weight += weights["class_balance"]

        # Normalize by total weight
        if total_weight > 0:
            similarity = similarity / total_weight

        return similarity

    def get_best_params(
        self, model_type: str, similar_runs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get best parameters for a model type from similar runs."""
        if not similar_runs:
            return None

        # Find best scoring run with this model type
        best_run = None
        best_score = -float("inf")

        for run in similar_runs:
            if run["best_model"] == model_type and run["best_score"] > best_score:
                best_run = run
                best_score = run["best_score"]

        return best_run["best_params"] if best_run else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored runs."""
        if not self.store["runs"]:
            return {"total_runs": 0}

        runs = self.store["runs"]
        task_types = [run["task_type"] for run in runs if run["task_type"]]
        model_types = [run["best_model"] for run in runs if run["best_model"]]

        return {
            "total_runs": len(runs),
            "task_types": list(set(task_types)),
            "model_types": list(set(model_types)),
            "avg_score": sum(run["best_score"] for run in runs if run["best_score"])
            / len(runs),
        }
