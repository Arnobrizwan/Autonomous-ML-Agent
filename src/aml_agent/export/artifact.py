"""
Artifact export functionality for the Autonomous ML Agent.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..logging import get_logger
from ..types import DatasetProfile, RunMetadata, TaskType, TrialResult

logger = get_logger()


class ArtifactExporter:
    """Export trained models and artifacts."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_pipeline(
        self,
        preprocessor: Any,
        model: Any,
        task_type: TaskType,
        feature_names: List[str],
        target_name: Optional[str] = None,
        format: str = "joblib",
    ) -> Dict[str, str]:
        """
        Export complete ML pipeline.

        Args:
            preprocessor: Fitted preprocessing pipeline
            model: Trained model
            task_type: Task type
            feature_names: List of feature names
            target_name: Target variable name
            format: Export format ("joblib" or "pickle")

        Returns:
            Dictionary with exported file paths
        """
        logger.info(f"Exporting pipeline in {format} format")

        exported_files = {}

        # Export preprocessor
        preprocessor_file = self.output_dir / f"preprocessor.{format}"
        self._export_object(preprocessor, preprocessor_file, format)
        exported_files["preprocessor"] = str(preprocessor_file)

        # Export model
        model_file = self.output_dir / f"model.{format}"
        self._export_object(model, model_file, format)
        exported_files["model"] = str(model_file)

        # Export metadata
        metadata = {
            "task_type": task_type.value,
            "feature_names": feature_names,
            "target_name": target_name,
            "n_features": len(feature_names),
            "export_timestamp": pd.Timestamp.now().isoformat(),
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        exported_files["metadata"] = str(metadata_file)

        # Export feature names
        feature_names_file = self.output_dir / "feature_names.json"
        with open(feature_names_file, "w") as f:
            json.dump(feature_names, f, indent=2)
        exported_files["feature_names"] = str(feature_names_file)

        logger.info(f"Pipeline exported to {self.output_dir}")
        return exported_files

    def export_trial_results(self, trial_results: List[TrialResult]) -> str:
        """Export trial results to CSV."""
        if not trial_results:
            return ""

        # Convert trial results to DataFrame
        data = []
        for result in trial_results:
            data.append(
                {
                    "trial_id": result.trial_id,
                    "model_type": result.model_type.value,
                    "score": result.score,
                    "metric": result.metric.value,
                    "cv_mean": np.mean(result.cv_scores) if result.cv_scores else 0,
                    "cv_std": np.std(result.cv_scores) if result.cv_scores else 0,
                    "fit_time": result.fit_time,
                    "predict_time": result.predict_time,
                    "status": result.status,
                    "timestamp": result.timestamp.isoformat(),
                    "params": json.dumps(result.params),
                }
            )

        df = pd.DataFrame(data)
        results_file = self.output_dir / "trial_results.csv"
        df.to_csv(results_file, index=False)

        logger.info(f"Trial results exported to {results_file}")
        return str(results_file)

    def export_leaderboard(self, leaderboard_data: List[Dict[str, Any]]) -> str:
        """Export leaderboard to CSV."""
        if not leaderboard_data:
            return ""

        df = pd.DataFrame(leaderboard_data)
        leaderboard_file = self.output_dir / "leaderboard.csv"
        df.to_csv(leaderboard_file, index=False)

        logger.info(f"Leaderboard exported to {leaderboard_file}")
        return str(leaderboard_file)

    def export_dataset_profile(self, dataset_profile: DatasetProfile) -> str:
        """Export dataset profile to JSON."""
        profile_data = {
            "n_rows": dataset_profile.n_rows,
            "n_cols": dataset_profile.n_cols,
            "n_numeric": dataset_profile.n_numeric,
            "n_categorical": dataset_profile.n_categorical,
            "n_datetime": dataset_profile.n_datetime,
            "n_text": dataset_profile.n_text,
            "missing_ratio": dataset_profile.missing_ratio,
            "class_balance": dataset_profile.class_balance,
            "task_type": (
                dataset_profile.task_type.value if dataset_profile.task_type else None
            ),
            "target_column": dataset_profile.target_column,
            "feature_columns": dataset_profile.feature_columns,
            "data_hash": dataset_profile.data_hash,
        }

        profile_file = self.output_dir / "dataset_profile.json"
        with open(profile_file, "w") as f:
            json.dump(profile_data, f, indent=2)

        logger.info(f"Dataset profile exported to {profile_file}")
        return str(profile_file)

    def export_run_metadata(self, metadata: RunMetadata) -> str:
        """Export run metadata to JSON."""
        metadata_data = {
            "run_id": metadata.run_id,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
            "status": metadata.status,
            "total_trials": metadata.total_trials,
            "best_score": metadata.best_score,
            "best_model": metadata.best_model,
            "config": metadata.config,
        }

        metadata_file = self.output_dir / "run_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata_data, f, indent=2, default=str)

        logger.info(f"Run metadata exported to {metadata_file}")
        return str(metadata_file)

    def export_feature_importance(
        self, feature_importance: Dict[str, float], model_name: str = "model"
    ) -> str:
        """Export feature importance to JSON."""
        # Sort by importance
        sorted_importance = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        importance_data = {
            "model_name": model_name,
            "feature_importance": dict(sorted_importance),
            "top_features": [feature for feature, _ in sorted_importance[:10]],
        }

        importance_file = self.output_dir / f"{model_name}_feature_importance.json"
        with open(importance_file, "w") as f:
            json.dump(importance_data, f, indent=2)

        logger.info(f"Feature importance exported to {importance_file}")
        return str(importance_file)

    def export_predictions(
        self,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
    ) -> str:
        """Export predictions to CSV."""
        data = {"prediction": predictions}

        if probabilities is not None:
            if probabilities.ndim == 1:
                data["probability"] = probabilities
            else:
                for i in range(probabilities.shape[1]):
                    data[f"probability_class_{i}"] = probabilities[:, i]

        if indices is not None:
            data["index"] = indices

        df = pd.DataFrame(data)
        predictions_file = self.output_dir / "predictions.csv"
        df.to_csv(predictions_file, index=False)

        logger.info(f"Predictions exported to {predictions_file}")
        return str(predictions_file)

    def _export_object(self, obj: Any, file_path: Path, format: str) -> None:
        """Export object to file."""
        if format == "joblib":
            joblib.dump(obj, file_path)
        elif format == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def create_export_summary(self, exported_files: Dict[str, str]) -> str:
        """Create export summary."""
        summary = f"# Export Summary\n\n"
        summary += f"Exported {len(exported_files)} files to {self.output_dir}:\n\n"

        for file_type, file_path in exported_files.items():
            summary += f"- **{file_type}**: `{file_path}`\n"

        summary += f"\nExport completed at {pd.Timestamp.now().isoformat()}\n"

        summary_file = self.output_dir / "export_summary.md"
        with open(summary_file, "w") as f:
            f.write(summary)

        return str(summary_file)


def export_pipeline(
    preprocessor: Any,
    model: Any,
    task_type: TaskType,
    feature_names: List[str],
    output_dir: Path,
    target_name: Optional[str] = None,
    format: str = "joblib",
) -> Dict[str, str]:
    """Export complete ML pipeline."""
    exporter = ArtifactExporter(output_dir)
    return exporter.export_pipeline(
        preprocessor, model, task_type, feature_names, target_name, format
    )
