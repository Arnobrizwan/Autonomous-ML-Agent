"""
Artifact export functionality for saving models and metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from ..logging import get_logger
from ..types import RunMetadata

logger = get_logger()


class ArtifactExporter:
    """Export trained models and metadata as reusable artifacts."""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def export_pipeline(
        self,
        pipeline: Pipeline,
        run_id: str,
        metadata: RunMetadata,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Export a complete ML pipeline as a reusable artifact.

        Args:
            pipeline: Fitted sklearn pipeline
            run_id: Unique run identifier
            metadata: Run metadata
            additional_data: Additional data to save

        Returns:
            Dictionary with paths to saved artifacts
        """
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save the main pipeline
        pipeline_path = run_dir / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        saved_paths["pipeline"] = str(pipeline_path)
        logger.info(f"Pipeline saved to {pipeline_path}")

        # Save metadata
        metadata_path = run_dir / "metadata.json"
        self._save_metadata(metadata, metadata_path)
        saved_paths["metadata"] = str(metadata_path)

        # Save additional data if provided
        if additional_data:
            additional_path = run_dir / "additional_data.json"
            with open(additional_path, "w") as f:
                json.dump(additional_data, f, indent=2, default=str)
            saved_paths["additional_data"] = str(additional_path)

        # Save pipeline info
        pipeline_info = self._extract_pipeline_info(pipeline)
        info_path = run_dir / "pipeline_info.json"
        with open(info_path, "w") as f:
            json.dump(pipeline_info, f, indent=2, default=str)
        saved_paths["pipeline_info"] = str(info_path)

        # Create requirements file
        requirements_path = run_dir / "requirements.txt"
        self._create_requirements_file(requirements_path)
        saved_paths["requirements"] = str(requirements_path)

        # Create usage example
        example_path = run_dir / "usage_example.py"
        self._create_usage_example(example_path, run_id)
        saved_paths["usage_example"] = str(example_path)

        logger.info(f"Artifacts exported to {run_dir}")
        return saved_paths

    def export_model_only(
        self, model: BaseEstimator, run_id: str, model_name: str = "model"
    ) -> str:
        """
        Export just the model without pipeline.

        Args:
            model: Trained model
            run_id: Unique run identifier
            model_name: Name for the model file

        Returns:
            Path to saved model
        """
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def export_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        run_id: str,
        data_name: str = "processed_data",
    ) -> Dict[str, str]:
        """
        Export processed data.

        Args:
            X: Feature matrix
            y: Target vector
            run_id: Unique run identifier
            data_name: Name for the data files

        Returns:
            Dictionary with paths to saved data
        """
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save features
        X_path = run_dir / f"{data_name}_X.parquet"
        X.to_parquet(X_path)
        saved_paths["features"] = str(X_path)

        # Save targets
        y_path = run_dir / f"{data_name}_y.parquet"
        y.to_frame().to_parquet(y_path)
        saved_paths["targets"] = str(y_path)

        # Save combined data
        combined_path = run_dir / f"{data_name}_combined.parquet"
        combined_data = pd.concat([X, y], axis=1)
        combined_data.to_parquet(combined_path)
        saved_paths["combined"] = str(combined_path)

        logger.info(f"Data exported to {run_dir}")
        return saved_paths

    def export_leaderboard(self, leaderboard: pd.DataFrame, run_id: str) -> str:
        """
        Export leaderboard results.

        Args:
            leaderboard: Leaderboard DataFrame
            run_id: Unique run identifier

        Returns:
            Path to saved leaderboard
        """
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        leaderboard_path = run_dir / "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        logger.info(f"Leaderboard saved to {leaderboard_path}")
        return str(leaderboard_path)

    def export_explanations(self, explanations: Dict[str, Any], run_id: str) -> str:
        """
        Export model explanations.

        Args:
            explanations: Explanation data
            run_id: Unique run identifier

        Returns:
            Path to saved explanations
        """
        run_dir = self.artifacts_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        explanations_path = run_dir / "explanations.json"
        with open(explanations_path, "w") as f:
            json.dump(explanations, f, indent=2, default=str)
        logger.info(f"Explanations saved to {explanations_path}")
        return str(explanations_path)

    def load_pipeline(self, run_id: str) -> Pipeline:
        """
        Load a saved pipeline.

        Args:
            run_id: Unique run identifier

        Returns:
            Loaded pipeline
        """
        # Try pipeline.joblib first, then model.joblib as fallback
        pipeline_path = self.artifacts_dir / run_id / "pipeline.joblib"
        if not pipeline_path.exists():
            pipeline_path = self.artifacts_dir / run_id / "model.joblib"
            if not pipeline_path.exists():
                raise FileNotFoundError(
                    f"Pipeline not found at {self.artifacts_dir / run_id / 'pipeline.joblib'} "
                    f"or {self.artifacts_dir / run_id / 'model.joblib'}"
                )

        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")
        return pipeline

    def load_preprocessor(self, run_id: str) -> Any:
        """
        Load a saved preprocessor.

        Args:
            run_id: Unique run identifier

        Returns:
            Loaded preprocessor
        """
        preprocessor_path = self.artifacts_dir / run_id / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        return preprocessor

    def load_model(self, run_id: str, model_name: str = "model") -> BaseEstimator:
        """
        Load a saved model.

        Args:
            run_id: Unique run identifier
            model_name: Name of the model file

        Returns:
            Loaded model
        """
        model_path = self.artifacts_dir / run_id / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model

    def load_metadata(self, run_id: str) -> Dict[str, Any]:
        """
        Load run metadata.

        Args:
            run_id: Unique run identifier

        Returns:
            Metadata dictionary
        """
        metadata_path = self.artifacts_dir / run_id / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        logger.info(f"Metadata loaded from {metadata_path}")
        return metadata

    def _save_metadata(self, metadata: RunMetadata, path: Path) -> None:
        """Save metadata to JSON file."""
        metadata_dict = {
            "run_id": metadata.run_id,
            "dataset_hash": metadata.dataset_hash,
            "task_type": metadata.task_type.value,
            "n_rows": metadata.n_rows,
            "n_features": metadata.n_features,
            "n_numeric": metadata.n_numeric,
            "n_categorical": metadata.n_categorical,
            "missing_ratio": metadata.missing_ratio,
            "class_balance": metadata.class_balance,
            "best_model": metadata.best_model,
            "best_score": metadata.best_score,
            "best_params": metadata.best_params,
            "timestamp": metadata.timestamp.isoformat(),
        }

        with open(path, "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

    def _extract_pipeline_info(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Extract information about the pipeline."""
        steps_info = []

        for name, step in pipeline.steps:
            step_info = {
                "name": name,
                "type": type(step).__name__,
                "module": type(step).__module__,
            }

            # Add parameters if available
            if hasattr(step, "get_params"):
                try:
                    params = step.get_params()
                    # Convert numpy types to Python types
                    serializable_params = {}
                    for key, value in params.items():
                        if hasattr(value, "item"):  # numpy scalar
                            serializable_params[key] = value.item()
                        elif isinstance(
                            value, (list, tuple, dict, str, int, float, bool)
                        ):
                            serializable_params[key] = value
                        else:
                            serializable_params[key] = str(value)
                    step_info["parameters"] = serializable_params
                except Exception as e:
                    step_info["parameters"] = {"error": str(e)}

            steps_info.append(step_info)

        return {
            "pipeline_type": type(pipeline).__name__,
            "n_steps": len(pipeline.steps),
            "steps": steps_info,
            "created_at": datetime.now().isoformat(),
        }

    def _create_requirements_file(self, path: Path) -> None:
        """Create requirements.txt file for the artifact."""
        requirements = [
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "joblib>=1.3.0",
        ]

        with open(path, "w") as f:
            f.write("\n".join(requirements))

    def _create_usage_example(self, path: Path, run_id: str) -> None:
        """Create usage example for the artifact."""
        example_code = f'''"""
Usage example for ML pipeline artifact: {run_id}
"""

import joblib
import pandas as pd
import numpy as np

# Load the pipeline
pipeline = joblib.load("pipeline.joblib")

# Example: Load new data
# new_data = pd.read_csv("new_data.csv")
# new_data = new_data.drop(columns=["target"])  # Remove target if present

# Example: Make predictions
# predictions = pipeline.predict(new_data)
# probabilities = pipeline.predict_proba(new_data)  # If classification

# Example: Get feature names after preprocessing
# feature_names = pipeline.get_feature_names_out()

print("Pipeline loaded successfully!")
print(f"Pipeline steps: {{[step[0] for step in pipeline.steps]}}")
'''

        with open(path, "w") as f:
            f.write(example_code)

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """List all available artifacts."""
        artifacts = []

        for run_dir in self.artifacts_dir.iterdir():
            if run_dir.is_dir():
                metadata_path = run_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        artifacts.append(
                            {
                                "run_id": run_dir.name,
                                "metadata": metadata,
                                "path": str(run_dir),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {run_dir}: {e}")

        return sorted(
            artifacts, key=lambda x: x["metadata"].get("timestamp", ""), reverse=True
        )

    def cleanup_old_artifacts(self, keep_last_n: int = 10) -> None:
        """Clean up old artifacts, keeping only the last N."""
        artifacts = self.list_artifacts()

        if len(artifacts) <= keep_last_n:
            return

        # Remove old artifacts
        for artifact in artifacts[keep_last_n:]:
            import shutil

            shutil.rmtree(artifact["path"])
            logger.info(f"Removed old artifact: {artifact['run_id']}")

        logger.info(f"Cleaned up {len(artifacts) - keep_last_n} old artifacts")
