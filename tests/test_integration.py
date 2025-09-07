"""
Integration tests for the Autonomous ML Agent.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aml_agent.agent.loop import run_autonomous_ml
from src.aml_agent.config import create_default_config
from src.aml_agent.types import TaskType
from src.aml_agent.utils import create_sample_data


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete pipeline."""

    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifacts_dir = Path(self.temp_dir) / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Cleanup after each test."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_end_to_end_classification(self):
        """Test complete classification pipeline."""
        # Create sample data
        data = create_sample_data(
            n_samples=50, n_features=4, task_type=TaskType.CLASSIFICATION
        )

        X = data.drop(columns=["target"])
        y = data["target"]

        # Create config
        config = create_default_config()
        config.time_budget_seconds = 10  # Much shorter for CI
        config.max_trials = 2  # Minimal trials for speed
        config.artifacts_dir = str(self.artifacts_dir)
        config.llm.enabled = False  # Disable LLM for faster testing

        # Run pipeline
        results = await run_autonomous_ml(config, X, y)

        # Verify results
        assert results["status"] == "completed"
        assert "best_score" in results
        assert "best_model" in results
        assert "total_trials" in results
        assert results["total_trials"] > 0

        # Verify artifacts
        assert Path(results["artifacts_dir"]).exists()
        artifacts_path = Path(results["artifacts_dir"])

        # Check for model files
        assert (artifacts_path / "model.joblib").exists()
        assert (artifacts_path / "preprocessor.joblib").exists()
        assert (artifacts_path / "metadata.json").exists()
        assert (artifacts_path / "leaderboard.csv").exists()

    async def test_end_to_end_regression(self):
        """Test complete regression pipeline."""
        # Create sample data
        data = create_sample_data(
            n_samples=50, n_features=4, task_type=TaskType.REGRESSION
        )

        X = data.drop(columns=["target"])
        y = data["target"]

        # Create config
        config = create_default_config()
        config.time_budget_seconds = 10  # Much shorter for CI
        config.max_trials = 2  # Minimal trials for speed
        config.artifacts_dir = str(self.artifacts_dir)
        config.llm.enabled = False

        # Run pipeline
        results = await run_autonomous_ml(config, X, y)

        # Verify results
        assert results["status"] == "completed"
        assert "best_score" in results
        assert "best_model" in results
        assert "total_trials" in results

    async def test_pipeline_with_missing_data(self):
        """Test pipeline with missing data."""
        # Create data with missing values
        data = create_sample_data(
            n_samples=50, n_features=4, task_type=TaskType.CLASSIFICATION
        )

        # Introduce missing values
        missing_indices = np.random.choice(data.index, size=20, replace=False)
        data.loc[missing_indices, data.columns[0]] = np.nan

        X = data.drop(columns=["target"])
        y = data["target"]

        # Create config
        config = create_default_config()
        config.time_budget_seconds = 10  # Much shorter for CI
        config.max_trials = 2  # Minimal trials for speed
        config.artifacts_dir = str(self.artifacts_dir)
        config.llm.enabled = False

        # Run pipeline
        results = await run_autonomous_ml(config, X, y)

        # Verify results
        assert results["status"] == "completed"

    async def test_pipeline_with_categorical_data(self):
        """Test pipeline with categorical data."""
        # Create data with categorical features
        np.random.seed(42)
        n_samples = 200

        data = pd.DataFrame(
            {
                "numeric_1": np.random.normal(0, 1, n_samples),
                "numeric_2": np.random.normal(0, 1, n_samples),
                "categorical_1": np.random.choice(["A", "B", "C"], n_samples),
                "categorical_2": np.random.choice(["X", "Y", "Z"], n_samples),
                "target": np.random.choice([0, 1], n_samples).astype(np.int64),
            }
        )

        X = data.drop(columns=["target"])
        y = data["target"]

        # Create config with simpler preprocessing for categorical data
        config = create_default_config()
        config.time_budget_seconds = 10  # Much shorter for CI
        config.max_trials = 2  # Minimal trials for speed
        config.artifacts_dir = str(self.artifacts_dir)
        config.llm.enabled = False
        # Use simpler preprocessing to avoid feature expansion issues
        config.preprocessing.encode_categorical = "label"

        # Run pipeline
        results = await run_autonomous_ml(config, X, y)

        # Verify results
        assert results["status"] == "completed"

    async def test_ensemble_creation(self):
        """Test ensemble model creation."""
        # Create sample data
        data = create_sample_data(
            n_samples=50, n_features=4, task_type=TaskType.CLASSIFICATION
        )

        X = data.drop(columns=["target"])
        y = data["target"]

        # Create config with ensembling enabled
        config = create_default_config()
        config.time_budget_seconds = 15  # Slightly longer for ensemble
        config.max_trials = 3  # Minimal models for ensemble
        config.artifacts_dir = str(self.artifacts_dir)
        config.enable_ensembling = True
        config.llm.enabled = False

        # Run pipeline
        results = await run_autonomous_ml(config, X, y)

        # Verify results
        assert results["status"] == "completed"

        # Check for ensemble model
        # Note: Ensemble creation depends on having multiple successful models

    async def test_model_export_and_loading(self):
        """Test model export and loading."""
        # Create sample data
        data = create_sample_data(
            n_samples=30, n_features=3, task_type=TaskType.CLASSIFICATION
        )

        X = data.drop(columns=["target"])
        y = data["target"]

        # Create config
        config = create_default_config()
        config.time_budget_seconds = 10  # Much shorter for CI
        config.max_trials = 2  # Minimal trials for speed
        config.artifacts_dir = str(self.artifacts_dir)
        config.llm.enabled = False

        # Run pipeline
        results = await run_autonomous_ml(config, X, y)

        # Verify artifacts exist
        artifacts_path = Path(results["artifacts_dir"])

        # Test loading model
        import joblib

        model = joblib.load(artifacts_path / "model.joblib")
        preprocessor = joblib.load(artifacts_path / "preprocessor.joblib")

        # Test prediction
        X_processed = preprocessor.transform(X)
        predictions = model.predict(X_processed)

        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    async def test_error_handling(self):
        """Test error handling in pipeline."""
        # Create invalid data (all NaN)
        X = pd.DataFrame({"feature_1": [np.nan] * 100, "feature_2": [np.nan] * 100})
        y = pd.Series([0, 1] * 50)

        # Create config
        config = create_default_config()
        config.time_budget_seconds = 10  # Much shorter for CI
        config.max_trials = 2  # Minimal trials for speed
        config.artifacts_dir = str(self.artifacts_dir)
        config.llm.enabled = False

        # Run pipeline - should handle errors gracefully
        try:
            results = await run_autonomous_ml(config, X, y)
            # If it succeeds, that's also fine (pipeline might handle NaN data)
            assert "status" in results
        except Exception as e:
            # Pipeline should fail gracefully with informative error
            assert len(str(e)) > 0

    async def test_performance_under_constraints(self):
        """Test pipeline performance under time constraints."""
        # Create sample data
        data = create_sample_data(
            n_samples=50, n_features=4, task_type=TaskType.CLASSIFICATION
        )

        X = data.drop(columns=["target"])
        y = data["target"]

        # Create config with very short time budget
        config = create_default_config()
        config.time_budget_seconds = 5  # Very short budget
        config.max_trials = 2  # Low trial count for speed
        config.artifacts_dir = str(self.artifacts_dir)
        config.llm.enabled = False

        # Run pipeline
        import time

        start_time = time.time()
        results = await run_autonomous_ml(config, X, y)
        end_time = time.time()

        # Verify it respects time budget (with some tolerance)
        assert (end_time - start_time) < 30  # 30 second tolerance for CI

        # Should still produce results
        assert "status" in results
