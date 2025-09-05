"""
Smoke tests for the Autonomous ML Agent.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.aml_agent.config import create_default_config
from src.aml_agent.agent.loop import run_autonomous_ml
from src.aml_agent.utils import create_sample_data
from src.aml_agent.types import TaskType


class TestSmoke:
    """Smoke tests to verify basic functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.artifacts_dir = Path(self.temp_dir) / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        # Test classification data
        data_cls = create_sample_data(n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION)
        assert data_cls.shape == (100, 6)  # 5 features + 1 target
        assert data_cls['target'].nunique() <= 2  # Binary classification
        
        # Test regression data
        data_reg = create_sample_data(n_samples=100, n_features=5, task_type=TaskType.REGRESSION)
        assert data_reg.shape == (100, 6)  # 5 features + 1 target
        assert data_reg['target'].dtype in ['float64', 'int64']  # Numeric target
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = create_default_config()
        assert config.time_budget_seconds > 0
        assert config.max_trials > 0
        assert config.cv_folds >= 2
        assert config.random_seed >= 0
    
    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline."""
        from src.aml_agent.preprocess import PreprocessingPipeline
        
        # Create sample data
        data = create_sample_data(n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION)
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Create and fit pipeline
        pipeline = PreprocessingPipeline()
        X_processed = pipeline.fit_transform(X, y)
        
        assert X_processed.shape[0] == X.shape[0]  # Same number of rows
        assert X_processed.shape[1] >= X.shape[1]  # May have more columns after encoding
        assert not X_processed.isnull().any().any()  # No missing values
    
    def test_model_training(self):
        """Test model training."""
        from src.aml_agent.models import ModelTrainer
        from src.aml_agent.types import ModelType, TaskType
        
        # Create sample data
        data = create_sample_data(n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION)
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Train a simple model
        trainer = ModelTrainer(task_type=TaskType.CLASSIFICATION)
        
        # Test single model training
        result = trainer.train_model(
            model_type=ModelType.LOGISTIC_REGRESSION,
            X=X,
            y=y,
            params={'random_state': 42}
        )
        
        assert result.status == "completed"
        assert result.score > 0
        assert result.fit_time > 0
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        from src.aml_agent.models import ModelTrainer
        from src.aml_agent.types import ModelType, TaskType
        
        # Create sample data
        data = create_sample_data(n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION)
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Run optimization
        trainer = ModelTrainer(task_type=TaskType.CLASSIFICATION)
        results = trainer.optimize_hyperparameters(
            model_type=ModelType.LOGISTIC_REGRESSION,
            X=X,
            y=y,
            n_trials=3
        )
        
        assert len(results) > 0
        assert all(result.status == "completed" for result in results)
        assert all(result.score > 0 for result in results)
    
    def test_leaderboard(self):
        """Test leaderboard functionality."""
        from src.aml_agent.ui.leaderboard import Leaderboard
        from src.aml_agent.types import LeaderboardEntry, ModelType, MetricType
        
        # Create leaderboard
        leaderboard = Leaderboard()
        
        # Add test entries
        entry1 = LeaderboardEntry(
            rank=1,
            model_type=ModelType.LOGISTIC_REGRESSION,
            score=0.85,
            metric=MetricType.ACCURACY,
            params={'C': 1.0},
            cv_mean=0.83,
            cv_std=0.02,
            fit_time=1.5,
            predict_time=0.01,
            trial_id=1
        )
        
        entry2 = LeaderboardEntry(
            rank=2,
            model_type=ModelType.RANDOM_FOREST,
            score=0.82,
            metric=MetricType.ACCURACY,
            params={'n_estimators': 100},
            cv_mean=0.80,
            cv_std=0.03,
            fit_time=2.1,
            predict_time=0.02,
            trial_id=2
        )
        
        leaderboard.add_entry(entry1)
        leaderboard.add_entry(entry2)
        
        # Test functionality
        assert len(leaderboard.entries) == 2
        assert leaderboard.get_best_entry().score == 0.85
        assert len(leaderboard.get_top_entries(1)) == 1
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        from src.aml_agent.interpret.importance import FeatureImportanceAnalyzer
        from src.aml_agent.models.registries import get_model_factory
        from src.aml_agent.types import ModelType, TaskType
        
        # Create sample data
        data = create_sample_data(n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION)
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Train model
        model = get_model_factory(ModelType.RANDOM_FOREST, TaskType.CLASSIFICATION)
        model.fit(X, y)
        
        # Calculate importance
        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.get_feature_importance(model, X, y)
        
        assert len(importance) == X.shape[1]
        assert all(score >= 0 for score in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 0.01  # Should sum to 1
    
    def test_ensemble_creation(self):
        """Test ensemble model creation."""
        from src.aml_agent.models.ensemble import EnsembleBuilder
        from src.aml_agent.types import TaskType, TrialResult, ModelType, MetricType
        from datetime import datetime
        
        # Create sample data
        data = create_sample_data(n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION)
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Create mock trial results
        trial_results = [
            TrialResult(
                trial_id=1,
                model_type=ModelType.LOGISTIC_REGRESSION,
                params={'C': 1.0},
                score=0.85,
                metric=MetricType.ACCURACY,
                cv_scores=[0.83, 0.84, 0.86, 0.85, 0.87],
                fit_time=1.5,
                predict_time=0.01,
                timestamp=datetime.now()
            ),
            TrialResult(
                trial_id=2,
                model_type=ModelType.RANDOM_FOREST,
                params={'n_estimators': 100},
                score=0.82,
                metric=MetricType.ACCURACY,
                cv_scores=[0.80, 0.81, 0.83, 0.82, 0.84],
                fit_time=2.1,
                predict_time=0.02,
                timestamp=datetime.now()
            )
        ]
        
        # Create ensemble
        builder = EnsembleBuilder(TaskType.CLASSIFICATION)
        ensemble = builder.create_ensemble(
            trial_results=trial_results,
            top_k=2,
            method="voting",
            X=X,
            y=y
        )
        
        assert ensemble is not None
        assert hasattr(ensemble, 'predict')
    
    def test_model_card_generation(self):
        """Test model card generation."""
        from src.aml_agent.export.model_card import ModelCardGenerator
        from src.aml_agent.types import TaskType, TrialResult, ModelType, MetricType, DatasetProfile
        from datetime import datetime
        
        # Create mock data
        trial_results = [
            TrialResult(
                trial_id=1,
                model_type=ModelType.LOGISTIC_REGRESSION,
                params={'C': 1.0},
                score=0.85,
                metric=MetricType.ACCURACY,
                cv_scores=[0.83, 0.84, 0.86, 0.85, 0.87],
                fit_time=1.5,
                predict_time=0.01,
                timestamp=datetime.now()
            )
        ]
        
        dataset_profile = DatasetProfile(
            n_rows=100,
            n_cols=6,
            n_numeric=5,
            n_categorical=0,
            n_datetime=0,
            n_text=0,
            missing_ratio=0.0,
            class_balance=0.5,
            task_type=TaskType.CLASSIFICATION,
            target_column='target',
            feature_columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'],
            data_hash='test_hash'
        )
        
        # Generate model card
        generator = ModelCardGenerator()
        model_card = generator.generate_card(
            trial_results=trial_results,
            task_type=TaskType.CLASSIFICATION,
            dataset_profile=dataset_profile
        )
        
        assert model_card.model_name is not None
        assert model_card.task_type == TaskType.CLASSIFICATION
        assert len(model_card.performance_metrics) > 0
        assert len(model_card.limitations) > 0
        assert len(model_card.recommendations) > 0
    
    def test_meta_store(self):
        """Test meta-learning store."""
        from src.aml_agent.meta.store import MetaStore
        from src.aml_agent.types import DatasetProfile, TrialResult, ModelType, MetricType
        from datetime import datetime
        
        # Create meta store
        store = MetaStore(str(self.artifacts_dir / "meta"))
        
        # Create mock data
        dataset_profile = DatasetProfile(
            n_rows=100,
            n_cols=6,
            n_numeric=5,
            n_categorical=0,
            n_datetime=0,
            n_text=0,
            missing_ratio=0.0,
            class_balance=0.5,
            task_type=TaskType.CLASSIFICATION,
            target_column='target',
            feature_columns=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'],
            data_hash='test_hash'
        )
        
        trial_results = [
            TrialResult(
                trial_id=1,
                model_type=ModelType.LOGISTIC_REGRESSION,
                params={'C': 1.0},
                score=0.85,
                metric=MetricType.ACCURACY,
                cv_scores=[0.83, 0.84, 0.86, 0.85, 0.87],
                fit_time=1.5,
                predict_time=0.01,
                timestamp=datetime.now()
            )
        ]
        
        # Store run
        store.store_run(
            run_id="test_run",
            dataset_profile=dataset_profile,
            trial_results=trial_results,
            best_params={'logistic_regression': {'C': 1.0}},
            performance_metrics={'accuracy': 0.85}
        )
        
        # Test retrieval
        similar_runs = store.find_similar_datasets(dataset_profile)
        assert len(similar_runs) > 0
        
        best_params = store.get_best_params_for_model(ModelType.LOGISTIC_REGRESSION, dataset_profile)
        assert best_params is not None
        assert 'C' in best_params
    
    def test_fastapi_service(self):
        """Test FastAPI service creation."""
        from src.aml_agent.service.app import create_app
        
        # Create test artifacts directory
        test_artifacts = self.artifacts_dir / "test_run"
        test_artifacts.mkdir(parents=True, exist_ok=True)
        
        # Create mock model file
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        model = LogisticRegression(random_state=42)
        model.fit([[1, 2], [3, 4]], [0, 1])
        joblib.dump(model, test_artifacts / "model.joblib")
        
        # Create mock metadata
        import json
        metadata = {
            "task_type": "classification",
            "model_type": "logistic_regression",
            "performance_metrics": {"accuracy": 0.85}
        }
        with open(test_artifacts / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create mock feature names
        feature_names = ["feature_0", "feature_1"]
        with open(test_artifacts / "feature_names.json", 'w') as f:
            json.dump(feature_names, f)
        
        # Create app
        app = create_app(test_artifacts)
        assert app is not None
        assert hasattr(app, 'routes')
    
    def test_end_to_end_smoke(self):
        """Test end-to-end pipeline with minimal configuration."""
        # Create sample data
        data = create_sample_data(n_samples=50, n_features=3, task_type=TaskType.CLASSIFICATION)
        
        # Create minimal config
        config = create_default_config()
        config.time_budget_seconds = 30  # Short budget for testing
        config.max_trials = 5
        config.data_path = str(Path(self.temp_dir) / "test_data.csv")
        
        # Save data
        data.to_csv(config.data_path, index=False)
        
        # Run pipeline
        try:
            results = run_autonomous_ml(config, data.drop(columns=['target']), data['target'])
            
            # Verify results
            assert results['status'] == 'completed'
            assert 'run_id' in results
            assert 'best_score' in results
            assert 'artifacts_dir' in results
            
            # Verify artifacts were created
            artifacts_dir = Path(results['artifacts_dir'])
            assert artifacts_dir.exists()
            assert (artifacts_dir / "metadata.json").exists()
            assert (artifacts_dir / "leaderboard.csv").exists()
            
        except Exception as e:
            pytest.fail(f"End-to-end test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
