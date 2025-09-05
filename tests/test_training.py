"""
Tests for model training and evaluation.
"""

import pytest

from src.aml_agent.models import ModelRegistry, ModelTrainer, get_model_factory
from src.aml_agent.models.ensemble import EnsembleBuilder, create_ensemble
from src.aml_agent.models.spaces import SearchSpaceGenerator, create_optuna_study
from src.aml_agent.types import ModelType, SearchStrategy, TaskType
from src.aml_agent.utils import create_sample_data


class TestModelTraining:
    """Test model training and evaluation."""

    def test_model_registry(self):
        """Test model registry functionality."""
        registry = ModelRegistry()

        # Test model class retrieval
        lr_classifier = registry.get_model_class(
            ModelType.LOGISTIC_REGRESSION, TaskType.CLASSIFICATION
        )
        assert lr_classifier is not None

        lr_regressor = registry.get_model_class(
            ModelType.LINEAR_REGRESSION, TaskType.REGRESSION
        )
        assert lr_regressor is not None

        # Test model support
        assert registry.supports_class_weight(ModelType.LOGISTIC_REGRESSION)
        assert not registry.supports_class_weight(ModelType.LINEAR_REGRESSION)

        # Test available models
        cls_models = registry.get_available_models(TaskType.CLASSIFICATION)
        reg_models = registry.get_available_models(TaskType.REGRESSION)

        assert ModelType.LOGISTIC_REGRESSION in cls_models
        assert ModelType.LINEAR_REGRESSION in reg_models

    def test_model_factory(self):
        """Test model factory creation."""
        # Test classification model
        model = get_model_factory(
            ModelType.LOGISTIC_REGRESSION, TaskType.CLASSIFICATION
        )
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

        # Test regression model
        model = get_model_factory(ModelType.LINEAR_REGRESSION, TaskType.REGRESSION)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_model_training(self):
        """Test model training."""
        # Create sample data
        data = create_sample_data(
            n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION
        )
        X = data.drop(columns=["target"])
        y = data["target"]

        # Test trainer
        trainer = ModelTrainer(task_type=TaskType.CLASSIFICATION)

        # Test single model training
        result = trainer.train_model(
            model_type=ModelType.LOGISTIC_REGRESSION,
            X=X,
            y=y,
            params={"random_state": 42, "max_iter": 100},
        )

        assert result.status == "completed"
        assert result.score > 0
        assert result.fit_time > 0
        assert result.predict_time >= 0
        assert len(result.cv_scores) > 0

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        # Create sample data
        data = create_sample_data(
            n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION
        )
        X = data.drop(columns=["target"])
        y = data["target"]

        # Test trainer
        trainer = ModelTrainer(task_type=TaskType.CLASSIFICATION)

        # Test optimization
        results = trainer.optimize_hyperparameters(
            model_type=ModelType.LOGISTIC_REGRESSION, X=X, y=y, n_trials=3
        )

        assert len(results) > 0
        assert all(result.status == "completed" for result in results)
        assert all(result.score > 0 for result in results)

    def test_search_space_generator(self):
        """Test search space generation."""
        generator = SearchSpaceGenerator()

        # Test search space retrieval
        space = generator.get_search_space(ModelType.LOGISTIC_REGRESSION)
        assert "C" in space
        assert "penalty" in space
        assert "solver" in space

        # Test Optuna space creation
        optuna_space = generator.create_optuna_space(ModelType.LOGISTIC_REGRESSION)
        assert "C" in optuna_space
        assert "penalty" in optuna_space

    def test_optuna_study_creation(self):
        """Test Optuna study creation."""
        # Test study creation
        study = create_optuna_study(
            model_type=ModelType.LOGISTIC_REGRESSION, strategy=SearchStrategy.BAYES
        )

        assert study is not None
        from optuna.study import StudyDirection

        assert study.direction == StudyDirection.MAXIMIZE

    def test_ensemble_creation(self):
        """Test ensemble model creation."""
        from datetime import datetime

        from src.aml_agent.types import MetricType, TrialResult

        # Create sample data
        data = create_sample_data(
            n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION
        )
        X = data.drop(columns=["target"])
        y = data["target"]

        # Create mock trial results
        trial_results = [
            TrialResult(
                trial_id=1,
                model_type=ModelType.LOGISTIC_REGRESSION,
                params={"C": 1.0, "random_state": 42},
                score=0.85,
                metric=MetricType.ACCURACY,
                cv_scores=[0.83, 0.84, 0.86, 0.85, 0.87],
                fit_time=1.5,
                predict_time=0.01,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id=2,
                model_type=ModelType.RANDOM_FOREST,
                params={"n_estimators": 100, "random_state": 42},
                score=0.82,
                metric=MetricType.ACCURACY,
                cv_scores=[0.80, 0.81, 0.83, 0.82, 0.84],
                fit_time=2.1,
                predict_time=0.02,
                timestamp=datetime.now(),
            ),
        ]

        # Test ensemble builder
        builder = EnsembleBuilder(TaskType.CLASSIFICATION)

        # Test voting ensemble
        voting_ensemble = builder.create_ensemble(
            trial_results=trial_results, top_k=2, method="voting", X=X, y=y
        )

        assert voting_ensemble is not None
        assert hasattr(voting_ensemble, "predict")

        # Test ensemble evaluation
        performance = builder.evaluate_ensemble(voting_ensemble, X, y)
        assert "cv_mean" in performance
        assert "cv_std" in performance

    def test_ensemble_with_different_methods(self):
        """Test different ensemble methods."""
        from datetime import datetime

        from src.aml_agent.types import MetricType, TrialResult

        # Create sample data
        data = create_sample_data(
            n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION
        )
        X = data.drop(columns=["target"])
        y = data["target"]

        # Create mock trial results
        trial_results = [
            TrialResult(
                trial_id=1,
                model_type=ModelType.LOGISTIC_REGRESSION,
                params={"C": 1.0, "random_state": 42},
                score=0.85,
                metric=MetricType.ACCURACY,
                cv_scores=[0.83, 0.84, 0.86, 0.85, 0.87],
                fit_time=1.5,
                predict_time=0.01,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id=2,
                model_type=ModelType.RANDOM_FOREST,
                params={"n_estimators": 100, "random_state": 42},
                score=0.82,
                metric=MetricType.ACCURACY,
                cv_scores=[0.80, 0.81, 0.83, 0.82, 0.84],
                fit_time=2.1,
                predict_time=0.02,
                timestamp=datetime.now(),
            ),
        ]

        # Test different ensemble methods
        methods = ["voting", "blending"]

        for method in methods:
            ensemble = create_ensemble(
                trial_results=trial_results,
                task_type=TaskType.CLASSIFICATION,
                top_k=2,
                method=method,
                X=X,
                y=y,
            )

            assert ensemble is not None
            assert hasattr(ensemble, "predict")

    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics."""
        from src.aml_agent.models.train_eval import cross_validate_model, evaluate_model

        # Create sample data
        data = create_sample_data(
            n_samples=100, n_features=5, task_type=TaskType.CLASSIFICATION
        )
        X = data.drop(columns=["target"])
        y = data["target"]

        # Train a model
        model = get_model_factory(
            ModelType.LOGISTIC_REGRESSION, TaskType.CLASSIFICATION
        )
        model.fit(X, y)

        # Test evaluation
        metrics = evaluate_model(model, X, y, TaskType.CLASSIFICATION)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(0 <= score <= 1 for score in metrics.values())

        # Test cross-validation
        cv_scores, cv_metrics = cross_validate_model(
            model, X, y, TaskType.CLASSIFICATION, cv_folds=3
        )

        assert len(cv_scores) == 3
        assert "accuracy" in cv_metrics
        assert all(0 <= score <= 1 for score in cv_scores)

    def test_regression_models(self):
        """Test regression models."""
        # Create regression data
        data = create_sample_data(
            n_samples=100, n_features=5, task_type=TaskType.REGRESSION
        )
        X = data.drop(columns=["target"])
        y = data["target"]

        # Test trainer
        trainer = ModelTrainer(task_type=TaskType.REGRESSION)

        # Test regression model training
        result = trainer.train_model(
            model_type=ModelType.LINEAR_REGRESSION, X=X, y=y, params={}
        )

        assert result.status == "completed"
        assert (
            result.score > -1
        )  # R2 score should be reasonable (can be negative for poor models)

    def test_model_performance_summary(self):
        """Test model performance summary."""
        from datetime import datetime

        from src.aml_agent.models.train_eval import get_model_performance_summary
        from src.aml_agent.types import MetricType, TrialResult

        # Create mock trial results
        trial_results = [
            TrialResult(
                trial_id=1,
                model_type=ModelType.LOGISTIC_REGRESSION,
                params={"C": 1.0},
                score=0.85,
                metric=MetricType.ACCURACY,
                cv_scores=[0.83, 0.84, 0.86, 0.85, 0.87],
                fit_time=1.5,
                predict_time=0.01,
                timestamp=datetime.now(),
            ),
            TrialResult(
                trial_id=2,
                model_type=ModelType.RANDOM_FOREST,
                params={"n_estimators": 100},
                score=0.82,
                metric=MetricType.ACCURACY,
                cv_scores=[0.80, 0.81, 0.83, 0.82, 0.84],
                fit_time=2.1,
                predict_time=0.02,
                timestamp=datetime.now(),
            ),
        ]

        # Test performance summary
        summary = get_model_performance_summary(trial_results)

        assert "total_trials" in summary
        assert "successful_trials" in summary
        assert "best_score" in summary
        assert "mean_score" in summary
        assert "best_model" in summary
        assert summary["total_trials"] == 2
        assert summary["successful_trials"] == 2
        assert summary["best_score"] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])
