"""
Model training and evaluation utilities.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# make_scorer imported later when needed
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from ..logging import get_logger
from ..types import (
    BudgetClock,
    MetricType,
    ModelType,
    TaskType,
    TrialResult,
)
from ..utils import calculate_metrics, select_metric
from .registries import get_model_factory, validate_model_params
from .spaces import get_search_space

logger = get_logger()


class ModelTrainer:
    """Train and evaluate models with cross-validation."""

    def __init__(
        self,
        task_type: TaskType,
        metric: MetricType,
        cv_folds: int = 5,
        random_seed: int = 42,
    ):
        self.task_type = task_type
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.scorer = self._create_scorer()

    def _create_scorer(self):
        """Create sklearn scorer for the metric."""
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
            make_scorer,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )

        metric_map = {
            MetricType.ACCURACY: make_scorer(accuracy_score),
            MetricType.PRECISION: make_scorer(precision_score, average="weighted"),
            MetricType.RECALL: make_scorer(recall_score, average="weighted"),
            MetricType.F1: make_scorer(f1_score, average="weighted"),
            MetricType.F1_MACRO: make_scorer(f1_score, average="macro"),
            MetricType.F1_WEIGHTED: make_scorer(f1_score, average="weighted"),
            MetricType.AUC: make_scorer(
                roc_auc_score, multi_class="ovr", average="weighted"
            ),
            MetricType.BALANCED_ACCURACY: make_scorer(balanced_accuracy_score),
            MetricType.MAE: make_scorer(mean_absolute_error, greater_is_better=False),
            MetricType.MSE: make_scorer(mean_squared_error, greater_is_better=False),
            MetricType.RMSE: make_scorer(
                mean_squared_error, greater_is_better=False, squared=False
            ),
            MetricType.R2: make_scorer(r2_score),
        }

        return metric_map.get(self.metric, make_scorer(f1_score, average="weighted"))

    def train_model(
        self,
        model_type: ModelType,
        X: pd.DataFrame,
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseEstimator:
        """
        Train a single model.

        Args:
            model_type: Type of model to train
            X: Feature matrix
            y: Target vector
            params: Model parameters

        Returns:
            Trained model
        """
        logger.info(f"Training {model_type.value} model")

        # Validate parameters
        if params:
            params = validate_model_params(model_type, params)

        # Create model
        model: Any = get_model_factory(model_type, self.task_type, params)

        # Train model
        start_time = time.time()
        model.fit(X, y)
        fit_time = time.time() - start_time

        logger.info(f"Model trained in {fit_time:.2f} seconds")
        return model

    def evaluate_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv: Whether to use cross-validation

        Returns:
            Evaluation results
        """
        if cv:
            return self._evaluate_with_cv(model, X, y)
        else:
            return self._evaluate_single_split(model, X, y)

    def _evaluate_with_cv(
        self, model: Any, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate model with cross-validation."""
        # Choose CV strategy
        if self.task_type == TaskType.CLASSIFICATION:
            cv_strategy: Any = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            )
        else:
            cv_strategy = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            )

        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, cv=cv_strategy, scoring=self.scorer, n_jobs=-1
        )

        # Calculate metrics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Fit model on full dataset for additional metrics
        model.fit(X, y)

        # Get additional metrics on full dataset
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)

        metrics = calculate_metrics(
            np.asarray(y.values), y_pred, y_prob, self.task_type
        )

        return {
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "metrics": metrics,
            "score": cv_mean,  # Primary score for optimization
        }

    def _evaluate_single_split(
        self, model: Any, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate model on single train/test split."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_seed,
            stratify=y if self.task_type == TaskType.CLASSIFICATION else None,
        )

        # Train on split
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_test.values, y_pred, y_prob, self.task_type)

        # Use primary metric as score
        score = metrics.get(self.metric.value, 0.0)

        return {
            "cv_scores": [score],
            "cv_mean": score,
            "cv_std": 0.0,
            "metrics": metrics,
            "score": score,
        }

    def _sample_hyperparameters(
        self, trial, search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        import optuna

        params = {}
        for param, distribution in search_space.items():
            if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                params[param] = trial.suggest_categorical(param, distribution.choices)
            elif isinstance(distribution, optuna.distributions.IntDistribution):
                params[param] = trial.suggest_int(
                    param, distribution.low, distribution.high, step=distribution.step
                )
            elif isinstance(distribution, optuna.distributions.FloatDistribution):
                if distribution.log:
                    params[param] = trial.suggest_float(
                        param, distribution.low, distribution.high, log=True
                    )
                else:
                    params[param] = trial.suggest_float(
                        param, distribution.low, distribution.high
                    )
        return params

    def _evaluate_trial(
        self,
        trial,
        model_type: ModelType,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: Dict[str, Any],
    ) -> float:
        """Evaluate a single trial."""

        # Sample parameters
        params = self._sample_hyperparameters(trial, search_space)
        params = validate_model_params(model_type, params)

        # Create and train model
        model: Any = get_model_factory(model_type, self.task_type, params)

        # Evaluate with CV
        start_time = time.time()
        try:
            eval_results = self._evaluate_with_cv(model, X, y)
            fit_time = time.time() - start_time

            # Calculate predict time (rough estimate)
            predict_start = time.time()
            model.predict(X.head(100))  # Sample prediction
            predict_time = (time.time() - predict_start) * (len(X) / 100)

            # Store trial results
            trial.set_user_attr("fit_time", fit_time)
            trial.set_user_attr("predict_time", predict_time)
            trial.set_user_attr("cv_scores", eval_results["cv_scores"])

            return eval_results["score"]
        except Exception as e:
            # Handle model-specific errors gracefully
            error_msg = str(e)
            if "constant" in error_msg.lower() or "ignored" in error_msg.lower():
                logger.warning(
                    f"Model {model_type} failed due to constant features: {error_msg}"
                )
                return 0.0
            else:
                raise e

    def _create_trial_result(
        self, trial, trial_id: int, model_type: ModelType, status: str
    ) -> TrialResult:
        """Create a TrialResult from a trial."""
        if status == "completed":
            return TrialResult(
                trial_id=trial_id,
                model_type=model_type,
                params=trial.params,
                score=trial.value,
                metric=self.metric,
                cv_scores=trial.user_attrs.get("cv_scores", []),
                fit_time=trial.user_attrs.get("fit_time", 0),
                predict_time=trial.user_attrs.get("predict_time", 0),
                timestamp=datetime.now(),
                status=status,
            )
        else:
            return TrialResult(
                trial_id=trial_id,
                model_type=model_type,
                params=trial.params,
                score=0.0,
                metric=self.metric,
                cv_scores=[],
                fit_time=0,
                predict_time=0,
                timestamp=datetime.now(),
                status=status,
            )

    def optimize_hyperparameters(
        self,
        model_type: ModelType,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        budget_clock: Optional[BudgetClock] = None,
        custom_space: Optional[Dict[str, Any]] = None,
    ) -> List[TrialResult]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            model_type: Type of model
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            budget_clock: Budget clock for time management
            custom_space: Custom search space

        Returns:
            List of trial results
        """
        logger.info(f"Starting hyperparameter optimization for {model_type.value}")

        # Get search space
        search_space = get_search_space(model_type, self.task_type, custom_space)

        # Create Optuna study
        import optuna

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )

        # Define objective function
        def objective(trial):
            # Check budget
            if budget_clock and budget_clock.is_expired():
                raise optuna.TrialPruned()
            return self._evaluate_trial(trial, model_type, X, y, search_space)

        # Run optimization
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=budget_clock.remaining_seconds() if budget_clock else None,
            )
        except Exception as e:
            logger.warning(f"Optimization interrupted: {e}")

        # Convert trials to results
        results = []
        for i, trial in enumerate(study.trials):
            status = (
                "completed"
                if trial.state == optuna.trial.TrialState.COMPLETE
                else "failed"
            )
            result = self._create_trial_result(trial, i, model_type, status)
            results.append(result)

        logger.info(f"Completed {len(results)} trials for {model_type.value}")
        return results

    def get_feature_importance(
        self, model: BaseEstimator, feature_names: List[str]
    ) -> Dict[str, float]:
        """Get feature importance from model."""
        importance_dict = {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
        elif hasattr(model, "coef_"):
            # For linear models, use absolute coefficients
            coef = np.abs(model.coef_)
            if coef.ndim > 1:
                coef = np.mean(coef, axis=0)
            importance_dict = dict(zip(feature_names, coef))
        else:
            logger.warning("Model does not support feature importance")
            return {}

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def train_and_evaluate(
    model_type: ModelType,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    metric: MetricType = MetricType.AUTO,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Convenience function to train and evaluate a model.

    Args:
        model_type: Type of model
        X: Feature matrix
        y: Target vector
        task_type: Task type
        metric: Evaluation metric
        params: Model parameters

    Returns:
        Tuple of (trained_model, evaluation_results)
    """
    # Select metric if auto
    if metric == MetricType.AUTO:
        metric = select_metric(task_type, metric)

    # Create trainer
    trainer = ModelTrainer(task_type=task_type, metric=metric)

    # Train model
    model = trainer.train_model(model_type, X, y, params)

    # Evaluate model
    results = trainer.evaluate_model(model, X, y)

    return model, results
