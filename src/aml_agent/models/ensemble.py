"""
Ensemble methods for the Autonomous ML Agent.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..logging import get_logger
from ..types import TaskType, TrialResult
from .registries import get_model_factory

logger = get_logger()


class EnsembleBuilder:
    """Build ensemble models from individual models."""

    def __init__(self, task_type: TaskType, random_seed: int = 42):
        self.task_type = task_type
        self.random_seed = random_seed
        self.ensemble_models = {}

    def create_ensemble(
        self,
        trial_results: List[TrialResult],
        top_k: int = 3,
        method: str = "voting",
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> BaseEstimator:
        """
        Create ensemble from top performing models.

        Args:
            trial_results: List of trial results
            top_k: Number of top models to include
            method: Ensemble method ("voting", "stacking", "blending")
            X: Training data for stacking
            y: Target data for stacking

        Returns:
            Ensemble model
        """
        # Get top K models
        top_models = self._get_top_models(trial_results, top_k)

        if len(top_models) < 2:
            logger.warning(
                "Not enough models for ensemble, returning best single model"
            )
            return self._create_single_model(top_models[0])

        logger.info(f"Creating {method} ensemble with {len(top_models)} models")

        if method == "voting":
            return self._create_voting_ensemble(top_models)
        elif method == "stacking":
            if X is None or y is None:
                raise ValueError("X and y required for stacking ensemble")
            return self._create_stacking_ensemble(top_models, X, y)
        elif method == "blending":
            return self._create_blending_ensemble(top_models)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def _get_top_models(
        self, trial_results: List[TrialResult], top_k: int
    ) -> List[TrialResult]:
        """Get top K performing models."""
        # Filter successful trials
        successful_results = [r for r in trial_results if r.status == "completed"]

        if not successful_results:
            raise ValueError("No successful trial results")

        # Sort by score and take top K
        top_models = sorted(successful_results, key=lambda x: x.score, reverse=True)[
            :top_k
        ]

        logger.info(
            f"Selected top {len(top_models)} models with scores: "
            f"{[f'{m.model_type.value}: {m.score:.4f}' for m in top_models]}"
        )

        return top_models

    def _create_single_model(self, trial_result: TrialResult) -> BaseEstimator:
        """Create single model from trial result."""
        model = get_model_factory(
            trial_result.model_type, self.task_type, trial_result.params
        )
        return model

    def _create_voting_ensemble(self, top_models: List[TrialResult]) -> BaseEstimator:
        """Create voting ensemble."""
        estimators = []

        for i, trial_result in enumerate(top_models):
            model = get_model_factory(
                trial_result.model_type, self.task_type, trial_result.params
            )
            estimator_name = f"{trial_result.model_type.value}_{i}"
            estimators.append((estimator_name, model))

        if self.task_type == TaskType.CLASSIFICATION:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting="soft" if self._supports_proba(top_models) else "hard",
            )
        else:
            ensemble = VotingRegressor(estimators=estimators)

        return ensemble

    def _create_stacking_ensemble(
        self, top_models: List[TrialResult], X: pd.DataFrame, y: pd.Series
    ) -> BaseEstimator:
        """Create stacking ensemble."""
        estimators = []

        for i, trial_result in enumerate(top_models):
            model = get_model_factory(
                trial_result.model_type, self.task_type, trial_result.params
            )
            estimator_name = f"{trial_result.model_type.value}_{i}"
            estimators.append((estimator_name, model))

        # Choose meta-learner
        if self.task_type == TaskType.CLASSIFICATION:
            meta_learner = LogisticRegression(random_state=self.random_seed)
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                random_state=self.random_seed,
            )
        else:
            meta_learner = LinearRegression()
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                random_state=self.random_seed,
            )

        return ensemble

    def _create_blending_ensemble(self, top_models: List[TrialResult]) -> BaseEstimator:
        """Create blending ensemble with weighted predictions."""
        # Calculate weights based on performance
        weights = [model.score for model in top_models]
        weights = np.array(weights) / sum(weights)

        # Create weighted ensemble
        if self.task_type == TaskType.CLASSIFICATION:
            ensemble = WeightedVotingClassifier(top_models, weights, self.random_seed)
        else:
            ensemble = WeightedVotingRegressor(top_models, weights, self.random_seed)

        return ensemble

    def _supports_proba(self, models: List[TrialResult]) -> bool:
        """Check if all models support probability prediction."""
        for model_result in models:
            model = get_model_factory(
                model_result.model_type, self.task_type, model_result.params
            )
            if not hasattr(model, "predict_proba"):
                return False
        return True

    def evaluate_ensemble(
        self, ensemble: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        from ..models.train_eval import cross_validate_model

        cv_scores, metrics = cross_validate_model(
            ensemble, X, y, self.task_type, cv_folds
        )

        return {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), **metrics}


class WeightedVotingClassifier(BaseEstimator, ClassifierMixin):
    """Weighted voting classifier for blending."""

    def __init__(
        self,
        trial_results: List[TrialResult],
        weights: np.ndarray,
        random_seed: int = 42,
    ):
        self.trial_results = trial_results
        self.weights = weights
        self.random_seed = random_seed
        self.models = []
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all models in the ensemble."""
        self.models = []

        for trial_result in self.trial_results:
            model = get_model_factory(
                trial_result.model_type, TaskType.CLASSIFICATION, trial_result.params
            )
            model.fit(X, y)
            self.models.append(model)

        # Store classes
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted voting."""
        predictions = []

        for model in self.models:
            if hasattr(model, "predict_proba"):
                pred_proba = model.predict_proba(X)
                predictions.append(pred_proba)
            else:
                pred = model.predict(X)
                # Convert to one-hot encoding
                pred_proba = np.zeros((len(pred), len(self.classes_)))
                for i, class_label in enumerate(self.classes_):
                    pred_proba[pred == class_label, i] = 1
                predictions.append(pred_proba)

        # Weighted average of probabilities
        weighted_proba = np.average(predictions, axis=0, weights=self.weights)

        # Return class with highest probability
        return self.classes_[np.argmax(weighted_proba, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        predictions = []

        for model in self.models:
            if hasattr(model, "predict_proba"):
                pred_proba = model.predict_proba(X)
                predictions.append(pred_proba)
            else:
                pred = model.predict(X)
                # Convert to one-hot encoding
                pred_proba = np.zeros((len(pred), len(self.classes_)))
                for i, class_label in enumerate(self.classes_):
                    pred_proba[pred == class_label, i] = 1
                predictions.append(pred_proba)

        # Weighted average of probabilities
        return np.average(predictions, axis=0, weights=self.weights)


class WeightedVotingRegressor(BaseEstimator, RegressorMixin):
    """Weighted voting regressor for blending."""

    def __init__(
        self,
        trial_results: List[TrialResult],
        weights: np.ndarray,
        random_seed: int = 42,
    ):
        self.trial_results = trial_results
        self.weights = weights
        self.random_seed = random_seed
        self.models = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all models in the ensemble."""
        self.models = []

        for trial_result in self.trial_results:
            model = get_model_factory(
                trial_result.model_type, TaskType.REGRESSION, trial_result.params
            )
            model.fit(X, y)
            self.models.append(model)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted voting."""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average of predictions
        return np.average(predictions, axis=0, weights=self.weights)


def create_ensemble(
    trial_results: List[TrialResult],
    task_type: TaskType,
    top_k: int = 3,
    method: str = "voting",
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
) -> BaseEstimator:
    """Create ensemble from trial results."""
    builder = EnsembleBuilder(task_type)
    return builder.create_ensemble(trial_results, top_k, method, X, y)


def evaluate_ensemble_performance(
    ensemble: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    cv_folds: int = 5,
) -> Dict[str, float]:
    """Evaluate ensemble performance."""
    builder = EnsembleBuilder(task_type)
    return builder.evaluate_ensemble(ensemble, X, y, cv_folds)


def get_ensemble_weights(
    trial_results: List[TrialResult], top_k: int = 3
) -> np.ndarray:
    """Get ensemble weights based on model performance."""
    top_models = sorted(trial_results, key=lambda x: x.score, reverse=True)[:top_k]
    weights = np.array([model.score for model in top_models])
    return weights / weights.sum()


def compare_ensemble_methods(
    trial_results: List[TrialResult],
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    top_k: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Compare different ensemble methods."""
    methods = ["voting", "blending"]
    results = {}

    builder = EnsembleBuilder(task_type)

    for method in methods:
        try:
            ensemble = builder.create_ensemble(trial_results, top_k, method, X, y)
            performance = builder.evaluate_ensemble(ensemble, X, y)
            results[method] = performance
        except Exception as e:
            logger.warning(f"Failed to create {method} ensemble: {e}")
            results[method] = {"error": str(e)}

    return results
