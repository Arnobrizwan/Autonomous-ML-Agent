"""
Ensemble learning utilities for combining multiple models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score

from ..logging import get_logger
from ..types import TaskType, TrialResult

logger = get_logger()


class EnsembleBuilder:
    """Build ensemble models from trial results."""

    def __init__(self, task_type: TaskType, random_seed: int = 42):
        self.task_type = task_type
        self.random_seed = random_seed

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
            y: Training labels for stacking

        Returns:
            Ensemble model
        """
        # Get top K models
        top_results = sorted(trial_results, key=lambda x: x.score, reverse=True)[:top_k]

        if len(top_results) < 2:
            logger.warning(
                "Not enough models for ensemble, returning best single model"
            )
            return self._create_single_model(top_results[0])

        logger.info(f"Creating {method} ensemble with {len(top_results)} models")

        if method == "voting":
            return self._create_voting_ensemble(top_results)
        elif method == "stacking":
            if X is None or y is None:
                logger.warning(
                    "Stacking requires training data, falling back to voting"
                )
                return self._create_voting_ensemble(top_results)
            return self._create_stacking_ensemble(top_results, X, y)
        elif method == "blending":
            return self._create_blending_ensemble(top_results)
        else:
            logger.warning(f"Unknown ensemble method: {method}, using voting")
            return self._create_voting_ensemble(top_results)

    def _create_single_model(self, trial_result: TrialResult) -> BaseEstimator:
        """Create single model from trial result."""
        from .registries import get_model_factory

        model = get_model_factory(
            trial_result.model_type, self.task_type, trial_result.params
        )
        return model

    def _create_voting_ensemble(
        self, trial_results: List[TrialResult]
    ) -> BaseEstimator:
        """Create voting ensemble."""
        from .registries import get_model_factory

        estimators = []
        for i, result in enumerate(trial_results):
            model = get_model_factory(result.model_type, self.task_type, result.params)
            estimators.append((f"model_{i}", model))

        if self.task_type == TaskType.CLASSIFICATION:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting="soft" if self._supports_proba(estimators) else "hard",
            )
        else:
            ensemble = VotingRegressor(estimators=estimators)

        return ensemble

    def _create_stacking_ensemble(
        self, trial_results: List[TrialResult], X: pd.DataFrame, y: pd.Series
    ) -> BaseEstimator:
        """Create stacking ensemble."""
        from sklearn.ensemble import StackingClassifier, StackingRegressor

        from .registries import get_model_factory

        estimators = []
        for i, result in enumerate(trial_results):
            model = get_model_factory(result.model_type, self.task_type, result.params)
            estimators.append((f"model_{i}", model))

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

    def _create_blending_ensemble(
        self, trial_results: List[TrialResult]
    ) -> BaseEstimator:
        """Create blending ensemble with learned weights."""
        from .registries import get_model_factory

        # Create weighted ensemble
        models = []
        weights = []

        for result in trial_results:
            model = get_model_factory(result.model_type, self.task_type, result.params)
            models.append(model)
            weights.append(result.score)  # Use score as weight

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        return WeightedEnsemble(models, weights, self.task_type)

    def _supports_proba(self, estimators: List[tuple]) -> bool:
        """Check if all estimators support predict_proba."""
        for _, estimator in estimators:
            if not hasattr(estimator, "predict_proba"):
                return False
        return True

    def evaluate_ensemble(
        self, ensemble: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        # Cross-validation evaluation
        if self.task_type == TaskType.CLASSIFICATION:
            cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="accuracy")
        else:
            cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="r2")

        return {
            "cv_mean": np.mean(cv_scores),
            "cv_std": np.std(cv_scores),
            "cv_scores": cv_scores.tolist(),
        }


class WeightedEnsemble(BaseEstimator):
    """Weighted ensemble of models."""

    def __init__(
        self, models: List[BaseEstimator], weights: np.ndarray, task_type: TaskType
    ):
        self.models = models
        self.weights = weights
        self.task_type = task_type
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using weighted average."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before predict")

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average
        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)

        if self.task_type == TaskType.CLASSIFICATION:
            return np.round(weighted_pred).astype(int)
        else:
            return weighted_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for classification."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before predict_proba")

        if self.task_type != TaskType.CLASSIFICATION:
            raise ValueError("predict_proba only available for classification")

        # Get probabilities from models that support it
        proba_models = [m for m in self.models if hasattr(m, "predict_proba")]
        if not proba_models:
            raise ValueError("No models support predict_proba")

        probabilities = []
        for model in proba_models:
            proba = model.predict_proba(X)
            probabilities.append(proba)

        # Weighted average of probabilities
        probabilities = np.array(probabilities)
        weighted_proba = np.average(
            probabilities, axis=0, weights=self.weights[: len(proba_models)]
        )

        return weighted_proba


class AdvancedEnsemble:
    """Advanced ensemble methods with dynamic weighting."""

    def __init__(self, task_type: TaskType, random_seed: int = 42):
        self.task_type = task_type
        self.random_seed = random_seed

    def create_dynamic_ensemble(
        self,
        trial_results: List[TrialResult],
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "performance_weighted",
    ) -> BaseEstimator:
        """Create ensemble with dynamic weighting based on performance."""
        from .registries import get_model_factory

        # Get top models
        top_results = sorted(trial_results, key=lambda x: x.score, reverse=True)[:5]

        models = []
        for result in top_results:
            model = get_model_factory(result.model_type, self.task_type, result.params)
            models.append(model)

        if method == "performance_weighted":
            return self._create_performance_weighted_ensemble(models, top_results, X, y)
        elif method == "diversity_weighted":
            return self._create_diversity_weighted_ensemble(models, X, y)
        else:
            # Fallback to simple voting
            return self._create_simple_voting_ensemble(models)

    def _create_performance_weighted_ensemble(
        self,
        models: List[BaseEstimator],
        trial_results: List[TrialResult],
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseEstimator:
        """Create ensemble weighted by individual model performance."""
        # Calculate performance weights
        scores = [result.score for result in trial_results]
        weights = np.array(scores)
        weights = weights / weights.sum()

        return WeightedEnsemble(models, weights, self.task_type)

    def _create_diversity_weighted_ensemble(
        self, models: List[BaseEstimator], X: pd.DataFrame, y: pd.Series
    ) -> BaseEstimator:
        """Create ensemble weighted by model diversity."""
        # Calculate diversity scores
        predictions = []
        for model in models:
            model.fit(X, y)
            pred = model.predict(X)
            predictions.append(pred)

        # Calculate pairwise diversity
        diversity_scores = []
        for i, pred1 in enumerate(predictions):
            diversity = 0
            for j, pred2 in enumerate(predictions):
                if i != j:
                    if self.task_type == TaskType.CLASSIFICATION:
                        diversity += 1 - accuracy_score(pred1, pred2)
                    else:
                        diversity += 1 - r2_score(pred1, pred2)
            diversity_scores.append(diversity)

        # Normalize weights
        weights = np.array(diversity_scores)
        weights = weights / weights.sum()

        return WeightedEnsemble(models, weights, self.task_type)

    def _create_simple_voting_ensemble(
        self, models: List[BaseEstimator]
    ) -> BaseEstimator:
        """Create simple voting ensemble."""
        estimators = [(f"model_{i}", model) for i, model in enumerate(models)]

        if self.task_type == TaskType.CLASSIFICATION:
            return VotingClassifier(estimators=estimators)
        else:
            return VotingRegressor(estimators=estimators)


def create_ensemble(
    trial_results: List[TrialResult],
    task_type: TaskType,
    method: str = "voting",
    top_k: int = 3,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
) -> BaseEstimator:
    """Convenience function to create ensemble."""
    builder = EnsembleBuilder(task_type)
    return builder.create_ensemble(trial_results, top_k, method, X, y)
