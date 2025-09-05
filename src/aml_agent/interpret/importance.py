"""
Feature importance analysis for model interpretability.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..logging import get_logger
from ..types import ModelType, TaskType

logger = get_logger()


class FeatureImportanceAnalyzer:
    """Analyze feature importance for trained models."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def get_feature_importance(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, method: str = "auto"
    ) -> Dict[str, float]:
        """
        Get feature importance for a model.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            method: Method to use ("auto", "builtin", "permutation")

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if method == "auto":
            method = self._select_best_method(model)

        if method == "builtin":
            return self._get_builtin_importance(model, X)
        elif method == "permutation":
            return self._get_permutation_importance(model, X, y)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _select_best_method(self, model: BaseEstimator) -> str:
        """Select best importance method for model type."""
        # Check if model has built-in feature importance
        if hasattr(model, "feature_importances_"):
            return "builtin"
        elif hasattr(model, "coef_"):
            return "builtin"
        else:
            return "permutation"

    def _get_builtin_importance(
        self, model: BaseEstimator, X: pd.DataFrame
    ) -> Dict[str, float]:
        """Get built-in feature importance."""
        importance_scores = {}

        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importances = model.feature_importances_
            for i, feature in enumerate(X.columns):
                importance_scores[feature] = float(importances[i])

        elif hasattr(model, "coef_"):
            # Linear models
            coef = model.coef_
            if coef.ndim > 1:
                # Multi-class or multi-output
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)

            for i, feature in enumerate(X.columns):
                importance_scores[feature] = float(coef[i])

        else:
            logger.warning(
                "Model has no built-in feature importance, falling back to permutation"
            )
            return {}

        # Normalize importance scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {
                feature: score / total_importance
                for feature, score in importance_scores.items()
            }

        return importance_scores

    def _get_permutation_importance(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5
    ) -> Dict[str, float]:
        """Get permutation importance."""
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model,
                X,
                y,
                n_repeats=n_repeats,
                random_state=self.random_seed,
                scoring=(
                    "neg_mean_squared_error" if len(np.unique(y)) > 2 else "accuracy"
                ),
            )

            # Extract importance scores
            importance_scores = {}
            for i, feature in enumerate(X.columns):
                importance_scores[feature] = float(perm_importance.importances_mean[i])

            # Normalize importance scores
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {
                    feature: score / total_importance
                    for feature, score in importance_scores.items()
                }

            return importance_scores

        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            return {}

    def get_top_features(
        self, importance_scores: Dict[str, float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top K most important features."""
        sorted_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:top_k]

    def analyze_feature_groups(
        self,
        importance_scores: Dict[str, float],
        feature_groups: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, float]:
        """Analyze importance by feature groups."""
        if feature_groups is None:
            # Default grouping by feature type
            feature_groups = self._create_default_groups(importance_scores.keys())

        group_importance = {}
        for group_name, features in feature_groups.items():
            group_score = sum(importance_scores.get(feature, 0) for feature in features)
            group_importance[group_name] = group_score

        return group_importance

    def _create_default_groups(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Create default feature groups based on naming patterns."""
        groups = {"numeric": [], "categorical": [], "datetime": [], "other": []}

        for feature in feature_names:
            if any(
                pattern in feature.lower()
                for pattern in ["_year", "_month", "_day", "_hour", "_dow"]
            ):
                groups["datetime"].append(feature)
            elif any(
                pattern in feature.lower() for pattern in ["_cat", "_cat_", "category"]
            ):
                groups["categorical"].append(feature)
            elif feature.replace(".", "").replace("-", "").isdigit():
                groups["numeric"].append(feature)
            else:
                groups["other"].append(feature)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def compare_models_importance(
        self, models: Dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Compare feature importance across multiple models."""
        model_importance = {}

        for model_name, model in models.items():
            try:
                importance = self.get_feature_importance(model, X, y)
                model_importance[model_name] = importance
            except Exception as e:
                logger.warning(f"Failed to get importance for {model_name}: {e}")
                model_importance[model_name] = {}

        return model_importance

    def get_importance_consensus(
        self, model_importance: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Get consensus feature importance across models."""
        if not model_importance:
            return {}

        # Get all features
        all_features = set()
        for importance_dict in model_importance.values():
            all_features.update(importance_dict.keys())

        # Calculate average importance
        consensus = {}
        for feature in all_features:
            scores = [
                importance_dict.get(feature, 0)
                for importance_dict in model_importance.values()
            ]
            consensus[feature] = np.mean(scores)

        # Normalize
        total_importance = sum(consensus.values())
        if total_importance > 0:
            consensus = {
                feature: score / total_importance
                for feature, score in consensus.items()
            }

        return consensus


def get_feature_importance(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, method: str = "auto"
) -> Dict[str, float]:
    """Get feature importance for a model."""
    analyzer = FeatureImportanceAnalyzer()
    return analyzer.get_feature_importance(model, X, y, method)


def get_top_features(
    importance_scores: Dict[str, float], top_k: int = 10
) -> List[Tuple[str, float]]:
    """Get top K most important features."""
    analyzer = FeatureImportanceAnalyzer()
    return analyzer.get_top_features(importance_scores, top_k)


def compare_models_importance(
    models: Dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series
) -> Dict[str, Dict[str, float]]:
    """Compare feature importance across multiple models."""
    analyzer = FeatureImportanceAnalyzer()
    return analyzer.compare_models_importance(models, X, y)
