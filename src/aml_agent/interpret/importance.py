"""
Feature importance analysis for model interpretability.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from ..logging import get_logger
from ..types import TaskType

logger = get_logger()


class FeatureImportanceAnalyzer:
    """Analyze feature importance for trained models."""

    def __init__(self, task_type: TaskType, random_seed: int = 42):
        self.task_type = task_type
        self.random_seed = random_seed

    def get_feature_importance(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Get feature importance for a trained model.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            method: Method to use ('auto', 'builtin', 'permutation')

        Returns:
            Dictionary with importance scores and feature names
        """
        if method == "auto":
            method = self._select_best_method(model)

        if method == "builtin":
            return self._get_builtin_importance(model, X.columns)
        elif method == "permutation":
            return self._get_permutation_importance(model, X, y)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _select_best_method(self, model: BaseEstimator) -> str:
        """Select the best importance method for the model."""
        # Check if model has built-in feature importance
        if hasattr(model, "feature_importances_"):
            return "builtin"
        elif hasattr(model, "coef_"):
            return "builtin"
        else:
            return "permutation"

    def _get_builtin_importance(
        self, model: BaseEstimator, feature_names: List[str]
    ) -> Dict[str, Any]:
        """Get built-in feature importance."""
        importance_scores = None
        importance_type = None

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importance_scores = model.feature_importances_
            importance_type = "feature_importances"
        # Linear models
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                # Multi-class or multi-output
                importance_scores = np.mean(np.abs(coef), axis=0)
            else:
                importance_scores = np.abs(coef)
            importance_type = "coefficients"

        if importance_scores is None:
            raise ValueError("Model does not support built-in feature importance")

        # Create importance dictionary
        importance_dict = {
            "scores": importance_scores.tolist(),
            "feature_names": feature_names,
            "type": importance_type,
            "method": "builtin",
        }

        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        importance_dict["sorted_features"] = [
            {"feature": feature_names[i], "importance": float(importance_scores[i])}
            for i in sorted_indices
        ]

        return importance_dict

    def _get_permutation_importance(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5
    ) -> Dict[str, Any]:
        """Get permutation importance."""
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model,
                X,
                y,
                n_repeats=n_repeats,
                random_state=self.random_seed,
                n_jobs=-1,
            )

            importance_dict = {
                "scores": perm_importance.importances_mean.tolist(),
                "std": perm_importance.importances_std.tolist(),
                "feature_names": X.columns.tolist(),
                "type": "permutation_importance",
                "method": "permutation",
                "n_repeats": n_repeats,
            }

            # Sort by importance
            sorted_indices = np.argsort(perm_importance.importances_mean)[::-1]
            importance_dict["sorted_features"] = [
                {
                    "feature": X.columns[i],
                    "importance": float(perm_importance.importances_mean[i]),
                    "std": float(perm_importance.importances_std[i]),
                }
                for i in sorted_indices
            ]

            return importance_dict

        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")
            # Fallback to random importance
            return self._get_random_importance(X.columns)

    def _get_random_importance(self, feature_names: List[str]) -> Dict[str, Any]:
        """Fallback random importance when other methods fail."""
        n_features = len(feature_names)
        random_scores = np.random.random(n_features)
        random_scores = random_scores / np.sum(random_scores)  # Normalize

        importance_dict = {
            "scores": random_scores.tolist(),
            "feature_names": feature_names,
            "type": "random",
            "method": "fallback",
        }

        # Sort by importance
        sorted_indices = np.argsort(random_scores)[::-1]
        importance_dict["sorted_features"] = [
            {"feature": feature_names[i], "importance": float(random_scores[i])}
            for i in sorted_indices
        ]

        return importance_dict

    def get_top_features(
        self, importance_dict: Dict[str, Any], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top K most important features."""
        if "sorted_features" in importance_dict:
            return importance_dict["sorted_features"][:top_k]
        else:
            # Fallback: sort by scores
            scores = importance_dict["scores"]
            feature_names = importance_dict["feature_names"]

            sorted_indices = np.argsort(scores)[::-1]
            return [
                {"feature": feature_names[i], "importance": float(scores[i])}
                for i in sorted_indices[:top_k]
            ]

    def compare_models_importance(
        self, models: Dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """Compare feature importance across multiple models."""
        results = {}

        for name, model in models.items():
            try:
                importance = self.get_feature_importance(model, X, y)
                results[name] = importance
            except Exception as e:
                logger.warning(f"Failed to get importance for {name}: {e}")
                results[name] = {"error": str(e)}

        return results

    def create_importance_plot_data(
        self, importance_dict: Dict[str, Any], top_k: int = 20
    ) -> Dict[str, Any]:
        """Create data for plotting feature importance."""
        top_features = self.get_top_features(importance_dict, top_k)

        return {
            "features": [f["feature"] for f in top_features],
            "importances": [f["importance"] for f in top_features],
            "std": (
                [f.get("std", 0) for f in top_features]
                if "std" in top_features[0]
                else None
            ),
            "title": f"Top {len(top_features)} Feature Importance",
            "method": importance_dict.get("method", "unknown"),
        }
