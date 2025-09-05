"""
Feature importance analysis for model interpretability.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

# SHAP integration
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..logging import get_logger

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
        elif method == "shap":
            return self._get_shap_importance(model, X, y)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _select_best_method(self, model: BaseEstimator) -> str:
        """Select best importance method for model type."""
        # Prefer SHAP if available and model supports it
        if SHAP_AVAILABLE and self._supports_shap(model):
            return "shap"
        # Check if model has built-in feature importance
        elif hasattr(model, "feature_importances_"):
            return "builtin"
        elif hasattr(model, "coef_"):
            return "builtin"
        else:
            return "permutation"

    def _supports_shap(self, model: BaseEstimator) -> bool:
        """Check if model supports SHAP explanation."""
        try:
            # Test if SHAP can create an explainer for this model
            if hasattr(model, "predict_proba") or hasattr(model, "predict"):
                return True
            return False
        except Exception:
            return False

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

    def _get_shap_importance(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        sample_size: int = 100,
    ) -> Dict[str, float]:
        """Get SHAP-based feature importance."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return {}

        try:
            # Sample data for SHAP calculation (for performance)
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X

            # Create SHAP explainer
            try:
                # Try TreeExplainer first (for tree-based models)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                # Fall back to other explainers
                try:
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_sample)
                    shap_values = shap_values.values
                except Exception:
                    # Last resort: use LinearExplainer
                    explainer = shap.LinearExplainer(model, X_sample)
                    shap_values = explainer.shap_values(X_sample)

            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = np.mean(shap_values, axis=0)

            # Calculate mean absolute SHAP values
            if shap_values.ndim > 1:
                mean_shap_values = np.mean(np.abs(shap_values), axis=0)
            else:
                mean_shap_values = np.abs(shap_values)

            # Create importance dictionary
            importance_dict = {}
            for i, feature_name in enumerate(X.columns):
                if i < len(mean_shap_values):
                    importance_dict[feature_name] = float(mean_shap_values[i])
                else:
                    importance_dict[feature_name] = 0.0

            # Normalize importance scores
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {
                    feature: score / total_importance
                    for feature, score in importance_dict.items()
                }

            return importance_dict

        except Exception as e:
            logger.error(f"Error calculating SHAP importance: {e}")
            # Try alternative SHAP approaches
            try:
                # Use KernelExplainer as fallback
                explainer = shap.KernelExplainer(model.predict, X_sample.iloc[:10])
                shap_values = explainer.shap_values(X_sample.iloc[:5])

                if isinstance(shap_values, list):
                    shap_values = np.mean(shap_values, axis=0)

                if shap_values.ndim > 1:
                    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
                else:
                    mean_shap_values = np.abs(shap_values)

                # Create importance dictionary
                importance_dict = {}
                for i, feature_name in enumerate(X.columns):
                    if i < len(mean_shap_values):
                        importance_dict[feature_name] = float(mean_shap_values[i])
                    else:
                        importance_dict[feature_name] = 0.0

                # Normalize
                total_importance = sum(importance_dict.values())
                if total_importance > 0:
                    importance_dict = {
                        feature: score / total_importance
                        for feature, score in importance_dict.items()
                    }

                return importance_dict

            except Exception as e2:
                logger.error(f"All SHAP methods failed: {e2}")
                raise RuntimeError("SHAP importance calculation failed completely")

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
