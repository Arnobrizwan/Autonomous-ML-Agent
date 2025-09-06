"""
Model explanation and interpretability features.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..logging import get_logger
from ..types import TaskType
from .importance import FeatureImportanceAnalyzer

logger = get_logger()

# Try to import SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("SHAP not available, using alternative explanation methods")

# Try to import ELI5
try:
    import eli5
    from eli5.sklearn import PermutationImportance

    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False
    logger.info("ELI5 not available, using alternative explanation methods")


class ModelExplainer:
    """Generate explanations for model predictions."""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.importance_analyzer = FeatureImportanceAnalyzer(task_type)

    def explain_model(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        explanation_type: str = "auto",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model explanation.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            explanation_type: Type of explanation ('auto', 'shap', 'eli5', 'basic')
            save_path: Path to save explanation results

        Returns:
            Dictionary with explanation results
        """
        if explanation_type == "auto":
            explanation_type = self._select_best_explanation_method(model)

        explanation = {
            "model_type": type(model).__name__,
            "task_type": self.task_type.value,
            "explanation_type": explanation_type,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
        }

        # Get feature importance
        try:
            importance = self.importance_analyzer.get_feature_importance(model, X, y)
            explanation["feature_importance"] = importance
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            explanation["feature_importance"] = {"error": str(e)}

        # Generate specific explanations based on type
        if explanation_type == "shap" and SHAP_AVAILABLE:
            explanation.update(self._generate_shap_explanation(model, X, y))
        elif explanation_type == "eli5" and ELI5_AVAILABLE:
            explanation.update(self._generate_eli5_explanation(model, X, y))
        else:
            explanation.update(self._generate_basic_explanation(model, X, y))

        # Save explanation if path provided
        if save_path:
            self._save_explanation(explanation, save_path)

        return explanation

    def _select_best_explanation_method(self, model: BaseEstimator) -> str:
        """Select the best explanation method for the model."""
        if SHAP_AVAILABLE:
            return "shap"
        elif ELI5_AVAILABLE:
            return "eli5"
        else:
            return "basic"

    def _generate_shap_explanation(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanation."""
        try:
            # Create SHAP explainer
            if hasattr(model, "predict_proba"):
                # For models with probability prediction
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)

                # Summary statistics
                summary = {
                    "shap_values_mean": np.mean(
                        np.abs(shap_values.values), axis=0
                    ).tolist(),
                    "shap_values_std": np.std(shap_values.values, axis=0).tolist(),
                    "base_value": (
                        float(shap_values.base_values[0])
                        if hasattr(shap_values, "base_values")
                        else None
                    ),
                }

                # Feature importance from SHAP
                feature_importance = np.mean(np.abs(shap_values.values), axis=0)
                sorted_indices = np.argsort(feature_importance)[::-1]

                summary["top_features"] = [
                    {
                        "feature": X.columns[i],
                        "importance": float(feature_importance[i]),
                    }
                    for i in sorted_indices[:10]
                ]

                return {"shap_explanation": summary}
            else:
                # For models without probability prediction
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)

                summary = {
                    "shap_values_mean": np.mean(
                        np.abs(shap_values.values), axis=0
                    ).tolist(),
                    "shap_values_std": np.std(shap_values.values, axis=0).tolist(),
                }

                return {"shap_explanation": summary}

        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {"shap_explanation": {"error": str(e)}}

    def _generate_eli5_explanation(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Generate ELI5-based explanation."""
        try:
            # Use ELI5 permutation importance
            perm_importance = PermutationImportance(model, random_state=42)
            perm_importance.fit(X, y)

            # Get feature weights
            feature_weights = eli5.explain_weights(
                perm_importance, feature_names=X.columns.tolist()
            )

            explanation = {
                "eli5_explanation": {
                    "feature_weights": str(feature_weights),
                    "top_features": [
                        {"feature": feature, "weight": weight}
                        for feature, weight in zip(
                            X.columns, perm_importance.feature_importances_
                        )
                    ][:10],
                }
            }

            return explanation

        except Exception as e:
            logger.warning(f"ELI5 explanation failed: {e}")
            return {"eli5_explanation": {"error": str(e)}}

    def _generate_basic_explanation(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Generate basic explanation without external libraries."""
        try:
            # Get model performance summary
            if hasattr(model, "score"):
                score = model.score(X, y)
            else:
                score = None

            # Get feature importance
            importance = self.importance_analyzer.get_feature_importance(model, X, y)

            # Generate prediction summary
            predictions = model.predict(X)
            prediction_summary = {
                "mean_prediction": float(np.mean(predictions)),
                "std_prediction": float(np.std(predictions)),
                "min_prediction": float(np.min(predictions)),
                "max_prediction": float(np.max(predictions)),
            }

            # Add probability summary if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X)
                prediction_summary["mean_probability"] = float(
                    np.mean(probabilities, axis=0).tolist()
                )
                prediction_summary["probability_std"] = float(
                    np.std(probabilities, axis=0).tolist()
                )

            return {
                "basic_explanation": {
                    "model_score": score,
                    "prediction_summary": prediction_summary,
                    "feature_importance": importance,
                    "model_parameters": self._get_model_parameters(model),
                }
            }

        except Exception as e:
            logger.warning(f"Basic explanation failed: {e}")
            return {"basic_explanation": {"error": str(e)}}

    def _get_model_parameters(self, model: BaseEstimator) -> Dict[str, Any]:
        """Extract model parameters for explanation."""
        try:
            params = model.get_params()
            # Convert numpy types to Python types for JSON serialization
            serializable_params = {}
            for key, value in params.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_params[key] = value.item()
                elif isinstance(value, np.ndarray):
                    serializable_params[key] = value.tolist()
                else:
                    serializable_params[key] = value
            return serializable_params
        except Exception as e:
            logger.warning(f"Failed to extract model parameters: {e}")
            return {"error": str(e)}

    def explain_prediction(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        instance_idx: int = 0,
        explanation_type: str = "auto",
    ) -> Dict[str, Any]:
        """
        Explain a specific prediction.

        Args:
            model: Trained model
            X: Feature matrix
            instance_idx: Index of instance to explain
            explanation_type: Type of explanation

        Returns:
            Dictionary with prediction explanation
        """
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")

        instance = X.iloc[instance_idx : instance_idx + 1]
        prediction = model.predict(instance)[0]

        explanation = {
            "instance_idx": instance_idx,
            "prediction": float(prediction),
            "feature_values": instance.iloc[0].to_dict(),
        }

        # Add probability if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(instance)[0]
            explanation["probabilities"] = probabilities.tolist()
            explanation["predicted_class"] = int(np.argmax(probabilities))

        # Add SHAP explanation if available
        if explanation_type in ["auto", "shap"] and SHAP_AVAILABLE:
            try:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(instance)
                explanation["shap_values"] = shap_values.values[0].tolist()
            except Exception as e:
                logger.warning(f"SHAP prediction explanation failed: {e}")

        return explanation

    def _save_explanation(self, explanation: Dict[str, Any], save_path: str) -> None:
        """Save explanation to file."""
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = path / "explanation.json"
        with open(json_path, "w") as f:
            json.dump(explanation, f, indent=2, default=str)

        logger.info(f"Explanation saved to {json_path}")

    def generate_model_card_data(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, Any]:
        """Generate data for model card."""
        explanation = self.explain_model(model, X, y)

        # Extract key metrics for model card
        model_card_data = {
            "model_type": type(model).__name__,
            "task_type": self.task_type.value,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_importance": explanation.get("feature_importance", {}),
            "model_parameters": explanation.get("basic_explanation", {}).get(
                "model_parameters", {}
            ),
        }

        return model_card_data
