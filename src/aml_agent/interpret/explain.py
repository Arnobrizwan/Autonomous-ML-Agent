"""
Model explanation and interpretability using SHAP and other methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
from pathlib import Path

from ..types import ModelType, TaskType, TrialResult
from ..logging import get_logger
from .importance import FeatureImportanceAnalyzer

logger = get_logger()

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("SHAP not available, using alternative explanation methods")


class ModelExplainer:
    """Generate model explanations and interpretability insights."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.importance_analyzer = FeatureImportanceAnalyzer(random_seed)
    
    def explain_model(self, 
                     model: Any,
                     X: pd.DataFrame,
                     y: pd.Series,
                     task_type: TaskType,
                     method: str = "auto") -> Dict[str, Any]:
        """
        Generate comprehensive model explanation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            task_type: Task type
            method: Explanation method ("auto", "shap", "importance", "lime")
            
        Returns:
            Dictionary with explanation results
        """
        if method == "auto":
            method = self._select_best_method(model)
        
        logger.info(f"Generating model explanation using {method}")
        
        explanation = {
            "method": method,
            "task_type": task_type.value,
            "n_features": X.shape[1],
            "n_samples": X.shape[0]
        }
        
        if method == "shap" and SHAP_AVAILABLE:
            explanation.update(self._explain_with_shap(model, X, y, task_type))
        else:
            explanation.update(self._explain_with_importance(model, X, y, task_type))
        
        return explanation
    
    def _select_best_method(self, model: Any) -> str:
        """Select best explanation method for model."""
        if SHAP_AVAILABLE:
            return "shap"
        else:
            return "importance"
    
    def _explain_with_shap(self, 
                          model: Any,
                          X: pd.DataFrame,
                          y: pd.Series,
                          task_type: TaskType) -> Dict[str, Any]:
        """Generate explanation using SHAP."""
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba') and task_type == TaskType.CLASSIFICATION:
                # Use TreeExplainer for tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use KernelExplainer for other models
                    explainer = shap.KernelExplainer(model.predict_proba, X.sample(100, random_state=self.random_seed))
            else:
                # Use TreeExplainer for tree-based models
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use KernelExplainer for other models
                    explainer = shap.KernelExplainer(model.predict, X.sample(100, random_state=self.random_seed))
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Multi-class: use first class for analysis
                shap_values = shap_values[0]
            
            # Calculate feature importance from SHAP values
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Normalize importance
            total_importance = feature_importance.sum()
            if total_importance > 0:
                feature_importance = feature_importance / total_importance
            
            # Create feature importance dictionary
            importance_dict = {
                feature: float(importance)
                for feature, importance in zip(X.columns, feature_importance)
            }
            
            # Get top features
            top_features = self.importance_analyzer.get_top_features(importance_dict, 10)
            
            return {
                "feature_importance": importance_dict,
                "top_features": top_features,
                "shap_values": shap_values.tolist(),
                "explainer_type": type(explainer).__name__
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}, falling back to importance")
            return self._explain_with_importance(model, X, y, task_type)
    
    def _explain_with_importance(self, 
                               model: Any,
                               X: pd.DataFrame,
                               y: pd.Series,
                               task_type: TaskType) -> Dict[str, Any]:
        """Generate explanation using feature importance."""
        # Get feature importance
        importance_dict = self.importance_analyzer.get_feature_importance(model, X, y)
        
        # Get top features
        top_features = self.importance_analyzer.get_top_features(importance_dict, 10)
        
        # Analyze feature groups
        feature_groups = self.importance_analyzer.analyze_feature_groups(importance_dict)
        
        return {
            "feature_importance": importance_dict,
            "top_features": top_features,
            "feature_groups": feature_groups,
            "explainer_type": "feature_importance"
        }
    
    def explain_prediction(self, 
                          model: Any,
                          X: pd.DataFrame,
                          instance_idx: int,
                          task_type: TaskType) -> Dict[str, Any]:
        """Explain a single prediction."""
        if instance_idx >= len(X):
            raise ValueError(f"Instance index {instance_idx} out of range")
        
        instance = X.iloc[instance_idx:instance_idx+1]
        prediction = model.predict(instance)[0]
        
        explanation = {
            "instance_idx": instance_idx,
            "prediction": float(prediction),
            "feature_values": instance.iloc[0].to_dict()
        }
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba') and task_type == TaskType.CLASSIFICATION:
            probabilities = model.predict_proba(instance)[0]
            explanation["probabilities"] = probabilities.tolist()
            explanation["predicted_class"] = int(np.argmax(probabilities))
        
        # Get feature importance for this instance
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, X.sample(100, random_state=self.random_seed))
                shap_values = explainer.shap_values(instance)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                explanation["shap_values"] = {
                    feature: float(value)
                    for feature, value in zip(X.columns, shap_values[0])
                }
            except Exception as e:
                logger.warning(f"SHAP instance explanation failed: {e}")
        
        return explanation
    
    def generate_explanation_report(self, 
                                  model: Any,
                                  X: pd.DataFrame,
                                  y: pd.Series,
                                  task_type: TaskType,
                                  output_dir: Path) -> Dict[str, str]:
        """Generate comprehensive explanation report with plots."""
        # Generate explanation
        explanation = self.explain_model(model, X, y, task_type)
        
        # Create plots
        plot_files = {}
        
        # Feature importance plot
        if explanation.get("feature_importance"):
            plot_files["feature_importance"] = self._plot_feature_importance(
                explanation["feature_importance"], output_dir
            )
        
        # Feature groups plot
        if explanation.get("feature_groups"):
            plot_files["feature_groups"] = self._plot_feature_groups(
                explanation["feature_groups"], output_dir
            )
        
        # SHAP summary plot
        if explanation.get("shap_values") and SHAP_AVAILABLE:
            plot_files["shap_summary"] = self._plot_shap_summary(
                explanation["shap_values"], X, output_dir
            )
        
        return plot_files
    
    def _plot_feature_importance(self, 
                               importance_dict: Dict[str, float],
                               output_dir: Path) -> str:
        """Plot feature importance."""
        if not PLOTTING_AVAILABLE:
            return ""
            
        # Get top 15 features
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if not top_features:
            return ""
        
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_file = output_dir / "feature_importance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _plot_feature_groups(self, 
                           feature_groups: Dict[str, float],
                           output_dir: Path) -> str:
        """Plot feature groups importance."""
        if not PLOTTING_AVAILABLE or not feature_groups:
            return ""
        
        groups, importances = zip(*feature_groups.items())
        
        plt.figure(figsize=(8, 6))
        plt.pie(importances, labels=groups, autopct='%1.1f%%')
        plt.title('Feature Importance by Group')
        plt.tight_layout()
        
        plot_file = output_dir / "feature_groups.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_file)
    
    def _plot_shap_summary(self, 
                         shap_values: np.ndarray,
                         X: pd.DataFrame,
                         output_dir: Path) -> str:
        """Plot SHAP summary."""
        if not SHAP_AVAILABLE:
            return ""
        
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False)
            plt.tight_layout()
            
            plot_file = output_dir / "shap_summary.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_file)
        except Exception as e:
            logger.warning(f"SHAP summary plot failed: {e}")
            return ""


def explain_model(model: Any,
                 X: pd.DataFrame,
                 y: pd.Series,
                 task_type: TaskType,
                 method: str = "auto") -> Dict[str, Any]:
    """Generate model explanation."""
    explainer = ModelExplainer()
    return explainer.explain_model(model, X, y, task_type, method)


def explain_prediction(model: Any,
                      X: pd.DataFrame,
                      instance_idx: int,
                      task_type: TaskType) -> Dict[str, Any]:
    """Explain a single prediction."""
    explainer = ModelExplainer()
    return explainer.explain_prediction(model, X, instance_idx, task_type)


def generate_explanation_report(model: Any,
                              X: pd.DataFrame,
                              y: pd.Series,
                              task_type: TaskType,
                              output_dir: Path) -> Dict[str, str]:
    """Generate comprehensive explanation report."""
    explainer = ModelExplainer()
    return explainer.generate_explanation_report(model, X, y, task_type, output_dir)
