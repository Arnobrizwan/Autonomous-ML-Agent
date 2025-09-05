"""
Model card generation for the Autonomous ML Agent.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..interpret.explain import ModelExplainer
from ..interpret.importance import FeatureImportanceAnalyzer
from ..logging import get_logger
from ..types import DatasetProfile, LLMConfig, ModelCard, TaskType, TrialResult

logger = get_logger()

# Try to import LLM libraries
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ModelCardGenerator:
    """Generate comprehensive model cards with LLM assistance."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config
        self.importance_analyzer = FeatureImportanceAnalyzer()
        self.explainer = ModelExplainer()

        # Initialize LLM if available
        self.llm_client = None
        if llm_config and llm_config.enabled:
            self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM client."""
        try:
            if self.llm_config.provider == "openai" and OPENAI_AVAILABLE:
                self.llm_client = openai.OpenAI(api_key=self.llm_config.api_key)
            elif self.llm_config.provider == "gemini" and GEMINI_AVAILABLE:
                genai.configure(api_key=self.llm_config.api_key)
                self.llm_client = genai.GenerativeModel(self.llm_config.model)
            else:
                logger.warning(f"LLM provider {self.llm_config.provider} not available")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

    def generate_card(
        self,
        trial_results: List[TrialResult],
        ensemble_model: Optional[Any] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        dataset_profile: Optional[DatasetProfile] = None,
        model: Optional[Any] = None,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
    ) -> ModelCard:
        """
        Generate comprehensive model card.

        Args:
            trial_results: List of trial results
            ensemble_model: Ensemble model (optional)
            task_type: Task type
            dataset_profile: Dataset profile
            model: Single model (optional)
            X: Feature matrix (optional)
            y: Target vector (optional)

        Returns:
            Model card object
        """
        logger.info("Generating model card")

        # Get best model
        best_result = (
            max(trial_results, key=lambda x: x.score) if trial_results else None
        )
        model_name = (
            "Ensemble"
            if ensemble_model
            else (best_result.model_type.value if best_result else "Unknown")
        )

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            trial_results, ensemble_model, model, X, y, task_type
        )

        # Get feature importance
        feature_importance = self._get_feature_importance(
            ensemble_model or model, X, y, task_type
        )

        # Get top features
        top_features = self._get_top_features(feature_importance)

        # Generate confusion matrix for classification
        confusion_mat = None
        if (
            task_type == TaskType.CLASSIFICATION
            and model
            and X is not None
            and y is not None
        ):
            try:
                y_pred = model.predict(X)
                confusion_mat = confusion_matrix(y, y_pred)
            except Exception as e:
                logger.warning(f"Failed to generate confusion matrix: {e}")

        # Generate LLM summary if available
        llm_summary = self._generate_llm_summary(
            trial_results, performance_metrics, dataset_profile, task_type
        )

        # Create model card
        model_card = ModelCard(
            model_name=model_name,
            model_type=best_result.model_type if best_result else None,
            task_type=task_type,
            performance_metrics=performance_metrics,
            feature_importance=feature_importance,
            confusion_matrix=confusion_mat,
            top_features=top_features,
            limitations=self._generate_limitations(dataset_profile, task_type),
            recommendations=self._generate_recommendations(
                performance_metrics, task_type
            ),
            created_at=datetime.now(),
        )

        # Add LLM summary if available
        if llm_summary:
            model_card.recommendations.append(f"LLM Analysis: {llm_summary}")

        return model_card

    def _calculate_performance_metrics(
        self,
        trial_results: List[TrialResult],
        ensemble_model: Optional[Any],
        model: Optional[Any],
        X: Optional[pd.DataFrame],
        y: Optional[pd.Series],
        task_type: TaskType,
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if trial_results:
            # Get metrics from best trial
            best_result = max(trial_results, key=lambda x: x.score)
            metrics.update(
                {
                    "best_score": best_result.score,
                    "cv_mean": (
                        np.mean(best_result.cv_scores) if best_result.cv_scores else 0
                    ),
                    "cv_std": (
                        np.std(best_result.cv_scores) if best_result.cv_scores else 0
                    ),
                    "fit_time": best_result.fit_time,
                    "predict_time": best_result.predict_time,
                }
            )

        # Calculate additional metrics if model and data available
        if model and X is not None and y is not None:
            try:
                from ..models.train_eval import evaluate_model

                additional_metrics = evaluate_model(model, X, y, task_type)
                metrics.update(additional_metrics)
            except Exception as e:
                logger.warning(f"Failed to calculate additional metrics: {e}")

        return metrics

    def _get_feature_importance(
        self,
        model: Optional[Any],
        X: Optional[pd.DataFrame],
        y: Optional[pd.Series],
        task_type: TaskType,
    ) -> Dict[str, float]:
        """Get feature importance."""
        if not model or X is None or y is None:
            return {}

        try:
            return self.importance_analyzer.get_feature_importance(model, X, y)
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return {}

    def _get_top_features(self, feature_importance: Dict[str, float]) -> List[str]:
        """Get top features."""
        if not feature_importance:
            return []

        top_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return [feature for feature, _ in top_features]

    def _generate_limitations(
        self, dataset_profile: Optional[DatasetProfile], task_type: TaskType
    ) -> List[str]:
        """Generate model limitations."""
        limitations = []

        if dataset_profile:
            if dataset_profile.missing_ratio > 0.1:
                limitations.append(
                    f"High missing data ratio ({dataset_profile.missing_ratio:.1%}) may affect performance"
                )

            if dataset_profile.n_rows < 1000:
                limitations.append("Small dataset size may limit model generalization")

            if (
                task_type == TaskType.CLASSIFICATION
                and dataset_profile.class_balance
                and dataset_profile.class_balance < 0.3
            ):
                limitations.append("Class imbalance may affect model performance")

        limitations.append(
            "Model performance may vary on data from different distributions"
        )
        limitations.append("Feature importance may not reflect causal relationships")

        return limitations

    def _generate_recommendations(
        self, performance_metrics: Dict[str, float], task_type: TaskType
    ) -> List[str]:
        """Generate model recommendations."""
        recommendations = []

        # Performance-based recommendations
        best_score = performance_metrics.get("best_score", 0)

        if best_score < 0.7:
            recommendations.append(
                "Consider collecting more data or feature engineering to improve performance"
            )

        if performance_metrics.get("cv_std", 0) > 0.1:
            recommendations.append(
                "High variance in CV scores suggests model instability - consider regularization"
            )

        # Task-specific recommendations
        if task_type == TaskType.CLASSIFICATION:
            if performance_metrics.get("precision", 0) < performance_metrics.get(
                "recall", 0
            ):
                recommendations.append(
                    "Low precision suggests many false positives - consider threshold tuning"
                )
            elif performance_metrics.get("recall", 0) < performance_metrics.get(
                "precision", 0
            ):
                recommendations.append(
                    "Low recall suggests many false negatives - consider class balancing"
                )

        recommendations.append("Monitor model performance on new data regularly")
        recommendations.append("Consider retraining with more recent data if available")

        return recommendations

    def _generate_llm_summary(
        self,
        trial_results: List[TrialResult],
        performance_metrics: Dict[str, float],
        dataset_profile: Optional[DatasetProfile],
        task_type: TaskType,
    ) -> Optional[str]:
        """Generate LLM summary of model performance."""
        if not self.llm_client or not self.llm_config.enabled:
            return None

        try:
            # Create prompt
            prompt = self._create_llm_prompt(
                trial_results, performance_metrics, dataset_profile, task_type
            )

            # Call LLM
            if self.llm_config.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                )
                return response.choices[0].message.content
            elif self.llm_config.provider == "gemini":
                response = self.llm_client.generate_content(prompt)
                return response.text
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")

        return None

    def _create_llm_prompt(
        self,
        trial_results: List[TrialResult],
        performance_metrics: Dict[str, float],
        dataset_profile: Optional[DatasetProfile],
        task_type: TaskType,
    ) -> str:
        """Create prompt for LLM summary."""
        prompt = f"""
Analyze this machine learning model performance and provide a concise summary:

Task Type: {task_type.value}
Dataset: {dataset_profile.n_rows if dataset_profile else 'Unknown'} rows, \
{dataset_profile.n_cols if dataset_profile else 'Unknown'} features
Best Score: {performance_metrics.get('best_score', 0):.4f}
CV Mean: {performance_metrics.get('cv_mean', 0):.4f} ± {performance_metrics.get('cv_std', 0):.4f}

Performance Metrics:
{json.dumps(performance_metrics, indent=2)}

Top Features: {', '.join(self._get_top_features(performance_metrics.get('feature_importance', {})))}

Provide a brief analysis of:
1. Model performance quality
2. Key strengths and weaknesses
3. Recommendations for improvement

Keep response under 200 words.
"""
        return prompt

    def save_card(self, model_card: ModelCard, output_path: Path) -> None:
        """Save model card to markdown file."""
        markdown_content = self._generate_markdown(model_card)

        with open(output_path, "w") as f:
            f.write(markdown_content)

        logger.info(f"Model card saved to {output_path}")

    def _generate_markdown(self, model_card: ModelCard) -> str:
        """Generate markdown content for model card."""
        content = f"""# Model Card: {model_card.model_name}

## Model Information
- **Model Type**: {model_card.model_type.value if model_card.model_type else 'Unknown'}
- **Task Type**: {model_card.task_type.value}
- **Created**: {model_card.created_at.strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics
"""

        # Add performance metrics
        for metric, value in model_card.performance_metrics.items():
            content += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"

        # Add confusion matrix for classification
        if model_card.confusion_matrix is not None:
            content += "\n## Confusion Matrix\n"
            content += "```\n"
            content += str(model_card.confusion_matrix)
            content += "\n```\n"

        # Add top features
        if model_card.top_features:
            content += "\n## Top Features\n"
            for i, feature in enumerate(model_card.top_features, 1):
                importance = model_card.feature_importance.get(feature, 0)
                content += f"{i}. **{feature}**: {importance:.4f}\n"

        # Add limitations
        if model_card.limitations:
            content += "\n## Limitations\n"
            for limitation in model_card.limitations:
                content += f"- {limitation}\n"

        # Add recommendations
        if model_card.recommendations:
            content += "\n## Recommendations\n"
            for recommendation in model_card.recommendations:
                content += f"- {recommendation}\n"

        return content


def generate_model_card(
    trial_results: List[TrialResult],
    ensemble_model: Optional[Any] = None,
    task_type: TaskType = TaskType.CLASSIFICATION,
    dataset_profile: Optional[DatasetProfile] = None,
    model: Optional[Any] = None,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    llm_config: Optional[LLMConfig] = None,
) -> ModelCard:
    """Generate model card."""
    generator = ModelCardGenerator(llm_config)
    return generator.generate_card(
        trial_results, ensemble_model, task_type, dataset_profile, model, X, y
    )
