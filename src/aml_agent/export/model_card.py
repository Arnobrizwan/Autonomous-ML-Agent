"""
Model card generation for ML models.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix

from ..logging import get_logger
from ..types import TaskType

logger = get_logger()


class ModelCardGenerator:
    """Generate comprehensive model cards for ML models."""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type

    def generate_model_card(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: Optional[pd.Series] = None,
        y_prob: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive model card.

        Args:
            model: Trained model
            X: Feature matrix
            y: True target values
            y_pred: Predicted values (if None, will predict)
            y_prob: Predicted probabilities (if None, will predict if available)
            metadata: Additional metadata
            feature_importance: Feature importance data
            save_path: Path to save model card

        Returns:
            Model card content as string
        """
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = model.predict(X)

        if y_prob is None and hasattr(model, "predict_proba"):
            y_prob = pd.DataFrame(model.predict_proba(X))

        # Generate model card sections
        card_sections = {
            "header": self._generate_header(model, metadata),
            "model_details": self._generate_model_details(model, X, y),
            "performance": self._generate_performance_section(y, y_pred, y_prob),
            "data_info": self._generate_data_info(X, y),
            "feature_importance": self._generate_feature_importance_section(
                feature_importance
            ),
            "usage": self._generate_usage_section(),
            "limitations": self._generate_limitations_section(),
            "footer": self._generate_footer(),
        }

        # Combine sections
        model_card = self._combine_sections(card_sections)

        # Save if path provided
        if save_path:
            self._save_model_card(model_card, save_path)

        return model_card

    def _generate_header(
        self, model: BaseEstimator, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate model card header."""
        model_name = type(model).__name__
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = f"""# Model Card: {model_name}

**Generated on:** {timestamp}  
**Task Type:** {self.task_type.value.title()}  
**Model Type:** {model_name}  

## Overview

This model card provides comprehensive information about the trained machine learning model, including its performance, limitations, and usage guidelines.

"""
        return header

    def _generate_model_details(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series
    ) -> str:
        """Generate model details section."""
        details = f"""## Model Details

### Architecture
- **Model Type:** {type(model).__name__}
- **Module:** {type(model).__module__}
- **Task:** {self.task_type.value.title()}

### Parameters
"""

        # Add model parameters
        try:
            params = model.get_params()
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    details += f"- **{key}:** {value}\n"
                elif isinstance(value, (list, tuple)) and len(str(value)) < 100:
                    details += f"- **{key}:** {value}\n"
        except Exception as e:
            details += f"- **Error loading parameters:** {e}\n"

        details += f"""
### Training Data
- **Number of samples:** {X.shape[0]:,}
- **Number of features:** {X.shape[1]:,}
- **Feature types:** {self._get_feature_types(X)}

"""
        return details

    def _generate_performance_section(
        self, y: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.DataFrame]
    ) -> str:
        """Generate performance metrics section."""
        performance = "## Performance Metrics\n\n"

        if self.task_type == TaskType.CLASSIFICATION:
            performance += self._generate_classification_metrics(y, y_pred, y_prob)
        else:
            performance += self._generate_regression_metrics(y, y_pred)

        return performance

    def _generate_classification_metrics(
        self, y: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.DataFrame]
    ) -> str:
        """Generate classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

        metrics = f"""### Classification Metrics
- **Accuracy:** {accuracy:.4f}
- **Precision (weighted):** {precision:.4f}
- **Recall (weighted):** {recall:.4f}
- **F1-Score (weighted):** {f1:.4f}

### Detailed Classification Report
```
{classification_report(y, y_pred, zero_division=0)}
```

### Confusion Matrix
```
{confusion_matrix(y, y_pred)}
```

"""

        # Add probability metrics if available
        if y_prob is not None:
            metrics += f"""### Probability Calibration
- **Mean predicted probability:** {y_prob.mean().mean():.4f}
- **Probability std:** {y_prob.std().mean():.4f}

"""

        return metrics

    def _generate_regression_metrics(self, y: pd.Series, y_pred: pd.Series) -> str:
        """Generate regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = mse**0.5
        r2 = r2_score(y, y_pred)

        metrics = f"""### Regression Metrics
- **Mean Absolute Error (MAE):** {mae:.4f}
- **Mean Squared Error (MSE):** {mse:.4f}
- **Root Mean Squared Error (RMSE):** {rmse:.4f}
- **R² Score:** {r2:.4f}

### Prediction Statistics
- **Mean prediction:** {y_pred.mean():.4f}
- **Std prediction:** {y_pred.std():.4f}
- **Min prediction:** {y_pred.min():.4f}
- **Max prediction:** {y_pred.max():.4f}

"""
        return metrics

    def _generate_data_info(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Generate data information section."""
        data_info = f"""## Data Information

### Dataset Characteristics
- **Total samples:** {X.shape[0]:,}
- **Features:** {X.shape[1]:,}
- **Target distribution:** {self._get_target_distribution(y)}

### Feature Statistics
"""

        # Add feature statistics
        numeric_features = X.select_dtypes(include=["number"]).columns
        if len(numeric_features) > 0:
            data_info += f"- **Numeric features:** {len(numeric_features)}\n"
            data_info += f"- **Mean values:** {X[numeric_features].mean().mean():.4f}\n"
            data_info += f"- **Std values:** {X[numeric_features].std().mean():.4f}\n"

        categorical_features = X.select_dtypes(include=["object", "category"]).columns
        if len(categorical_features) > 0:
            data_info += f"- **Categorical features:** {len(categorical_features)}\n"

        data_info += f"- **Missing values:** {X.isnull().sum().sum():,} ({X.isnull().sum().sum() / (X.shape[0] * X.shape[1]) * 100:.2f}%)\n"

        data_info += "\n"
        return data_info

    def _generate_feature_importance_section(
        self, feature_importance: Optional[Dict[str, Any]]
    ) -> str:
        """Generate feature importance section."""
        if not feature_importance:
            return (
                "## Feature Importance\n\n*Feature importance data not available.*\n\n"
            )

        importance_section = "## Feature Importance\n\n"

        if "sorted_features" in feature_importance:
            importance_section += "### Top 10 Most Important Features\n\n"
            for i, feature_info in enumerate(
                feature_importance["sorted_features"][:10], 1
            ):
                importance_section += f"{i}. **{feature_info['feature']}**: {feature_info['importance']:.4f}\n"
        else:
            importance_section += (
                f"**Method:** {feature_importance.get('method', 'unknown')}\n"
            )
            importance_section += (
                f"**Type:** {feature_importance.get('type', 'unknown')}\n"
            )

        importance_section += "\n"
        return importance_section

    def _generate_usage_section(self) -> str:
        """Generate usage section."""
        usage = """## Usage

### Loading the Model
```python
import joblib
model = joblib.load('pipeline.joblib')
```

### Making Predictions
```python
# For new data
predictions = model.predict(new_data)

# For probabilities (classification only)
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(new_data)
```

### Preprocessing
The model includes built-in preprocessing. Ensure your input data has the same structure as the training data.

"""
        return usage

    def _generate_limitations_section(self) -> str:
        """Generate limitations section."""
        limitations = """## Limitations and Considerations

### Model Limitations
- This model was trained on a specific dataset and may not generalize to different data distributions
- Performance may degrade on data with significantly different characteristics
- The model assumes similar preprocessing steps for new data

### Data Requirements
- Input data should have the same feature structure as training data
- Missing values should be handled consistently with training preprocessing
- Feature types should match the expected input format

### Performance Considerations
- Model performance is based on the specific evaluation metrics used during training
- Cross-validation results may not reflect performance on completely unseen data
- Consider retraining if data distribution changes significantly

"""
        return limitations

    def _generate_footer(self) -> str:
        """Generate model card footer."""
        footer = f"""---

*This model card was automatically generated by the Autonomous ML Agent on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.*

For questions or issues, please refer to the model documentation or contact the development team.
"""
        return footer

    def _get_feature_types(self, X: pd.DataFrame) -> str:
        """Get feature type summary."""
        numeric_count = len(X.select_dtypes(include=["number"]).columns)
        categorical_count = len(X.select_dtypes(include=["object", "category"]).columns)
        datetime_count = len(X.select_dtypes(include=["datetime64"]).columns)

        types = []
        if numeric_count > 0:
            types.append(f"{numeric_count} numeric")
        if categorical_count > 0:
            types.append(f"{categorical_count} categorical")
        if datetime_count > 0:
            types.append(f"{datetime_count} datetime")

        return ", ".join(types) if types else "unknown"

    def _get_target_distribution(self, y: pd.Series) -> str:
        """Get target distribution summary."""
        if self.task_type == TaskType.CLASSIFICATION:
            value_counts = y.value_counts()
            if len(value_counts) <= 5:
                return f"Classes: {dict(value_counts)}"
            else:
                return f"{len(value_counts)} classes, most common: {value_counts.iloc[0]} ({value_counts.iloc[0]/len(y)*100:.1f}%)"
        else:
            return f"Mean: {y.mean():.4f}, Std: {y.std():.4f}, Range: [{y.min():.4f}, {y.max():.4f}]"

    def _combine_sections(self, sections: Dict[str, str]) -> str:
        """Combine all sections into final model card."""
        return "".join(sections.values())

    def _save_model_card(self, model_card: str, save_path: str) -> None:
        """Save model card to file."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(model_card)

        logger.info(f"Model card saved to {path}")

    def generate_summary_card(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: pd.Series,
        score: float,
    ) -> str:
        """Generate a brief summary model card."""
        summary = f"""# Model Summary

**Model:** {type(model).__name__}  
**Task:** {self.task_type.value.title()}  
**Score:** {score:.4f}  
**Samples:** {X.shape[0]:,}  
**Features:** {X.shape[1]:,}  

## Quick Stats
- **Accuracy/R²:** {score:.4f}
- **Data size:** {X.shape[0]:,} samples × {X.shape[1]:,} features
- **Model type:** {type(model).__name__}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return summary
