"""
Monitoring and metrics collection for ML models.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..logging import get_logger
from ..types import MetricType, TaskType

logger = get_logger()


class HealthChecker:
    """Health checker for ML models and system components."""

    def __init__(self):
        self.checks = []

    def add_check(self, name: str, check_func):
        """Add a health check function."""
        self.checks.append((name, check_func))

    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return status."""
        results = {"status": "healthy", "checks": {}}
        
        for name, check_func in self.checks:
            try:
                result = check_func()
                results["checks"][name] = {"status": "pass", "result": result}
        except Exception as e:
                results["checks"][name] = {"status": "fail", "error": str(e)}
                results["status"] = "unhealthy"
        
        return results


class PerformanceMonitor:
    """Monitor performance metrics for ML models."""

    def __init__(self):
        self.metrics = []

    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics.append({
            "name": name,
            "value": value,
            "timestamp": timestamp
        })

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics:
            return {"status": "no_data"}
        
        # Group metrics by name
        metric_groups = {}
        for metric in self.metrics:
            name = metric["name"]
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric["value"])
        
        summary = {}
        for name, values in metric_groups.items():
            summary[name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
                }

        return summary


# Global instances
health_checker = HealthChecker()
performance_monitor = PerformanceMonitor()


class MetricsCollector:
    """Collect and track ML model metrics."""

    def __init__(self, task_type: TaskType, metrics_dir: str = "artifacts/metrics"):
        self.task_type = task_type
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: Optional[pd.DataFrame] = None,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for model evaluation.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_prob: Predicted probabilities (optional)
            model_name: Name of the model
            run_id: Run identifier

        Returns:
            Dictionary with calculated metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "run_id": run_id,
            "task_type": self.task_type.value,
            "n_samples": len(y_true),
        }

        if self.task_type == TaskType.CLASSIFICATION:
            metrics.update(
                self._calculate_classification_metrics(y_true, y_pred, y_prob)
            )
        else:
            metrics.update(self._calculate_regression_metrics(y_true, y_pred))

        # Store in history
        self.metrics_history.append(metrics)

        # Save to file
        self._save_metrics(metrics)

        return metrics

    def _calculate_classification_metrics(
        self, y_true: pd.Series, y_pred: pd.Series, y_prob: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate classification-specific metrics."""
        metrics = {}

        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["precision_weighted"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["recall_weighted"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Binary classification metrics
        if len(y_true.unique()) == 2:
            metrics["precision_binary"] = precision_score(
                y_true, y_pred, zero_division=0
            )
            metrics["recall_binary"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1_binary"] = f1_score(y_true, y_pred, zero_division=0)

        # Probability-based metrics
        if y_prob is not None:
            try:
                if len(y_true.unique()) == 2:
                    # Binary classification
                    metrics["roc_auc"] = roc_auc_score(y_true, y_prob.iloc[:, 1])
                else:
                    # Multi-class classification
                    metrics["roc_auc_ovr"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="weighted"
                    )
                    metrics["roc_auc_ovo"] = roc_auc_score(
                        y_true, y_prob, multi_class="ovo", average="weighted"
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate ROC AUC: {e}")

        # Class distribution
        metrics["class_distribution_true"] = y_true.value_counts().to_dict()
        metrics["class_distribution_pred"] = pd.Series(y_pred).value_counts().to_dict()

        return metrics

    def _calculate_regression_metrics(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, Any]:
        """Calculate regression-specific metrics."""
        metrics = {}

        # Basic regression metrics
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
        metrics["r2"] = r2_score(y_true, y_pred)

        # Additional regression metrics
        residuals = y_true - y_pred
        metrics["mean_residual"] = residuals.mean()
        metrics["std_residual"] = residuals.std()
        metrics["max_residual"] = residuals.max()
        metrics["min_residual"] = residuals.min()

        # Prediction statistics
        metrics["mean_prediction"] = y_pred.mean()
        metrics["std_prediction"] = y_pred.std()
        metrics["min_prediction"] = y_pred.min()
        metrics["max_prediction"] = y_pred.max()

        return metrics

    def track_training_metrics(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv_scores: List[float],
        training_time: float,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Track metrics during model training.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv_scores: Cross-validation scores
            training_time: Training time in seconds
            model_name: Name of the model
            run_id: Run identifier

        Returns:
            Dictionary with training metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = pd.DataFrame(model.predict_proba(X))

        # Calculate basic metrics
        metrics = self.calculate_metrics(
            y_true=y,
            y_pred=pd.Series(y_pred),
            y_prob=y_prob,
            model_name=model_name,
            run_id=run_id,
        )

        # Add training-specific metrics
        metrics.update(
            {
                "cv_scores": cv_scores,
                "cv_mean": sum(cv_scores) / len(cv_scores),
                "cv_std": pd.Series(cv_scores).std(),
                "training_time": training_time,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        )

        return metrics

    def track_prediction_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: Optional[pd.DataFrame] = None,
        prediction_time: Optional[float] = None,
        model_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Track metrics for model predictions.

        Args:
            y_true: True target values
            y_pred: Predicted values
            y_prob: Predicted probabilities (optional)
            prediction_time: Prediction time in seconds
            model_name: Name of the model
            run_id: Run identifier

        Returns:
            Dictionary with prediction metrics
        """
        metrics = self.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=model_name,
            run_id=run_id,
        )

        if prediction_time is not None:
            metrics["prediction_time"] = prediction_time

        return metrics

    def get_metrics_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary of collected metrics.

        Args:
            last_n: Number of recent metrics to include (None for all)

        Returns:
            Dictionary with metrics summary
        """
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}

        metrics_to_analyze = (
            self.metrics_history[-last_n:] if last_n else self.metrics_history
        )

        summary = {
            "total_metrics": len(metrics_to_analyze),
            "time_range": {
                "start": metrics_to_analyze[0]["timestamp"],
                "end": metrics_to_analyze[-1]["timestamp"],
            },
            "models": list(
                set(
                    m.get("model_name")
                    for m in metrics_to_analyze
                    if m.get("model_name")
                )
            ),
            "runs": list(
                set(m.get("run_id") for m in metrics_to_analyze if m.get("run_id"))
            ),
        }

        # Calculate average metrics
        if self.task_type == TaskType.CLASSIFICATION:
            summary["average_metrics"] = self._calculate_average_classification_metrics(
                metrics_to_analyze
            )
        else:
            summary["average_metrics"] = self._calculate_average_regression_metrics(
                metrics_to_analyze
            )

        return summary

    def _calculate_average_classification_metrics(
        self, metrics_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate average classification metrics."""
        avg_metrics = {}

        metric_keys = [
            "accuracy",
            "balanced_accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ]

        for key in metric_keys:
            values = [m.get(key) for m in metrics_list if m.get(key) is not None]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                avg_metrics[f"std_{key}"] = pd.Series(values).std()

        return avg_metrics

    def _calculate_average_regression_metrics(
        self, metrics_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate average regression metrics."""
        avg_metrics = {}

        metric_keys = ["mae", "mse", "rmse", "r2"]

        for key in metric_keys:
            values = [m.get(key) for m in metrics_list if m.get(key) is not None]
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                avg_metrics[f"std_{key}"] = pd.Series(values).std()

        return avg_metrics

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save metrics to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.json"
        filepath = self.metrics_dir / filename

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.debug(f"Metrics saved to {filepath}")

    def export_metrics(self, output_path: str) -> str:
        """Export all metrics to a single file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

        logger.info(f"Metrics exported to {output_file}")
        return str(output_file)

    def load_metrics(self, file_path: str) -> None:
        """Load metrics from file."""
        with open(file_path, "r") as f:
            self.metrics_history = json.load(f)

        logger.info(f"Metrics loaded from {file_path}")

    def get_model_performance_trend(self, model_name: str) -> Dict[str, Any]:
        """Get performance trend for a specific model."""
        model_metrics = [
            m for m in self.metrics_history if m.get("model_name") == model_name
        ]

        if not model_metrics:
            return {"message": f"No metrics found for model {model_name}"}

        trend = {
            "model_name": model_name,
            "n_evaluations": len(model_metrics),
            "timestamps": [m["timestamp"] for m in model_metrics],
        }

        # Add performance trends
        if self.task_type == TaskType.CLASSIFICATION:
            trend["accuracy_trend"] = [m.get("accuracy") for m in model_metrics]
            trend["f1_trend"] = [m.get("f1_weighted") for m in model_metrics]
        else:
            trend["r2_trend"] = [m.get("r2") for m in model_metrics]
            trend["rmse_trend"] = [m.get("rmse") for m in model_metrics]

        return trend
