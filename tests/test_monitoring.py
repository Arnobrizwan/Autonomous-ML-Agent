"""
Comprehensive tests for monitoring and metrics system.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.aml_agent.monitoring.metrics import (
    HealthChecker,
    MetricsCollector,
    PerformanceMonitor,
    health_checker,
    performance_monitor,
)
from src.aml_agent.types import TaskType


@pytest.mark.unit
class TestHealthChecker:
    """Test health checking functionality."""

    def test_health_checker_initialization(self):
        """Test health checker initialization."""
        checker = HealthChecker()
        assert checker is not None

    def test_run_health_checks(self):
        """Test running health checks."""
        checker = HealthChecker()
        health_status = checker.run_health_checks()
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "checks" in health_status


@pytest.mark.unit
class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        monitor = PerformanceMonitor()
        summary = monitor.get_performance_summary()
        assert isinstance(summary, dict)


@pytest.mark.unit
class TestMetricsCollector:
    """Test metrics collection functionality."""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly."""
        collector = MetricsCollector(TaskType.CLASSIFICATION)
        assert collector is not None
        assert collector.task_type == TaskType.CLASSIFICATION
        assert hasattr(collector, "metrics_history")

    def test_calculate_metrics_classification(self):
        """Test metrics calculation for classification."""
        collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Create sample data
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_pred = pd.Series([0, 1, 0, 1, 0])
        y_prob = pd.DataFrame(
            {0: [0.8, 0.2, 0.9, 0.1, 0.3], 1: [0.2, 0.8, 0.1, 0.9, 0.7]}
        )

        metrics = collector.calculate_metrics(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "precision_weighted" in metrics
        assert "recall_weighted" in metrics
        assert "f1_weighted" in metrics

    def test_calculate_metrics_regression(self):
        """Test metrics calculation for regression."""
        collector = MetricsCollector(TaskType.REGRESSION)

        # Create sample data
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.1, 1.9, 3.1, 3.9, 5.1])

        metrics = collector.calculate_metrics(y_true, y_pred)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "rmse" in metrics

    def test_track_training_metrics(self):
        """Test tracking training metrics."""
        from sklearn.ensemble import RandomForestClassifier

        collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Create sample data and model
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )
        y = pd.Series([0, 1, 0, 1, 1])
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        # Mock CV scores and training time
        cv_scores = [0.9, 0.85, 0.95]
        training_time = 1.5

        metrics = collector.track_training_metrics(
            model, X, y, cv_scores, training_time, "test_model", "run_123"
        )
        assert len(collector.metrics_history) == 1
        assert "accuracy" in metrics

    def test_track_prediction_metrics(self):
        """Test tracking prediction metrics."""
        collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Create sample data
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_pred = pd.Series([0, 1, 0, 1, 0])
        y_prob = pd.DataFrame(
            {0: [0.8, 0.2, 0.9, 0.1, 0.3], 1: [0.2, 0.8, 0.1, 0.9, 0.7]}
        )

        metrics = collector.track_prediction_metrics(
            y_true, y_pred, y_prob, 0.05, "test_model", "run_123"
        )
        assert len(collector.metrics_history) == 1
        assert "accuracy" in metrics

    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        from sklearn.ensemble import RandomForestClassifier

        collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Create sample data and model
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )
        y = pd.Series([0, 1, 0, 1, 1])
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        # Add some metrics
        collector.track_training_metrics(
            model, X, y, [0.9, 0.85], 1.0, "model1", "run1"
        )
        collector.track_training_metrics(
            model, X, y, [0.95, 0.9], 1.2, "model1", "run2"
        )

        summary = collector.get_metrics_summary()
        assert "total_metrics" in summary
        assert summary["total_metrics"] == 2

    def test_export_metrics(self):
        """Test exporting metrics."""
        from sklearn.ensemble import RandomForestClassifier

        collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Create sample data and model
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )
        y = pd.Series([0, 1, 0, 1, 1])
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        # Add some metrics
        collector.track_training_metrics(
            model, X, y, [0.9, 0.85], 1.0, "model1", "run1"
        )

        export_path = collector.export_metrics("test_metrics.json")
        assert Path(export_path).exists()

    def test_load_metrics(self):
        """Test loading metrics from file."""
        from sklearn.ensemble import RandomForestClassifier

        collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Create sample data and model
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]}
        )
        y = pd.Series([0, 1, 0, 1, 1])
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        # First export some metrics
        collector.track_training_metrics(
            model, X, y, [0.9, 0.85], 1.0, "model1", "run1"
        )
        export_path = collector.export_metrics("test_load_metrics.json")

        # Create new collector and load metrics
        new_collector = MetricsCollector(TaskType.CLASSIFICATION)
        new_collector.load_metrics(export_path)
        assert len(new_collector.metrics_history) == 1


@pytest.mark.unit
class TestGlobalInstances:
    """Test global monitoring instances."""

    def test_global_instances_exist(self):
        """Test that global instances are properly initialized."""
        assert health_checker is not None
        assert performance_monitor is not None
        assert MetricsCollector(TaskType.CLASSIFICATION) is not None

    def test_global_instances_functionality(self):
        """Test that global instances work correctly."""
        # Test health checker
        health_status = health_checker.run_health_checks()
        assert isinstance(health_status, dict)

        # Test performance monitor
        summary = performance_monitor.get_performance_summary()
        assert isinstance(summary, dict)

        # Test metrics collector
        from sklearn.ensemble import RandomForestClassifier

        collector = MetricsCollector(TaskType.CLASSIFICATION)
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        y = pd.Series([0, 1, 0])
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        collector.track_training_metrics(
            model, X, y, [0.9, 0.85], 1.0, "test_model", "test_run"
        )
        assert len(collector.metrics_history) == 1


@pytest.mark.integration
class TestIntegration:
    """Integration tests for monitoring system."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        # Initialize components
        health_checker = HealthChecker()
        performance_monitor = PerformanceMonitor()
        metrics_collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Run health checks
        health_status = health_checker.run_health_checks()
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]

        # Get performance summary
        perf_summary = performance_monitor.get_performance_summary()
        assert isinstance(perf_summary, dict)

        # Track metrics
        from sklearn.ensemble import RandomForestClassifier

        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        y = pd.Series([0, 1, 0])
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        metrics_collector.track_training_metrics(
            model, X, y, [0.95, 0.9], 1.5, "test_model", "test_run"
        )
        assert len(metrics_collector.metrics_history) == 1

    def test_monitoring_under_load(self):
        """Test monitoring system under simulated load."""
        from sklearn.ensemble import RandomForestClassifier

        collector = MetricsCollector(TaskType.CLASSIFICATION)

        # Create sample data
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5] * 2, "feature2": [0.1, 0.2, 0.3, 0.4, 0.5] * 2}
        )
        y = pd.Series([0, 1, 0, 1, 1] * 2)

        # Simulate multiple training runs
        for i in range(10):
            model = RandomForestClassifier(random_state=42 + i)
            model.fit(X, y)
            cv_scores = [0.9 + (i * 0.01), 0.85 + (i * 0.01)]
            training_time = 1.0 + (i * 0.1)

            collector.track_training_metrics(
                model, X, y, cv_scores, training_time, f"model_{i}", f"run_{i}"
            )

        assert len(collector.metrics_history) == 10

        # Test summary
        summary = collector.get_metrics_summary()
        assert summary["total_metrics"] == 10
