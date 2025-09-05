"""
Comprehensive tests for monitoring and metrics system.
"""

import pytest
import time
import json
from unittest.mock import patch, MagicMock

from src.aml_agent.monitoring.metrics import (
    HealthChecker,
    PerformanceMonitor,
    MetricsCollector,
    health_checker,
    performance_monitor,
    metrics_collector,
)


class TestHealthChecker:
    """Test health checking functionality."""

    def test_health_checker_initialization(self):
        """Test health checker initializes correctly."""
        checker = HealthChecker()
        assert checker is not None

    def test_run_health_checks(self):
        """Test health checks return expected structure."""
        checker = HealthChecker()
        health_status = checker.run_health_checks()

        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "checks" in health_status
        assert health_status["status"] in ["healthy", "unhealthy"]

    def test_system_health_check(self):
        """Test system health check specifically."""
        checker = HealthChecker()
        system_health = checker._check_system_health()

        assert isinstance(system_health, dict)
        assert "status" in system_health
        assert system_health["status"] in ["healthy", "unhealthy"]

    def test_memory_health_check(self):
        """Test memory health check."""
        checker = HealthChecker()
        memory_health = checker._check_memory_health()

        assert isinstance(memory_health, dict)
        assert "status" in memory_health
        assert memory_health["status"] in ["healthy", "unhealthy", "warning"]

    def test_disk_health_check(self):
        """Test disk health check."""
        checker = HealthChecker()
        disk_health = checker._check_disk_health()

        assert isinstance(disk_health, dict)
        assert "status" in disk_health
        assert disk_health["status"] in ["healthy", "unhealthy", "warning"]

    def test_dependencies_health_check(self):
        """Test dependencies health check."""
        checker = HealthChecker()
        deps_health = checker._check_dependencies_health()

        assert isinstance(deps_health, dict)
        assert "status" in deps_health
        assert deps_health["status"] in ["healthy", "unhealthy"]


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_performance_monitor_initialization(self):
        """Test performance monitor initializes correctly."""
        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_get_performance_summary(self):
        """Test performance summary generation."""
        monitor = PerformanceMonitor()
        summary = monitor.get_performance_summary()

        assert isinstance(summary, dict)
        assert "uptime" in summary
        assert "metrics" in summary

        # All values should be numeric or dict
        for key, value in summary.items():
            assert isinstance(value, (int, float, dict))


class TestMetricsCollector:
    """Test metrics collection functionality."""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initializes correctly."""
        collector = MetricsCollector()
        assert collector is not None
        assert collector.max_history == 1000

    def test_increment_counter(self):
        """Test counter increment in collector."""
        collector = MetricsCollector()
        collector.increment_counter("test_counter")
        collector.increment_counter("test_counter", 3)

        assert collector.counters["test_counter"] == 4

    def test_record_timer(self):
        """Test timer recording in collector."""
        collector = MetricsCollector()
        collector.record_timer("test_timer", 2.5)

        assert "test_timer" in collector.timers
        assert collector.timers["test_timer"] == [2.5]

    def test_record_gauge(self):
        """Test gauge recording in collector."""
        collector = MetricsCollector()
        collector.record_gauge("test_gauge", 75.0)

        assert "gauge_test_gauge" in collector.metrics
        assert len(collector.metrics["gauge_test_gauge"]) == 1

    def test_get_metric_summary(self):
        """Test metric summary generation."""
        collector = MetricsCollector()

        # Add some data
        collector.record_timer("test_timer", 1.0)
        collector.record_timer("test_timer", 2.0)
        collector.record_timer("test_timer", 3.0)

        summary = collector.get_metric_summary("timer_test_timer")

        assert summary is not None
        assert summary["name"] == "timer_test_timer"
        assert summary["count"] == 3
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["mean"] == 2.0

    def test_get_all_metrics(self):
        """Test getting all metrics summary."""
        collector = MetricsCollector()

        # Add various types of metrics
        collector.increment_counter("test_counter")
        collector.record_timer("test_timer", 1.5)
        collector.record_gauge("test_gauge", 50.0)

        all_metrics = collector.get_all_metrics()

        assert isinstance(all_metrics, dict)
        assert "counter_test_counter" in all_metrics
        assert "timer_test_timer" in all_metrics
        assert "gauge_test_gauge" in all_metrics

    def test_get_metric_trends(self):
        """Test metric trend analysis."""
        collector = MetricsCollector()

        # Add some historical data
        for i in range(10):
            collector.record_gauge("test_trend", float(i))
            time.sleep(0.01)  # Small delay to ensure different timestamps

        trends = collector.get_metric_trends("gauge_test_trend", hours=1)

        assert isinstance(trends, dict)
        assert "trend" in trends
        assert "change" in trends
        assert "values" in trends
        assert "current" in trends
        assert "average" in trends

    def test_export_metrics_json(self):
        """Test JSON export of metrics."""
        collector = MetricsCollector()
        collector.record_gauge("test_export", 100.0)

        json_export = collector.export_metrics("json")

        assert isinstance(json_export, str)
        data = json.loads(json_export)
        assert isinstance(data, dict)

    def test_export_metrics_csv(self):
        """Test CSV export of metrics."""
        collector = MetricsCollector()
        collector.record_gauge("test_export", 100.0)

        csv_export = collector.export_metrics("csv")

        assert isinstance(csv_export, str)
        assert "metric_name" in csv_export
        assert "type" in csv_export
        assert "value" in csv_export

    def test_export_metrics_invalid_format(self):
        """Test export with invalid format raises error."""
        collector = MetricsCollector()

        with pytest.raises(ValueError):
            collector.export_metrics("invalid_format")

    def test_get_health_score(self):
        """Test health score calculation."""
        collector = MetricsCollector()

        with patch.object(health_checker, "run_health_checks") as mock_health:
            with patch.object(
                performance_monitor, "get_performance_summary"
            ) as mock_perf:
                mock_health.return_value = {
                    "system": True,
                    "memory": True,
                    "disk": True,
                    "dependencies": True,
                }
                mock_perf.return_value = {
                    "cpu_usage": 50,
                    "memory_usage": 60,
                    "disk_usage": 70,
                }

                score = collector.get_health_score()

                assert isinstance(score, (int, float))
                assert 0 <= score <= 100

    def test_get_performance_alerts(self):
        """Test performance alert generation."""
        collector = MetricsCollector()

        with patch.object(performance_monitor, "get_performance_summary") as mock_perf:
            mock_perf.return_value = {
                "cpu_usage": 95,  # Should trigger critical alert
                "memory_usage": 85,  # Should trigger warning alert
                "disk_usage": 75,  # Should not trigger alert
            }

            alerts = collector.get_performance_alerts()

            assert isinstance(alerts, list)
            # Check alert structure if any alerts exist
            for alert in alerts:
                assert "type" in alert
                assert "metric" in alert
                assert "value" in alert
                assert "message" in alert
                assert alert["type"] in ["critical", "warning"]

    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        collector = MetricsCollector()

        # Add some data
        collector.increment_counter("test_counter")
        collector.record_timer("test_timer", 1.0)
        collector.record_gauge("test_gauge", 50.0)

        # Reset
        collector.reset_metrics()

        # Check everything is reset
        assert collector.counters == {}
        assert collector.timers == {}
        assert collector.metrics == {}


class TestGlobalInstances:
    """Test global monitoring instances."""

    def test_global_instances_exist(self):
        """Test that global instances are properly initialized."""
        assert health_checker is not None
        assert performance_monitor is not None
        assert metrics_collector is not None

    def test_global_instances_functionality(self):
        """Test that global instances work correctly."""
        # Test health checker
        health_status = health_checker.run_health_checks()
        assert isinstance(health_status, dict)

        # Test performance monitor
        summary = performance_monitor.get_performance_summary()
        assert isinstance(summary, dict)

        # Test metrics collector
        metrics_collector.increment_counter("global_test")
        assert metrics_collector.counters["global_test"] == 1


class TestIntegration:
    """Integration tests for monitoring system."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        collector = MetricsCollector()

        # Simulate some activity
        collector.increment_counter("requests_processed")
        collector.record_timer("request_duration", 0.5)
        collector.record_gauge("active_connections", 25)

        # Get health status
        health_status = health_checker.run_health_checks()

        # Get performance summary
        performance = performance_monitor.get_performance_summary()

        # Get all metrics
        all_metrics = collector.get_all_metrics()

        # Verify everything works together
        assert isinstance(health_status, dict)
        assert isinstance(performance, dict)
        assert isinstance(all_metrics, dict)

        # Verify metrics were recorded
        assert "counter_requests_processed" in all_metrics
        assert "timer_request_duration" in all_metrics
        assert "gauge_active_connections" in all_metrics

    def test_monitoring_under_load(self):
        """Test monitoring system under simulated load."""
        collector = MetricsCollector()

        # Simulate high load
        for i in range(100):
            collector.increment_counter("high_load_counter")
            collector.record_timer("high_load_timer", i * 0.01)
            collector.record_gauge("high_load_gauge", i)

        # Verify system still works
        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) >= 3

        # Test trend analysis
        trends = collector.get_metric_trends("gauge_high_load_gauge")
        assert trends["trend"] in ["increasing", "stable", "decreasing"]

        # Test health score
        health_score = collector.get_health_score()
        assert 0 <= health_score <= 100
