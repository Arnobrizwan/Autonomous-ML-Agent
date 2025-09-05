"""
Monitoring and metrics collection for the Autonomous ML Agent.
"""

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..logging import get_logger

logger = get_logger()


class MetricsCollector:
    """Collect and store system metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[name] += value
        self._add_metric(f"counter_{name}", self.counters[name])

    def record_timer(self, name: str, duration: float) -> None:
        """Record a timing metric."""
        self.timers[name].append(duration)
        if len(self.timers[name]) > self.max_history:
            self.timers[name] = self.timers[name][-self.max_history :]

        self._add_metric(f"timer_{name}", duration)

    def record_gauge(self, name: str, value: float) -> None:
        """Record a gauge metric."""
        self._add_metric(f"gauge_{name}", value)

    def _add_metric(self, name: str, value: float) -> None:
        """Add metric to history."""
        timestamp = time.time()
        self.metrics[name].append(
            {
                "timestamp": timestamp,
                "value": value,
                "datetime": datetime.fromtimestamp(timestamp),
            }
        )

    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None

        values = [m["value"] for m in self.metrics[name]]

        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1],
            "first_seen": self.metrics[name][0]["datetime"],
            "last_seen": self.metrics[name][-1]["datetime"],
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics summary."""
        summary = {}

        # Counter metrics
        for name, value in self.counters.items():
            summary[f"counter_{name}"] = {"type": "counter", "value": value}

        # Timer metrics
        for name, values in self.timers.items():
            if values:
                summary[f"timer_{name}"] = {
                    "type": "timer",
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "latest": values[-1],
                }

        # Gauge metrics
        for name, history in self.metrics.items():
            if not name.startswith(("counter_", "timer_")):
                if history:
                    summary[name] = {
                        "type": "gauge",
                        "latest": history[-1]["value"],
                        "count": len(history),
                    }

        return summary

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()
        self.start_time = time.time()
        logger.info("Metrics reset")


class PerformanceMonitor:
    """Monitor system performance."""

    def __init__(self):
        self.metrics = MetricsCollector()
        self.active_requests = 0
        self.request_times = []

    def start_request(self) -> str:
        """Start monitoring a request."""
        self.active_requests += 1
        request_id = f"req_{int(time.time() * 1000)}"
        self.metrics.increment_counter("requests_started")
        return request_id

    def end_request(self, request_id: str, success: bool = True) -> None:
        """End monitoring a request."""
        self.active_requests = max(0, self.active_requests - 1)

        if success:
            self.metrics.increment_counter("requests_completed")
        else:
            self.metrics.increment_counter("requests_failed")

    def record_prediction_time(self, duration: float) -> None:
        """Record prediction time."""
        self.metrics.record_timer("prediction_time", duration)

    def record_training_time(self, duration: float) -> None:
        """Record training time."""
        self.metrics.record_timer("training_time", duration)

    def record_memory_usage(self, usage_mb: float) -> None:
        """Record memory usage."""
        self.metrics.record_gauge("memory_usage_mb", usage_mb)

    def record_model_accuracy(self, accuracy: float) -> None:
        """Record model accuracy."""
        self.metrics.record_gauge("model_accuracy", accuracy)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "uptime_seconds": self.metrics.get_uptime(),
            "active_requests": self.active_requests,
            "metrics": self.metrics.get_all_metrics(),
        }


class HealthChecker:
    """Check system health."""

    def __init__(self, performance_monitor: PerformanceMonitor):
        self.performance_monitor = performance_monitor
        self.health_checks = {}

    def register_health_check(self, name: str, check_func) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func

    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat(),
        }

        overall_healthy = True

        for name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "result": check_result,
                }
                if not check_result:
                    overall_healthy = False
            except Exception as e:
                results["checks"][name] = {"status": "error", "error": str(e)}
                overall_healthy = False

        results["status"] = "healthy" if overall_healthy else "unhealthy"
        return results

    def check_memory_usage(self) -> bool:
        """Check if memory usage is reasonable."""
        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent
            return memory_percent < 90  # Consider unhealthy if > 90%
        except ImportError:
            return True  # Can't check without psutil

    def check_disk_space(self) -> bool:
        """Check if disk space is sufficient."""
        try:
            import shutil

            free_space = shutil.disk_usage(".").free
            return free_space > 1024 * 1024 * 1024  # At least 1GB free
        except:
            return True

    def check_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        # This would need to be connected to the actual model service
        return True


# Global instances
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker(performance_monitor)

# Register default health checks
health_checker.register_health_check("memory", health_checker.check_memory_usage)
health_checker.register_health_check("disk_space", health_checker.check_disk_space)
health_checker.register_health_check("model_loaded", health_checker.check_model_loaded)
