"""
Monitoring and metrics collection for the Autonomous ML Agent.
"""

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..logging import get_logger

logger = get_logger()


class HealthChecker:
    """Health check system for monitoring system status."""

    def __init__(self):
        self.start_time = time.time()

    def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks."""
        checks = {
            "system": self._check_system_health(),
            "memory": self._check_memory_health(),
            "disk": self._check_disk_health(),
            "dependencies": self._check_dependencies_health(),
        }

        overall_status = (
            "healthy"
            if all(check["status"] == "healthy" for check in checks.values())
            else "unhealthy"
        )

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
        }

    def _check_system_health(self) -> Dict[str, Any]:
        """Check basic system health."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            if cpu_percent > 90:
                return {
                    "status": "unhealthy",
                    "message": f"High CPU usage: {cpu_percent}%",
                }
            elif memory.percent > 90:
                return {
                    "status": "unhealthy",
                    "message": f"High memory usage: {memory.percent}%",
                }
            else:
                return {
                    "status": "healthy",
                    "message": f"CPU: {cpu_percent}%, Memory: {memory.percent}%",
                }
        except ImportError:
            return {"status": "warning", "message": "psutil not available"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"System check failed: {e}"}

    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return {"status": "unhealthy", "message": "Critical memory usage"}
            elif memory.percent > 80:
                return {"status": "warning", "message": "High memory usage"}
            else:
                return {
                    "status": "healthy",
                    "message": f"Memory usage: {memory.percent}%",
                }
        except ImportError:
            return {"status": "warning", "message": "psutil not available"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Memory check failed: {e}"}

    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk health."""
        try:
            import psutil

            disk = psutil.disk_usage("/")
            free_percent = (disk.free / disk.total) * 100

            if free_percent < 5:
                return {"status": "unhealthy", "message": "Critical disk space"}
            elif free_percent < 20:
                return {"status": "warning", "message": "Low disk space"}
            else:
                return {
                    "status": "healthy",
                    "message": f"Disk free: {free_percent:.1f}%",
                }
        except ImportError:
            return {"status": "warning", "message": "psutil not available"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Disk check failed: {e}"}

    def _check_dependencies_health(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        try:
            import numpy
            import pandas
            import sklearn

            return {
                "status": "healthy",
                "message": f"Core dependencies available: sklearn {sklearn.__version__}, pandas {pandas.__version__}, numpy {numpy.__version__}",
            }
        except ImportError as e:
            return {"status": "unhealthy", "message": f"Missing dependency: {e}"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Dependency check failed: {e}"}


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""

    def __init__(self):
        self.start_time = time.time()
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "uptime": self.get_uptime(),
            "metrics": {},
        }

        # Process counters
        for name, value in self.counters.items():
            summary["metrics"][name] = {"type": "counter", "value": value}

        # Process timers
        for name, times in self.timers.items():
            if times:
                summary["metrics"][name] = {
                    "type": "timer",
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times),
                }

        return summary

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric."""
        self.counters[name] += value

    def record_timer(self, name: str, duration: float):
        """Record a timer metric."""
        self.timers[name].append(duration)

    def record_metric(self, name: str, value: float):
        """Record a general metric."""
        self.metrics[name].append(value)

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()


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

    def get_metric_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Analyze trends for a specific metric."""
        if metric_name not in self.metrics:
            return {"trend": "no_data", "change": 0, "values": []}

        cutoff_time = time.time() - (hours * 3600)
        recent_values = [
            m for m in self.metrics[metric_name] if m["timestamp"] > cutoff_time
        ]

        if len(recent_values) < 2:
            return {"trend": "insufficient_data", "change": 0, "values": []}

        values = [m["value"] for m in recent_values]

        # Calculate trend
        if len(values) >= 10:
            recent_avg = sum(values[-5:]) / 5
            older_avg = sum(values[:5]) / 5
        else:
            recent_avg = values[-1]
            older_avg = values[0]

        change = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0

        if change > 5:
            trend = "increasing"
        elif change < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "change": change,
            "values": values,
            "current": values[-1] if values else 0,
            "average": sum(values) / len(values),
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        metrics = self.get_all_metrics()

        if format == "json":
            import json

            return json.dumps(metrics, indent=2, default=str)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(
                ["metric_name", "type", "value", "count", "min", "max", "mean"]
            )

            for name, data in metrics.items():
                if isinstance(data, dict) and "type" in data:
                    writer.writerow(
                        [
                            name,
                            data.get("type", ""),
                            data.get("value", data.get("latest", "")),
                            data.get("count", ""),
                            data.get("min", ""),
                            data.get("max", ""),
                            data.get("mean", ""),
                        ]
                    )

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            health_checks = health_checker.run_health_checks()
            performance = performance_monitor.get_performance_summary()

            # Calculate health score based on various factors
            score = 100.0

            # CPU usage penalty
            cpu_usage = performance.get("cpu_usage", 0)
            if cpu_usage > 90:
                score -= 30
            elif cpu_usage > 80:
                score -= 20
            elif cpu_usage > 70:
                score -= 10

            # Memory usage penalty
            memory_usage = performance.get("memory_usage", 0)
            if memory_usage > 90:
                score -= 30
            elif memory_usage > 80:
                score -= 20
            elif memory_usage > 70:
                score -= 10

            # Disk usage penalty
            disk_usage = performance.get("disk_usage", 0)
            if disk_usage > 90:
                score -= 20
            elif disk_usage > 80:
                score -= 10

            # Health check failures
            failed_checks = sum(
                1 for check, status in health_checks.items() if not status
            )
            score -= failed_checks * 10

            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50.0  # Default neutral score

    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts based on current metrics."""
        alerts = []
        performance = performance_monitor.get_performance_summary()

        # CPU alert
        cpu_usage = performance.get("cpu_usage", 0)
        if cpu_usage > 90:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "cpu_usage",
                    "value": cpu_usage,
                    "message": f"CPU usage is critically high: {cpu_usage:.1f}%",
                }
            )
        elif cpu_usage > 80:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "cpu_usage",
                    "value": cpu_usage,
                    "message": f"CPU usage is high: {cpu_usage:.1f}%",
                }
            )

        # Memory alert
        memory_usage = performance.get("memory_usage", 0)
        if memory_usage > 90:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "memory_usage",
                    "value": memory_usage,
                    "message": f"Memory usage is critically high: {memory_usage:.1f}%",
                }
            )
        elif memory_usage > 80:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "memory_usage",
                    "value": memory_usage,
                    "message": f"Memory usage is high: {memory_usage:.1f}%",
                }
            )

        # Disk alert
        disk_usage = performance.get("disk_usage", 0)
        if disk_usage > 90:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "disk_usage",
                    "value": disk_usage,
                    "message": f"Disk usage is critically high: {disk_usage:.1f}%",
                }
            )
        elif disk_usage > 80:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "disk_usage",
                    "value": disk_usage,
                    "message": f"Disk usage is high: {disk_usage:.1f}%",
                }
            )

        return alerts


# Global instances
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker()

# Additional global instances
metrics_collector = MetricsCollector()
