"""
Budget management for time-limited optimization.
"""

import signal
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from ..logging import get_logger
from ..types import BudgetClock

logger = get_logger()


class BudgetManager:
    """Manage budget constraints for optimization."""

    def __init__(self, time_budget_seconds: float):
        self.time_budget_seconds = time_budget_seconds
        self.start_time = datetime.now()
        self.budget_clock = BudgetClock(self.start_time, time_budget_seconds)
        self.is_expired = False
        self.callbacks = []

    def add_callback(self, callback: Callable[[], None]):
        """Add callback to be called when budget expires."""
        self.callbacks.append(callback)

    def check_budget(self) -> bool:
        """Check if budget is still available."""
        self.budget_clock.update_elapsed()

        if self.budget_clock.is_expired() and not self.is_expired:
            self.is_expired = True
            logger.warning(
                f"Budget expired after {self.budget_clock.elapsed_seconds:.1f}s"
            )

            # Call all callbacks
            for callback in self.callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in budget callback: {e}")

        return not self.is_expired

    def remaining_time(self) -> float:
        """Get remaining time in seconds."""
        return self.budget_clock.remaining_seconds()

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.budget_clock.elapsed_seconds

    def progress(self) -> float:
        """Get budget progress as percentage."""
        return (self.elapsed_time() / self.time_budget_seconds) * 100


@contextmanager
def budget_context(time_budget_seconds: float):
    """Context manager for budget management."""
    manager = BudgetManager(time_budget_seconds)

    def timeout_handler(signum, frame):
        logger.warning("Budget timeout signal received")
        manager.is_expired = True

    # Set up signal handler for timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(time_budget_seconds))

    try:
        yield manager
    finally:
        # Restore original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class TimeoutException(Exception):
    """Exception raised when budget is exceeded."""

    pass


class BudgetAwareOptimizer:
    """Wrapper for optimization that respects budget constraints."""

    def __init__(self, time_budget_seconds: float):
        self.manager = BudgetManager(time_budget_seconds)
        self.optimization_start = None

    def optimize(self, objective_func: Callable, n_trials: int = 100) -> Any:
        """
        Run optimization with budget constraints.

        Args:
            objective_func: Function to optimize
            n_trials: Maximum number of trials

        Returns:
            Optimization result
        """
        self.optimization_start = time.time()

        # Calculate trials per second based on budget
        remaining_time = self.manager.remaining_time()
        if remaining_time <= 0:
            raise TimeoutException("No time remaining for optimization")

        # Estimate trials per second (conservative estimate)
        estimated_trials_per_second = 0.5
        max_trials = min(n_trials, int(remaining_time * estimated_trials_per_second))

        logger.info(
            f"Starting optimization with {max_trials} trials "
            f"and {remaining_time:.1f}s remaining"
        )

        results = []
        for i in range(max_trials):
            if not self.manager.check_budget():
                logger.info(f"Budget expired after {i} trials")
                break

            try:
                result = objective_func(i)
                results.append(result)
            except Exception as e:
                logger.warning(f"Trial {i} failed: {e}")
                continue

        logger.info(
            f"Completed {len(results)} trials in {time.time() - self.optimization_start:.1f}s"
        )
        return results

    def should_continue(self) -> bool:
        """Check if optimization should continue."""
        return self.manager.check_budget()

    def get_progress(self) -> dict:
        """Get optimization progress."""
        return {
            "elapsed_time": self.manager.elapsed_time(),
            "remaining_time": self.manager.remaining_time(),
            "progress_percent": self.manager.progress(),
            "is_expired": self.manager.is_expired,
        }


def create_budget_manager(time_budget_seconds: float) -> BudgetManager:
    """Create a budget manager."""
    return BudgetManager(time_budget_seconds)


def estimate_trial_time(model_type: str, n_samples: int, n_features: int) -> float:
    """Estimate time for a single trial."""
    # Base time estimates in seconds
    base_times = {
        "logistic_regression": 0.1,
        "linear_regression": 0.05,
        "random_forest": 0.5,
        "gradient_boosting": 1.0,
        "knn": 0.2,
        "mlp": 0.8,
    }

    base_time = base_times.get(model_type, 0.5)

    # Scale with data size
    size_factor = (n_samples * n_features) / 10000

    return base_time * size_factor


def calculate_optimal_trials(
    time_budget_seconds: float, model_types: list, n_samples: int, n_features: int
) -> dict:
    """Calculate optimal number of trials per model based on budget."""
    total_estimated_time = 0
    model_times = {}

    for model_type in model_types:
        trial_time = estimate_trial_time(model_type, n_samples, n_features)
        model_times[model_type] = trial_time
        total_estimated_time += trial_time

    # Allocate trials proportionally
    trials_per_model = {}
    for model_type in model_types:
        proportion = model_times[model_type] / total_estimated_time
        trials = max(1, int(time_budget_seconds * proportion / model_times[model_type]))
        trials_per_model[model_type] = trials

    return trials_per_model


def monitor_budget(manager: BudgetManager, interval: float = 10.0):
    """Monitor budget in a separate thread."""

    def monitor():
        while not manager.is_expired:
            time.sleep(interval)
            if manager.check_budget():
                remaining = manager.remaining_time()
                progress = manager.progress()
                logger.info(
                    f"Budget status: {progress:.1f}% used, "
                    f"{remaining:.1f}s remaining"
                )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread
