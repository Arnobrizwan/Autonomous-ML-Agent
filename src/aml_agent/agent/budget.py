"""
Budget management for time-limited optimization.
"""

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from ..logging import get_logger
from ..types import BudgetClock

logger = get_logger()


class BudgetManager:
    """Manage time budget for optimization."""

    def __init__(self, time_budget_seconds: float):
        self.time_budget_seconds = time_budget_seconds
        self.budget_clock = BudgetClock(
            start_time=datetime.now(),
            time_budget_seconds=time_budget_seconds,
        )
        self.is_expired = False

    def check_budget(self) -> bool:
        """Check if budget is still available."""
        if self.is_expired:
            return False

        self.budget_clock.update_elapsed()

        if self.budget_clock.is_expired():
            self.is_expired = True
            logger.info("Time budget expired")
            return False

        return True

    def get_remaining_time(self) -> float:
        """Get remaining time in seconds."""
        self.budget_clock.update_elapsed()
        return self.budget_clock.remaining_seconds()

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        self.budget_clock.update_elapsed()
        return self.budget_clock.elapsed_seconds

    def get_progress(self) -> float:
        """Get progress as percentage."""
        elapsed = self.get_elapsed_time()
        return min(100.0, (elapsed / self.time_budget_seconds) * 100)

    def log_status(self):
        """Log current budget status."""
        elapsed = self.get_elapsed_time()
        remaining = self.get_remaining_time()
        progress = self.get_progress()

        logger.info(
            f"Budget status: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining "
            f"({progress:.1f}% complete)"
        )


@contextmanager
def budget_context(time_budget_seconds: float):
    """Context manager for budget tracking."""
    manager = BudgetManager(time_budget_seconds)

    try:
        yield manager
    finally:
        manager.log_status()


def create_budget_manager(time_budget_seconds: float) -> BudgetManager:
    """Create budget manager instance."""
    return BudgetManager(time_budget_seconds)


class AdaptiveBudgetManager(BudgetManager):
    """Advanced budget manager with adaptive allocation."""

    def __init__(self, time_budget_seconds: float, min_trial_time: float = 5.0):
        super().__init__(time_budget_seconds)
        self.min_trial_time = min_trial_time
        self.trial_times = []
        self.estimated_remaining_trials = 0

    def estimate_remaining_trials(self) -> int:
        """Estimate number of trials that can be completed."""
        if not self.trial_times:
            return 0

        avg_trial_time = sum(self.trial_times) / len(self.trial_times)
        remaining_time = self.get_remaining_time()

        # Add buffer for safety
        safe_remaining_time = remaining_time * 0.8
        estimated_trials = int(safe_remaining_time / avg_trial_time)

        return max(0, estimated_trials)

    def record_trial_time(self, trial_time: float):
        """Record time taken for a trial."""
        self.trial_times.append(trial_time)

        # Keep only recent trials for better estimation
        if len(self.trial_times) > 10:
            self.trial_times = self.trial_times[-10:]

    def should_continue(self, current_trial: int, total_planned_trials: int) -> bool:
        """Check if optimization should continue."""
        if not self.check_budget():
            return False

        # Estimate if we can complete remaining trials
        remaining_trials = total_planned_trials - current_trial
        estimated_remaining = self.estimate_remaining_trials()

        if estimated_remaining < remaining_trials:
            logger.info(
                f"Stopping early: estimated {estimated_remaining} trials remaining, "
                f"but {remaining_trials} planned"
            )
            return False

        return True

    def get_adaptive_trial_budget(self, model_type: str) -> int:
        """Get adaptive trial budget for model type."""
        base_trials = 10

        if not self.trial_times:
            return base_trials

        # Adjust based on remaining time and model complexity
        remaining_time = self.get_remaining_time()
        avg_trial_time = sum(self.trial_times) / len(self.trial_times)

        # Complex models get fewer trials
        complexity_multiplier = 1.0
        if model_type in ["xgboost", "lightgbm", "catboost", "mlp"]:
            complexity_multiplier = 0.7
        elif model_type in ["random_forest", "gradient_boosting"]:
            complexity_multiplier = 0.8

        max_trials = int(remaining_time / avg_trial_time * complexity_multiplier)
        return min(max_trials, base_trials * 2)  # Cap at 2x base


class TimeoutHandler:
    """Handle timeouts gracefully."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def check_timeout(self) -> bool:
        """Check if timeout has been reached."""
        elapsed = time.time() - self.start_time
        return elapsed < self.timeout_seconds

    def get_remaining_time(self) -> float:
        """Get remaining time before timeout."""
        elapsed = time.time() - self.start_time
        return max(0, self.timeout_seconds - elapsed)

    def is_expired(self) -> bool:
        """Check if timeout has expired."""
        return not self.check_timeout()


def create_adaptive_budget_manager(time_budget_seconds: float) -> AdaptiveBudgetManager:
    """Create adaptive budget manager."""
    return AdaptiveBudgetManager(time_budget_seconds)


def create_timeout_handler(timeout_seconds: float) -> TimeoutHandler:
    """Create timeout handler."""
    return TimeoutHandler(timeout_seconds)
