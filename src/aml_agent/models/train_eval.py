"""
Model training and evaluation for the Autonomous ML Agent.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, balanced_accuracy_score
)
import optuna
from optuna import Trial

from ..types import (
    ModelType, TaskType, MetricType, TrialResult, BudgetClock
)
from ..utils import calculate_metrics, select_metric, set_random_seed
from ..logging import get_logger
from .registries import get_model_factory, validate_model_params
from .spaces import suggest_parameters, validate_parameters

logger = get_logger()


class ModelTrainer:
    """Train and evaluate models with hyperparameter optimization."""
    
    def __init__(self, 
                 task_type: TaskType,
                 metric: MetricType = MetricType.AUTO,
                 cv_folds: int = 5,
                 random_seed: int = 42):
        self.task_type = task_type
        self.metric = select_metric(task_type, metric)
        self.metric_name = self.metric.value
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.trial_results = []
        
        # Set random seed
        set_random_seed(random_seed)
    
    def train_model(self, 
                   model_type: ModelType,
                   X: pd.DataFrame,
                   y: pd.Series,
                   params: Dict[str, Any],
                   trial: Optional[Trial] = None) -> TrialResult:
        """
        Train a single model with given parameters.
        
        Args:
            model_type: Type of model to train
            X: Feature matrix
            y: Target vector
            params: Model parameters
            trial: Optuna trial (optional)
            
        Returns:
            Trial result
        """
        start_time = time.time()
        trial_id = trial.number if trial else len(self.trial_results)
        
        try:
            # Validate parameters
            validated_params = validate_parameters(model_type, params)
            
            # Create model
            model = get_model_factory(model_type, self.task_type, validated_params)
            
            # Train model
            model.fit(X, y)
            
            # Evaluate model
            cv_scores, metrics = self._evaluate_model(model, X, y)
            
            # Calculate timing
            fit_time = time.time() - start_time
            predict_time = self._measure_predict_time(model, X)
            
            # Create trial result
            result = TrialResult(
                trial_id=trial_id,
                model_type=model_type,
                params=validated_params,
                score=cv_scores.mean(),
                metric=self.metric,
                cv_scores=cv_scores.tolist(),
                fit_time=fit_time,
                predict_time=predict_time,
                timestamp=datetime.now(),
                status="completed"
            )
            
            # Log result
            logger.info(f"Trial {trial_id} ({model_type}): score={result.score:.4f}, "
                       f"fit_time={fit_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {str(e)}")
            
            # Create failed trial result
            result = TrialResult(
                trial_id=trial_id,
                model_type=model_type,
                params=params,
                score=0.0,
                metric=self.metric,
                cv_scores=[],
                fit_time=time.time() - start_time,
                predict_time=0.0,
                timestamp=datetime.now(),
                status="failed"
            )
            
            return result
    
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
        """Evaluate model using cross-validation."""
        # Create CV strategy
        if self.task_type == TaskType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        
        # Calculate CV scores
        cv_scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring=self.metric_name,
            n_jobs=-1
        )
        
        # Calculate additional metrics on full dataset
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Get probabilities for classification
        y_prob = None
        if self.task_type == TaskType.CLASSIFICATION and hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
        
        # Calculate comprehensive metrics
        metrics = calculate_metrics(y.values, y_pred, y_prob, self.task_type)
        
        return cv_scores, metrics
    
    def _measure_predict_time(self, model: Any, X: pd.DataFrame) -> float:
        """Measure prediction time for model."""
        start_time = time.time()
        model.predict(X)
        return time.time() - start_time
    
    def optimize_hyperparameters(self,
                                model_type: ModelType,
                                X: pd.DataFrame,
                                y: pd.Series,
                                n_trials: int = 50,
                                budget_clock: Optional[BudgetClock] = None) -> List[TrialResult]:
        """
        Optimize hyperparameters for a model type.
        
        Args:
            model_type: Type of model
            X: Feature matrix
            y: Target vector
            n_trials: Number of trials
            budget_clock: Budget clock for time management
            
        Returns:
            List of trial results
        """
        logger.info(f"Starting hyperparameter optimization for {model_type} "
                   f"with {n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_seed)
        )
        
        def objective(trial):
            # Check budget
            if budget_clock and budget_clock.is_expired():
                raise optuna.TrialPruned()
            
            # Suggest parameters
            params = suggest_parameters(trial, model_type)
            
            # Train and evaluate model
            result = self.train_model(model_type, X, y, params, trial)
            
            # Return score for optimization
            return result.score
        
        # Optimize
        try:
            study.optimize(objective, n_trials=n_trials, timeout=None)
        except optuna.TrialPruned:
            logger.info("Optimization pruned due to budget constraints")
        
        # Collect results
        results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = self.train_model(model_type, X, y, trial.params, trial)
                results.append(result)
        
        logger.info(f"Completed optimization for {model_type}: "
                   f"{len(results)} successful trials")
        
        return results
    
    def train_multiple_models(self,
                             model_types: List[ModelType],
                             X: pd.DataFrame,
                             y: pd.Series,
                             n_trials_per_model: int = 10,
                             budget_clock: Optional[BudgetClock] = None) -> List[TrialResult]:
        """
        Train multiple models with hyperparameter optimization.
        
        Args:
            model_types: List of model types to train
            X: Feature matrix
            y: Target vector
            n_trials_per_model: Number of trials per model
            budget_clock: Budget clock for time management
            
        Returns:
            List of all trial results
        """
        all_results = []
        
        for model_type in model_types:
            if budget_clock and budget_clock.is_expired():
                logger.info("Budget expired, stopping model training")
                break
            
            logger.info(f"Training {model_type} model")
            
            # Calculate remaining trials based on budget
            remaining_trials = n_trials_per_model
            if budget_clock:
                remaining_time = budget_clock.remaining_seconds()
                if remaining_time < 60:  # Less than 1 minute remaining
                    remaining_trials = min(remaining_trials, 5)
            
            # Optimize hyperparameters
            model_results = self.optimize_hyperparameters(
                model_type, X, y, remaining_trials, budget_clock
            )
            
            all_results.extend(model_results)
            
            # Update budget clock
            if budget_clock:
                budget_clock.update_elapsed()
        
        return all_results


def evaluate_model(model: Any, 
                  X: pd.DataFrame, 
                  y: pd.Series,
                  task_type: TaskType,
                  metric: str = 'auto') -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        task_type: Task type
        metric: Metric to use
        
    Returns:
        Dictionary of metrics
    """
    # Select metric
    selected_metric = select_metric(task_type, MetricType(metric))
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Get probabilities for classification
    y_prob = None
    if task_type == TaskType.CLASSIFICATION and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
    
    # Calculate metrics
    metrics = calculate_metrics(y.values, y_pred, y_prob, task_type)
    
    return metrics


def cross_validate_model(model: Any,
                        X: pd.DataFrame,
                        y: pd.Series,
                        task_type: TaskType,
                        cv_folds: int = 5,
                        metric: str = 'auto') -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Cross-validate a model.
    
    Args:
        model: Model to validate
        X: Feature matrix
        y: Target vector
        task_type: Task type
        cv_folds: Number of CV folds
        metric: Metric to use
        
    Returns:
        Tuple of CV scores and metrics
    """
    # Select metric
    selected_metric = select_metric(task_type, MetricType(metric))
    
    # Create CV strategy
    if task_type == TaskType.CLASSIFICATION:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Calculate CV scores
    cv_scores = cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring=selected_metric,
        n_jobs=-1
    )
    
    # Calculate additional metrics (fit model first)
    model.fit(X, y)
    metrics = evaluate_model(model, X, y, task_type, metric)
    
    return cv_scores, metrics


def get_model_performance_summary(results: List[TrialResult]) -> Dict[str, Any]:
    """Get performance summary from trial results."""
    if not results:
        return {}
    
    # Filter successful trials
    successful_results = [r for r in results if r.status == "completed"]
    
    if not successful_results:
        return {"error": "No successful trials"}
    
    # Calculate statistics
    scores = [r.score for r in successful_results]
    fit_times = [r.fit_time for r in successful_results]
    predict_times = [r.predict_time for r in successful_results]
    
    # Group by model type
    model_performance = {}
    for result in successful_results:
        model_type = result.model_type.value
        if model_type not in model_performance:
            model_performance[model_type] = []
        model_performance[model_type].append(result.score)
    
    # Calculate best model
    best_result = max(successful_results, key=lambda x: x.score)
    
    return {
        "total_trials": len(results),
        "successful_trials": len(successful_results),
        "best_score": max(scores),
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "best_model": best_result.model_type.value,
        "best_params": best_result.params,
        "mean_fit_time": np.mean(fit_times),
        "mean_predict_time": np.mean(predict_times),
        "model_performance": {
            model_type: {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "n_trials": len(scores)
            }
            for model_type, scores in model_performance.items()
        }
    }
