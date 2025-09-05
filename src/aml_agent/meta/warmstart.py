"""
Warm-start capabilities for hyperparameter optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import optuna
from optuna import Trial

from ..types import ModelType, DatasetProfile, TrialResult
from ..logging import get_logger
from .store import MetaStore
from ..models.spaces import suggest_parameters, validate_parameters

logger = get_logger()


class WarmStartManager:
    """Manage warm-start for hyperparameter optimization."""
    
    def __init__(self, meta_store: MetaStore):
        self.meta_store = meta_store
    
    def warm_start_study(self, 
                        study: optuna.Study,
                        model_type: ModelType,
                        dataset_profile: DatasetProfile,
                        n_warm_start_trials: int = 5) -> None:
        """
        Warm-start a study with parameters from similar datasets.
        
        Args:
            study: Optuna study to warm-start
            model_type: Model type
            dataset_profile: Current dataset profile
            n_warm_start_trials: Number of warm-start trials
        """
        # Get best parameters from similar datasets
        best_params = self.meta_store.get_best_params_for_model(model_type, dataset_profile)
        
        if not best_params:
            logger.info(f"No warm-start parameters found for {model_type}")
            return
        
        logger.info(f"Warm-starting {model_type} with {n_warm_start_trials} trials")
        
        # Create warm-start trials
        warm_start_trials = self._create_warm_start_trials(
            best_params, n_warm_start_trials, model_type
        )
        
        # Add trials to study
        for trial_params in warm_start_trials:
            try:
                # Create a trial with suggested parameters
                trial = study.ask()
                
                # Set parameters
                for param, value in trial_params.items():
                    trial.set_user_attr(param, value)
                
                # Complete the trial (will be evaluated later)
                study.tell(trial, 0.0)  # Placeholder score
                
            except Exception as e:
                logger.warning(f"Failed to add warm-start trial: {e}")
    
    def _create_warm_start_trials(self, 
                                 best_params: Dict[str, Any],
                                 n_trials: int,
                                 model_type: ModelType) -> List[Dict[str, Any]]:
        """Create warm-start trials with parameter variations."""
        trials = []
        
        # Add exact best parameters
        trials.append(best_params.copy())
        
        # Create variations
        for i in range(n_trials - 1):
            trial_params = self._create_parameter_variation(best_params, model_type)
            trials.append(trial_params)
        
        return trials
    
    def _create_parameter_variation(self, 
                                   base_params: Dict[str, Any],
                                   model_type: ModelType) -> Dict[str, Any]:
        """Create parameter variation for warm-start."""
        variation = base_params.copy()
        
        # Apply variations based on parameter type
        for param, value in base_params.items():
            if isinstance(value, (int, float)):
                # Numeric parameter - add small random variation
                if isinstance(value, int):
                    # Integer parameter
                    variation[param] = max(1, int(value * np.random.uniform(0.8, 1.2)))
                else:
                    # Float parameter
                    variation[param] = value * np.random.uniform(0.8, 1.2)
            elif isinstance(value, str):
                # String parameter - keep same or try alternatives
                if param == "solver" and value in ["liblinear", "saga"]:
                    variation[param] = np.random.choice(["liblinear", "saga"])
                elif param == "penalty" and value in ["l1", "l2", "elasticnet"]:
                    variation[param] = np.random.choice(["l1", "l2", "elasticnet"])
                elif param == "weights" and value in ["uniform", "distance"]:
                    variation[param] = np.random.choice(["uniform", "distance"])
                # Keep other string parameters as is
        
        return variation
    
    def get_warm_start_suggestions(self, 
                                  model_type: ModelType,
                                  dataset_profile: DatasetProfile) -> Dict[str, Any]:
        """
        Get warm-start suggestions for a model type.
        
        Args:
            model_type: Model type
            dataset_profile: Current dataset profile
            
        Returns:
            Dictionary with warm-start suggestions
        """
        # Get similar datasets
        similar_runs = self.meta_store.find_similar_datasets(dataset_profile)
        
        if not similar_runs:
            return {}
        
        # Analyze performance across similar datasets
        model_performance = {}
        
        for run in similar_runs:
            trial_summary = run.get("trial_summary", {})
            run_model_perf = trial_summary.get("model_performance", {})
            
            if model_type.value in run_model_perf:
                perf = run_model_perf[model_type.value]
                model_performance[run["run_id"]] = {
                    "mean_score": perf["mean_score"],
                    "std_score": perf["std_score"],
                    "n_trials": perf["n_trials"],
                    "best_params": run.get("best_params", {}).get(model_type.value, {})
                }
        
        if not model_performance:
            return {}
        
        # Calculate statistics
        scores = [perf["mean_score"] for perf in model_performance.values()]
        best_run_id = max(model_performance.keys(), 
                         key=lambda x: model_performance[x]["mean_score"])
        
        return {
            "model_type": model_type.value,
            "similar_datasets": len(similar_runs),
            "performance_stats": {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "best_score": max(scores)
            },
            "best_params": model_performance[best_run_id]["best_params"],
            "confidence": min(1.0, len(similar_runs) / 5.0)  # Confidence based on number of similar datasets
        }
    
    def should_use_warm_start(self, 
                            model_type: ModelType,
                            dataset_profile: DatasetProfile,
                            min_confidence: float = 0.3) -> bool:
        """
        Determine if warm-start should be used for a model type.
        
        Args:
            model_type: Model type
            dataset_profile: Current dataset profile
            min_confidence: Minimum confidence threshold
            
        Returns:
            True if warm-start should be used
        """
        suggestions = self.get_warm_start_suggestions(model_type, dataset_profile)
        
        if not suggestions:
            return False
        
        confidence = suggestions.get("confidence", 0.0)
        return confidence >= min_confidence
    
    def get_optimized_search_space(self, 
                                 model_type: ModelType,
                                 dataset_profile: DatasetProfile,
                                 base_search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimized search space based on similar datasets.
        
        Args:
            model_type: Model type
            dataset_profile: Current dataset profile
            base_search_space: Base search space
            
        Returns:
            Optimized search space
        """
        suggestions = self.get_warm_start_suggestions(model_type, dataset_profile)
        
        if not suggestions or suggestions.get("confidence", 0) < 0.5:
            return base_search_space
        
        # Get best parameters from similar datasets
        best_params = suggestions.get("best_params", {})
        
        if not best_params:
            return base_search_space
        
        # Create optimized search space around best parameters
        optimized_space = {}
        
        for param, value in best_params.items():
            if param not in base_search_space:
                continue
            
            base_spec = base_search_space[param]
            
            if isinstance(value, (int, float)):
                # Numeric parameter - create range around best value
                if isinstance(value, int):
                    # Integer parameter
                    range_factor = 0.5
                    low = max(1, int(value * (1 - range_factor)))
                    high = int(value * (1 + range_factor))
                    optimized_space[param] = (low, high, "int")
                else:
                    # Float parameter
                    range_factor = 0.3
                    low = value * (1 - range_factor)
                    high = value * (1 + range_factor)
                    optimized_space[param] = (low, high, "uniform")
            else:
                # Keep original specification for non-numeric parameters
                optimized_space[param] = base_spec
        
        # Add any missing parameters from base space
        for param, spec in base_search_space.items():
            if param not in optimized_space:
                optimized_space[param] = spec
        
        logger.info(f"Optimized search space for {model_type} based on {suggestions['similar_datasets']} similar datasets")
        
        return optimized_space


def create_warmstart_manager(meta_store: MetaStore) -> WarmStartManager:
    """Create warm-start manager."""
    return WarmStartManager(meta_store)


def warm_start_optuna_study(study: optuna.Study,
                           model_type: ModelType,
                           dataset_profile: DatasetProfile,
                           meta_store: MetaStore,
                           n_warm_start_trials: int = 5) -> None:
    """Warm-start an Optuna study."""
    manager = WarmStartManager(meta_store)
    manager.warm_start_study(study, model_type, dataset_profile, n_warm_start_trials)


def get_warm_start_recommendations(dataset_profile: DatasetProfile,
                                 meta_store: MetaStore) -> Dict[str, Any]:
    """Get warm-start recommendations for all model types."""
    recommendations = {}
    
    for model_type in ModelType:
        manager = WarmStartManager(meta_store)
        suggestions = manager.get_warm_start_suggestions(model_type, dataset_profile)
        
        if suggestions:
            recommendations[model_type.value] = suggestions
    
    return recommendations
