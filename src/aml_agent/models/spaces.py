"""
Hyperparameter search space definitions for the Autonomous ML Agent.
"""

import optuna
from typing import Dict, Any, List, Optional, Union
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner

from ..types import ModelType, SearchStrategy, TaskType
from ..logging import get_logger

logger = get_logger()


class SearchSpaceGenerator:
    """Generate hyperparameter search spaces for different models."""
    
    def __init__(self):
        self.spaces = {}
        self._initialize_spaces()
    
    def _initialize_spaces(self):
        """Initialize search spaces for all models."""
        # Logistic Regression
        self.spaces[ModelType.LOGISTIC_REGRESSION] = {
            'C': (0.001, 1000, 'log'),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': (0.0, 1.0, 'uniform')  # Only used with elasticnet
        }
        
        # Linear Regression
        self.spaces[ModelType.LINEAR_REGRESSION] = {
            'fit_intercept': [True, False],
            'normalize': [True, False]  # Deprecated in newer sklearn versions
        }
        
        # Random Forest
        self.spaces[ModelType.RANDOM_FOREST] = {
            'n_estimators': (10, 200, 'int'),
            'max_depth': (3, 20, 'int'),
            'min_samples_split': (2, 20, 'int'),
            'min_samples_leaf': (1, 10, 'int'),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Gradient Boosting
        self.spaces[ModelType.GRADIENT_BOOSTING] = {
            'n_estimators': (10, 200, 'int'),
            'learning_rate': (0.01, 0.3, 'log'),
            'max_depth': (3, 10, 'int'),
            'min_samples_split': (2, 20, 'int'),
            'min_samples_leaf': (1, 10, 'int'),
            'subsample': (0.8, 1.0, 'uniform')
        }
        
        # k-NN
        self.spaces[ModelType.KNN] = {
            'n_neighbors': (3, 20, 'int'),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': (1, 3, 'int')  # Only used with minkowski
        }
        
        # MLP
        self.spaces[ModelType.MLP] = {
            'hidden_layer_sizes': [
                (50,), (100,), (50, 50), (100, 50), (100, 100),
                (50, 50, 50), (100, 50, 25)
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'learning_rate': (0.001, 0.1, 'log'),
            'alpha': (0.0001, 0.1, 'log'),
            'batch_size': (32, 256, 'int')
        }
    
    def get_search_space(self, model_type: ModelType, 
                        custom_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get search space for model type.
        
        Args:
            model_type: Type of model
            custom_space: Custom search space to override defaults
            
        Returns:
            Search space dictionary
        """
        if model_type not in self.spaces:
            raise ValueError(f"No search space defined for {model_type}")
        
        space = self.spaces[model_type].copy()
        
        if custom_space:
            space.update(custom_space)
        
        return space
    
    def create_optuna_space(self, model_type: ModelType, 
                           custom_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create Optuna-compatible search space.
        
        Args:
            model_type: Type of model
            custom_space: Custom search space
            
        Returns:
            Optuna search space
        """
        space = self.get_search_space(model_type, custom_space)
        optuna_space = {}
        
        for param, spec in space.items():
            if isinstance(spec, (list, tuple)):
                if len(spec) == 3 and isinstance(spec[0], (int, float)) and isinstance(spec[1], (int, float)) and isinstance(spec[2], str):
                    # Range specification: (low, high, type)
                    low, high, param_type = spec
                    if param_type == 'int':
                        optuna_space[param] = optuna.distributions.IntDistribution(low, high)
                    elif param_type == 'log':
                        optuna_space[param] = optuna.distributions.FloatDistribution(low, high, log=True)
                    elif param_type == 'uniform':
                        optuna_space[param] = optuna.distributions.FloatDistribution(low, high)
                else:
                    # Categorical specification
                    optuna_space[param] = optuna.distributions.CategoricalDistribution(spec)
            else:
                # Single value
                optuna_space[param] = spec
        
        return optuna_space
    
    def suggest_parameters(self, trial: optuna.Trial, model_type: ModelType,
                          custom_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Suggest parameters for a trial.
        
        Args:
            trial: Optuna trial
            model_type: Type of model
            custom_space: Custom search space
            
        Returns:
            Suggested parameters
        """
        space = self.create_optuna_space(model_type, custom_space)
        params = {}
        
        for param, distribution in space.items():
            if isinstance(distribution, optuna.distributions.IntDistribution):
                params[param] = trial.suggest_int(param, distribution.low, distribution.high)
            elif isinstance(distribution, optuna.distributions.FloatDistribution):
                if distribution.log:
                    params[param] = trial.suggest_float(param, distribution.low, distribution.high, log=True)
                else:
                    params[param] = trial.suggest_float(param, distribution.low, distribution.high)
            elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
                params[param] = trial.suggest_categorical(param, distribution.choices)
            else:
                params[param] = distribution
        
        return params


def get_search_space(model_type: ModelType, 
                    custom_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get search space for model type."""
    generator = SearchSpaceGenerator()
    return generator.get_search_space(model_type, custom_space)


def create_optuna_study(model_type: ModelType, 
                       strategy: SearchStrategy = SearchStrategy.BAYES,
                       custom_space: Optional[Dict[str, Any]] = None) -> optuna.Study:
    """
    Create Optuna study for hyperparameter optimization.
    
    Args:
        model_type: Type of model
        strategy: Search strategy
        custom_space: Custom search space
        
    Returns:
        Optuna study
    """
    generator = SearchSpaceGenerator()
    
    # Create sampler based on strategy
    if strategy == SearchStrategy.BAYES:
        sampler = TPESampler(seed=42)
    else:
        sampler = RandomSampler(seed=42)
    
    # Create pruner
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',  # Will be overridden based on metric
        sampler=sampler,
        pruner=pruner
    )
    
    return study


def suggest_parameters(trial: optuna.Trial, model_type: ModelType,
                      custom_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Suggest parameters for a trial."""
    generator = SearchSpaceGenerator()
    return generator.suggest_parameters(trial, model_type, custom_space)


def validate_parameters(model_type: ModelType, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean parameters for model type.
    
    Args:
        model_type: Type of model
        params: Parameters to validate
        
    Returns:
        Validated parameters
    """
    validated = params.copy()
    
    # Model-specific validations
    if model_type == ModelType.LOGISTIC_REGRESSION:
        # Validate solver and penalty combination
        if 'penalty' in validated and 'solver' in validated:
            penalty = validated['penalty']
            solver = validated['solver']
            
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                logger.warning(f"L1 penalty requires liblinear or saga solver, got {solver}")
                validated['solver'] = 'liblinear'
            elif penalty == 'elasticnet' and solver != 'saga':
                logger.warning(f"Elasticnet penalty requires saga solver, got {solver}")
                validated['solver'] = 'saga'
    
    elif model_type == ModelType.KNN:
        # Validate metric and p parameter
        if 'metric' in validated and 'p' in validated:
            metric = validated['metric']
            if metric != 'minkowski' and 'p' in validated:
                validated.pop('p')  # Remove p parameter for non-minkowski metrics
    
    elif model_type == ModelType.MLP:
        # Validate hidden layer sizes
        if 'hidden_layer_sizes' in validated:
            hidden_sizes = validated['hidden_layer_sizes']
            if not isinstance(hidden_sizes, (tuple, list)):
                validated['hidden_layer_sizes'] = (hidden_sizes,)
    
    return validated


def get_parameter_importance(study: optuna.Study) -> Dict[str, float]:
    """Get parameter importance from study."""
    try:
        importance = optuna.importance.get_param_importances(study)
        return importance
    except Exception as e:
        logger.warning(f"Could not calculate parameter importance: {e}")
        return {}


def get_best_parameters(study: optuna.Study) -> Dict[str, Any]:
    """Get best parameters from study."""
    return study.best_params


def get_optimization_history(study: optuna.Study) -> List[Dict[str, Any]]:
    """Get optimization history from study."""
    history = []
    for trial in study.trials:
        history.append({
            'trial_number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start,
            'datetime_complete': trial.datetime_complete
        })
    return history
