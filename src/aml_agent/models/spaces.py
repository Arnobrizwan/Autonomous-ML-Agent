"""
Hyperparameter search spaces for Optuna optimization.
"""

from typing import Any, Dict, Optional

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from ..logging import get_logger
from ..types import ModelType, TaskType

logger = get_logger()


class SearchSpaceBuilder:
    """Build Optuna search spaces for different model types."""

    def __init__(self):
        self.spaces = {}

    def get_search_space(
        self,
        model_type: ModelType,
        task_type: TaskType,
        custom_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get search space for model type.

        Args:
            model_type: Type of model
            task_type: Task type (classification/regression)
            custom_space: Custom search space overrides

        Returns:
            Optuna search space dictionary
        """
        if custom_space:
            return self._validate_custom_space(custom_space, model_type)

        if model_type == ModelType.LOGISTIC_REGRESSION:
            return self._logistic_regression_space()
        elif model_type == ModelType.LINEAR_REGRESSION:
            return self._linear_regression_space()
        elif model_type == ModelType.RANDOM_FOREST:
            return self._random_forest_space()
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return self._gradient_boosting_space()
        elif model_type == ModelType.KNN:
            return self._knn_space()
        elif model_type == ModelType.MLP:
            return self._mlp_space()
        elif model_type == ModelType.XGBOOST:
            return self._xgboost_space()
        elif model_type == ModelType.LIGHTGBM:
            return self._lightgbm_space()
        elif model_type == ModelType.CATBOOST:
            return self._catboost_space()
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return {}

    def _logistic_regression_space(self) -> Dict[str, Any]:
        """Logistic regression search space."""
        return {
            "C": FloatDistribution(0.001, 1000, log=True),
            "penalty": CategoricalDistribution(["l1", "l2"]),
            "solver": CategoricalDistribution(["liblinear", "saga"]),
            "max_iter": IntDistribution(100, 2000),
        }

    def _linear_regression_space(self) -> Dict[str, Any]:
        """Linear regression search space."""
        return {
            "fit_intercept": CategoricalDistribution([True, False]),
        }

    def _random_forest_space(self) -> Dict[str, Any]:
        """Random forest search space."""
        return {
            "n_estimators": IntDistribution(10, 500),
            "max_depth": IntDistribution(3, 30),
            "min_samples_split": IntDistribution(2, 20),
            "min_samples_leaf": IntDistribution(1, 10),
            "max_features": CategoricalDistribution(["sqrt", "log2", None]),
            "bootstrap": CategoricalDistribution([True, False]),
        }

    def _gradient_boosting_space(self) -> Dict[str, Any]:
        """Gradient boosting search space."""
        return {
            "n_estimators": IntDistribution(10, 500),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "max_depth": IntDistribution(3, 15),
            "min_samples_split": IntDistribution(2, 20),
            "min_samples_leaf": IntDistribution(1, 10),
            "subsample": FloatDistribution(0.5, 1.0),
            "max_features": CategoricalDistribution(["sqrt", "log2", None]),
        }

    def _knn_space(self) -> Dict[str, Any]:
        """k-NN search space."""
        return {
            "n_neighbors": IntDistribution(3, 50),
            "weights": CategoricalDistribution(["uniform", "distance"]),
            "metric": CategoricalDistribution(["euclidean", "manhattan", "minkowski"]),
            "p": IntDistribution(1, 3),  # For Minkowski distance
        }

    def _mlp_space(self) -> Dict[str, Any]:
        """Multi-layer perceptron search space."""
        return {
            "hidden_layer_sizes": CategoricalDistribution(
                [
                    "50",
                    "100",
                    "50,50",
                    "100,50",
                    "100,100",
                    "50,50,50",
                    "100,50,25",
                    "200,100",
                ]
            ),
            "activation": CategoricalDistribution(["relu", "tanh", "logistic"]),
            "learning_rate": CategoricalDistribution(
                ["constant", "invscaling", "adaptive"]
            ),
            "learning_rate_init": FloatDistribution(0.0001, 0.1, log=True),
            "alpha": FloatDistribution(0.0001, 0.1, log=True),
            "max_iter": IntDistribution(200, 2000),
            "early_stopping": CategoricalDistribution([True, False]),
            "validation_fraction": FloatDistribution(0.05, 0.2),
        }

    def _xgboost_space(self) -> Dict[str, Any]:
        """XGBoost search space."""
        return {
            "n_estimators": IntDistribution(10, 1000),
            "max_depth": IntDistribution(3, 15),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "subsample": FloatDistribution(0.5, 1.0),
            "colsample_bytree": FloatDistribution(0.5, 1.0),
            "colsample_bylevel": FloatDistribution(0.5, 1.0),
            "reg_alpha": FloatDistribution(0.01, 10, log=True),  # Fixed: was 0
            "reg_lambda": FloatDistribution(0.01, 10, log=True),  # Fixed: was 0
            "gamma": FloatDistribution(0, 5),
            "min_child_weight": IntDistribution(1, 10),
        }

    def _lightgbm_space(self) -> Dict[str, Any]:
        """LightGBM search space."""
        return {
            "n_estimators": IntDistribution(10, 1000),
            "max_depth": IntDistribution(3, 15),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "subsample": FloatDistribution(0.5, 1.0),
            "colsample_bytree": FloatDistribution(0.5, 1.0),
            "reg_alpha": FloatDistribution(0.01, 10, log=True),  # Fixed: was 0
            "reg_lambda": FloatDistribution(0.01, 10, log=True),  # Fixed: was 0
            "min_child_samples": IntDistribution(5, 100),
            "num_leaves": IntDistribution(10, 300),
        }

    def _catboost_space(self) -> Dict[str, Any]:
        """CatBoost search space."""
        return {
            "iterations": IntDistribution(10, 1000),
            "depth": IntDistribution(3, 15),
            "learning_rate": FloatDistribution(0.01, 0.3, log=True),
            "l2_leaf_reg": FloatDistribution(1, 10, log=True),
            "bootstrap_type": CategoricalDistribution(
                ["Bernoulli", "MVS"]
            ),  # Removed Bayesian
            "subsample": FloatDistribution(0.5, 1.0),
            "colsample_bylevel": FloatDistribution(0.5, 1.0),
            "min_child_samples": IntDistribution(5, 100),
        }

    def _validate_custom_space(
        self, custom_space: Dict[str, Any], model_type: ModelType
    ) -> Dict[str, Any]:
        """Validate and convert custom search space to Optuna format."""
        validated_space = {}

        for param, value in custom_space.items():
            if isinstance(value, list):
                if all(isinstance(x, (int, float)) for x in value):
                    if all(isinstance(x, int) for x in value):
                        validated_space[param] = CategoricalDistribution(value)
                    else:
                        # Convert to float distribution if mixed types
                        min_val, max_val = min(value), max(value)
                        validated_space[param] = FloatDistribution(min_val, max_val)
                else:
                    validated_space[param] = CategoricalDistribution(value)
            elif isinstance(value, tuple) and len(value) == 2:
                min_val, max_val = value
                if isinstance(min_val, int) and isinstance(max_val, int):
                    validated_space[param] = IntDistribution(min_val, max_val)
                else:
                    validated_space[param] = FloatDistribution(min_val, max_val)
            else:
                logger.warning(f"Invalid parameter format for {param}: {value}")

        return validated_space

    def create_study(
        self,
        model_type: ModelType,
        task_type: TaskType,
        direction: str = "maximize",
        custom_space: Optional[Dict[str, Any]] = None,
    ) -> optuna.Study:
        """
        Create Optuna study with search space.

        Args:
            model_type: Type of model
            task_type: Task type
            direction: Optimization direction
            custom_space: Custom search space

        Returns:
            Optuna study
        """
        search_space = self.get_search_space(model_type, task_type, custom_space)

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Add search space to study
        study.enqueue_trial(
            {param: dist.suggest() for param, dist in search_space.items()}
        )

        return study


def get_search_space(
    model_type: ModelType,
    task_type: TaskType,
    custom_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to get search space."""
    builder = SearchSpaceBuilder()
    return builder.get_search_space(model_type, task_type, custom_space)


def create_optuna_study(
    model_type: ModelType,
    task_type: TaskType,
    direction: str = "maximize",
    custom_space: Optional[Dict[str, Any]] = None,
) -> optuna.Study:
    """Convenience function to create Optuna study."""
    builder = SearchSpaceBuilder()
    return builder.create_study(model_type, task_type, direction, custom_space)
