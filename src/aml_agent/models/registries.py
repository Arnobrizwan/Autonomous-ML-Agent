"""
Model registry and factory for the Autonomous ML Agent.
"""

from typing import Any, Dict, List, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Advanced ML Models
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from ..logging import get_logger
from ..types import ModelType, TaskType

logger = get_logger()


class ModelRegistry:
    """Registry for available models and their configurations."""

    def __init__(self):
        self.models = {}
        self._register_models()

    def _register_models(self):
        """Register all available models."""
        # Classification models
        self.models[ModelType.LOGISTIC_REGRESSION] = {
            "classifier": LogisticRegression,
            "regressor": None,
            "supports_class_weight": True,
            "supports_early_stopping": False,
        }

        self.models[ModelType.RANDOM_FOREST] = {
            "classifier": RandomForestClassifier,
            "regressor": RandomForestRegressor,
            "supports_class_weight": True,
            "supports_early_stopping": False,
        }

        self.models[ModelType.GRADIENT_BOOSTING] = {
            "classifier": GradientBoostingClassifier,
            "regressor": GradientBoostingRegressor,
            "supports_class_weight": False,
            "supports_early_stopping": True,
        }

        self.models[ModelType.KNN] = {
            "classifier": KNeighborsClassifier,
            "regressor": KNeighborsRegressor,
            "supports_class_weight": False,
            "supports_early_stopping": False,
        }

        self.models[ModelType.MLP] = {
            "classifier": MLPClassifier,
            "regressor": MLPRegressor,
            "supports_class_weight": False,
            "supports_early_stopping": True,
        }

        # Linear regression
        self.models[ModelType.LINEAR_REGRESSION] = {
            "classifier": None,
            "regressor": LinearRegression,
            "supports_class_weight": False,
            "supports_early_stopping": False,
        }

        # Advanced ML Models
        if XGBOOST_AVAILABLE:
            self.models[ModelType.XGBOOST] = {
                "classifier": xgb.XGBClassifier,
                "regressor": xgb.XGBRegressor,
                "supports_class_weight": True,
                "supports_early_stopping": True,
            }
        else:
            logger.warning("XGBoost not available, skipping registration")

        if LIGHTGBM_AVAILABLE:
            self.models[ModelType.LIGHTGBM] = {
                "classifier": lgb.LGBMClassifier,
                "regressor": lgb.LGBMRegressor,
                "supports_class_weight": True,
                "supports_early_stopping": True,
            }
        else:
            logger.warning("LightGBM not available, skipping registration")

        if CATBOOST_AVAILABLE:
            self.models[ModelType.CATBOOST] = {
                "classifier": cb.CatBoostClassifier,
                "regressor": cb.CatBoostRegressor,
                "supports_class_weight": True,
                "supports_early_stopping": True,
            }
        else:
            logger.warning("CatBoost not available, skipping registration")

    def get_model_class(
        self, model_type: ModelType, task_type: TaskType
    ) -> BaseEstimator:
        """
        Get model class for given type and task.

        Args:
            model_type: Type of model
            task_type: Task type (classification/regression)

        Returns:
            Model class
        """
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_info = self.models[model_type]

        if task_type == TaskType.CLASSIFICATION:
            if model_info["classifier"] is None:
                raise ValueError(f"Model {model_type} does not support classification")
            return model_info["classifier"]
        else:
            if model_info["regressor"] is None:
                raise ValueError(f"Model {model_type} does not support regression")
            return model_info["regressor"]

    def supports_class_weight(self, model_type: ModelType) -> bool:
        """Check if model supports class_weight parameter."""
        return self.models[model_type]["supports_class_weight"]

    def supports_early_stopping(self, model_type: ModelType) -> bool:
        """Check if model supports early stopping."""
        return self.models[model_type]["supports_early_stopping"]

    def get_available_models(self, task_type: TaskType) -> List[ModelType]:
        """Get list of available models for task type."""
        available = []
        for model_type, info in self.models.items():
            if task_type == TaskType.CLASSIFICATION and info["classifier"] is not None:
                available.append(model_type)
            elif task_type == TaskType.REGRESSION and info["regressor"] is not None:
                available.append(model_type)
        return available


def get_model_factory(
    model_type: ModelType, task_type: TaskType, params: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create model instance with given parameters.

    Args:
        model_type: Type of model
        task_type: Task type
        params: Model parameters

    Returns:
        Model instance
    """
    registry = ModelRegistry()
    model_class: Any = registry.get_model_class(model_type, task_type)

    # Set default parameters
    default_params = _get_default_params(model_type, task_type)
    if params:
        default_params.update(params)

    # Create model instance
    model: Any = model_class(**default_params)

    logger.info(
        f"Created {model_type} model for {task_type} with params: {default_params}"
    )
    return model


def _get_default_params(model_type: ModelType, task_type: TaskType) -> Dict[str, Any]:
    """Get default parameters for model type."""
    defaults = {
        ModelType.LOGISTIC_REGRESSION: {"random_state": 42, "max_iter": 5000},
        ModelType.LINEAR_REGRESSION: {},
        ModelType.RANDOM_FOREST: {"random_state": 42, "n_estimators": 100},
        ModelType.GRADIENT_BOOSTING: {"random_state": 42, "n_estimators": 100},
        ModelType.KNN: {"n_neighbors": 5},
        ModelType.MLP: {"random_state": 42, "max_iter": 1000},
        # Advanced ML Models
        ModelType.XGBOOST: {
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": (
                "logloss" if task_type == TaskType.CLASSIFICATION else "rmse"
            ),
        },
        ModelType.LIGHTGBM: {
            "random_state": 42,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        },
        ModelType.CATBOOST: {
            "random_state": 42,
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
            "verbose": False,
        },
    }

    result = defaults.get(model_type, {})
    return result if isinstance(result, dict) else {}


def get_model_info(model_type: ModelType) -> Dict[str, Any]:
    """Get information about a model type."""
    registry = ModelRegistry()
    return {
        "model_type": model_type,
        "supports_classification": registry.models[model_type]["classifier"]
        is not None,
        "supports_regression": registry.models[model_type]["regressor"] is not None,
        "supports_class_weight": registry.supports_class_weight(model_type),
        "supports_early_stopping": registry.supports_early_stopping(model_type),
    }


def validate_model_params(
    model_type: ModelType, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and clean model parameters.

    Args:
        model_type: Type of model
        params: Parameters to validate

    Returns:
        Validated parameters
    """
    registry = ModelRegistry()
    validated_params = params.copy()

    # Remove unsupported parameters
    if (
        not registry.supports_class_weight(model_type)
        and "class_weight" in validated_params
    ):
        logger.warning(f"Removing unsupported class_weight parameter for {model_type}")
        validated_params.pop("class_weight")

    if (
        not registry.supports_early_stopping(model_type)
        and "early_stopping" in validated_params
    ):
        logger.warning(
            f"Removing unsupported early_stopping parameter for {model_type}"
        )
        validated_params.pop("early_stopping")

    # Validate specific parameters
    if model_type == ModelType.LOGISTIC_REGRESSION:
        if "solver" in validated_params and "penalty" in validated_params:
            solver = validated_params["solver"]
            penalty = validated_params["penalty"]
            if penalty == "l1" and solver not in ["liblinear", "saga"]:
                logger.warning(
                    f"L1 penalty requires liblinear or saga solver, got {solver}"
                )
                validated_params["solver"] = "liblinear"

    # Convert string hidden_layer_sizes to tuple for MLP
    if model_type == ModelType.MLP and "hidden_layer_sizes" in validated_params:
        hidden_sizes_str = validated_params["hidden_layer_sizes"]
        if isinstance(hidden_sizes_str, str):
            # Convert string like "50,50" to tuple (50, 50)
            try:
                hidden_sizes = tuple(
                    int(x.strip()) for x in hidden_sizes_str.split(",")
                )
                validated_params["hidden_layer_sizes"] = hidden_sizes
            except ValueError:
                logger.warning(f"Invalid hidden_layer_sizes format: {hidden_sizes_str}")
                validated_params["hidden_layer_sizes"] = (100,)

    return validated_params


def get_model_complexity(model_type: ModelType) -> str:
    """Get model complexity level."""
    complexity_map = {
        ModelType.LINEAR_REGRESSION: "low",
        ModelType.LOGISTIC_REGRESSION: "low",
        ModelType.KNN: "low",
        ModelType.RANDOM_FOREST: "medium",
        ModelType.GRADIENT_BOOSTING: "high",
        ModelType.MLP: "high",
        # Advanced ML Models
        ModelType.XGBOOST: "high",
        ModelType.LIGHTGBM: "high",
        ModelType.CATBOOST: "high",
    }
    return complexity_map.get(model_type, "medium")


def estimate_training_time(
    model_type: ModelType, n_samples: int, n_features: int
) -> float:
    """Estimate training time in seconds."""
    # Rough estimates based on model complexity
    base_time = 0.1  # Base time in seconds

    complexity_multipliers = {"low": 1.0, "medium": 2.0, "high": 5.0}

    complexity = get_model_complexity(model_type)
    multiplier = complexity_multipliers[complexity]

    # Scale with data size
    size_factor = (
        n_samples * n_features
    ) / 10000  # Normalize to 10k samples * features

    return base_time * multiplier * size_factor
