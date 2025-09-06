"""
Meta-learning warm start functionality for hyperparameter optimization.
"""

from typing import Any, Dict, List, Optional

import optuna
from optuna.samplers import TPESampler

from ..logging import get_logger
from ..types import DatasetProfile, ModelType
from .store import MetaStore

logger = get_logger()


class WarmStartManager:
    """Manages warm start functionality for hyperparameter optimization."""

    def __init__(self, meta_store: MetaStore):
        self.meta_store = meta_store

    def get_warm_start_params(
        self, model_type: ModelType, profile: DatasetProfile, top_k: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Get warm start parameters for a model type based on similar datasets.

        Args:
            model_type: Type of model to get parameters for
            profile: Dataset profile to find similar datasets
            top_k: Number of similar runs to consider

        Returns:
            Dictionary of warm start parameters or None
        """
        # Query similar datasets
        similar_runs = self.meta_store.query_similar(profile, top_k=top_k)

        if not similar_runs:
            logger.info(f"No similar datasets found for {model_type.value}")
            return None

        # Get best parameters for this model type
        best_params = self.meta_store.get_best_params(model_type.value, similar_runs)

        if best_params:
            logger.info(
                f"Found warm start parameters for {model_type.value} from {len(similar_runs)} similar runs"
            )
            return best_params
        else:
            logger.info(f"No warm start parameters found for {model_type.value}")
            return None

    def create_warm_start_sampler(
        self,
        model_type: ModelType,
        profile: DatasetProfile,
        base_sampler: Optional[TPESampler] = None,
    ) -> TPESampler:
        """
        Create a TPE sampler with warm start parameters.

        Args:
            model_type: Type of model
            profile: Dataset profile
            base_sampler: Base TPE sampler to extend

        Returns:
            TPE sampler with warm start parameters
        """
        warm_start_params = self.get_warm_start_params(model_type, profile)

        if warm_start_params:
            # Create sampler with warm start
            sampler = TPESampler(
                n_startup_trials=5,  # Reduced startup trials due to warm start
                n_ei_candidates=24,
                gamma=lambda n: min(25, n // 4),
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=False,
                multivariate=True,
                warn_independent_sampling=True,
                constant_liar=True,
                constraints_func=None,
                seed=None,
            )

            # Add warm start parameters as suggestions
            if hasattr(sampler, "_warm_start_params"):
                sampler._warm_start_params = warm_start_params

            logger.info(f"Created warm start sampler for {model_type.value}")
            return sampler
        else:
            # Return base sampler or create new one
            if base_sampler:
                return base_sampler
            else:
                return TPESampler(
                    n_startup_trials=10,
                    n_ei_candidates=24,
                    gamma=lambda n: min(25, n // 4),
                    prior_weight=1.0,
                    consider_magic_clip=True,
                    consider_endpoints=False,
                    multivariate=True,
                    warn_independent_sampling=True,
                    constant_liar=True,
                    constraints_func=None,
                    seed=None,
                )

    def suggest_warm_start_trials(
        self,
        study: optuna.Study,
        model_type: ModelType,
        profile: DatasetProfile,
        n_trials: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Suggest warm start trials for a study.

        Args:
            study: Optuna study
            model_type: Type of model
            profile: Dataset profile
            n_trials: Number of warm start trials to suggest

        Returns:
            List of suggested parameter dictionaries
        """
        warm_start_params = self.get_warm_start_params(model_type, profile)

        if not warm_start_params:
            return []

        suggestions = []

        # Add the best parameters as first suggestion
        suggestions.append(warm_start_params)

        # Add variations of the best parameters
        for i in range(1, min(n_trials, 3)):
            variation = self._create_parameter_variation(
                warm_start_params, variation_factor=0.1 * i
            )
            if variation:
                suggestions.append(variation)

        logger.info(
            f"Suggested {len(suggestions)} warm start trials for {model_type.value}"
        )
        return suggestions

    def _create_parameter_variation(
        self, base_params: Dict[str, Any], variation_factor: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """
        Create a variation of base parameters.

        Args:
            base_params: Base parameters to vary
            variation_factor: Factor for variation (0.1 = 10% variation)

        Returns:
            Varied parameters or None
        """
        try:
            variation = {}

            for key, value in base_params.items():
                if isinstance(value, (int, float)):
                    # Add random variation
                    import random

                    variation_amount = abs(value) * variation_factor
                    variation[key] = value + random.uniform(
                        -variation_amount, variation_amount
                    )

                    # Ensure reasonable bounds
                    if key in ["C", "alpha", "learning_rate"]:
                        variation[key] = max(0.001, variation[key])
                    elif key in [
                        "n_estimators",
                        "max_depth",
                        "min_samples_split",
                        "min_samples_leaf",
                    ]:
                        variation[key] = max(1, int(variation[key]))
                else:
                    # Keep categorical parameters as is
                    variation[key] = value

            return variation
        except Exception as e:
            logger.warning(f"Failed to create parameter variation: {e}")
            return None

    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        return self.meta_store.get_statistics()
