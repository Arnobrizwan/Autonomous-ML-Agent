"""
Main agent loop for autonomous ML pipeline.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..agent.budget import create_budget_manager
from ..agent.planner import LLMPlanner, create_planning_context
from ..config import Config
from ..export.model_card import ModelCardGenerator
from ..logging import get_logger
from ..models import EnsembleBuilder, ModelTrainer
from ..preprocess import PreprocessingPipeline
from ..types import (
    LeaderboardEntry,
    RunMetadata,
    TaskType,
)
from ..ui.leaderboard import Leaderboard
from ..utils import (
    create_artifacts_dir,
    detect_task_type,
    generate_run_id,
    profile_dataset,
    save_metadata,
    select_metric,
)

logger = get_logger()


class AgentLoop:
    """Main agent loop for autonomous ML pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.run_id = generate_run_id()
        self.artifacts_dir = create_artifacts_dir(self.run_id)
        self.metadata = RunMetadata(
            run_id=self.run_id, start_time=datetime.now(), config=config.to_dict()
        )

        # Initialize components
        self.preprocessor: Optional[Any] = None
        self.trainer: Optional[Any] = None
        self.planner: Optional[Any] = None
        self.budget_manager: Optional[Any] = None
        self.leaderboard = Leaderboard()

        # Results storage
        self.trial_results: List[Any] = []
        self.ensemble_model = None
        self.final_pipeline = None

    async def run(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run the complete autonomous ML pipeline.

        Args:
            X: Feature matrix
            y: Target vector (optional, will be auto-detected if not provided)

        Returns:
            Pipeline results
        """
        logger.info(f"Starting autonomous ML pipeline (run_id: {self.run_id})")

        try:
            # Step 1: Data profiling and preprocessing
            logger.info("Step 1: Data profiling and preprocessing")
            X_processed, y_processed, task_type = self._preprocess_data(X, y)

            # Step 2: Initialize components
            logger.info("Step 2: Initializing components")
            self._initialize_components(task_type)

            # Step 3: Create optimization plan
            logger.info("Step 3: Creating optimization plan")
            plan = await self._create_optimization_plan(
                X_processed, y_processed, task_type
            )

            # Step 4: Run hyperparameter optimization
            logger.info("Step 4: Running hyperparameter optimization")
            self._run_optimization(X_processed, y_processed, plan)

            # Step 5: Create ensemble (if enabled)
            if self.config.enable_ensembling and len(self.trial_results) >= 2:
                logger.info("Step 5: Creating ensemble model")
                self._create_ensemble(X_processed, y_processed, plan)

            # Step 6: Export results
            logger.info("Step 6: Exporting results")
            self._export_results(X, y, X_processed, y_processed, task_type)

            # Update metadata
            self.metadata.end_time = datetime.now()
            self.metadata.status = "completed"
            self.metadata.total_trials = len(self.trial_results)

            if self.trial_results:
                best_result = max(self.trial_results, key=lambda x: x.score)
                self.metadata.best_score = best_result.score
                self.metadata.best_model = best_result.model_type.value

            logger.info(f"Pipeline completed successfully (run_id: {self.run_id})")

            return {
                "run_id": self.run_id,
                "status": "completed",
                "best_score": self.metadata.best_score,
                "best_model": self.metadata.best_model,
                "total_trials": self.metadata.total_trials,
                "artifacts_dir": str(self.artifacts_dir),
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.metadata.status = "failed"
            self.metadata.end_time = datetime.now()
            raise

    def _preprocess_data(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, pd.Series, TaskType]:
        """Preprocess data and detect task type."""
        # Detect target if not provided
        if y is None:
            if self.config.target:
                # Extract target column
                y = X[self.config.target]
                X = X.drop(columns=[self.config.target])
            else:
                # Auto-detect target (last column)
                y = X.iloc[:, -1]
                X = X.iloc[:, :-1]

        # Detect task type
        if self.config.task_type == TaskType.AUTO:
            task_type = detect_task_type(y)
        else:
            task_type = self.config.task_type

        # Profile dataset
        target_name = str(y.name) if y is not None and hasattr(y, "name") else "target"
        dataset_profile = profile_dataset(X, target_name)
        self.metadata.dataset_profile = dataset_profile

        # Create preprocessing pipeline
        from ..preprocess.config import PreprocessingConfig
        from ..preprocess.pipeline import PreprocessingPipeline

        # Convert PreprocessingSettings to PreprocessingConfig
        if hasattr(self.config.preprocessing, "model_dump"):
            # It's a Pydantic model, convert to dict then to PreprocessingConfig
            preprocessing_dict = self.config.preprocessing.model_dump()
            preprocessing_config = PreprocessingConfig.from_dict(preprocessing_dict)
        elif isinstance(self.config.preprocessing, dict):
            preprocessing_config = PreprocessingConfig.from_dict(
                self.config.preprocessing
            )
        else:
            preprocessing_config = self.config.preprocessing
        self.preprocessor = PreprocessingPipeline(preprocessing_config)

        # Fit and transform data
        if self.preprocessor is not None:
            X_processed = self.preprocessor.fit_transform(X, y)
        else:
            X_processed = X

        logger.info(f"Preprocessed data: {X.shape} -> {X_processed.shape}")
        logger.info(f"Detected task type: {task_type}")

        return X_processed, y, task_type

    def _initialize_components(self, task_type: TaskType):
        """Initialize pipeline components."""
        # Initialize trainer
        logger.info(
            f"Config metric: {self.config.metric}, type: {type(self.config.metric)}"
        )
        from ..types import MetricType

        metric_name = str(self.config.metric)
        metric = (
            MetricType(metric_name)
            if hasattr(MetricType, metric_name)
            else MetricType.ACCURACY
        )
        logger.info(f"Selected metric: {metric}")
        self.trainer = ModelTrainer(
            task_type=task_type,
            metric=metric,
            cv_folds=self.config.cv_folds,
            random_seed=self.config.random_seed,
        )

        # Initialize planner
        logger.info("Initializing planner...")
        llm_config = self.config.get_llm_config()
        logger.info(f"LLM config: {llm_config}")
        self.planner = LLMPlanner(llm_config)
        logger.info("Planner initialized")

        # Initialize budget manager
        logger.info("Initializing budget manager...")
        self.budget_manager = create_budget_manager(self.config.time_budget_seconds)
        logger.info("Budget manager initialized")

    async def _create_optimization_plan(
        self, X: pd.DataFrame, y: pd.Series, task_type: TaskType
    ) -> Any:
        """Create optimization plan using planner."""
        # Get available models
        from ..models.registries import ModelRegistry

        registry = ModelRegistry()
        available_models = registry.get_available_models(task_type)

        # Create planning context
        if self.metadata.dataset_profile is None:
            raise ValueError("Dataset profile not available")
        context = create_planning_context(
            dataset_profile=self.metadata.dataset_profile,
            task_type=task_type,
            available_models=available_models,
            time_budget_seconds=self.config.time_budget_seconds,
        )

        # Create plan
        if self.planner is not None:
            plan = await self.planner.create_plan(context)
        else:
            raise ValueError("Planner not initialized")

        logger.info(
            f"Created optimization plan: {len(plan.candidate_models)} models, "
            f"metric: {plan.metric.value}"
        )

        return plan

    def _run_optimization(self, X: pd.DataFrame, y: pd.Series, plan: Any):
        """Run hyperparameter optimization."""
        for model_type in plan.candidate_models:
            if (
                self.budget_manager is not None
                and not self.budget_manager.check_budget()
            ):
                logger.info("Budget expired, stopping optimization")
                break

            # Get trials for this model
            n_trials = plan.search_budgets.get(model_type, 10)

            logger.info(f"Optimizing {model_type.value} with {n_trials} trials")

            # Run optimization
            if self.trainer is not None and self.budget_manager is not None:
                model_results = self.trainer.optimize_hyperparameters(
                    model_type=model_type,
                    X=X,
                    y=y,
                    n_trials=n_trials,
                    budget_clock=self.budget_manager.budget_clock,
                )
            else:
                logger.warning(
                    "Trainer or budget manager not initialized, skipping optimization"
                )
                continue

            # Add to results
            self.trial_results.extend(model_results)

            # Update leaderboard
            for result in model_results:
                if result.status == "completed":
                    entry = LeaderboardEntry(
                        rank=0,  # Will be updated by leaderboard
                        model_type=result.model_type,
                        score=result.score,
                        metric=result.metric,
                        params=result.params,
                        cv_mean=result.cv_scores[0] if result.cv_scores else 0,
                        cv_std=0,  # Will be calculated by leaderboard
                        fit_time=result.fit_time,
                        predict_time=result.predict_time,
                        trial_id=result.trial_id,
                    )
                    self.leaderboard.add_entry(entry)

            # Update budget
            self.budget_manager.budget_clock.update_elapsed()

    def _create_ensemble(self, X: pd.DataFrame, y: pd.Series, plan: Any):
        """Create ensemble model."""
        if not plan.ensemble_strategy:
            return

        try:
            if self.trainer is not None:
                ensemble_builder = EnsembleBuilder(
                    task_type=self.trainer.task_type,
                    random_seed=self.config.random_seed,
                )
            else:
                logger.warning("Trainer not initialized, skipping ensemble creation")
                return

            self.ensemble_model: Any = ensemble_builder.create_ensemble(
                trial_results=self.trial_results,
                top_k=plan.ensemble_strategy.top_k,
                method=plan.ensemble_strategy.method,
                X=X,
                y=y,
            )

            # Evaluate ensemble
            ensemble_performance = ensemble_builder.evaluate_ensemble(
                self.ensemble_model, X, y
            )

            logger.info(
                f"Created {plan.ensemble_strategy.method} ensemble: "
                f"score={ensemble_performance.get('cv_mean', 0):.4f}"
            )

        except Exception as e:
            logger.warning(f"Failed to create ensemble: {e}")

    def _export_results(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        X_processed: pd.DataFrame,
        y_processed: Optional[pd.Series],
        task_type: TaskType,
    ):
        """Export pipeline results."""
        # Save metadata
        save_metadata(self.metadata.__dict__, self.artifacts_dir)

        # Save leaderboard
        self.leaderboard.save_csv(self.artifacts_dir / "leaderboard.csv")

        # Save preprocessing pipeline
        if self.preprocessor:
            import joblib

            joblib.dump(self.preprocessor, self.artifacts_dir / "preprocessor.joblib")

        # Save original feature names (before preprocessing)
        # Get original feature names from the data
        original_feature_names = list(X.columns)

        import json

        with open(self.artifacts_dir / "feature_names.json", "w") as f:
            json.dump(original_feature_names, f, indent=2)

        # Save best model or ensemble
        best_model = None
        if self.ensemble_model:
            import joblib

            joblib.dump(self.ensemble_model, self.artifacts_dir / "model.joblib")
            best_model = self.ensemble_model
        elif self.trial_results:
            # Save best single model
            best_result = max(self.trial_results, key=lambda x: x.score)
            # Extract the actual model from the trial result
            from ..models.registries import get_model_factory, validate_model_params

            # Validate parameters before creating model
            validated_params = validate_model_params(
                best_result.model_type, best_result.params
            )
            best_model: Any = get_model_factory(
                best_result.model_type, task_type, validated_params
            )
            best_model.fit(X_processed, y_processed)

            import joblib

            joblib.dump(best_model, self.artifacts_dir / "model.joblib")

        # Generate model card
        if self.config.export.generate_model_card and best_model is not None:
            card_generator = ModelCardGenerator(task_type)
            model_card = card_generator.generate_model_card(
                model=best_model,
                X=X_processed,
                y=(
                    y_processed
                    if y_processed is not None
                    else pd.Series([], dtype=float)
                ),
            )
            with open(self.artifacts_dir / "model_card.md", "w") as f:
                f.write(model_card)

        logger.info(f"Results exported to {self.artifacts_dir}")


async def run_autonomous_ml(
    config: Config, X: pd.DataFrame, y: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Run autonomous ML pipeline with given configuration.

    Args:
        config: Configuration object
        X: Feature matrix
        y: Target vector (optional)

    Returns:
        Pipeline results
    """
    agent = AgentLoop(config)
    return await agent.run(X, y)
