"""
Main agent loop for autonomous ML pipeline.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

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
        self.preprocessor = None
        self.trainer = None
        self.planner = None
        self.budget_manager = None
        self.leaderboard = Leaderboard()

        # Results storage
        self.trial_results = []
        self.ensemble_model = None
        self.final_pipeline = None

    def run(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
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
            plan = self._create_optimization_plan(X_processed, y_processed, task_type)

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
                y = X[self.config.target]
                X = X.drop(columns=[self.config.target])
            else:
                # Auto-detect target (last column if categorical/regression sensible)
                y = X.iloc[:, -1]
                X = X.iloc[:, :-1]

        # Detect task type
        if self.config.task_type == TaskType.AUTO:
            task_type = detect_task_type(y)
        else:
            task_type = self.config.task_type

        # Profile dataset
        dataset_profile = profile_dataset(X, y.name if hasattr(y, "name") else "target")
        self.metadata.dataset_profile = dataset_profile

        # Create preprocessing pipeline
        self.preprocessor = PreprocessingPipeline(self.config.preprocessing)

        # Fit and transform data
        X_processed = self.preprocessor.fit_transform(X, y)

        logger.info(f"Preprocessed data: {X.shape} -> {X_processed.shape}")
        logger.info(f"Detected task type: {task_type}")

        return X_processed, y, task_type

    def _initialize_components(self, task_type: TaskType):
        """Initialize pipeline components."""
        # Initialize trainer
        logger.info(
            f"Config metric: {self.config.metric}, type: {type(self.config.metric)}"
        )
        metric = select_metric(task_type, self.config.metric)
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

    def _create_optimization_plan(
        self, X: pd.DataFrame, y: pd.Series, task_type: TaskType
    ) -> Any:
        """Create optimization plan using planner."""
        # Get available models
        from ..models.registries import ModelRegistry

        registry = ModelRegistry()
        available_models = registry.get_available_models(task_type)

        # Create planning context
        context = create_planning_context(
            dataset_profile=self.metadata.dataset_profile,
            task_type=task_type,
            available_models=available_models,
            time_budget_seconds=self.config.time_budget_seconds,
        )

        # Create plan
        plan = self.planner.create_plan(context)

        logger.info(
            f"Created optimization plan: {len(plan.candidate_models)} models, "
            f"metric: {plan.metric.value}"
        )

        return plan

    def _run_optimization(self, X: pd.DataFrame, y: pd.Series, plan: Any):
        """Run hyperparameter optimization."""
        for model_type in plan.candidate_models:
            if not self.budget_manager.check_budget():
                logger.info("Budget expired, stopping optimization")
                break

            # Get trials for this model
            n_trials = plan.search_budgets.get(model_type, 10)

            logger.info(f"Optimizing {model_type.value} with {n_trials} trials")

            # Run optimization
            model_results = self.trainer.optimize_hyperparameters(
                model_type=model_type,
                X=X,
                y=y,
                n_trials=n_trials,
                budget_clock=self.budget_manager.budget_clock,
            )

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
            ensemble_builder = EnsembleBuilder(
                task_type=self.trainer.task_type, random_seed=self.config.random_seed
            )

            self.ensemble_model = ensemble_builder.create_ensemble(
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
        y: pd.Series,
        X_processed: pd.DataFrame,
        y_processed: pd.Series,
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
        if self.ensemble_model:
            import joblib

            joblib.dump(self.ensemble_model, self.artifacts_dir / "model.joblib")
        elif self.trial_results:
            # Save best single model
            best_result = max(self.trial_results, key=lambda x: x.score)
            self.trainer.train_model(best_result.model_type, X, y, best_result.params)
            # Extract the actual model from the trial result
            from ..models.registries import get_model_factory

            best_model = get_model_factory(
                best_result.model_type, task_type, best_result.params
            )
            best_model.fit(X_processed, y)

            import joblib

            joblib.dump(best_model, self.artifacts_dir / "model.joblib")

        # Generate model card
        if self.config.export.generate_model_card:
            card_generator = ModelCardGenerator()
            model_card = card_generator.generate_card(
                trial_results=self.trial_results,
                ensemble_model=self.ensemble_model,
                task_type=task_type,
                dataset_profile=self.metadata.dataset_profile,
            )
            card_generator.save_card(model_card, self.artifacts_dir / "model_card.md")

        logger.info(f"Results exported to {self.artifacts_dir}")


def run_autonomous_ml(
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
    return agent.run(X, y)
