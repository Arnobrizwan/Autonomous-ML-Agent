"""
LLM-guided planning for hyperparameter optimization strategy.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..logging import get_logger
from ..types import (
    DatasetProfile,
    EnsembleConfig,
    LLMConfig,
    MetricType,
    ModelType,
    PlannerProposal,
    TaskType,
)
from ..utils import get_memory_usage

logger = get_logger()


@dataclass
class PlanningContext:
    """Context for LLM planning."""

    dataset_profile: DatasetProfile
    task_type: TaskType
    available_models: List[ModelType]
    time_budget_seconds: float
    memory_usage_mb: float
    historical_performance: Optional[Dict[str, float]] = None


class LLMPlanner:
    """LLM-guided planning for optimization strategy."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm_client = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM client if available."""
        if not self.config.enabled:
            return

        try:
            if self.config.provider == "openai":
                import openai

                self.llm_client = openai.OpenAI(api_key=self.config.api_key)
            elif self.config.provider == "gemini":
                import google.generativeai as genai

                genai.configure(api_key=self.config.api_key)
                self.llm_client = genai.GenerativeModel(self.config.model)
            else:
                logger.warning(f"Unknown LLM provider: {self.config.provider}")
        except ImportError as e:
            logger.warning(f"LLM library not available: {e}")
            self.llm_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")
            self.llm_client = None

    def create_plan(self, context: PlanningContext) -> PlannerProposal:
        """
        Create optimization plan based on context.

        Args:
            context: Planning context

        Returns:
            Planner proposal
        """
        if self.llm_client and self.config.enabled:
            return self._llm_guided_plan(context)
        else:
            return self._heuristic_plan(context)

    def _llm_guided_plan(self, context: PlanningContext) -> PlannerProposal:
        """Create plan using LLM guidance."""
        try:
            prompt = self._create_planning_prompt(context)
            response = self._call_llm(prompt)
            return self._parse_llm_response(response, context)
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, falling back to heuristics")
            return self._heuristic_plan(context)

    def _heuristic_plan(self, context: PlanningContext) -> PlannerProposal:
        """Create plan using heuristic rules."""
        logger.info("Using heuristic planning strategy")

        # Select models based on task type and dataset characteristics
        candidate_models = self._select_models_heuristic(context)

        # Allocate search budget
        search_budgets = self._allocate_budget_heuristic(
            candidate_models, context.time_budget_seconds
        )

        # Select metric
        metric = self._select_metric_heuristic(context)

        # Determine ensemble strategy
        ensemble_strategy = self._determine_ensemble_heuristic(context)

        return PlannerProposal(
            candidate_models=candidate_models,
            search_budgets=search_budgets,
            metric=metric,
            ensemble_strategy=ensemble_strategy,
            reasoning="Heuristic-based planning",
        )

    def _select_models_heuristic(self, context: PlanningContext) -> List[ModelType]:
        """Select models using heuristic rules."""
        models = []

        # Always include linear models for baseline
        if context.task_type == TaskType.CLASSIFICATION:
            models.append(ModelType.LOGISTIC_REGRESSION)
        else:
            models.append(ModelType.LINEAR_REGRESSION)

        # Add tree-based models for non-linear patterns
        models.append(ModelType.RANDOM_FOREST)
        models.append(ModelType.GRADIENT_BOOSTING)

        # Add k-NN for local patterns
        models.append(ModelType.KNN)

        # Add MLP for complex patterns (if dataset is large enough)
        if context.dataset_profile.n_rows > 1000:
            models.append(ModelType.MLP)

        return models

    def _allocate_budget_heuristic(
        self, models: List[ModelType], total_budget: float
    ) -> Dict[ModelType, int]:
        """Allocate search budget using heuristics."""
        # Base trials per model
        base_trials = {
            ModelType.LOGISTIC_REGRESSION: 10,
            ModelType.LINEAR_REGRESSION: 5,
            ModelType.RANDOM_FOREST: 15,
            ModelType.GRADIENT_BOOSTING: 20,
            ModelType.KNN: 10,
            ModelType.MLP: 15,
        }

        # Scale based on total budget
        total_base_trials = sum(base_trials.get(model, 10) for model in models)
        scale_factor = min(1.0, total_budget / 300)  # Scale based on 5-minute budget

        budgets = {}
        for model in models:
            trials = int(base_trials.get(model, 10) * scale_factor)
            budgets[model] = max(1, trials)

        return budgets

    def _select_metric_heuristic(self, context: PlanningContext) -> MetricType:
        """Select metric using heuristics."""
        if context.task_type == TaskType.CLASSIFICATION:
            # Check for class imbalance
            if (
                context.dataset_profile.class_balance
                and context.dataset_profile.class_balance < 0.3
            ):
                return MetricType.BALANCED_ACCURACY
            else:
                return MetricType.F1
        else:
            return MetricType.R2

    def _determine_ensemble_heuristic(
        self, context: PlanningContext
    ) -> Optional[EnsembleConfig]:
        """Determine ensemble strategy using heuristics."""
        # Enable ensembling if we have enough models and time
        if len(context.available_models) >= 3 and context.time_budget_seconds > 300:
            return EnsembleConfig(method="voting", top_k=3)
        return None

    def _create_planning_prompt(self, context: PlanningContext) -> str:
        """Create prompt for LLM planning."""
        prompt = f"""
You are an expert ML engineer planning a hyperparameter optimization strategy.

Dataset Profile:
- Rows: {context.dataset_profile.n_rows:,}
- Features: {context.dataset_profile.n_cols}
- Numeric features: {context.dataset_profile.n_numeric}
- Categorical features: {context.dataset_profile.n_categorical}
- Missing data: {context.dataset_profile.missing_ratio:.1%}
- Task type: {context.task_type.value}
- Class balance: {context.dataset_profile.class_balance or 'N/A'}

Constraints:
- Time budget: {context.time_budget_seconds}s
- Memory usage: {context.memory_usage_mb:.1f} MB
- Available models: {[m.value for m in context.available_models]}

Please provide a JSON response with:
1. candidate_models: List of model types to try
2. search_budgets: Number of trials per model
3. metric: Primary metric to optimize
4. ensemble_strategy: Whether to use ensembling and which method
5. reasoning: Brief explanation of your choices

Focus on efficiency and effectiveness given the constraints.
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if self.config.provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        elif self.config.provider == "gemini":
            response = self.llm_client.generate_content(prompt)
            return response.text
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")

    def _parse_llm_response(
        self, response: str, context: PlanningContext
    ) -> PlannerProposal:
        """Parse LLM response into planner proposal."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Parse candidate models
            candidate_models = []
            for model_str in data.get("candidate_models", []):
                try:
                    candidate_models.append(ModelType(model_str))
                except ValueError:
                    logger.warning(f"Unknown model type: {model_str}")

            # Parse search budgets
            search_budgets = {}
            for model_str, budget in data.get("search_budgets", {}).items():
                try:
                    search_budgets[ModelType(model_str)] = int(budget)
                except (ValueError, KeyError):
                    logger.warning(f"Invalid budget for model {model_str}")

            # Parse metric
            metric = MetricType(data.get("metric", "auto"))

            # Parse ensemble strategy
            ensemble_strategy = None
            ensemble_data = data.get("ensemble_strategy")
            if ensemble_data:
                ensemble_strategy = EnsembleConfig(
                    method=ensemble_data.get("method", "voting"),
                    top_k=ensemble_data.get("top_k", 3),
                )

            return PlannerProposal(
                candidate_models=candidate_models,
                search_budgets=search_budgets,
                metric=metric,
                ensemble_strategy=ensemble_strategy,
                reasoning=data.get("reasoning", "LLM-generated plan"),
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._heuristic_plan(context)

    def update_plan(
        self, context: PlanningContext, current_results: List[Dict[str, Any]]
    ) -> PlannerProposal:
        """Update plan based on current results."""
        # This could be implemented to dynamically adjust strategy
        # based on intermediate results
        return self.create_plan(context)


def create_planner(config: LLMConfig) -> LLMPlanner:
    """Create LLM planner with configuration."""
    return LLMPlanner(config)


def create_planning_context(
    dataset_profile: DatasetProfile,
    task_type: TaskType,
    available_models: List[ModelType],
    time_budget_seconds: float,
    historical_performance: Optional[Dict[str, float]] = None,
) -> PlanningContext:
    """Create planning context."""
    return PlanningContext(
        dataset_profile=dataset_profile,
        task_type=task_type,
        available_models=available_models,
        time_budget_seconds=time_budget_seconds,
        memory_usage_mb=get_memory_usage(),
        historical_performance=historical_performance,
    )
