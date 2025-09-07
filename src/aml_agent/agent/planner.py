"""
LLM-guided planning for autonomous ML optimization.
"""

import json
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from ..logging import get_logger
from ..types import (
    DatasetProfile,
    LLMConfig,
    MetricType,
    ModelType,
    PlannerProposal,
    TaskType,
)

logger = get_logger()


class PlanningContext(BaseModel):
    """Context for planning optimization strategy."""

    dataset_profile: DatasetProfile
    task_type: TaskType
    available_models: List[ModelType]
    time_budget_seconds: int
    current_metric: Optional[MetricType] = None
    previous_results: Optional[Dict[str, Any]] = None


class LLMPlanner:
    """LLM-guided planner for optimization strategy."""

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.client = None
        if llm_config.enabled and llm_config.api_key:
            self._initialize_client()

    def _initialize_client(self):
        """Initialize HTTP client for LLM API."""
        if self.llm_config.provider == "openai":
            self.client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {self.llm_config.api_key}"},
            )
        elif self.llm_config.provider == "gemini":
            self.client = httpx.AsyncClient(
                base_url="https://generativelanguage.googleapis.com/v1beta",
                headers={"Authorization": f"Bearer {self.llm_config.api_key}"},
            )

    async def create_plan(self, context: PlanningContext) -> PlannerProposal:
        """
        Create optimization plan based on context.

        Args:
            context: Planning context

        Returns:
            Optimization plan proposal
        """
        if self.llm_config.enabled and self.client:
            return await self._create_llm_plan(context)
        else:
            return self._create_heuristic_plan(context)

    async def _create_llm_plan(self, context: PlanningContext) -> PlannerProposal:
        """Create plan using LLM."""
        try:
            prompt = self._build_planning_prompt(context)
            response = await self._call_llm(prompt)
            return self._parse_llm_response(response, context)
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, falling back to heuristics")
            return self._create_heuristic_plan(context)

    def _create_heuristic_plan(self, context: PlanningContext) -> PlannerProposal:
        """Create plan using heuristic rules."""
        logger.info("Creating heuristic optimization plan")

        # Select models based on dataset characteristics
        candidate_models = self._select_models_heuristic(context)

        # Allocate search budget
        search_budgets = self._allocate_budget_heuristic(candidate_models, context)

        # Select metric
        metric = self._select_metric_heuristic(context)

        # Determine ensemble strategy
        ensemble_strategy = self._determine_ensemble_strategy_heuristic(context)

        # Convert ensemble_strategy dict to EnsembleConfig if provided
        ensemble_config = None
        if ensemble_strategy:
            from ..types import EnsembleConfig

            ensemble_config = EnsembleConfig(
                method=ensemble_strategy.get("method", "voting"),
                top_k=ensemble_strategy.get("top_k", 3),
                meta_learner=ensemble_strategy.get("meta_learner"),
                weights=ensemble_strategy.get("weights"),
            )

        return PlannerProposal(
            candidate_models=candidate_models,
            search_budgets=search_budgets,
            metric=metric,
            ensemble_strategy=ensemble_config,
            reasoning="Heuristic-based plan using dataset characteristics",
        )

    def _select_models_heuristic(self, context: PlanningContext) -> List[ModelType]:
        """Select models using heuristic rules - following original prompt exactly."""
        # As per original prompt: ALL these models must be tested
        # (Logistic/Linear Regression, RandomForest, GradientBoosting, kNN,
        # MLPClassifier/Regressor, XGBoost, LightGBM, CatBoost)

        all_models = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.LINEAR_REGRESSION,
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING,
            ModelType.KNN,
            ModelType.MLP,
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
            ModelType.CATBOOST,
        ]

        # Filter to available models only
        available_models = [m for m in all_models if m in context.available_models]

        return available_models

    def _allocate_budget_heuristic(
        self, models: List[ModelType], context: PlanningContext
    ) -> Dict[ModelType, int]:
        """Allocate search budget across models."""
        # Ensure minimum trials per model as per original prompt
        n_models = len(models)
        min_trials_per_model = max(1, 50 // n_models)  # At least 1 trial per model
        total_trials = min(
            100, context.time_budget_seconds // 5
        )  # More generous estimate

        # Allocate more trials to promising models
        budgets = {}
        base_trials = max(min_trials_per_model, total_trials // n_models)

        for i, model in enumerate(models):
            # Give more trials to ensemble methods
            if model in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST]:
                multiplier = 1.5
            elif model in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                multiplier = 1.2
            else:
                multiplier = 1.0

            budgets[model] = max(1, int(base_trials * multiplier))

        return budgets

    def _select_metric_heuristic(self, context: PlanningContext) -> MetricType:
        """Select metric using heuristic rules."""
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

    def _determine_ensemble_strategy_heuristic(
        self, context: PlanningContext
    ) -> Optional[Dict[str, Any]]:
        """Determine ensemble strategy using heuristics."""
        # Enable ensembling for larger datasets
        if context.dataset_profile.n_rows > 500:
            return {
                "method": "voting",
                "top_k": 3,
                "meta_learner": None,
            }
        return None

    def _build_planning_prompt(self, context: PlanningContext) -> str:
        """Build prompt for LLM planning."""
        prompt = f"""
You are an expert ML engineer tasked with creating an optimization plan for an autonomous ML system.

Dataset Profile:
- Rows: {context.dataset_profile.n_rows}
- Features: {context.dataset_profile.n_cols}
- Numeric features: {context.dataset_profile.n_numeric}
- Categorical features: {context.dataset_profile.n_categorical}
- Missing ratio: {context.dataset_profile.missing_ratio:.2%}
- Class balance: {context.dataset_profile.class_balance}
- Task type: {context.task_type.value}

Available models: {[m.value for m in context.available_models]}
Time budget: {context.time_budget_seconds} seconds

Please provide a JSON response with:
1. candidate_models: List of 3-5 model types to try
2. search_budgets: Dictionary mapping model types to number of trials
3. metric: Primary evaluation metric
4. ensemble_strategy: Optional ensemble configuration
5. reasoning: Brief explanation of choices

Example response:
{{
    "candidate_models": ["random_forest", "xgboost", "logistic_regression"],
    "search_budgets": {{"random_forest": 15, "xgboost": 20, "logistic_regression": 10}},
    "metric": "f1",
    "ensemble_strategy": {{"method": "voting", "top_k": 3}},
    "reasoning": "Selected ensemble methods for complex dataset with mixed features"
}}
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        if self.llm_config.provider == "openai":
            return await self._call_openai(prompt)
        elif self.llm_config.provider == "gemini":
            return await self._call_gemini(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if self.client is None:
            raise ValueError("Client not initialized")
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.llm_config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.llm_config.temperature,
                "max_tokens": self.llm_config.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        if self.client is None:
            raise ValueError("Client not initialized")
        response = await self.client.post(
            f"/models/{self.llm_config.model}:generateContent",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.llm_config.temperature,
                    "maxOutputTokens": self.llm_config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _parse_llm_response(
        self, response: str, context: PlanningContext
    ) -> PlannerProposal:
        """Parse LLM response into proposal."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]

            data = json.loads(json_str)

            # Convert model names to ModelType enums
            candidate_models = []
            for model_name in data.get("candidate_models", []):
                try:
                    model_type = ModelType(model_name)
                    if model_type in context.available_models:
                        candidate_models.append(model_type)
                except ValueError:
                    logger.warning(f"Unknown model type: {model_name}")

            # Convert search budgets
            search_budgets = {}
            for model_name, budget in data.get("search_budgets", {}).items():
                try:
                    model_type = ModelType(model_name)
                    search_budgets[model_type] = int(budget)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid budget for {model_name}: {budget}")

            # Convert metric
            metric = MetricType.F1  # Default
            metric_name = data.get("metric", "f1")
            try:
                metric = MetricType(metric_name)
            except ValueError:
                logger.warning(f"Unknown metric: {metric_name}")

            # Parse ensemble strategy
            ensemble_strategy = data.get("ensemble_strategy")

            return PlannerProposal(
                candidate_models=candidate_models,
                search_budgets=search_budgets,
                metric=metric,
                ensemble_strategy=ensemble_strategy,
                reasoning=data.get("reasoning", "LLM-generated plan"),
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response: {response}")
            raise

    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()


def create_planning_context(
    dataset_profile: DatasetProfile,
    task_type: TaskType,
    available_models: List[ModelType],
    time_budget_seconds: int,
) -> PlanningContext:
    """Create planning context."""
    return PlanningContext(
        dataset_profile=dataset_profile,
        task_type=task_type,
        available_models=available_models,
        time_budget_seconds=time_budget_seconds,
    )


def create_planner(llm_config: LLMConfig) -> LLMPlanner:
    """Create planner instance."""
    return LLMPlanner(llm_config)
