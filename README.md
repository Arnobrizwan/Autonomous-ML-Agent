# Autonomous ML Agent

An intelligent machine learning agent that automatically processes tabular data, trains multiple models with LLM-guided hyperparameter optimization, and provides comprehensive model analysis and deployment capabilities.

## Features

- **Autonomous Data Processing**: Automatic type inference, missing value handling, categorical encoding, and feature engineering
- **LLM-Guided Optimization**: Intelligent hyperparameter search with budget-aware optimization
- **Multi-Model Training**: Support for Logistic/Linear Regression, RandomForest, GradientBoosting, kNN, and MLP
- **Ensemble Learning**: Automatic stacking and blending of top-performing models
- **Meta-Learning**: Warm-start optimization using historical run data
- **Comprehensive Analysis**: Feature importance, SHAP explanations, and natural language model cards
- **Production Ready**: FastAPI service, Docker deployment, and CI/CD pipeline
- **Zero GPU Dependency**: Runs entirely on CPU with local dependencies

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd autonomous-ml-agent
make setup

# Or manual installation
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```bash
# Run with default config
make run

# Run with custom data
aml run --data data/your_data.csv --target target_column --time-budget 600

# View results
make leaderboard

# Get explanations
make explain

# Start API service
make serve

# Make predictions
curl -X POST localhost:8000/predict_one \
  -H "Content-Type: application/json" \
  -d '{"feature_a": 1, "feature_b": 2.5}'
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Preprocessing   │───▶│  Model Training │
│  (CSV/Parquet)  │    │   & Profiling    │    │   & HPO Loop    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Model Cards   │◀───│   Interpretability│◀───│   Ensembling    │
│  & Export       │    │   & Analysis     │    │   & Selection   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  FastAPI Service │
                       │  & Deployment    │
                       └──────────────────┘
```

## Commands

### Core Commands
- `aml run` - Start autonomous ML pipeline
- `aml leaderboard` - View model performance leaderboard
- `aml explain` - Generate model explanations and feature importance
- `aml export` - Export trained models and artifacts
- `aml serve` - Start FastAPI prediction service
- `aml predict` - Make single or batch predictions

### Development Commands
- `make setup` - Install dependencies
- `make run` - Run with default config
- `make test` - Run test suite
- `make lint` - Run code formatting and linting
- `make clean` - Clean artifacts and cache

## Configuration

Edit `configs/default.yaml` to customize behavior:

```yaml
data_path: "data/sample.csv"
target: null                    # Auto-detect if null
task_type: "auto"              # "auto" | "classification" | "regression"
time_budget_seconds: 900       # Maximum runtime
max_trials: 60                 # Maximum HPO trials
cv_folds: 5                    # Cross-validation folds
metric: "auto"                 # Auto-select based on task
search_strategy: "bayes"       # "random" | "bayes"
enable_ensembling: true        # Enable model ensembling
top_k_for_ensemble: 3          # Number of models to ensemble
random_seed: 42                # Reproducibility seed
use_mlflow: false              # Enable MLflow tracking
```

## Project Structure

```
autonomous-ml-agent/
├── src/aml_agent/           # Core package
│   ├── config.py           # Configuration management
│   ├── types.py            # Type definitions
│   ├── preprocess/         # Data preprocessing
│   ├── models/             # Model registry and training
│   ├── agent/              # LLM-guided planning
│   ├── meta/               # Meta-learning and warm-start
│   ├── interpret/          # Model interpretability
│   ├── export/             # Model export and cards
│   ├── ui/                 # CLI and leaderboard
│   └── service/            # FastAPI service
├── configs/                # Configuration files
├── tests/                  # Test suite
├── docker/                 # Docker deployment
└── artifacts/              # Generated artifacts (gitignored)
```

## Supported Models

- **Linear Models**: Logistic Regression, Linear Regression
- **Tree-based**: Random Forest, Gradient Boosting
- **Distance-based**: k-Nearest Neighbors
- **Neural Networks**: Multi-layer Perceptron

## Meta-Learning

The agent learns from previous runs to improve future performance:
- Stores dataset fingerprints and best hyperparameters
- Identifies similar datasets for warm-start optimization
- Adapts search strategies based on historical success

## Deployment

### Docker
```bash
# Build and run
docker build -f docker/Dockerfile -t aml-agent .
docker-compose up

# Or with custom config
docker run -p 8000:8000 -v $(pwd)/data:/app/data aml-agent
```

### Production
- FastAPI service with automatic schema validation
- Batch and single prediction endpoints
- Health checks and monitoring
- Configurable resource limits

## Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_smoke.py -v

# Run with coverage
pytest --cov=src/aml_agent tests/
```

## Local Verification

Run the full local harness (dataset → deps → tests → quick train → artifacts → serve & probe):

```bash
make verify
```

This comprehensive verification script will:
1. Create a synthetic dataset (200 rows with realistic features)
2. Set up a virtual environment and install dependencies
3. Run the test suite
4. Execute a quick training run with 60-second budget
5. Verify all critical artifacts are generated
6. Start the FastAPI service and test prediction endpoints
7. Validate the complete end-to-end pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `make lint` and `make test`
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v0.1.0
- Initial release
- Core ML pipeline with LLM-guided optimization
- FastAPI service and Docker deployment
- Comprehensive model analysis and interpretability
- Meta-learning and warm-start capabilities

