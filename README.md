# Autonomous ML Agent

An intelligent machine learning agent that automatically processes tabular data, trains multiple models with LLM-guided hyperparameter optimization, and provides comprehensive model analysis and deployment capabilities.

## Features

- **Autonomous Data Processing**: Automatic type inference, missing value handling, categorical encoding, and feature engineering
- **LLM-Guided Optimization**: Intelligent hyperparameter search with budget-aware optimization
- **Multi-Model Training**: Support for Logistic/Linear Regression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, kNN, and MLP
- **Advanced Preprocessing**: Text processing, embeddings, polynomial features, outlier detection, and feature selection (temporarily disabled for stability)
- **Ensemble Learning**: Automatic stacking and blending of top-performing models
- **Meta-Learning**: Warm-start optimization using historical run data
- **Comprehensive Analysis**: Feature importance, SHAP explanations, and natural language model cards
- **Production Ready**: FastAPI service, Docker deployment, and CI/CD pipeline
- **Zero GPU Dependency**: Runs entirely on CPU with local dependencies
- **Clean Codebase**: Optimized Docker builds, comprehensive testing, and code quality checks

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

### Quick Verification

```bash
# Run smoke tests to verify everything works
pytest tests/test_smoke.py -v

# Expected output: 12 tests should pass
# ========================= 12 passed in 5.76s =========================
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
# Build and run (optimized)
make docker-build
make docker-run

# Or with docker-compose
docker-compose up

# Or with custom config
docker run -p 8000:8000 -v $(pwd)/data:/app/data aml-agent
```

### Docker Performance Optimizations

The Docker build has been optimized for **maximum speed and reliability**:

- **Single-stage build** for simplicity and speed
- **Layer caching optimization** with requirements.txt copied first
- **Network resilience** with retry and timeout handling
- **Minimal dependencies** to reduce image size
- **BuildKit enabled** for parallel builds
- **GitHub Actions caching** for CI/CD speed

### Build Performance

| Build Type | Expected Time | Cache Strategy |
|------------|---------------|----------------|
| First Build | 8-12 minutes | No cache |
| Cached Build | 2-4 minutes | Layer caching |
| CI/CD Build | 3-6 minutes | GitHub Actions cache |

### Docker Commands

| Command | Description | Performance |
|---------|-------------|-------------|
| `make docker-build` | Build with cache optimization | ⚡ Fast |
| `make docker-build-no-cache` | Clean build without cache | 🐌 Slow |
| `make docker-build-fast` | Quick build using existing cache | ⚡⚡ Very Fast |
| `make docker-run` | Run container | ⚡ Fast |
| `make docker-test` | Test Docker image | ⚡ Fast |
| `make docker-performance` | Analyze build performance | 📊 Metrics |

## 🌐 Web Interface

### Streamlit Web UI

Launch the interactive web interface:

```bash
# Launch web UI
aml web

# Or directly with streamlit
streamlit run src/aml_agent/ui/web.py
```

**Features:**
- 📊 **Dashboard**: Overview of runs and performance
- 🚀 **Pipeline Runner**: Interactive pipeline execution
- 🏆 **Model Leaderboard**: Visual performance comparison
- 📋 **Model Cards**: Detailed model documentation
- 📊 **Monitoring**: Real-time system health and metrics
- ⚙️ **Settings**: Configuration management

## 🔒 Security Features

### API Authentication

```bash
# Generate API key
aml security

# Use API key in requests
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/predict_one \
     -d '{"features": {...}}'
```

**Security Features:**
- 🔑 **API Key Management**: Secure authentication
- 🚦 **Rate Limiting**: Request throttling
- 🛡️ **Input Validation**: Data sanitization
- 📊 **Access Logging**: Request monitoring

## 📊 Monitoring & Metrics

### System Monitoring

```bash
# Check system health
aml monitor

# View performance metrics
aml monitor --detailed
```

**Monitoring Features:**
- 💓 **Health Checks**: System status monitoring
- 📈 **Performance Metrics**: Real-time performance tracking
- 🚨 **Alerting**: Automated issue detection
- 📊 **Dashboards**: Visual monitoring interface

## 📁 Data Format Support

### Supported Formats

```python
# Load data from various formats
from aml_agent.utils import load_data, save_data

# CSV, JSON, Parquet, Excel, Feather, Pickle
data = load_data("data.parquet")
save_data(data, "output.xlsx")
```

**Supported Formats:**
- 📄 **CSV**: Comma-separated values
- 📋 **JSON**: JavaScript Object Notation
- 🗜️ **Parquet**: Columnar storage
- 📊 **Excel**: .xlsx and .xls files
- 🪶 **Feather**: Fast columnar format
- 🥒 **Pickle**: Python serialization

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

## What's Included

### Core Features
- **Automatic Data Processing**: Handles missing values, categorical encoding, and feature scaling
- **Multiple ML Models**: Logistic/Linear Regression, RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost, kNN, MLP
- **Smart Optimization**: Intelligent hyperparameter search with time budget management
- **Ensemble Learning**: Automatically combines top-performing models for better accuracy
- **Model Analysis**: Feature importance analysis and model explanations
- **Production Ready**: FastAPI service with authentication and Docker deployment
- **Web Interface**: User-friendly dashboard for running and monitoring experiments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `make lint` and `make test`
6. Submit a pull request

### Development Guidelines
- Follow black formatting standards
- Use isort for import sorting
- Write comprehensive tests
- Update documentation for new features
- Ensure all smoke tests pass before submitting PRs

## License

MIT License - see LICENSE file for details.


## Version History

### v0.1.1 (Current)
- Enhanced data preprocessing with better categorical handling
- Improved model training stability and performance
- Optimized Docker builds for faster deployment
- Enhanced code quality and testing coverage

### v0.1.0
- Initial release with core ML pipeline
- FastAPI service and Docker deployment
- Model analysis and interpretability features

