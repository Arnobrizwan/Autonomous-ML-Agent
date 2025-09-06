# 🤖 Autonomous ML Agent

**Transform your data into production-ready machine learning models automatically!**

The Autonomous ML Agent is an intelligent system that takes your tabular data and automatically builds, trains, and deploys the best possible machine learning models without any manual intervention.

## 🎯 What This System Does

**Input**: Your CSV/Excel data file  
**Output**: Production-ready ML models with API endpoints

### ✨ Key Capabilities

- **🧠 Smart Data Processing**: Automatically handles missing values, categorical data, and feature engineering
- **🚀 Multi-Model Training**: Tests 8+ different ML algorithms to find the best one for your data
- **🎯 Intelligent Optimization**: Uses AI to find optimal model settings automatically
- **🏆 Ensemble Learning**: Combines multiple models for maximum accuracy
- **📊 Model Analysis**: Provides detailed insights into why models make decisions
- **🌐 Web Interface**: Easy-to-use dashboard for running experiments
- **🔧 Production Ready**: Deploy models as APIs with Docker containers
- **💻 No GPU Required**: Runs on any computer with standard hardware

## ⚡ Super Quick Start (2 Minutes)

**Just want to see it work? Run this:**
```bash
git clone https://github.com/Arnobrizwan/Autonomous-ML-Agent.git
cd autonomous-ml-agent
./start_demo.sh
# Open http://localhost:8000 in your browser
```

**That's it!** Your ML Agent is running with a web interface.

## 🚀 Quick Start (5 Minutes)

### Step 1: Install
```bash
git clone https://github.com/Arnobrizwan/Autonomous-ML-Agent.git
cd autonomous-ml-agent
pip install -r requirements.txt
pip install -e .
```

### Step 2: Quick Demo
```bash
# Start demo (one command)
./start_demo.sh

# Or run full interactive demo
./demo.sh
```

### Step 3: Use Your Data
```bash
# Run with your data
aml run --data your_data.csv --target target_column --time-budget 300

# View results in web interface
aml web
```

## 🎬 Demo & Testing

### Quick Health Check
```bash
# Verify everything is working
./quick_check.sh
```

### Live Demo
```bash
# Start demo environment
./start_demo.sh

# Access demo URLs:
# • Web Interface: http://localhost:8000
# • API Documentation: http://localhost:8000/docs
# • Health Check: http://localhost:8000/healthz
```

### Full Interactive Demo
```bash
# Run complete demonstration with sample data
./demo.sh
```

## ✨ What Makes This Special

- **🤖 Fully Automated**: No ML expertise required - just provide your data
- **🚀 Production Ready**: Deploy models as APIs immediately
- **✅ Fully Tested**: 66 passing tests with comprehensive coverage
- **🐳 Docker Ready**: Multi-platform containerized deployment
- **🔄 CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **📊 Real-time Monitoring**: Health checks, metrics, and performance tracking
- **🎯 AI-Powered**: Uses advanced AI to find the best model settings
- **📊 Complete Solution**: From data to deployed API in minutes
- **🔧 Enterprise Grade**: Built with security, monitoring, and scalability

## 🎯 Project Status

**✅ Production Ready** - Fully functional and tested  
**✅ CI/CD Pipeline** - Automated testing and deployment  
**✅ Docker Support** - Multi-platform containerization  
**✅ API Ready** - FastAPI with automatic documentation  
**✅ Web Interface** - User-friendly dashboard  
**✅ Monitoring** - Health checks and performance metrics  
**✅ Documentation** - Comprehensive guides and examples  

### Recent Updates
- **🎬 Demo Scripts**: Complete demonstration tools (`demo.sh`, `start_demo.sh`, `quick_check.sh`)
- **🐳 Docker Optimization**: Multi-platform builds (linux/amd64, linux/arm64)
- **🔄 GitHub Actions**: Automated CI/CD pipeline with GitHub Container Registry
- **📊 Monitoring**: Advanced health checks and performance tracking
- **🧪 Testing**: 66 comprehensive tests with 100% critical path coverage
- **📚 Documentation**: Complete demo guide and API documentation

## 🎬 Demo Files

The project includes several demo scripts to help you get started quickly:

| File | Purpose | Usage |
|------|---------|-------|
| `start_demo.sh` | One-command demo starter | `./start_demo.sh` |
| `demo.sh` | Full interactive demonstration | `./demo.sh` |
| `quick_check.sh` | Pre-demo health verification | `./quick_check.sh` |
| `DEMO_GUIDE.md` | Complete demo presentation guide | Read for demo tips |

### Demo Features
- **🌐 Web Interface**: Interactive dashboard at http://localhost:8000
- **📚 API Documentation**: Auto-generated docs at http://localhost:8000/docs
- **🔍 Health Monitoring**: Real-time health checks at http://localhost:8000/healthz
- **📊 Sample Data**: Automatic generation of test datasets
- **🤖 Live Pipeline**: Real ML model training and evaluation
- **📈 Results Display**: Model leaderboard and performance metrics

## 🎁 What You Get

### For Business Users
- **Zero ML Knowledge Required**: Just provide your data, get results
- **Automatic Model Selection**: System finds the best algorithm for your data
- **Production-Ready APIs**: Deploy models immediately with Docker
- **Web Dashboard**: Easy-to-use interface for running experiments
- **Detailed Reports**: Understand why models make decisions

### For Technical Users
- **8+ ML Algorithms**: Logistic/Linear Regression, RandomForest, XGBoost, LightGBM, CatBoost, kNN, MLP
- **AI-Guided Optimization**: LLM-powered hyperparameter search
- **Ensemble Learning**: Automatic model combination for better accuracy
- **FastAPI Service**: RESTful API with authentication and monitoring
- **Docker Deployment**: Containerized deployment with CI/CD pipeline
- **Comprehensive Testing**: 41 passing tests with 43% code coverage

## 🚀 Quick Start

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

## 💼 Perfect For

### Business Use Cases
- **Sales Forecasting**: Predict customer behavior and sales trends
- **Customer Segmentation**: Automatically group customers by behavior
- **Risk Assessment**: Identify high-risk customers or transactions
- **Quality Control**: Detect defects in manufacturing processes
- **Price Optimization**: Set optimal prices based on market data
- **Churn Prediction**: Identify customers likely to leave

### Technical Use Cases
- **Rapid Prototyping**: Quickly test ML ideas with real data
- **Model Benchmarking**: Compare multiple algorithms automatically
- **Feature Engineering**: Discover important data patterns
- **API Development**: Build ML-powered microservices
- **A/B Testing**: Deploy and test different models easily

## 🏗️ How It Works

```
Your Data → Smart Processing → AI Model Selection → Best Model → Production API
    ↓              ↓                    ↓              ↓           ↓
  CSV/Excel    Auto-clean        Test 8+ models    Ensemble    Docker + FastAPI
```

**Step 1**: Upload your data (CSV, Excel, etc.)  
**Step 2**: System automatically cleans and prepares data  
**Step 3**: AI tests multiple ML algorithms to find the best one  
**Step 4**: Combines top models for maximum accuracy  
**Step 5**: Deploys as production-ready API with Docker

## 🎮 Easy Commands

### For Everyone
```bash
# Run ML pipeline on your data
aml run --data your_data.csv --target target_column

# View results in web browser
aml web

# Start prediction API
aml serve
```

### For Developers
```bash
# Install everything
make setup

# Test the system
make verify

# Run with custom settings
aml run --data data.csv --time-budget 600 --max-trials 100
```

## 📋 Supported Data Formats

- **CSV files** (.csv)
- **Excel files** (.xlsx, .xls)
- **JSON files** (.json)
- **Parquet files** (.parquet)
- **Feather files** (.feather)

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

## 🏢 Enterprise Features

- **🔐 Security**: API authentication and rate limiting
- **📊 Monitoring**: Real-time system health and performance tracking
- **🐳 Docker**: One-click deployment with containerization
- **🔄 CI/CD**: Automated testing and deployment pipeline
- **📈 Scalability**: Handles large datasets and high-traffic APIs
- **🛡️ Reliability**: 66 passing tests with comprehensive error handling
- **🎬 Demo Ready**: Complete demonstration tools and guides
- **📚 Documentation**: Comprehensive API docs and user guides

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

### Quick Start with Docker
```bash
# One-command demo
./start_demo.sh

# Or manual Docker deployment
docker run -d --name aml-agent -p 8000:8000 aml-agent:latest

# Access the service
# Web Interface: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Production Deployment
```bash
# Build and run (optimized)
make docker-build
make docker-run

# Or with docker-compose
docker-compose up

# Or with custom config
docker run -p 8000:8000 -v $(pwd)/data:/app/data aml-agent:latest
```

### GitHub Container Registry
```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/arnobrizwan/autonomous-ml-agent:latest

# Run the image
docker run -p 8000:8000 ghcr.io/arnobrizwan/autonomous-ml-agent:latest
```

**Note**: The image will be available at `ghcr.io/arnobrizwan/autonomous-ml-agent:latest` after the GitHub Actions workflow completes.

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

