# 🤖 Autonomous ML Agent

**Transform your data into production-ready machine learning models automatically!**

Just upload your CSV file and get a complete ML solution with the best model, explanations, and ready-to-use API - no coding required!

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=FastAPI)](https://fastapi.tiangolo.com/)

## ✨ What You Get

- **📊 Upload & Go**: Drop your CSV file, specify the target column, and you're done
- **🤖 Smart Training**: Automatically tests 8+ ML algorithms to find the best one
- **🎯 Auto-Optimization**: AI finds the perfect settings for your data
- **🏆 Best Model**: Gets the highest accuracy model automatically
- **📈 Insights**: See why the model makes decisions with feature importance
- **🌐 Ready API**: Get a working API endpoint for predictions
- **📱 Web Dashboard**: Beautiful interface to explore everything
- **🔒 Production Ready**: Deploy anywhere with Docker

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AUTONOMOUS ML AGENT ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA INPUT    │    │  PREPROCESSING  │    │  TASK DETECTION │    │  LLM PLANNING   │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ CSV File  │──┼────┼──│ Data      │──┼────┼──│ Auto      │──┼────┼──│ GPT-4     │  │
│  │ Upload    │  │    │  │ Profiling │  │    │  │ Detect    │  │    │  │ Planning  │  │
│  └───────────┘  │    │  └───────────┘  │    │  │ Task Type │  │    │  └───────────┘  │
│                 │    │                 │    │  └───────────┘  │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │                 │    │  ┌───────────┐  │
│  │ Target    │──┼────┼──│ Clean     │  │    │                 │    │  │ Model     │  │
│  │ Column    │  │    │  │ Handle    │  │    │                 │    │  │ Selection │  │
│  └───────────┘  │    │  │ Missing   │  │    │                 │    │  └───────────┘  │
│                 │    │  │ Encode    │  │    │                 │    │                 │
│  ┌───────────┐  │    │  │ Scale     │  │    │                 │    │  ┌───────────┐  │
│  │ Config    │──┼────┼──│ Features  │  │    │                 │    │  │ Hyperopt  │  │
│  │ YAML      │  │    │  └───────────┘  │    │                 │    │  │ Strategy  │  │
│  └───────────┘  │    └─────────────────┘    └─────────────────┘    │  └───────────┘  │
└─────────────────┘                                                └─────────────────┘
                                                                              │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MODEL TRAINING │    │  HYPEROPTIMIZE  │    │  MODEL SELECTION│    │  ENSEMBLE BUILD │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ 8+ Models │──┼────┼──│ Optuna    │──┼────┼──│ Best      │──┼────┼──│ Top K     │  │
│  │ Training  │  │    │  │ Bayesian  │  │    │  │ Model     │  │    │  │ Models    │  │
│  │           │  │    │  │ Search    │  │    │  │ Selection │  │    │  │           │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Cross     │  │    │  │ Time      │  │    │  │ Cross     │  │    │  │ Voting    │  │
│  │ Validation│  │    │  │ Budget    │  │    │  │ Validation│  │    │  │ Stacking  │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                                              │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  MODEL EXPORT   │    │  API SERVICE    │    │  WEB DASHBOARD  │    │  CLI INTERFACE  │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Save      │──┼────┼──│ FastAPI   │──┼────┼──│ Streamlit │──┼────┼──│ Command   │  │
│  │ Model     │  │    │  │ REST API  │  │    │  │ Dashboard │  │    │  │ Line     │  │
│  │ Artifacts │  │    │  │           │  │    │  │           │  │    │  │ Interface│  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Model     │  │    │  │ Health    │  │    │  │ Data      │  │    │  │ Local     │  │
│  │ Card      │  │    │  │ Monitoring│  │    │  │ Viz       │  │    │  │ Testing   │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CORE COMPONENTS                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA LAYER    │    │   ML LAYER      │    │   API LAYER     │    │   UI LAYER      │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Pandas    │  │    │  │ Scikit    │  │    │  │ FastAPI   │  │    │  │ Streamlit │  │
│  │ DataFrames│  │    │  │ Learn     │  │    │  │ Uvicorn   │  │    │  │ Plotly    │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ NumPy     │  │    │  │ XGBoost   │  │    │  │ Pydantic  │  │    │  │ Pandas    │  │
│  │ Arrays    │  │    │  │ LightGBM  │  │    │  │ Schemas   │  │    │  │ Styling   │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Data      │  │    │  │ Optuna    │  │    │  │ CORS      │  │    │  │ Responsive│  │
│  │ Validation│  │    │  │ Hyperopt  │  │    │  │ Security  │  │    │  │ Design    │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

    CSV Input ──┐
                │
                ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    DATA PREPROCESSING PIPELINE                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
    │  │   Profiling │──│   Cleaning  │──│  Encoding   │──│   Scaling   │    │
    │  │   Analysis  │  │   Missing   │  │ Categorical │  │  Features   │    │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      ML TRAINING PIPELINE                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
    │  │   Task      │──│   Model     │──│ Hyperparam  │──│  Ensemble   │    │
    │  │ Detection   │  │ Selection   │  │ Optimization│  │  Creation   │    │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      DEPLOYMENT PIPELINE                               │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
    │  │   Model     │──│   API       │──│   Web       │──│   CLI       │    │
    │  │   Export    │  │  Service    │  │ Dashboard   │  │ Interface   │    │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Option 1: Web Interface (Easiest)
```bash
# Clone and start
git clone https://github.com/Arnobrizwan/Autonomous-ML-Agent.git
cd Autonomous-ML-Agent

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start web interface
aml web

# Open http://localhost:8501 in your browser
# Upload your CSV file and click "Train Model"
```

### Option 2: Command Line
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run with your data
aml run --data data/sample.csv --target target_column

# View results in web browser
aml web
```

### Option 3: Docker (Production)
```bash
# One command deployment
docker run -p 8000:8000 aml-agent:latest
```

## 📊 Supported Data Formats

- **CSV files** (.csv) - Most common format
- **Excel files** (.xlsx, .xls) - Spreadsheet data  
- **Parquet files** (.parquet) - Efficient columnar format
- **Any tabular data** with rows and columns
- **Automatic detection** of your target column
- **Handles missing values** automatically
- **Works with text, numbers, dates** - everything!

## 🤖 Supported ML Models

- **Linear Models**: Logistic Regression, Linear Regression, Ridge, Lasso
- **Tree-based**: Random Forest, Gradient Boosting, Extra Trees
- **Distance-based**: k-Nearest Neighbors
- **Neural Networks**: Multi-layer Perceptron
- **Advanced**: XGBoost, LightGBM, CatBoost
- **Ensemble Methods**: Voting, Stacking, Blending

## 🧠 Smart Features

- **Auto-Detection**: Automatically figures out if it's classification or regression
- **Smart Preprocessing**: Handles missing values, text, categories automatically
- **AI Optimization**: Uses Optuna for hyperparameter optimization
- **Ensemble Learning**: Combines multiple models for better accuracy
- **LLM Planning**: GPT-4 guided model selection and strategy
- **Model Cards**: Automatic generation of model documentation
- **Feature Importance**: Understand what drives predictions

## ⚙️ Configuration

Create a `configs/custom.yaml` file to customize settings:

```yaml
# Basic settings
time_budget_seconds: 300        # Time limit in seconds (5 minutes)
max_trials: 50                  # Maximum optimization trials
cv_folds: 5                     # Cross-validation folds
metric: "auto"                  # Auto-select based on task
search_strategy: "bayes"        # "random" | "bayes"
enable_ensembling: true         # Enable model ensembling
top_k_for_ensemble: 3           # Number of models to ensemble
random_seed: 42                 # Reproducibility seed
use_mlflow: false               # Enable MLflow tracking

# Preprocessing settings
preprocessing:
  handle_missing: true
  impute_numeric: "median"
  impute_categorical: "most_frequent"
  encode_categorical: "onehot"
  scale_features: true
  handle_outliers: true
  outlier_method: "iqr"

# Model settings
models:
  logistic_regression:
    enabled: true
  random_forest:
    enabled: true
  xgboost:
    enabled: true
  lightgbm:
    enabled: true
```

## 🌐 Web Dashboard Features

The web dashboard provides:
- **Drag & Drop Upload**: Easy data file upload
- **Real-time Progress**: Live training progress updates
- **Model Rankings**: Performance comparison across models
- **Feature Importance**: Visual explanations of model decisions
- **Prediction Testing**: Try predictions with sample data
- **Model Download**: Export trained models and artifacts
- **API Testing**: Test REST API endpoints directly

## 🐳 Deployment Options

### Local Development
```bash
# Install and run locally
pip install -e .
aml run --data your_data.csv --target target_column
```

### Docker Compose
```bash
# Use docker-compose for full stack
docker-compose up -d
```

### Cloud Deployment
- **AWS ECS**: Deploy as containerized service
- **Google Cloud Run**: Serverless container deployment
- **Azure Container Instances**: Managed container service
- **Kubernetes**: Full orchestration support

## 📡 API Endpoints

Once running, access these endpoints:

- `GET /health` - Health check
- `POST /predict` - Make predictions
- `GET /models` - List available models
- `GET /metrics` - Performance metrics
- `POST /upload` - Upload new data
- `GET /status` - Training status

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_models.py -v
```

## 📁 Project Structure

```
Autonomous-ML-Agent/
├── src/aml_agent/           # Main source code
│   ├── agent/               # Core ML agent logic
│   ├── models/              # ML model implementations
│   ├── ui/                  # CLI and web interfaces
│   ├── export/              # Model export and artifacts
│   └── utils/               # Utility functions
├── configs/                 # Configuration files
├── data/                    # Sample data
├── docker/                  # Docker configuration
├── scripts/                 # Helper scripts
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## 💼 Use Cases

- **Business Users**: No coding required, just upload and get results
- **Data Scientists**: Quick prototyping and model comparison
- **Developers**: Ready-to-use API for your applications
- **Students**: Learn ML concepts hands-on
- **Researchers**: Experiment with different algorithms
- **Startups**: Rapid ML model development and deployment

## 🆘 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Kill existing process
pkill -f "aml_agent"
# Or use different port
aml serve --port 8001
```

**Memory issues:**
```bash
# Reduce data size or model complexity
# Edit configs/default.yaml
max_trials: 10
time_budget_seconds: 60
```

**Docker issues:**
```bash
# Rebuild container
docker build --no-cache -t aml-agent:latest -f docker/Dockerfile .
```

### Getting Help

- **Quick Demo**: Run `./start_demo.sh` and go to http://localhost:8000
- **Issues**: Report problems on [GitHub Issues](https://github.com/Arnobrizwan/Autonomous-ML-Agent/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/Arnobrizwan/Autonomous-ML-Agent/discussions)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn** for ML algorithms
- **Optuna** for hyperparameter optimization
- **FastAPI** for the REST API
- **Streamlit** for the web dashboard
- **XGBoost, LightGBM, CatBoost** for advanced models

---

**Built with ❤️ for everyone who wants to use ML without the complexity**

[![GitHub stars](https://img.shields.io/github/stars/Arnobrizwan/Autonomous-ML-Agent?style=social)](https://github.com/Arnobrizwan/Autonomous-ML-Agent)
[![GitHub forks](https://img.shields.io/github/forks/Arnobrizwan/Autonomous-ML-Agent?style=social)](https://github.com/Arnobrizwan/Autonomous-ML-Agent)
[![GitHub issues](https://img.shields.io/github/issues/Arnobrizwan/Autonomous-ML-Agent)](https://github.com/Arnobrizwan/Autonomous-ML-Agent/issues)