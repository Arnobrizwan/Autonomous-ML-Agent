# 🤖 Autonomous ML Agent

**Transform your data into production-ready machine learning models automatically!**

Just upload your CSV file and get a complete ML solution with the best model, explanations, and ready-to-use API - no coding required!

## ✨ What You Get

- **📊 Upload & Go**: Drop your CSV file, specify the target column, and you're done
- **🤖 Smart Training**: Automatically tests 8+ ML algorithms to find the best one
- **🎯 Auto-Optimization**: AI finds the perfect settings for your data
- **🏆 Best Model**: Gets the highest accuracy model automatically
- **📈 Insights**: See why the model makes decisions with feature importance
- **🌐 Ready API**: Get a working API endpoint for predictions
- **📱 Web Dashboard**: Beautiful interface to explore everything
- **🔒 Production Ready**: Deploy anywhere with Docker

## 🚀 Super Quick Start (2 Minutes)

**Just want to see it work? Run this:**
```bash
git clone https://github.com/Arnobrizwan/Autonomous-ML-Agent.git
cd autonomous-ml-agent
./start_demo.sh
# Open http://localhost:8000 in your browser
```

**That's it!** Your ML Agent is running with a web interface.

## 🎬 Try It Now

### Option 1: Web Interface (Easiest)
```bash
# Start the web dashboard
./start_demo.sh
# Go to http://localhost:8000
# Upload your CSV file and click "Train Model"
```

### Option 2: Command Line
```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run with your data
aml run --data your_data.csv --target target_column

# View results in web browser
aml web
```

### Option 3: Docker (Production)
```bash
# One command deployment
docker run -p 8000:8000 ghcr.io/arnobrizwan/autonomous-ml-agent:latest
# Go to http://localhost:8000
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

## 🎮 Easy Commands

```bash
# Quick start
./start_demo.sh

# Run with your data
aml run --data your_data.csv --target target_column

# View results
aml web

# Check health
./quick_check.sh

# Full demo
./demo.sh
```

## 📊 What Data Works

- **CSV files** (.csv) - Most common format
- **Excel files** (.xlsx, .xls) - Spreadsheet data
- **Any tabular data** with rows and columns
- **Automatic detection** of your target column
- **Handles missing values** automatically
- **Works with text, numbers, dates** - everything!

## ⚙️ Customize (Optional)

Create a `config.yaml` file to change settings:

```yaml
# Basic settings
time_budget: 300              # Time limit in seconds (5 minutes)
cv_folds: 5                   # Cross-validation folds
metric: "auto"                # Auto-select based on task
search_strategy: "bayes"      # "random" | "bayes"
enable_ensembling: true       # Enable model ensembling
top_k_for_ensemble: 3         # Number of models to ensemble
random_seed: 42               # Reproducibility seed
use_mlflow: false             # Enable MLflow tracking
```

## 🤖 What Models It Tests

- **Linear Models**: Logistic Regression, Linear Regression
- **Tree-based**: Random Forest, Gradient Boosting  
- **Distance-based**: k-Nearest Neighbors
- **Neural Networks**: Multi-layer Perceptron
- **Advanced**: XGBoost, LightGBM, CatBoost

## 🧠 Smart Features

- **Auto-Detection**: Automatically figures out if it's classification or regression
- **Smart Preprocessing**: Handles missing values, text, categories automatically
- **AI Optimization**: Uses AI to find the best model settings
- **Ensemble Learning**: Combines multiple models for better accuracy
- **Meta-Learning**: Learns from previous runs to improve future performance

## 🐳 Deploy Anywhere

### Quick Deploy
```bash
# One command deployment
docker run -p 8000:8000 ghcr.io/arnobrizwan/autonomous-ml-agent:latest
# Go to http://localhost:8000
```

### Production Deploy
```bash
# Build and run
docker build -f docker/Dockerfile -t aml-agent .
docker-compose up

# Or with custom config
docker run -p 8000:8000 -v $(pwd)/data:/app/data aml-agent
```

## 🌐 Web Interface

The web dashboard lets you:
- **Upload data** with drag-and-drop
- **See training progress** in real-time
- **View model rankings** and performance
- **Get explanations** for predictions
- **Download results** and models
- **Test the API** directly

## 💼 Perfect For

- **Business Users**: No coding required, just upload and get results
- **Data Scientists**: Quick prototyping and model comparison
- **Developers**: Ready-to-use API for your applications
- **Students**: Learn ML concepts hands-on
- **Researchers**: Experiment with different algorithms

## 🆘 Need Help?

- **Quick Demo**: Run `./start_demo.sh` and go to http://localhost:8000
- **Full Guide**: Check the demo guide for step-by-step instructions
- **Issues**: Report problems on GitHub
- **Questions**: Ask in discussions

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for everyone who wants to use ML without the complexity**