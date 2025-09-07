# 🎯 Autonomous ML Agent - Demo Guide

## 🚀 **Quick Start Demo (5 minutes)**

### **Step 1: Health Check** ✅
```bash
# Check if dependencies are installed
pip list | grep -E "(streamlit|fastapi|scikit-learn)"

# Check if the package is installed
python -c "import aml_agent; print('✅ Package installed successfully')"
```
**Expected Output:** All dependencies found and package imports successfully ✅

### **Step 2: Start the Service** 🌐
```bash
# Option A: Docker (Recommended for demo)
docker run -d --name aml-demo -p 8000:8000 aml-agent:latest

# Option B: Local Python Web Interface
aml web --port 8501 --host localhost

# Option C: Local Python API Server
aml serve --host 0.0.0.0 --port 8000
```

### **Step 3: Verify API is Running** 🔍
```bash
# Check health
curl http://localhost:8000/healthz

# Expected: {"status":"healthy","version":"0.1.0"}
```

### **Step 4: Open Web Interface** 🖥️
- **Web UI:** http://localhost:8501 (if using `aml web`)
- **API Server:** http://localhost:8000 (if using `aml serve`)
- **API Docs:** http://localhost:8000/docs
- **Health Dashboard:** http://localhost:8000/health

---

## 🎬 **Live Demo Script**

### **Opening (30 seconds)**
> "Today I'll demonstrate the Autonomous ML Agent - a system that automatically builds, trains, and deploys machine learning models from your data with zero manual intervention."

### **Demo 1: Web Interface (2 minutes)**
1. **Open browser:** http://localhost:8000
2. **Show features:**
   - Upload data file
   - Configure parameters
   - Start pipeline
   - View results
   - Download models

### **Demo 2: CLI Interface (2 minutes)**
```bash
# Create sample data
python -c "
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_informative=8, 
                          n_redundant=2, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y
df.to_csv('demo_data.csv', index=False)
print('Sample dataset created: 1000 samples, 10 features')
"

# Run ML pipeline
aml run --data demo_data.csv --target target --time-budget 60

# Show results
ls -la artifacts/
cat artifacts/leaderboard.csv
```

### **Demo 3: API Endpoints (1 minute)**
```bash
# Health check
curl http://localhost:8000/healthz

# API documentation
open http://localhost:8000/docs

# Make prediction (if model exists)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"feature_0": 1.2, "feature_1": 0.8, "feature_2": -1.1}'
```

---

## 🎯 **Key Talking Points**

### **What Makes This Special:**
- **🤖 Fully Automated:** No ML expertise required
- **🚀 Production Ready:** Docker + API + Web UI
- **📊 Multi-Model:** Tests 8+ algorithms automatically
- **🏆 Ensemble Learning:** Combines best models
- **🔍 Explainable AI:** Shows why models make decisions
- **⚡ Fast:** 5-minute time budget for quick results
- **🛡️ Reliable:** 66 passing tests, comprehensive error handling

### **Technical Highlights:**
- **Docker:** Multi-platform builds (linux/amd64, linux/arm64)
- **API:** FastAPI with automatic documentation
- **CI/CD:** GitHub Actions with automated testing
- **Monitoring:** Health checks, metrics, performance tracking
- **Security:** Input validation, error handling
- **Scalability:** Handles large datasets efficiently

---

## 🎪 **Interactive Demo Scenarios**

### **Scenario 1: Business User**
> "I have sales data and want to predict customer churn"
1. Upload CSV file
2. Select target column
3. Click "Run Pipeline"
4. View results in 5 minutes

### **Scenario 2: Data Scientist**
> "I want to compare multiple models quickly"
1. Use CLI for batch processing
2. Show model comparison
3. Export best model
4. Deploy as API

### **Scenario 3: Developer**
> "I need to integrate ML into my application"
1. Show API endpoints
2. Demonstrate prediction calls
3. Show Docker deployment
4. Explain monitoring

---

## 🛠️ **Troubleshooting**

### **If Something Goes Wrong:**

**Container won't start:**
```bash
docker logs aml-demo
docker restart aml-demo
```

**API not responding:**
```bash
curl -v http://localhost:8000/healthz
docker exec aml-demo ps aux
```

**Pipeline fails:**
```bash
aml run --data demo_data.csv --target target --verbose
```

**Web interface not loading:**
```bash
# Check if port 8000 is free
lsof -i :8000
# Kill process if needed
kill -9 $(lsof -t -i:8000)
```

---

## 📊 **Demo Data Examples**

### **Classification Example:**
```python
# Customer churn prediction
import pandas as pd
import numpy as np

data = {
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'tenure': np.random.randint(0, 10, 1000),
    'satisfaction': np.random.uniform(1, 5, 1000),
    'churned': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
}
df = pd.DataFrame(data)
df.to_csv('customer_data.csv', index=False)
```

### **Regression Example:**
```python
# House price prediction
import pandas as pd
import numpy as np

data = {
    'size': np.random.randint(500, 5000, 1000),
    'bedrooms': np.random.randint(1, 6, 1000),
    'bathrooms': np.random.randint(1, 4, 1000),
    'age': np.random.randint(0, 50, 1000),
    'price': np.random.normal(300000, 100000, 1000)
}
df = pd.DataFrame(data)
df.to_csv('house_data.csv', index=False)
```

---

## 🎉 **Demo Conclusion**

### **Call to Action:**
1. **Try it yourself:** Upload your own data
2. **Deploy it:** Use Docker for production
3. **Integrate it:** Use the API in your apps
4. **Customize it:** Modify for your needs

### **Next Steps:**
- **Documentation:** Full docs at `/docs`
- **Support:** GitHub issues for questions
- **Contributing:** Pull requests welcome
- **Enterprise:** Contact for custom solutions

---

## 🚀 **Quick Commands Reference**

```bash
# Health check
./quick_check.sh

# Full demo
./demo.sh

# Run pipeline
aml run --data your_data.csv --target target_column

# Web interface
aml web

# API docs
open http://localhost:8000/docs

# Stop service
docker stop aml-demo && docker rm aml-demo
```

**🎯 Your Autonomous ML Agent is ready to impress! 🎯**
