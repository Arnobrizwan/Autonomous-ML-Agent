#!/bin/bash

# 🚀 Quick Health Check for Autonomous ML Agent
# Run this to verify everything is working before your demo

echo "🔍 Quick Health Check - Autonomous ML Agent"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1${NC}"
        return 1
    fi
}

echo "1. Checking Python environment..."
python -c "import aml_agent; print('AML Agent imported successfully')" 2>/dev/null
check_status "Python environment OK"

echo ""
echo "2. Checking CLI installation..."
aml --help > /dev/null 2>&1
check_status "CLI installed and working"

echo ""
echo "3. Checking Docker image..."
if docker images | grep -q "aml-agent.*latest"; then
    echo -e "${GREEN}✅ Docker image found${NC}"
    docker images aml-agent:latest --format "   Image: {{.Repository}}:{{.Tag}} ({{.Size}})"
else
    echo -e "${YELLOW}⚠️  Building Docker image...${NC}"
    docker build -f docker/Dockerfile -t aml-agent:latest . > /dev/null 2>&1
    check_status "Docker image built"
fi

echo ""
echo "4. Testing Docker container..."
CONTAINER_ID=$(docker run -d --name aml-agent-check -p 8000:8000 aml-agent:latest 2>/dev/null)
if [ $? -eq 0 ]; then
    sleep 3
    if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Container running and healthy${NC}"
        echo "   Health check: $(curl -s http://localhost:8000/healthz)"
    else
        echo -e "${RED}❌ Container health check failed${NC}"
    fi
    docker stop aml-agent-check > /dev/null 2>&1
    docker rm aml-agent-check > /dev/null 2>&1
else
    echo -e "${RED}❌ Failed to start container${NC}"
fi

echo ""
echo "5. Checking test suite..."
python -m pytest tests/ -q --tb=no > /dev/null 2>&1
check_status "All tests passing"

echo ""
echo "6. Checking code quality..."
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics > /dev/null 2>&1
check_status "No critical linting errors"

echo ""
echo "🎯 Quick Check Summary:"
echo "======================"
echo "• Python environment: $(python -c "import aml_agent; print('OK')" 2>/dev/null || echo 'FAILED')"
echo "• CLI installation: $(aml --help > /dev/null 2>&1 && echo 'OK' || echo 'FAILED')"
echo "• Docker image: $(docker images | grep -q "aml-agent.*latest" && echo 'OK' || echo 'FAILED')"
echo "• Container health: $(curl -s http://localhost:8000/healthz > /dev/null 2>&1 && echo 'OK' || echo 'FAILED')"
echo "• Test suite: $(python -m pytest tests/ -q --tb=no > /dev/null 2>&1 && echo 'OK' || echo 'FAILED')"
echo "• Code quality: $(flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics > /dev/null 2>&1 && echo 'OK' || echo 'FAILED')"

echo ""
echo "🚀 Ready for demo: $(curl -s http://localhost:8000/healthz > /dev/null 2>&1 && echo 'YES' || echo 'NO')"
echo ""
echo "💡 To run full demo: ./demo.sh"
echo "💡 To start web interface: aml web"
echo "💡 To run with your data: aml run --data your_data.csv --target target_column"
