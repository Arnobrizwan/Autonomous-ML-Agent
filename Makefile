.PHONY: setup run test lint clean leaderboard explain serve predict verify help

# Default target
help:
	@echo "Available commands:"
	@echo "  setup       - Install dependencies and setup environment"
	@echo "  run         - Run autonomous ML pipeline with default config"
	@echo "  test        - Run test suite"
	@echo "  lint        - Run code formatting and linting"
	@echo "  clean       - Clean artifacts and cache"
	@echo "  leaderboard - Show model performance leaderboard"
	@echo "  explain     - Generate model explanations"
	@echo "  serve       - Start FastAPI prediction service"
	@echo "  predict     - Make sample prediction"
	@echo "  verify      - Run full local verification harness"
	@echo ""
	@echo "Docker commands:"
	@echo "  docker-build      - Build Docker image locally"
	@echo "  docker-build-prod - Build multi-platform production image"
	@echo "  docker-run        - Run container with volume mounts"
	@echo "  docker-compose-up - Start all services with docker-compose"
	@echo "  docker-compose-down - Stop all services"
	@echo "  docker-compose-logs - View service logs"
	@echo "  docker-test       - Test Docker image functionality"
	@echo "  docker-push       - Push image to Docker Hub (set DOCKER_USERNAME)"
	@echo "  docker-clean      - Clean Docker system and volumes"

# Setup environment
setup:
	pip install -r requirements.txt
	pip install -e .
	@echo "Setup complete! Run 'make run' to start."

# Run with default config
run:
	aml run --config configs/default.yaml

# Run tests
test:
	pytest tests/ -v

# Lint code
lint:
	black src/ tests/ --check
	isort src/ tests/ --check-only
	flake8 src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean artifacts
clean:
	rm -rf artifacts/
	rm -rf __pycache__/
	rm -rf src/**/__pycache__/
	rm -rf tests/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

# Show leaderboard
leaderboard:
	aml leaderboard

# Generate explanations
explain:
	aml explain

# Start API service
serve:
	aml serve --host 0.0.0.0 --port 8000

# Make sample prediction
predict:
	@echo "Making sample prediction..."
	@echo '{"feature_a": 1, "feature_b": 2.5}' | aml predict --run-id latest --stdin

# Run full local verification harness
verify:
	@bash scripts/verify_local.sh

# Docker commands
docker-build:
	docker build -f docker/Dockerfile -t aml-agent:latest .

docker-build-prod:
	docker buildx build --platform linux/amd64,linux/arm64 -f docker/Dockerfile -t aml-agent:latest .

docker-run:
	docker run --rm -p 8000:8000 -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts aml-agent:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

docker-test:
	docker run --rm aml-agent:latest python -c "import aml_agent; print('Import successful')"
	docker run --rm aml-agent:latest python -c "from aml_agent.service.app import create_app; print('FastAPI app creation successful')"

docker-push:
	docker tag aml-agent:latest $(DOCKER_USERNAME)/aml-agent:latest
	docker push $(DOCKER_USERNAME)/aml-agent:latest

docker-clean:
	docker system prune -f
	docker volume prune -f

# Development
dev-setup: setup
	pip install -e ".[mlflow,openai,gemini]"

# Generate sample data
sample-data:
	python -c "import pandas as pd; import numpy as np; np.random.seed(42); df = pd.DataFrame({'feature_a': np.random.randn(100), 'feature_b': np.random.randn(100), 'target': np.random.randint(0, 2, 100)}); df.to_csv('data/sample.csv', index=False); print('Sample data created at data/sample.csv')"

