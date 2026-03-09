.PHONY: setup install download-data train evaluate test clean run-streamlit run-api docker-build docker-run pipeline help

PROJECT_NAME := retinal_disease_classifier
PYTHON := python
PIP := pip
VERSION := 1.0.0

help:
	@echo "$(PROJECT_NAME) - Retinal Disease Classification System"
	@echo "Version: $(VERSION)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Setup development environment
	@echo "Setting up environment..."
	$(PYTHON) -m venv venv
	. venv/Scripts/activate && pip install --upgrade pip setuptools wheel
	. venv/Scripts/activate && pip install -r requirements.txt
	@echo "Environment setup complete!"

install: ## Install dependencies
	$(PIP) install -r requirements.txt

download-data: ## Download and prepare dataset
	$(PYTHON) download_dataset.py --num-samples 200

download-data-full: ## Download full dataset (requires manual Kaggle API setup)
	$(PYTHON) download_dataset.py --dataset-path /path/to/aptos2019

train: ## Train all models
	$(PYTHON) train.py --epochs 30 --batch-size 32 --models "baseline_cnn,custom_cnn"

train-single: ## Train single model (baseline)
	$(PYTHON) train.py --epochs 30 --batch-size 32 --models "baseline_cnn"

train-advanced: ## Train advanced models
	$(PYTHON) train.py --epochs 50 --batch-size 32 --models "resnet50,efficientnet"

train-extended: ## Extended training with more epochs
	$(PYTHON) train.py --epochs 100 --batch-size 16 --models "baseline_cnn,custom_cnn,resnet50"

evaluate: ## Evaluate trained models
	$(PYTHON) evaluate.py --models "baseline_cnn,custom_cnn"

predict: ## Make prediction on sample image
	$(PYTHON) predict.py --image sample_image.jpg --model baseline_cnn

predict-gradcam: ## Make prediction with Grad-CAM
	$(PYTHON) predict.py --image sample_image.jpg --model baseline_cnn --gradcam

test: ## Run tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint: ## Check code style
	flake8 src/ --max-line-length=120
	black src/ --line-length=120

format: ## Format code
	black src/ --line-length=120
	isort src/

run-streamlit: ## Run Streamlit web application
	streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0

run-api: ## Run FastAPI server
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-api-prod: ## Run FastAPI in production mode
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

run-both: ## Run Streamlit and FastAPI together
	@echo "Starting FastAPI (port 8000) and Streamlit (port 8501)..."
	@echo "API: http://localhost:8000"
	@echo "App: http://localhost:8501"
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
	streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0

docker-build: ## Build Docker image
	docker build -t $(PROJECT_NAME):$(VERSION) -f docker/Dockerfile .
	@echo "Docker image built: $(PROJECT_NAME):$(VERSION)"

docker-run: ## Run Docker container
	docker run -p 8000:8000 -p 8501:8501 \
		-v $$(pwd)/data:/app/data \
		-v $$(pwd)/models:/app/models \
		-v $$(pwd)/reports:/app/reports \
		$(PROJECT_NAME):$(VERSION)

docker-run-detached: ## Run Docker container in background
	docker run -d -p 8000:8000 -p 8501:8501 \
		-v $$(pwd)/data:/app/data \
		-v $$(pwd)/models:/app/models \
		--name $(PROJECT_NAME)-container \
		$(PROJECT_NAME):$(VERSION)
	@echo "Container running as: $(PROJECT_NAME)-container"

docker-stop: ## Stop running Docker container
	docker stop $(PROJECT_NAME)-container

docker-logs: ## View Docker container logs
	docker logs $(PROJECT_NAME)-container -f

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Cleanup complete!"

clean-all: clean ## Clean all including models and data
	rm -rf data/processed/* models/* logs/*
	@echo "All generated content cleaned!"

pipeline: ## Run complete pipeline (data → train → evaluate)
	@echo "Running complete pipeline..."
	$(MAKE) download-data
	$(MAKE) train
	$(MAKE) evaluate
	@echo "Pipeline complete!"

report: ## Generate project report
	@echo "Generating project report..."
	mkdir -p reports
	$(PYTHON) -c "from src.utils.logger import setup_logger; print('Report generation would run here')"

docs: ## Generate documentation
	@echo "Project documentation available in README.md and reports/"

status: ## Show project status
	@echo "$(PROJECT_NAME) - Project Status"
	@echo "Version: $(VERSION)"
	@echo ""
	@echo "Checking components..."
	@test -f requirements.txt && echo "✓ Requirements file" || echo "✗ Requirements file"
	@test -f README.md && echo "✓ README" || echo "✗ README"
	@test -d src && echo "✓ Source code" || echo "✗ Source code"
	@test -d models && echo "✓ Models directory" || echo "✗ Models directory"
	@test -d data && echo "✓ Data directory" || echo "✗ Data directory"

requirements-update: ## Update requirements.txt with current environment
	pip freeze > requirements.txt
	@echo "requirements.txt updated"

init: ## Initialize project (one-time setup)
	$(PYTHON) -c "from src.utils.config import Config; Config.create_all_directories()"
	@echo "Project initialized successfully!"

info: ## Display project information
	@echo "$(PROJECT_NAME)"
	@echo "Version: $(VERSION)"
	@echo ""
	@echo "Project Structure:"
	@echo "  src/              - Source code modules"
	@echo "  data/             - Dataset directory"
	@echo "  models/           - Trained model files"
	@echo "  reports/          - Generated reports and results"
	@echo "  streamlit_app/    - Web interface"
	@echo "  tests/            - Test suite"
	@echo "  docker/           - Docker configuration"
	@echo ""
	@echo "Main Scripts:"
	@echo "  download_dataset.py  - Prepare dataset"
	@echo "  train.py             - Train models"
	@echo "  evaluate.py          - Evaluate models"
	@echo "  predict.py           - Make predictions"
	@echo ""
	@echo "Use 'make help' for all available commands"

.DEFAULT_GOAL := help
