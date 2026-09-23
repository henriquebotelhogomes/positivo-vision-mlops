.PHONY: help install lint test generate-data train run mlflow-ui docker-build docker-up docker-down deploy

help:
	@echo "Comandos disponíveis no projeto Positivo Vision MLOps:"
	@echo "  make install        - Instala dependências de runtime e dev com uv"
	@echo "  make lint           - Executa linter estático (ruff)"
	@echo "  make test           - Executa suíte de testes determinísticos (pytest)"
	@echo "  make generate-data  - Gera dataset industrial sintético de PCBs"
	@echo "  make train          - Executa pipeline de treino, benchmark ONNX e MLflow"
	@echo "  make run            - Inicia a API FastAPI localmente (porta 8000)"
	@echo "  make mlflow-ui      - Inicia interface do MLflow Tracking (porta 5000)"
	@echo "  make docker-build   - Constrói a imagem Docker multi-stage de produção"
	@echo "  make docker-up      - Sobe containers da API e MLflow via Docker Compose"
	@echo "  make docker-down    - Encerra os containers do Docker Compose"
	@echo "  make deploy         - Realiza deploy serverless no Google Cloud Run"

install:
	uv sync --extra dev

lint:
	uv run ruff check .

test:
	uv run pytest -v

generate-data:
	uv run python scripts/generate_industrial_pcbs.py

train:
	uv run python -m src.training.train

run:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

docker-build:
	docker build -t positivo-vision-mlops:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

deploy:
	bash scripts/deploy_cloudrun.sh
