# ==============================================================================
# Multi-Stage Production Dockerfile for Positivo Vision MLOps
# Hardened, Non-Root, Slim Base, Scaled-to-Zero Cloud Run Ready
# ==============================================================================

# Stage 1: Build Dependencies via uv
FROM python:3.12-slim AS builder

WORKDIR /app

# Instala curl para baixar uv ou copia o binário oficial do uv
COPY --from=ghcr.io/astral-sh/uv:0.4.15 /uv /bin/uv

# Copia arquivos de definição de pacotes
COPY pyproject.toml uv.lock ./

# Compila o virtualenv sem dependências de desenvolvimento
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv sync --frozen --no-dev --no-install-project


# Stage 2: Runtime Minimal & Seguro
FROM python:3.12-slim AS runtime

LABEL maintainer="Henrique <developer@positivovision.local>" \
      description="Positivo Vision MLOps - Inspeção Visual de PCBs com ONNX Runtime e GenAI" \
      version="0.1.0"

# Instala apenas dependências de sistema essenciais (curl para healthcheck, libglib/libgomp para opencv/torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-privilegiado (appuser) com UID 10001
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -m -s /bin/bash appuser

WORKDIR /app

# Copia o ambiente virtual compilado
COPY --from=builder /app/.venv /app/.venv

# Copia os artefatos do projeto com ownership direto de appuser
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser data/sample_pool/ /app/data/sample_pool/
COPY --chown=appuser:appuser pyproject.toml /app/pyproject.toml

# Cria pasta de telemetria com permissões para appuser (evita chown no .venv pesado)
RUN mkdir -p /app/data/telemetry && \
    chown -R appuser:appuser /app/data

# Configura variáveis de ambiente
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    PORT=8000

# Troca para usuário não-privilegiado
USER appuser

# Healthcheck em conformidade com o padrão Cloud Native
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/healthz || exit 1

EXPOSE 8000

# Execução assíncrona com suporte dinâmico ao $PORT do Cloud Run
CMD ["sh", "-c", "exec /app/.venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
