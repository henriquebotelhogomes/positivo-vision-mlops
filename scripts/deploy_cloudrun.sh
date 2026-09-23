#!/usr/bin/env bash
# ==============================================================================
# Script de Deploy Automatizado Serverless no Google Cloud Run
# FinOps: Scale-to-Zero ($0/mês ocioso), 1 vCPU, 1Gi RAM, Max 2 instâncias
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "🚀 Iniciando Deploy Serverless: Positivo Vision MLOps (Cloud Run)"
echo "=================================================================="

# 1. Validação de pré-requisitos
if ! command -v gcloud &> /dev/null; then
    echo "❌ Erro: CLI 'gcloud' não encontrada no PATH."
    echo "Instale o Google Cloud SDK ou execute via Cloud Shell."
    exit 1
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo "⚠️ Nenhum projeto ativo configurado no gcloud."
    read -rp "Digite o ID do Projeto Google Cloud (ex: positivo-vision-demo): " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="positivo-vision"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "📌 Projeto GCP: $PROJECT_ID"
echo "📌 Região:      $REGION"
echo "📌 Serviço:     $SERVICE_NAME"
echo "📌 Imagem:      $IMAGE_TAG"
echo "------------------------------------------------------------------"

# 2. Habilita APIs necessárias no GCP
echo "⚙️  Verificando e habilitando APIs requeridas (Cloud Run & Cloud Build)..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com

# 3. Compilação do container na nuvem via Cloud Build (dispensa Docker local)
echo "📦 Compilando imagem de produção via Google Cloud Build..."
gcloud builds submit --tag "$IMAGE_TAG" .

# 4. Deploy no Cloud Run com política Scale-to-Zero (Custo Zero)
echo "🚀 Publicando microsserviço no Google Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_TAG" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --port 8000 \
  --set-env-vars APP_ENV=production,PORT=8000,OPENCODE_API_KEY="${OPENCODE_API_KEY:-}"

# 5. Obtém e exibe a URL pública ativa
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format 'value(status.url)')

echo "=================================================================="
echo "🎉 DEPLOY CONCLUÍDO COM SUCESSO!"
echo "🌐 URL Pública da Demonstração: ${SERVICE_URL}/demo"
echo "📄 Documentação Scalar (OpenAPI): ${SERVICE_URL}/docs"
echo "🩺 Healthcheck Probe:          ${SERVICE_URL}/healthz"
echo "💰 Custo quando ocioso:         \$0.00 / mês (Scale-to-Zero garantido)"
echo "=================================================================="
