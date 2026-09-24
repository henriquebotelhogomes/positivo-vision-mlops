#!/usr/bin/env bash
# ==============================================================================
# Script de Deploy Automatizado Serverless no Azure Container Apps (ACA)
# Multi-Cloud & FinOps: Scale-to-Zero ($0/mês ocioso), 0.5 CPU, 1.0Gi RAM
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo "🚀 Iniciando Deploy Multi-Cloud: Positivo Vision MLOps (Azure ACA)"
echo "=================================================================="

# 1. Validação de pré-requisitos
if ! command -v az &> /dev/null; then
    echo "❌ Erro: CLI 'az' não encontrada no PATH."
    echo "Instale a Azure CLI: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
fi

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-positivo-vision}"
LOCATION="${AZURE_LOCATION:-brazilsouth}"
CONTAINERAPPS_ENV="${AZURE_ACA_ENV:-env-positivo-vision}"
APP_NAME="positivo-vision"
ACR_NAME="${AZURE_ACR_NAME:-acrpositivovision}"

echo "📌 Resource Group: $RESOURCE_GROUP"
echo "📌 Localização:    $LOCATION (Brasil South - São Paulo)"
echo "📌 Ambiente ACA:   $CONTAINERAPPS_ENV"
echo "📌 Aplicação:      $APP_NAME"
echo "------------------------------------------------------------------"

# 2. Criar Resource Group caso não exista
echo "⚙️  Verificando Resource Group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

# 3. Criar Azure Container Registry (ACR)
echo "📦 Verificando Azure Container Registry (ACR)..."
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true --output none || true

# 4. Build da Imagem na Nuvem (Azure ACR Build)
echo "🔨 Compilando imagem multi-stage via ACR Build..."
az acr build --registry "$ACR_NAME" --image "${APP_NAME}:latest" .

# 5. Criar Ambiente de Container Apps
echo "🌐 Configurando Azure Container Apps Environment..."
az containerapp env create --name "$CONTAINERAPPS_ENV" --resource-group "$RESOURCE_GROUP" --location "$LOCATION" --output none || true

# 6. Deploy do Container App com Scale-to-Zero
ACR_SERVER="${ACR_NAME}.azurecr.io"
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo "🚀 Publicando aplicação no Azure Container Apps..."
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINERAPPS_ENV" \
  --image "${ACR_SERVER}/${APP_NAME}:latest" \
  --target-port 8000 \
  --ingress external \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --min-replicas 0 \
  --max-replicas 3 \
  --cpu 0.5 \
  --memory 1.0Gi \
  --env-vars \
    APP_ENV=production \
    PORT=8000 \
    MODEL_PATH=models/champion.onnx \
    MLFLOW_TRACKING_URI="https://dagshub.com/henriquebotelhogomes/positivo-vision-mlops.mlflow" \
  --output none

# 7. URL do Serviço
FQDN=$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv)

echo "=================================================================="
echo "✅ Deploy no Azure concluído com sucesso!"
echo "🔗 URL Pública Ativa: https://${FQDN}/demo"
echo "📑 Documentação Scalar: https://${FQDN}/docs"
echo "=================================================================="
