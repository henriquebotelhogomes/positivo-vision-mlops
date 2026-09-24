# ==============================================================================
# Script PowerShell de Deploy Automatizado no Azure Container Apps (ACA)
# Multi-Cloud & FinOps: Scale-to-Zero ($0/mês ocioso), 0.5 CPU, 1.0Gi RAM
# ==============================================================================

[CmdletBinding()]
param (
    [string]$ResourceGroup = "rg-positivo-vision",
    [string]$Location = "brazilsouth",
    [string]$ContainerAppEnv = "env-positivo-vision",
    [string]$AppName = "positivo-vision",
    [string]$AcrName = "acrpositivovision"
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "🚀 Iniciando Deploy Multi-Cloud: Positivo Vision MLOps (Azure ACA)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Validação de pré-requisitos
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Error "❌ CLI 'az' não encontrada no PATH. Instale a Azure CLI: https://docs.microsoft.com/cli/azure/install-azure-cli"
    exit 1
}

Write-Host "📌 Resource Group: $ResourceGroup" -ForegroundColor Yellow
Write-Host "📌 Localização:    $Location (Brasil South - São Paulo)" -ForegroundColor Yellow
Write-Host "📌 Ambiente ACA:   $ContainerAppEnv" -ForegroundColor Yellow
Write-Host "📌 Aplicação:      $AppName" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------------"

# 2. Criar Resource Group caso não exista
Write-Host "⚙️  Verificando Resource Group..." -ForegroundColor Green
az group create --name $ResourceGroup --location $Location --output none

# 3. Criar Azure Container Registry (ACR)
Write-Host "📦 Verificando Azure Container Registry (ACR)..." -ForegroundColor Green
az acr create --resource-group $ResourceGroup --name $AcrName --sku Basic --admin-enabled true --output none 2>$null

# 4. Build da Imagem na Nuvem
Write-Host "🔨 Compilando imagem multi-stage via ACR Build..." -ForegroundColor Green
az acr build --registry $AcrName --image "${AppName}:latest" .

# 5. Criar Ambiente de Container Apps
Write-Host "🌐 Configurando Azure Container Apps Environment..." -ForegroundColor Green
az containerapp env create --name $ContainerAppEnv --resource-group $ResourceGroup --location $Location --output none 2>$null

# 6. Deploy do Container App com Scale-to-Zero
$AcrServer = "${AcrName}.azurecr.io"
$AcrPassword = (az acr credential show --name $AcrName --query "passwords[0].value" -o tsv)

Write-Host "🚀 Publicando aplicação no Azure Container Apps..." -ForegroundColor Green
az containerapp create `
  --name $AppName `
  --resource-group $ResourceGroup `
  --environment $ContainerAppEnv `
  --image "${AcrServer}/${AppName}:latest" `
  --target-port 8000 `
  --ingress external `
  --registry-server $AcrServer `
  --registry-username $AcrName `
  --registry-password $AcrPassword `
  --min-replicas 0 `
  --max-replicas 3 `
  --cpu 0.5 `
  --memory 1.0Gi `
  --env-vars `
    APP_ENV=production `
    PORT=8000 `
    MODEL_PATH=models/champion.onnx `
    MLFLOW_TRACKING_URI="https://dagshub.com/henriquebotelhogomes/positivo-vision-mlops.mlflow" `
  --output none

# 7. URL do Serviço
$Fqdn = (az containerapp show --name $AppName --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv)

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "✅ Deploy no Azure concluído com sucesso!" -ForegroundColor Green
Write-Host "🔗 URL Pública Ativa: https://${Fqdn}/demo" -ForegroundColor Cyan
Write-Host "📑 Documentação Scalar: https://${Fqdn}/docs" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
