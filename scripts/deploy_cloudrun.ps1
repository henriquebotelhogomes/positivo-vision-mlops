# ==============================================================================
# Script de Deploy Automatizado Serverless no Google Cloud Run (PowerShell)
# FinOps: Scale-to-Zero ($0/mês ocioso), 1 vCPU, 1Gi RAM, Max 2 instâncias
# ==============================================================================

[CmdletBinding()]
param (
    [string]$ProjectID = "retainiq-prod",
    [string]$Region = "us-central1",
    [string]$ServiceName = "positivo-vision"
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "🚀 Iniciando Deploy Serverless: Positivo Vision MLOps (Cloud Run)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Valida existência da CLI gcloud
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "❌ CLI 'gcloud' não encontrada no PATH do sistema. Instale o Google Cloud SDK."
    exit 1
}

# 2. Configura projeto ativo
Write-Host "📌 Configurando projeto ativo: $ProjectID..." -ForegroundColor Yellow
& gcloud config set project $ProjectID

$ImageTag = "gcr.io/$ProjectID/$ServiceName`:latest"

Write-Host "📌 Projeto GCP: $ProjectID" -ForegroundColor Green
Write-Host "📌 Região:      $Region" -ForegroundColor Green
Write-Host "📌 Serviço:     $ServiceName" -ForegroundColor Green
Write-Host "📌 Imagem:      $ImageTag" -ForegroundColor Green
Write-Host "------------------------------------------------------------------"

# 3. Compilação do container na nuvem via Cloud Build
Write-Host "📦 Compilando imagem de produção via Google Cloud Build..." -ForegroundColor Yellow
& gcloud builds submit --tag $ImageTag .

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Falha ao compilar a imagem no Cloud Build."
    exit 1
}

# 4. Deploy no Cloud Run com política Scale-to-Zero
Write-Host "🚀 Publicando microsserviço no Google Cloud Run..." -ForegroundColor Yellow
& gcloud run deploy $ServiceName `
  --image $ImageTag `
  --region $Region `
  --platform managed `
  --allow-unauthenticated `
  --memory 1Gi `
  --cpu 1 `
  --min-instances 0 `
  --max-instances 2 `
  --port 8000 `
  --set-env-vars "APP_ENV=production,PORT=8000"

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Falha no deploy do Cloud Run."
    exit 1
}

# 5. Obtém a URL pública do serviço
$ServiceUrl = (& gcloud run services describe $ServiceName --region $Region --format "value(status.url)").Trim()

Write-Host "==================================================================" -ForegroundColor Green
Write-Host "🎉 DEPLOY CONCLUÍDO COM SUCESSO!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "🌐 URL Pública:    $ServiceUrl" -ForegroundColor Cyan
Write-Host "📱 Live Demo UI:   $ServiceUrl/demo" -ForegroundColor Cyan
Write-Host "📑 Scalar Docs:    $ServiceUrl/docs" -ForegroundColor Cyan
Write-Host "🏥 Health Check:   $ServiceUrl/healthz" -ForegroundColor Cyan
Write-Host "📊 SQL Seis Sigma: $ServiceUrl/api/v1/telemetry/sql-metrics" -ForegroundColor Cyan
Write-Host "------------------------------------------------------------------"
Write-Host "💡 FinOps: Scale-to-Zero ATIVO (--min-instances 0)." -ForegroundColor Magenta
Write-Host "   Custo de ociosidade: \$0,00 / mês." -ForegroundColor Magenta
