<#
.SYNOPSIS
    Script de Deploy Automatizado Serverless no Google Cloud Run (PowerShell)
    FinOps: Scale-to-Zero ($0/mês ocioso), 1 vCPU, 1Gi RAM, Max 2 instâncias
#>

$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "🚀 Iniciando Deploy Serverless: Positivo Vision MLOps (Cloud Run)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Validação de pré-requisitos
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "❌ Erro: CLI 'gcloud' não encontrada no PATH. Instale o Google Cloud SDK."
    exit 1
}

$ProjectId = $env:GCP_PROJECT_ID
if (-not $ProjectId) {
    $CurrentProject = (& gcloud config get-value project 2>$null).Trim()
    if ($CurrentProject -and $CurrentProject -ne "(unset)") {
        $ProjectId = $CurrentProject
    } else {
        $ProjectId = Read-Host "Digite o ID do Projeto Google Cloud (ex: positivo-vision-demo)"
        & gcloud config set project $ProjectId
    }
}

$Region = if ($env:GCP_REGION) { $env:GCP_REGION } else { "us-central1" }
$ServiceName = "positivo-vision"
$ImageTag = "gcr.io/${ProjectId}/${ServiceName}:latest"

Write-Host "📌 Projeto GCP: $ProjectId" -ForegroundColor Yellow
Write-Host "📌 Região:      $Region" -ForegroundColor Yellow
Write-Host "📌 Serviço:     $ServiceName" -ForegroundColor Yellow
Write-Host "📌 Imagem:      $ImageTag" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------------"

# 2. Habilita APIs necessárias no GCP
Write-Host "⚙️  Verificando e habilitando APIs requeridas (Cloud Run & Cloud Build)..." -ForegroundColor Green
& gcloud services enable run.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com

# 3. Compilação do container na nuvem via Cloud Build
Write-Host "📦 Compilando imagem de produção via Google Cloud Build..." -ForegroundColor Green
& gcloud builds submit --tag $ImageTag .

# 4. Deploy no Cloud Run com política Scale-to-Zero
Write-Host "🚀 Publicando microsserviço no Google Cloud Run..." -ForegroundColor Green
$ApiKey = if ($env:OPENCODE_API_KEY) { $env:OPENCODE_API_KEY } else { "" }
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
  --set-env-vars APP_ENV=production,PORT=8000,OPENCODE_API_KEY=$ApiKey

# 5. Obtém e exibe a URL pública ativa
$ServiceUrl = (& gcloud run services describe $ServiceName --region $Region --format "value(status.url)").Trim()

Write-Host "==================================================================" -ForegroundColor Green
Write-Host "🎉 DEPLOY CONCLUÍDO COM SUCESSO!" -ForegroundColor Green
Write-Host "🌐 URL Pública da Demonstração: ${ServiceUrl}/demo" -ForegroundColor Cyan
Write-Host "📄 Documentação Scalar (OpenAPI): ${ServiceUrl}/docs" -ForegroundColor Cyan
Write-Host "🩺 Healthcheck Probe:          ${ServiceUrl}/healthz" -ForegroundColor Cyan
Write-Host "💰 Custo quando ocioso:         `$0.00 / mês (Scale-to-Zero garantido)" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Green
