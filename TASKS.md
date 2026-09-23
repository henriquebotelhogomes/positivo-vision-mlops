# 📋 TASKS: Backlog de Implementação do Projeto (Padrão Staff / Cloud-Ready)

> **Objetivo:** Ter o projeto **Positivo Vision MLOps** 100% funcional, testado, conteinerizado, publicado online na nuvem (Google Cloud Run / Azure) e municiado com diferenciais de nível Staff (Open-Set Anomaly Head, Laudo Multimodal GenAI e Human-in-the-Loop) para uma apresentação arrebatadora na Positivo Tecnologia (POSI3).  
> **Status:** Especificação Técnica Atualizada — Pronto para Execução.

---

## 🚀 Roadmap de Execução por Fases

### Fase 1: Fundação do Ambiente, Dependências & Estrutura de Diretórios
- [x] Criar estrutura de diretórios em `d:\positivo-vision-mlops/` (`data/`, `src/`, `tests/`, `scripts/`, `.github/`).
- [x] Configurar gerenciamento de dependências com **`uv`** e **`pyproject.toml`** (PyTorch, ONNX Runtime, FastAPI, MLflow, Pydantic v2, Polars, Structlog, Scalar, OpenAI).
- [x] Criar arquivo de configuração `.env.example` e módulo de settings tipado em `src/core/config.py` com `pydantic-settings`.
- [x] Implementar logger estruturado em `src/core/logging.py` via `structlog`.
- [x] Popular `data/sample_pool/` com amostras industriais balanceadas de placas de circuito impresso (classes `NORMAL` e os 4 defeitos industriais + anomalias inéditas).

### Fase 2: Modelagem Visual, Open-Set Anomaly Head, Grad-CAM & ONNX
- [x] Desenvolver pipeline de pré-processamento, augmentations e triagem estatística OOD (blur com Laplaciano e luminosidade) em `src/data/transforms.py`.
- [x] Implementar arquitetura de rede neural com Transfer Learning (ResNet50 / MobileNetV3) em `src/models/vision_net.py`.
- [x] Implementar a cabeça de detecção de defeitos inéditos (**Open-Set Anomaly Head**) baseada em distância no espaço latente em `src/models/anomaly_head.py`.
- [x] Implementar gerador de mapas de calor **Grad-CAM** acoplado às camadas convolucionais finais em `src/models/gradcam.py`.
- [x] Construir módulo de compilação e exportação para **ONNX Runtime** com quantização dinâmica em `src/models/onnx_exporter.py`.
- [x] Construir gerador de **Laudo Técnico Multimodal de Causa-Raiz (GenAI)** em `src/models/llm_report.py` integrado à API do **OpenCode Go** (modelo **`DeepSeek V4.1 Flash`**).

### Fase 3: Treinamento, Governança MLflow & Calibração de Custos (RMA Shield)
- [x] Desenvolver pipeline de treinamento desacoplado integrado ao **MLflow Tracking** (`src/training/train.py`):
  - Log de hiperparâmetros, loss, acurácia, precision, recall e F1 ponderado.
  - Benchmark de latência e ganho de throughput: PyTorch vs. ONNX Runtime (latência ONNX de 4.03 ms / 248 FPS).
  - Log da matriz de confusão e amostras de Grad-CAM como artefatos de auditoria.
- [x] Implementar calibração de limiar de decisão sensível a custo (*Cost-Sensitive Threshold Tuning*) em `src/training/cost_optimizer.py` para blindagem contra devoluções RMA.
- [x] Desenvolver lógica de governança de modelos no MLflow Registry (`src/registry/promote_model.py`):
  - Comparação out-of-sample entre nova versão (*Challenger*) e o modelo atual (*Champion*).
  - Promoção automática via Model Alias (`@champion`) baseada em ganho estatístico.

### Fase 4: Microsserviço FastAPI, GenAI, HITL, Rate Limiting & Telemetria Parquet
- [x] Definir schemas Pydantic v2 de entrada e saída em `src/api/schemas.py`.
- [x] Implementar middleware de **Rate Limiting em Memória (FinOps)** em `src/api/rate_limiter.py`:
  - Janela deslizante de **25 requisições / 2 horas por IP** para inferência e 10 para o laudo LLM.
  - Bypass de segurança para a entrevista via `ADMIN_BYPASS_KEY`.
  - Respostas amigáveis com status `HTTP 429 Too Many Requests`.
- [x] Construir o motor da API com gerenciamento de memória via **`lifespan` context manager** em `src/api/main.py`:
  - `GET /healthz` e `GET /ready` (Liveness e Readiness para orquestradores K8s/Cloud Run).
  - `POST /api/v1/predict` (Inferência via upload com latência ONNX, Open-Set flag e Grad-CAM em Base64).
  - `GET /api/v1/predict-random` (Sorteio instantâneo do pool industrial para a Live Demo em 1-clique).
  - `POST /api/v1/generate-report` (Geração do Laudo Técnico SMT via OpenCode Go DeepSeek V4.1 Flash).
  - `POST /api/v1/feedback` (Gravação de validação/correção humana do operador de bancada - HITL).
  - `GET /api/v1/telemetry/stats` (Agregação de métricas da esteira fabril lidas do stream Parquet).
  - `GET /docs` (Documentação interativa com **Scalar OpenAPI**).
- [x] Implementar gravação assíncrona de telemetria de produção (`BackgroundTasks`) gerando arquivos particionados em `.parquet` em `src/monitoring/telemetry.py`.

### Fase 5: Interface de Demonstração Executiva (Live Demo UI)
- [x] Criar interface web moderna, responsiva e industrial (HTML5/Tailwind/JS moderno) servida na rota `/demo`:
  - **Badge de Cota de Testes:** Exibição do contador restante de testes (25 / 2 horas).
  - **Botão 1 (Ação Rápida na Entrevista):** *"🎲 Simular Leitura da Câmera na Esteira (Amostra Aleatória)"*.
  - **Botão 2:** *"📁 Fazer Upload de Imagem Externa"*.
  - **Chave de Alternância (Toggle):** Imagem original da placa vs. Heatmap Grad-CAM sobreposto destacando o defeito.
  - **Botão GenAI:** *"📄 Gerar Laudo Técnico SMT (DeepSeek V4.1)"* abrindo modal com análise técnica detalhada.
  - **Botões HITL:** *"✅ Validar Diagnóstico"* ou *"✏️ Corrigir (Falso Positivo)"* com feedback visual imediato.
  - Indicadores em tempo real: Status (Conforme / Reprovada / Anomalia Inédita), Confiança (%), Latência ONNX (ms) e Metadados do Modelo (`@champion`).

### Fase 6: Conteinerização Segura, Testes & CI/CD
- [x] Escrever `Dockerfile` multi-stage build seguro (usuário `non-root`, base slim, sem compiladores no runtime).
- [x] Configurar `docker-compose.yml` para orquestração local (Servidor MLflow com SQLite + API de Inferência).
- [x] Desenvolver suíte de testes unitários determinísticos com `pytest` em `tests/` cobrindo pré-processamento, inferência ONNX, anomaly head, schemas da API e Grad-CAM (13/13 testes passando).
- [x] Configurar pipeline de CI/CD no GitHub Actions (`.github/workflows/ci.yml`) com `ruff`, `pytest` e build da imagem Docker.
- [x] Padronizar comandos operacionais em `Makefile` (`make install`, `make lint`, `make test`, `make train`, `make run`, `make docker-up`, `make deploy`).

### Fase 7: Publicação Online em Nuvem (Cloud Run) & Ensaio Geral da Entrevista
- [x] Criar scripts de deploy automatizado serverless para o **Google Cloud Run** (`scripts/deploy_cloudrun.sh` e `scripts/deploy_cloudrun.ps1`) com política de Scale-to-Zero.
- [ ] Executar deploy online no Google Cloud Run via `scripts/deploy_cloudrun.ps1` (ou bash) e homologar a **URL Pública HTTPS ativa**.
- [ ] Realizar teste cego de estresse (10 execuções consecutivas na Live Demo web) garantindo latência estável (< 25ms ONNX).
- [ ] Ensaio geral dos pontos táticos do `INTERVIEW_PLAYBOOK.md` antes da reunião com o Tech Lead da Positivo.
