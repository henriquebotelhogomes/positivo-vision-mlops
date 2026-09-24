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

### Fase 7: Otimizações de Borda, Tiling Scanner & Governança DagsHub (Concluído)
- [x] **Sliding Window Tiling Scanner (Inspeção Gigapixel):**
  - Implementação de varredura por mosaicos para placas de alta resolução (4.8 Megapixels).
  - Re-exportação do modelo ONNX com dimensões dinâmicas de batch (`dynamic_axes={"input": {0: "batch_size"}}`).
  - Resolução do gap de escala: detecção de defeitos microscópicos (0.07% da área da PCB) sem perda por interpolação de redimensionamento.
- [x] **Unificação dos Datasets Reais (DeepPCB + Kaggle):**
  - Taxonomia padronizada de 7 classes industriais (`NORMAL`, `DEFECT_SHORT`, `DEFECT_OPEN`, `DEFECT_MISSING_HOLE`, `DEFECT_MOUSEBITE`, `DEFECT_SPUR`, `DEFECT_SPURIOUS_COPPER`).
  - Treinamento validado out-of-sample com F1 ponderado de 0.8354.
- [x] **Governança MLflow Enterprise & Model Signatures:**
  - Contrato de tensores estrito via `infer_signature` (`[None, 3, 224, 224]` -> `[None, 7]`) e `input_example`.
  - Linhagem de datasets com `mlflow.data.from_pandas` e `mlflow.log_input`.
  - Tags de compliance industrial: `standard="IPC-A-610-Class-3"`, `git.commit`, `git.branch`.
- [x] **Infraestrutura em Nuvem DagsHub:**
  - Instalação e autenticação do cliente `dagshub`.
  - Sincronização remota da run `@champion`, artefatos (matriz de confusão, centróide de anomalia, ONNX) e Model Registry no DagsHub.
  - Link de 1-clique incorporado no cabeçalho executivo da Live Demo (`/demo`) e badges no `README.md`.
- [x] **Versionamento Privado no GitHub:**
  - Repositório privado configurado e sincronizado em `henriquebotelhogomes/positivo-vision-mlops`.

---

### Fase 8: Fechamento dos Gaps Técnicos — Padrão Nota 10 Absoluta
- [ ] **Melhoria 1: Agente de Causa-Raiz SMT com LangGraph & Knowledge Graph (`Graph / LLM / LangChain`):**
  - Criar `src/models/smt_graph_agent.py` utilizando **LangGraph** (`StateGraph`).
  - Modelar o fluxo de decisão de processo SMT:
    - Nó 1: Extração de Telemetria e Coordenadas XAI (Grad-CAM).
    - Nó 2: Consulta a regras de processo SMT / IPC-A-610 (Stencil, Pasta de Solda, Zonas 1-8 do Forno de Refluxo).
    - Nó 3: Decisor Condicional / Roteador (Inspeção Humana vs. Intervenção Imediata na Linha).
    - Nó 4: Síntese Estruturada do Laudo Técnico com DeepSeek LLM.
- [ ] **Melhoria 2: Camada Analítica SQL com DuckDB sobre Streams Parquet (`SQL Industrial`):**
  - Implementar módulo `src/monitoring/sql_analytics.py` utilizando **DuckDB** para consultas ANSI SQL sobre arquivos `.parquet`.
  - Implementar queries analíticas industriais:
    - Cálculo de **PPM (Partes Por Milhão)** de defeitos por período.
    - Taxa de concordância do operador **Human-in-the-Loop (HITL Agreement Rate)**.
    - Distribuição de percentis de latência (P50, P90, P99) por classe inspecionada.
  - Expor endpoint `GET /api/v1/telemetry/sql-metrics` e integrar no painel de auditoria.
- [ ] **Melhoria 3: Orquestração Cloud Native Kubernetes (`k8s/` Manifests):**
  - Criar a pasta `k8s/` com manifestos de produção:
    - `deployment.yaml`: Recursos com `limits`/`requests` (CPU/RAM), `runAsNonRoot: true`, sondagens `livenessProbe` e `readinessProbe` em `/healthz` e `/ready`.
    - `service.yaml`: Serviço de rede industrial (ClusterIP / LoadBalancer).
    - `hpa.yaml`: **Horizontal Pod Autoscaler** com escalonamento automático de 2 a 10 réplicas baseado em utilização de CPU.
    - `kustomization.yaml`: Configuração declarativa para GitOps (ArgoCD / Kustomize).
- [ ] **Melhoria 4: Automação Multi-Cloud (Azure Container Apps):**
  - Criar script de deploy automatizado para Azure (`scripts/deploy_azure.sh`).
