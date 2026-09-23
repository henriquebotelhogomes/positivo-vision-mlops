# 🏭 Diretrizes do Projeto: Positivo Vision MLOps

> **Projeto Local:** Inspeção Visual Automatizada de Hardware & PCBs para a Positivo Tecnologia (POSI3).  
> **Classificação:** Tier 2 (MLOps de Produção / Deep Learning Industrial).

---

## 1. Stack & Padrões Normativos do Backend (Python)
* **Manifesto & Pacotes:** Uso obrigatório de **`uv`** com **`pyproject.toml`** (padrão oficial PEP 621) e lockfile determinístico (`uv.lock`). Proibido `setup.py` imperativo ou depender de `requirements.txt` solto.
* **Framework Web:** **FastAPI** assíncrono. O ciclo de vida e estado do modelo na memória (GPU/CPU) deve ser gerenciado estritamente via **`lifespan` context manager** (`@asynccontextmanager`), proibindo o uso dos eventos legados `@app.on_event`.
* **Validação & Configurações:** **Pydantic v2**. Variáveis de ambiente exclusivamente via **`pydantic-settings` (`BaseSettings`)** com validação estrita no startup (*Fail-Fast*). Proibido espalhar `os.getenv` soltos pelo código.
* **Engenharia de Dados:** Arquivos colunares **`.parquet`** (Apache Arrow) obrigatórios para datasets e features intermediárias (proibido CSV puro para dados de treino). Priorizar **`polars`** para manipulação pesada multithreaded.
* **Documentação de API:** **Scalar obrigatório** servido em `/docs` ou `/api/docs`. Proibido Swagger UI tradicional.
* **Logging Estruturado:** Proibido `logging.basicConfig()` em `__init__.py` ou emitir texto puro. Usar **`structlog`** (padrão ouro em JSON para Kubernetes/Cloud) ou **`loguru`** (chave-valor), centralizado em `src/core/logging.py`.

---

## 2. MLOps, Model Registry, Borda & Observabilidade (Visão Industrial)
* **Modelagem & Transfer Learning:** Redes neurais convolucionais (ResNet50 / MobileNetV3) pré-treinadas para classificação e localização de defeitos em placas de circuito impresso (PCBs).
* **Detecção de Defeitos Inéditos (Open-Set Anomaly Head):** Camada de detecção de anomalias no espaço latente de embeddings (Distância de Mahalanobis / Cosseno em relação ao cluster da PCB de referência), sinalizando `UNKNOWN_ANOMALY` para defeitos que nunca apareceram no dataset de treino.
* **Compilação & Otimização de Borda (Edge Inference):** Obrigatória a exportação e otimização para **ONNX Runtime** com quantização dinâmica (FP16/INT8), garantindo inferência industrial ultrarrápida (< 25ms em CPU fabril). Benchmark comparativo PyTorch vs. ONNX registrado no MLflow.
* **Explicabilidade Visual (XAI) com Grad-CAM:** Geração de mapas de ativação de classes (**Grad-CAM**) sobrepostos à PCB para demonstrar visualmente ao operador/técnico onde está o defeito (curto-circuito, trilha rompida, furo ausente). Retornado em Base64 na API e exibido na Demo.
* **Laudo Técnico de Causa-Raiz Multimodal (GenAI Industrial):** Endpoint (`/api/v1/generate-report`) que consome a imagem, a classe predita e as coordenadas do Grad-CAM para gerar um laudo técnico formatado para engenharia de processos SMT (causa-raiz provável em fornos/stencil e ação corretiva recomendada). Provedor padrão: **OpenCode Go** (modelo **`DeepSeek V4.1 Flash`** via client compatível com OpenAI, ou fallback configurável no `.env`).
* **Rate Limiting & FinOps Guardrails:** Proteção ativa contra exaustão de tokens e DDoS na demonstração pública via rate limiting por IP de **25 requisições a cada 2 horas** para endpoints de teste e 10 para o gerador de laudo GenAI, com bypass de administrador (`ADMIN_BYPASS_KEY`) e resposta `HTTP 429 Too Many Requests`.
* **Human-in-the-Loop & Active Learning:** Endpoint (`/api/v1/feedback`) permitindo que o operador de bancada valide ou corrija diagnósticos, gravando no stream Parquet com flag `requires_retrain=True` para ciclo fechado de melhoria contínua.
* **Métricas de Negócio & Calibração de Limiar (RMA Guardrail):** Calibração de limiar de decisão (*Cost-Sensitive Threshold Tuning*) priorizando Recall na classe de defeitos para mitigar custos de devolução (RMA da Positivo/Vaio) sem explodir falsos positivos de retrabalho.
* **MLflow:** Rastreamento rigoroso de parâmetros, métricas (Loss, F1-weighted, Recall, Precision, Matriz de Confusão e Latência ONNX).
* **Model Registry & Promoção:** Adotar política de **Champion vs. Challenger** com Model Aliases (`@champion`), promovendo novos modelos apenas após validação de ganho estatístico em dados de teste out-of-sample.
* **Detecção Ativa de Drift & Triagem OOD:** Monitoramento contínuo de *Data Drift* e anomalias de imagem (*Out-of-Distribution - OOD*) via checagem de luminância, blur (variância Laplaciana) e Evidently AI.
* **Telemetria de Inferência em Parquet:** Gravação assíncrona em background (`BackgroundTasks`) das inferências e metadados industriais (timestamp, latência, predição, OOD, feedback HITL) em arquivos colunares `.parquet` para retreinamento contínuo e auditoria fabril.

---

## 3. Demonstração, Git, Conteinerização & Deploy Cloud Serverless
* **Live Demo:** Endpoint dedicado (`/api/v1/predict-random`) para sorteio de amostras industriais do `sample_pool/` com botão de 1-clique para teste à prova de falhas na entrevista, alternância de visualização original vs. heatmap Grad-CAM, botão de geração de laudo multimodal, botões de validação HITL do operador e mostrador de cota de testes restante (25/2h).
* **Deploy Cloud Serverless & FinOps (Scale-to-Zero):** A solução deve estar 100% pronta e com script automatizado para deploy em **Google Cloud Run** ou **Azure Container Apps**, operando com escala zero (custo \$0/mês quando ocioso) e link público acessível por qualquer dispositivo durante a entrevista.
* **Docker de Produção:** Multi-stage build, imagem base slim, execução obrigatória como usuário não-privilegiado (`non-root` / `appuser`), `.dockerignore` rigoroso e endpoints de sondagem (`/healthz` e `/ready`) prontos para orquestração em Kubernetes/Cloud Run.
* **Git, CI/CD & Automação:** Conventional Commits, `.env.example` versionado, automação via `Makefile` (`make train`, `make test`, `make docker-up`, `make deploy`) e workflow no GitHub Actions (`.github/workflows/ci.yml`) com linting, testes e deploy contínuo.
