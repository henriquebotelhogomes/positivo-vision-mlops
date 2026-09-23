"""Schemas tipados de dados para a API utilizando Pydantic v2.

Define os contratos estritos de entrada, saída, telemetria, feedback HITL
e laudos técnicos multimodais.
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class OODFlagsSchema(BaseModel):
    """Flags de triagem Out-of-Distribution."""

    is_ood: bool = Field(description="Indica se a imagem falhou na triagem estatística")
    is_blurred: bool = Field(description="Indica se a imagem apresenta desfoque ou trepidação mecânica")
    laplacian_variance: float = Field(description="Variância do operador Laplaciano")
    mean_luminance: float = Field(description="Luminância média no canal V do espaço HSV")
    luminance_ok: bool = Field(description="Indica se a iluminação da esteira está na faixa operacional")


class PredictionResponse(BaseModel):
    """Contrato de resposta da inferência industrial."""

    status: str = Field(default="success", description="Status da requisição")
    inference_id: str = Field(description="Identificador único da inspeção fabril")
    prediction: str = Field(description="Classe predita pelo modelo")
    confidence: float = Field(description="Grau de certeza da predição (0.0 a 1.0)")
    is_defective: bool = Field(description="Indica se a placa possui defeito industrial")
    is_unknown_anomaly: bool = Field(description="Indica se foi detectada uma anomalia inédita (Open-Set)")
    anomaly_score: float = Field(description="Score contínuo de distância do cluster de normalidade")
    probabilities: dict[str, float] = Field(description="Probabilidades calculadas para cada classe")
    inference_time_ms: float = Field(description="Tempo de processamento da inferência em milissegundos")
    engine: str = Field(default="onnxruntime-cpu", description="Motor de inferência utilizado")
    model_version: str = Field(default="v1", description="Versão do modelo registrada no MLflow")
    model_alias: str = Field(default="@champion", description="Alias de governança ativo")
    gradcam_base64: str = Field(description="Mapa de calor Grad-CAM sobreposto codificado em Base64")
    ood_flags: OODFlagsSchema = Field(description="Métricas de integridade de sinal da imagem")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class RandomSampleResponse(PredictionResponse):
    """Resposta estendida para o sorteio de amostras industriais da Live Demo."""

    sample_name: str = Field(description="Nome do arquivo da amostra industrial no pool")
    original_image_base64: str = Field(description="Imagem original da placa em Base64")


class TechnicalReportRequest(BaseModel):
    """Requisição para geração de Laudo Técnico de Causa-Raiz SMT."""

    inference_id: str = Field(description="ID da inspeção a ser auditada")
    prediction: str = Field(description="Diagnóstico da falha")
    confidence: float = Field(description="Nível de confiança do modelo")
    is_unknown_anomaly: bool = Field(default=False, description="Flag de anomalia inédita")
    latency_ms: float = Field(default=20.0, description="Latência registrada na inferência")


class TechnicalReportResponse(BaseModel):
    """Resposta com laudo técnico gerado via IA Multimodal."""

    status: str = Field(default="success")
    inference_id: str = Field(description="ID da inspeção")
    source: str = Field(description="Origem do laudo (opencode_go ou industrial_kb_fallback)")
    model: str = Field(description="Modelo que gerou o parecer")
    ipc_standard: str = Field(description="Norma de montagem eletrônica aplicável")
    report_markdown: str = Field(description="Parecer técnico completo formatado em Markdown")
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class FeedbackRequest(BaseModel):
    """Contrato de feedback do operador de bancada (Human-in-the-Loop)."""

    inference_id: str = Field(description="ID da inspeção avaliada")
    is_correct: bool = Field(description="Confirmação do diagnóstico emitido pela IA")
    corrected_class: str | None = Field(default=None, description="Classe correta caso tenha ocorrido falso positivo")
    operator_comments: str | None = Field(default=None, description="Observações do técnico de bancada")


class FeedbackResponse(BaseModel):
    """Resposta da gravação de feedback HITL."""

    status: str = Field(default="success")
    inference_id: str
    requires_retrain: bool
    recorded_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TelemetryStatsResponse(BaseModel):
    """Estatísticas agregadas de telemetria da esteira fabril."""

    total_inspections: int = Field(description="Total de placas inspecionadas registradas no Parquet")
    defective_count: int = Field(description="Quantidade de peças reprovadas por defeito")
    defect_rate_pct: float = Field(description="Taxa percentual de defeitos na linha SMT")
    unknown_anomalies_count: int = Field(description="Total de anomalias inéditas enviadas para quarentena")
    operator_agreements_count: int = Field(description="Validações humanas confirmadas pelo operador")
    operator_divergences_count: int = Field(default=0, description="Divergências registradas pelo operador")
    pending_audits_count: int = Field(default=0, description="Auditorias pendentes de validação humana")
    agreement_rate_pct: float = Field(default=100.0, description="Taxa de concordância Homem-Máquina (%)")
    requires_retrain_count: int = Field(default=0, description="Amostras marcadas para retreino contínuo")
    latency_p95_ms: float = Field(description="Latência de inferência percentil 95 em ms")
    active_champion: str = Field(default="models/champion.onnx")


class AuditRecord(BaseModel):
    """Registro individual de auditoria fabril da telemetria."""

    inference_id: str
    timestamp: str | None = None
    prediction: str
    confidence: float
    is_defective: bool
    is_unknown_anomaly: bool
    anomaly_score: float
    inference_time_ms: float
    is_ood: bool
    operator_confirmed: bool | None = None
    operator_corrected_class: str | None = None
    requires_retrain: bool = False
    status: str = "PENDENTE"


class AuditsResponse(BaseModel):
    """Lista de auditorias recentes para o painel operacional."""

    status: str = "success"
    total_returned: int
    stats: TelemetryStatsResponse
    records: list[AuditRecord]

