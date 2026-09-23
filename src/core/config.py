"""Configurações centrais do sistema utilizando Pydantic v2 e pydantic-settings.

Fail-Fast: Todas as variáveis são validadas na inicialização do microsserviço.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Configurações globais tipadas para o Positivo Vision MLOps."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Identificação da Aplicação
    APP_NAME: str = Field(default="Positivo Vision MLOps", description="Nome da aplicação")
    APP_ENV: Literal["development", "staging", "production", "test"] = Field(
        default="development", description="Ambiente de execução"
    )
    HOST: str = Field(default="0.0.0.0", description="Host de binding da API")
    PORT: int = Field(default=8000, description="Porta da API")
    LOG_LEVEL: str = Field(default="INFO", description="Nível de log (DEBUG, INFO, WARNING, ERROR)")

    # Rate Limiting & FinOps Guardrails
    RATE_LIMIT_INFERENCE: str = Field(
        default="25/2hours", description="Limite de testes de inferência por IP a cada 2 horas"
    )
    RATE_LIMIT_REPORT: str = Field(
        default="10/2hours", description="Limite de laudos GenAI por IP a cada 2 horas"
    )
    ADMIN_BYPASS_KEY: str = Field(
        default="positivo_audit_bypass_key_2026",
        description="Chave mestra para contornar o rate limiting durante a entrevista",
    )

    # LLM Industrial (OpenCode Go / DeepSeek V4.1 Flash)
    OPENCODE_BASE_URL: str = Field(
        default="https://api.opencode.go/v1", description="Endpoint compatível com OpenAI"
    )
    OPENCODE_API_KEY: str = Field(default="", description="Token de acesso da API do OpenCode")
    OPENCODE_MODEL: str = Field(
        default="DeepSeek V4.1 Flash", description="Modelo para geração de laudos técnicos"
    )

    # MLOps & MLflow Tracking
    MLFLOW_TRACKING_URI: str = Field(
        default="sqlite:///mlflow.db", description="URI do backend store do MLflow"
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="positivo-pcb-inspection", description="Nome do experimento no MLflow"
    )

    # Modelos e Caminhos de Borda
    MODEL_PATH: str = Field(
        default="models/champion.onnx", description="Caminho relativo para o modelo ONNX champion"
    )
    MODEL_ALIAS: str = Field(
        default="@champion", description="Alias do modelo em produção no MLflow"
    )
    DATA_SAMPLE_POOL_DIR: str = Field(
        default="data/sample_pool", description="Diretório contendo imagens para a Live Demo"
    )
    DATA_TELEMETRY_DIR: str = Field(
        default="data/telemetry", description="Diretório onde streams Parquet são gravados"
    )

    # Limiar Sensível a Custo (RMA Shield)
    COST_FP_REWORK: float = Field(
        default=0.50, description="Custo financeiro estimado de um Falso Positivo (R$)"
    )
    COST_FN_RMA: float = Field(
        default=500.00, description="Custo financeiro estimado de um Falso Negativo / RMA (R$)"
    )
    ANOMALY_LATENT_THRESHOLD: float = Field(
        default=1.65, description="Limiar de distância de embedding para Open-Set Anomaly"
    )

    @property
    def is_production(self) -> bool:
        """Indica se a aplicação está rodando em ambiente de produção."""
        return self.APP_ENV == "production"

    @property
    def resolved_model_path(self) -> Path:
        """Retorna o caminho absoluto do modelo ONNX."""
        return BASE_DIR / self.MODEL_PATH

    @property
    def resolved_sample_pool_dir(self) -> Path:
        """Retorna o caminho absoluto do diretório de sample pool."""
        return BASE_DIR / self.DATA_SAMPLE_POOL_DIR

    @property
    def resolved_telemetry_dir(self) -> Path:
        """Retorna o caminho absoluto do diretório de telemetria."""
        return BASE_DIR / self.DATA_TELEMETRY_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância singleton imutável das configurações."""
    return Settings()
