from datetime import UTC, datetime
from typing import Any

import polars as pl
import structlog

from src.core.config import get_settings

logger = structlog.get_logger("positivo_vision.telemetry")


class TelemetryManager:
    """Gerencia a persistência e consulta colunar de telemetria industrial."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.telemetry_dir = self.settings.resolved_telemetry_dir
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_file = self.telemetry_dir / "inferences.parquet"

    def record_inference(
        self,
        inference_id: str,
        prediction: str,
        confidence: float,
        is_defective: bool,
        is_unknown_anomaly: bool,
        anomaly_score: float,
        inference_time_ms: float,
        is_ood: bool,
        timestamp: str | None = None,
    ) -> None:
        """Grava uma nova inferência no stream Parquet."""
        ts = timestamp or datetime.now(UTC).isoformat()
        row_dict = {
            "inference_id": [inference_id],
            "timestamp": [ts],
            "prediction": [prediction],
            "confidence": [float(confidence)],
            "is_defective": [bool(is_defective)],
            "is_unknown_anomaly": [bool(is_unknown_anomaly)],
            "anomaly_score": [float(anomaly_score)],
            "inference_time_ms": [float(inference_time_ms)],
            "is_ood": [bool(is_ood)],
            "operator_confirmed": [None],
            "operator_corrected_class": [None],
            "requires_retrain": [False],
        }

        new_df = pl.DataFrame(
            row_dict,
            schema={
                "inference_id": pl.Utf8,
                "timestamp": pl.Utf8,
                "prediction": pl.Utf8,
                "confidence": pl.Float64,
                "is_defective": pl.Boolean,
                "is_unknown_anomaly": pl.Boolean,
                "anomaly_score": pl.Float64,
                "inference_time_ms": pl.Float64,
                "is_ood": pl.Boolean,
                "operator_confirmed": pl.Boolean,
                "operator_corrected_class": pl.Utf8,
                "requires_retrain": pl.Boolean,
            },
        )

        try:
            if self.telemetry_file.exists():
                existing_df = pl.read_parquet(self.telemetry_file)
                combined_df = pl.concat([existing_df, new_df])
                combined_df.write_parquet(self.telemetry_file)
            else:
                new_df.write_parquet(self.telemetry_file)
        except Exception as exc:
            logger.error("Erro ao gravar telemetria em Parquet", error=str(exc))

    def record_feedback(
        self,
        inference_id: str,
        is_correct: bool,
        corrected_class: str | None = None,
    ) -> bool:
        """Atualiza a linha de telemetria com a validação humana do operador."""
        if not self.telemetry_file.exists():
            return False

        try:
            df = pl.read_parquet(self.telemetry_file)
            requires_retrain = not is_correct

            updated_df = df.with_columns([
                pl.when(pl.col("inference_id") == inference_id)
                .then(pl.lit(is_correct))
                .otherwise(pl.col("operator_confirmed"))
                .alias("operator_confirmed"),
                pl.when(pl.col("inference_id") == inference_id)
                .then(pl.lit(corrected_class))
                .otherwise(pl.col("operator_corrected_class"))
                .alias("operator_corrected_class"),
                pl.when(pl.col("inference_id") == inference_id)
                .then(pl.lit(requires_retrain))
                .otherwise(pl.col("requires_retrain"))
                .alias("requires_retrain"),
            ])
            updated_df.write_parquet(self.telemetry_file)
            return True
        except Exception as exc:
            logger.error("Erro ao atualizar feedback humano no Parquet", error=str(exc))
            return False

    def get_stats(self) -> dict[str, Any]:
        """Calcula estatísticas agregadas da esteira lendo o Parquet via Polars."""
        if not self.telemetry_file.exists():
            return {
                "total_inspections": 0,
                "defective_count": 0,
                "defect_rate_pct": 0.0,
                "unknown_anomalies_count": 0,
                "operator_agreements_count": 0,
                "latency_p95_ms": 0.0,
                "active_champion": str(self.settings.MODEL_PATH),
            }

        try:
            df = pl.read_parquet(self.telemetry_file)
            total = df.height
            if total == 0:
                return {
                    "total_inspections": 0,
                    "defective_count": 0,
                    "defect_rate_pct": 0.0,
                    "unknown_anomalies_count": 0,
                    "operator_agreements_count": 0,
                    "latency_p95_ms": 0.0,
                    "active_champion": str(self.settings.MODEL_PATH),
                }

            defective = df.filter(pl.col("is_defective")).height
            defect_rate = round((defective / total) * 100.0, 2)
            unknowns = df.filter(pl.col("is_unknown_anomaly")).height
            agreements = df.filter(pl.col("operator_confirmed")).height
            lat_p95 = round(float(df["inference_time_ms"].quantile(0.95)), 2)

            return {
                "total_inspections": total,
                "defective_count": defective,
                "defect_rate_pct": defect_rate,
                "unknown_anomalies_count": unknowns,
                "operator_agreements_count": agreements,
                "latency_p95_ms": lat_p95,
                "active_champion": str(self.settings.MODEL_PATH),
            }
        except Exception as exc:
            logger.error("Erro ao calcular estatísticas de telemetria", error=str(exc))
            return {
                "total_inspections": 0,
                "defective_count": 0,
                "defect_rate_pct": 0.0,
                "unknown_anomalies_count": 0,
                "operator_agreements_count": 0,
                "latency_p95_ms": 0.0,
                "active_champion": str(self.settings.MODEL_PATH),
            }


telemetry_manager = TelemetryManager()
