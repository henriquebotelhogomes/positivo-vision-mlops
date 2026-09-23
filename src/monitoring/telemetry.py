import threading
import uuid
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
        self._lock = threading.Lock()

    def get_default_stats(self) -> dict[str, Any]:
        """Retorna estrutura de métricas padrão e segura para inicialização ou fallbacks."""
        return {
            "total_inspections": 0,
            "defective_count": 0,
            "defect_rate_pct": 0.0,
            "unknown_anomalies_count": 0,
            "operator_agreements_count": 0,
            "operator_divergences_count": 0,
            "pending_audits_count": 0,
            "agreement_rate_pct": 100.0,
            "requires_retrain_count": 0,
            "latency_p95_ms": 0.0,
            "active_champion": str(self.settings.MODEL_PATH),
        }

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
        """Grava uma nova inferência no stream Parquet com escrita atômica thread-safe."""
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

        with self._lock:
            temp_file = self.telemetry_dir / f"tmp_inf_{uuid.uuid4().hex[:8]}.parquet"
            try:
                if self.telemetry_file.exists() and self.telemetry_file.stat().st_size > 0:
                    try:
                        existing_df = pl.read_parquet(self.telemetry_file)
                        combined_df = pl.concat([existing_df, new_df])
                    except BaseException as read_err:
                        logger.warning("Arquivo Parquet recuperado de fallback", error=str(read_err))
                        combined_df = new_df
                else:
                    combined_df = new_df

                combined_df.write_parquet(temp_file)
                temp_file.replace(self.telemetry_file)
            except BaseException as exc:
                logger.error("Erro ao gravar telemetria em Parquet", error=str(exc))
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)

    def record_feedback(
        self,
        inference_id: str,
        is_correct: bool,
        corrected_class: str | None = None,
    ) -> bool:
        """Atualiza a linha de telemetria com a validação humana de forma atômica."""
        if not self.telemetry_file.exists():
            return False

        with self._lock:
            temp_file = self.telemetry_dir / f"tmp_feed_{uuid.uuid4().hex[:8]}.parquet"
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
                updated_df.write_parquet(temp_file)
                temp_file.replace(self.telemetry_file)
                return True
            except BaseException as exc:
                logger.error("Erro ao atualizar feedback humano no Parquet", error=str(exc))
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
                return False

    def get_stats(self) -> dict[str, Any]:
        """Calcula estatísticas agregadas da esteira lendo o Parquet via Polars."""
        default_stats = self.get_default_stats()

        if not self.telemetry_file.exists():
            return default_stats

        with self._lock:
            try:
                if not self.telemetry_file.exists() or self.telemetry_file.stat().st_size == 0:
                    return default_stats

                df = pl.read_parquet(self.telemetry_file)
                total = df.height
                if total == 0:
                    return default_stats

                defective = df.filter(pl.col("is_defective")).height
                defect_rate = round((defective / total) * 100.0, 2)
                unknowns = df.filter(pl.col("is_unknown_anomaly")).height
                agreements = df.filter(pl.col("operator_confirmed") == True).height  # noqa: E712
                divergences = df.filter(pl.col("operator_confirmed") == False).height  # noqa: E712
                pending = total - (agreements + divergences)
                total_audited = agreements + divergences
                agreement_rate = round((agreements / total_audited) * 100.0, 1) if total_audited > 0 else 100.0
                retrain_count = df.filter(pl.col("requires_retrain") == True).height  # noqa: E712
                lat_p95 = round(float(df["inference_time_ms"].quantile(0.95)), 2)

                return {
                    "total_inspections": total,
                    "defective_count": defective,
                    "defect_rate_pct": defect_rate,
                    "unknown_anomalies_count": unknowns,
                    "operator_agreements_count": agreements,
                    "operator_divergences_count": divergences,
                    "pending_audits_count": pending,
                    "agreement_rate_pct": agreement_rate,
                    "requires_retrain_count": retrain_count,
                    "latency_p95_ms": lat_p95,
                    "active_champion": str(self.settings.MODEL_PATH),
                }
            except BaseException as exc:
                logger.error("Erro ao calcular estatísticas de telemetria", error=str(exc))
                return default_stats

    def get_recent_audits(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retorna os registros mais recentes de auditoria fabril e feedback do operador."""
        if not self.telemetry_file.exists():
            return []

        with self._lock:
            try:
                if not self.telemetry_file.exists() or self.telemetry_file.stat().st_size == 0:
                    return []

                df = pl.read_parquet(self.telemetry_file)
                if df.height == 0:
                    return []

                recent_df = df.tail(limit).reverse()
                records = []
                for row in recent_df.iter_rows(named=True):
                    op_confirmed = row.get("operator_confirmed")
                    if op_confirmed is True:
                        status_text = "CONFIRMADO"
                    elif op_confirmed is False:
                        status_text = "DIVERGENCIA"
                    else:
                        status_text = "PENDENTE"

                    records.append({
                        "inference_id": row.get("inference_id"),
                        "timestamp": row.get("timestamp"),
                        "prediction": row.get("prediction"),
                        "confidence": round(float(row.get("confidence", 0.0)), 4),
                        "is_defective": bool(row.get("is_defective", False)),
                        "is_unknown_anomaly": bool(row.get("is_unknown_anomaly", False)),
                        "anomaly_score": round(float(row.get("anomaly_score", 0.0)), 4),
                        "inference_time_ms": round(float(row.get("inference_time_ms", 0.0)), 2),
                        "is_ood": bool(row.get("is_ood", False)),
                        "operator_confirmed": op_confirmed,
                        "operator_corrected_class": row.get("operator_corrected_class"),
                        "requires_retrain": bool(row.get("requires_retrain", False)),
                        "status": status_text,
                    })
                return records
            except BaseException as exc:
                logger.error("Erro ao ler auditorias recentes do Parquet", error=str(exc))
                return []


telemetry_manager = TelemetryManager()

