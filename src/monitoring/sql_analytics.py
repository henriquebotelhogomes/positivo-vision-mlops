"""Motor Analítico SQL (DuckDB sobre Parquet): Consultas Industriais de Alta Performance.

Permite executar consultas analíticas ANSI SQL em microssegundos diretamente sobre o lago
colunar de telemetria da esteira fabril (SQL-on-Parquet), sem lock de banco e com zero overhead.
"""

from pathlib import Path
from typing import Any

import duckdb
import structlog

from src.core.config import get_settings

logger = structlog.get_logger("positivo_vision.sql_analytics")


class IndustrialSQLAnalytics:
    """Executa queries analíticas SQL industriais via DuckDB sobre arquivos Parquet."""

    def __init__(self, parquet_path: Path | None = None) -> None:
        self.settings = get_settings()
        self.parquet_path = parquet_path or (self.settings.resolved_telemetry_dir / "inferences.parquet")

    def _has_data(self) -> bool:
        """Verifica se o arquivo Parquet existe e possui dados."""
        return self.parquet_path.exists() and self.parquet_path.stat().st_size > 500

    def get_industrial_ppm_report(self) -> list[dict[str, Any]]:
        """Calcula o índice de Defeitos por Partes Por Milhão (PPM) via Window Functions SQL.

        Métrica clássica de manufatura de alta precisão (Seis Sigma / Linhas SMT).
        """
        if not self._has_data():
            return []

        query = """
        WITH class_stats AS (
            SELECT
                prediction AS defect_class,
                COUNT(*) AS total_samples,
                SUM(CASE WHEN is_defective THEN 1 ELSE 0 END) AS defective_units
            FROM read_parquet(?)
            GROUP BY prediction
        ),
        grand_total AS (
            SELECT SUM(total_samples) AS total_inspected FROM class_stats
        )
        SELECT
            defect_class,
            total_samples,
            defective_units,
            ROUND(CAST(defective_units AS DOUBLE) / NULLIF(total_samples, 0) * 100.0, 2) AS defect_rate_pct,
            ROUND(CAST(defective_units AS DOUBLE) / NULLIF((SELECT total_inspected FROM grand_total), 0) * 1000000.0, 1) AS ppm
        FROM class_stats
        ORDER BY defective_units DESC, total_samples DESC;
        """

        try:
            con = duckdb.connect(database=":memory:")
            result = con.execute(query, [str(self.parquet_path)]).fetchdf()
            con.close()
            return result.to_dict(orient="records")
        except Exception as exc:
            logger.error("Erro ao calcular PPM industrial via DuckDB", error=str(exc))
            return []

    def get_latency_sla_percentiles(self) -> list[dict[str, Any]]:
        """Calcula percentis de latência (P50, P90, P95, P99) agrupados por classe via SQL."""
        if not self._has_data():
            return []

        query = """
        SELECT
            prediction AS defect_class,
            COUNT(*) AS total_inspections,
            ROUND(quantile_cont(inference_time_ms, 0.50), 2) AS p50_latency_ms,
            ROUND(quantile_cont(inference_time_ms, 0.90), 2) AS p90_latency_ms,
            ROUND(quantile_cont(inference_time_ms, 0.95), 2) AS p95_latency_ms,
            ROUND(quantile_cont(inference_time_ms, 0.99), 2) AS p99_latency_ms,
            ROUND(AVG(inference_time_ms), 2) AS avg_latency_ms,
            ROUND(MAX(inference_time_ms), 2) AS max_latency_ms
        FROM read_parquet(?)
        GROUP BY prediction
        ORDER BY total_inspections DESC;
        """

        try:
            con = duckdb.connect(database=":memory:")
            result = con.execute(query, [str(self.parquet_path)]).fetchdf()
            con.close()
            return result.to_dict(orient="records")
        except Exception as exc:
            logger.error("Erro ao calcular percentis de latencia via DuckDB", error=str(exc))
            return []

    def get_hitl_operator_audit(self) -> dict[str, Any]:
        """Calcula a taxa de concordância do operador da esteira (Human-in-the-Loop) via SQL."""
        if not self._has_data():
            return {
                "total_audited": 0,
                "confirmed_by_operator": 0,
                "diverged_by_operator": 0,
                "operator_agreement_pct": 100.0,
                "queued_for_retraining": 0,
            }

        query = """
        SELECT
            COUNT(*) AS total_audited,
            COALESCE(SUM(CASE WHEN operator_confirmed = true THEN 1 ELSE 0 END), 0) AS confirmed_by_operator,
            COALESCE(SUM(CASE WHEN operator_confirmed = false THEN 1 ELSE 0 END), 0) AS diverged_by_operator,
            ROUND(
                CAST(COALESCE(SUM(CASE WHEN operator_confirmed = true THEN 1 ELSE 0 END), 0) AS DOUBLE) /
                NULLIF(COUNT(*), 0) * 100.0,
                2
            ) AS operator_agreement_pct,
            COALESCE(SUM(CASE WHEN requires_retrain = true THEN 1 ELSE 0 END), 0) AS queued_for_retraining
        FROM read_parquet(?)
        WHERE operator_confirmed IS NOT NULL;
        """

        try:
            con = duckdb.connect(database=":memory:")
            result = con.execute(query, [str(self.parquet_path)]).fetchdf()
            con.close()
            records = result.to_dict(orient="records")
            return records[0] if records else {}
        except Exception as exc:
            logger.error("Erro ao auditar HITL via DuckDB", error=str(exc))
            return {}

    def get_consolidated_sql_summary(self) -> dict[str, Any]:
        """Retorna sumário executivo com todas as análises industriais via SQL."""
        return {
            "engine": "DuckDB ANSI SQL-on-Parquet",
            "source_parquet": str(self.parquet_path),
            "ppm_report": self.get_industrial_ppm_report(),
            "latency_sla": self.get_latency_sla_percentiles(),
            "operator_hitl_audit": self.get_hitl_operator_audit(),
        }
