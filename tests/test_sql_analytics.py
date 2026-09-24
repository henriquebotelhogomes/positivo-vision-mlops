"""Testes unitários determinísticos para o motor analítico DuckDB (SQL-on-Parquet)."""

from pathlib import Path

import polars as pl
import pytest

from src.monitoring.sql_analytics import IndustrialSQLAnalytics


@pytest.fixture
def mock_telemetry_parquet(tmp_path: Path) -> Path:
    """Cria um arquivo Parquet temporário com dados industriais para testes SQL."""
    parquet_path = tmp_path / "inferences.parquet"

    data = {
        "inference_id": [f"test-{i}" for i in range(10)],
        "timestamp": ["2026-09-24T00:00:00Z"] * 10,
        "prediction": [
            "NORMAL", "NORMAL", "NORMAL", "NORMAL", "NORMAL",
            "DEFECT_SHORT", "DEFECT_SHORT", "DEFECT_OPEN", "DEFECT_MISSING_HOLE", "NORMAL"
        ],
        "confidence": [0.95, 0.98, 0.92, 0.99, 0.91, 0.88, 0.94, 0.85, 0.78, 0.96],
        "is_defective": [False, False, False, False, False, True, True, True, True, False],
        "is_unknown_anomaly": [False] * 10,
        "anomaly_score": [0.1] * 10,
        "inference_time_ms": [15.2, 14.8, 16.1, 15.0, 14.5, 18.2, 19.0, 17.5, 22.0, 14.9],
        "is_ood": [False] * 10,
        "operator_confirmed": [True, True, True, None, None, True, False, True, None, True],
        "operator_corrected_class": [None, None, None, None, None, None, "DEFECT_SPUR", None, None, None],
        "requires_retrain": [False, False, False, False, False, False, True, False, False, False],
    }

    df = pl.DataFrame(data)
    df.write_parquet(parquet_path)
    return parquet_path


def test_sql_ppm_report(mock_telemetry_parquet: Path):
    analytics = IndustrialSQLAnalytics(parquet_path=mock_telemetry_parquet)
    ppm_report = analytics.get_industrial_ppm_report()

    assert len(ppm_report) > 0
    # Verifica cálculo de PPM
    short_row = next((r for r in ppm_report if r["defect_class"] == "DEFECT_SHORT"), None)
    assert short_row is not None
    assert short_row["defective_units"] == 2
    assert short_row["ppm"] == 200000.0  # 2 defeitos em 10 total = 200.000 PPM


def test_sql_latency_percentiles(mock_telemetry_parquet: Path):
    analytics = IndustrialSQLAnalytics(parquet_path=mock_telemetry_parquet)
    latency_stats = analytics.get_latency_sla_percentiles()

    assert len(latency_stats) > 0
    normal_row = next((r for r in latency_stats if r["defect_class"] == "NORMAL"), None)
    assert normal_row is not None
    assert normal_row["total_inspections"] == 6
    assert 14.0 <= normal_row["p50_latency_ms"] <= 17.0


def test_sql_hitl_audit(mock_telemetry_parquet: Path):
    analytics = IndustrialSQLAnalytics(parquet_path=mock_telemetry_parquet)
    hitl_audit = analytics.get_hitl_operator_audit()

    assert hitl_audit["total_audited"] == 7
    assert hitl_audit["confirmed_by_operator"] == 6
    assert hitl_audit["diverged_by_operator"] == 1
    assert hitl_audit["operator_agreement_pct"] == pytest.approx(85.71, 0.1)
    assert hitl_audit["queued_for_retraining"] == 1
