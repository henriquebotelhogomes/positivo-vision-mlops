"""Testes de integração assíncronos para os endpoints da API FastAPI."""

from io import BytesIO

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from src.api.main import app


@pytest.mark.asyncio
async def test_healthz_and_ready_probes():
    """Valida as sondas de liveness e readiness para orquestração."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Liveness
        res_live = await client.get("/healthz")
        assert res_live.status_code == 200
        assert res_live.json() == {"status": "healthy"}

        # Readiness
        res_ready = await client.get("/ready")
        assert res_ready.status_code in (200, 503)


@pytest.mark.asyncio
async def test_scalar_docs_endpoint():
    """Valida que a documentação servida é o Scalar OpenAPI e não o Swagger clássico."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/docs")
        assert res.status_code == 200
        assert "@scalar/api-reference" in res.text


@pytest.mark.asyncio
async def test_predict_random_sample():
    """Valida o sorteio de amostras industriais em 1-clique para a Live Demo."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/predict-random", headers={"X-Admin-Bypass": "positivo_audit_bypass_key_2026"})
        assert res.status_code == 200
        data = res.json()

        assert data["status"] == "success"
        assert "inference_id" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "inference_time_ms" in data
        assert "gradcam_base64" in data
        assert "original_image_base64" in data
        assert data["inference_time_ms"] > 0


@pytest.mark.asyncio
async def test_predict_manual_upload():
    """Valida a inferência via upload direto de arquivo de imagem."""
    # Cria uma imagem temporária em memória
    img = Image.new("RGB", (224, 224), color=(20, 75, 40))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test_pcb.png", buf, "image/png")}
        res = await client.post(
            "/api/v1/predict",
            files=files,
            headers={"X-Admin-Bypass": "positivo_audit_bypass_key_2026"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["prediction"] in ["NORMAL", "DEFECT_SHORT", "DEFECT_OPEN", "DEFECT_MISSING_HOLE", "DEFECT_SPURIOUS"]


@pytest.mark.asyncio
async def test_generate_technical_report():
    """Valida a geração de laudo técnico SMT com fallback estruturado."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "inference_id": "inf_test123",
            "prediction": "DEFECT_SHORT",
            "confidence": 0.985,
            "is_unknown_anomaly": False,
            "latency_ms": 14.2,
        }
        res = await client.post(
            "/api/v1/generate-report",
            json=payload,
            headers={"X-Admin-Bypass": "positivo_audit_bypass_key_2026"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "IPC-A-610" in data["ipc_standard"]
        assert len(data["report_markdown"]) > 50


@pytest.mark.asyncio
async def test_feedback_and_telemetry_flow():
    """Valida o fluxo Human-in-the-Loop gravando no stream Parquet."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Envia feedback
        feedback_payload = {
            "inference_id": "inf_hitl_test",
            "is_correct": True,
            "operator_comments": "Validação em bancada de teste",
        }
        res_fb = await client.post("/api/v1/feedback", json=feedback_payload)
        assert res_fb.status_code == 200

        # Consulta telemetria
        res_stats = await client.get("/api/v1/telemetry/stats")
        assert res_stats.status_code == 200
        stats = res_stats.json()
        assert "total_inspections" in stats
        assert "defect_rate_pct" in stats


@pytest.mark.asyncio
async def test_telemetry_audits_endpoint():
    """Valida o endpoint de auditorias detalhadas para o painel HITL."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/telemetry/audits?limit=10")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "stats" in data
        assert "records" in data
        assert isinstance(data["records"], list)


@pytest.mark.asyncio
async def test_download_samples_zip_endpoint():
    """Valida a geração e download do pacote ZIP de amostras de teste."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/download-samples")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/zip"
        assert len(res.content) > 1000  # Pacote com imagens reais


@pytest.mark.asyncio
async def test_predict_random_with_category_normal():
    """Valida o sorteio com filtro específico de categoria (ex: normal)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/predict-random?category=normal")
        assert res.status_code == 200
        data = res.json()
        assert "sample_name" in data
        assert data["sample_name"].startswith("sample_normal")

