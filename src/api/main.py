"""Microsserviço de Inferência em Alta Performance e Demonstração Industrial.

Construído com FastAPI assíncrono, ciclo de vida estritamente gerenciado
via lifespan context manager, documentação moderna via Scalar OpenAPI
e aceleração de borda com ONNX Runtime.
"""

import random
import time
import uuid
import zipfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

from src.api.rate_limiter import rate_limiter
from src.api.schemas import (
    AuditRecord,
    AuditsResponse,
    FeedbackRequest,
    FeedbackResponse,
    PredictionResponse,
    RandomSampleResponse,
    TechnicalReportRequest,
    TechnicalReportResponse,
    TelemetryStatsResponse,
)
from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.data.transforms import check_out_of_distribution, get_inference_transforms
from src.models.anomaly_head import OpenSetAnomalyDetector
from src.models.gradcam import GradCAM
from src.models.vision_net import CLASS_NAMES, build_model
from src.monitoring.telemetry import telemetry_manager

settings = get_settings()
logger = get_logger("positivo_vision.api")


def load_model_engine(app_state: Any) -> None:
    """Carrega o motor ONNX, modelo PyTorch, Grad-CAM e Anomaly Head no app_state."""
    models_dir = Path("models")
    model_path = settings.resolved_model_path

    if not model_path.exists():
        candidate_path = models_dir / "candidate_quantized.onnx"
        if candidate_path.exists():
            model_path = candidate_path

    # 1. Carregamento da Sessão ONNX Runtime
    if model_path.exists():
        ort_opts = ort.SessionOptions()
        ort_opts.intra_op_num_threads = 4
        app_state.onnx_session = ort.InferenceSession(
            str(model_path), ort_opts, providers=["CPUExecutionProvider"]
        )
        app_state.onnx_input_name = app_state.onnx_session.get_inputs()[0].name
        logger.info("ONNX Runtime Engine carregado com sucesso", path=str(model_path))
    else:
        app_state.onnx_session = None
        app_state.onnx_input_name = None
        logger.warning("Arquivo do modelo ONNX não encontrado", path=str(model_path))

    # 2. Carregamento do PyTorch Model para Grad-CAM e Embeddings
    device = torch.device("cpu")
    pytorch_model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    pt_path = models_dir / "best_model.pt"
    if pt_path.exists():
        try:
            pytorch_model.load_state_dict(torch.load(str(pt_path), map_location=device))
        except Exception:
            pass
    pytorch_model.to(device)
    pytorch_model.eval()

    app_state.pytorch_model = pytorch_model
    app_state.gradcam = GradCAM(model=pytorch_model, target_layer=pytorch_model.get_last_conv_layer())

    # 3. Carregamento do Open-Set Anomaly Detector
    anomaly_detector = OpenSetAnomalyDetector(threshold=settings.ANOMALY_LATENT_THRESHOLD)
    centroid_path = models_dir / "anomaly_centroid.npz"
    if centroid_path.exists():
        anomaly_detector.load(centroid_path)
        logger.info("Open-Set Anomaly Head carregada", threshold=anomaly_detector.threshold)
    app_state.anomaly_detector = anomaly_detector

    # 4. Warm-up
    if getattr(app_state, "onnx_session", None) is not None:
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        _ = app_state.onnx_session.run(None, {app_state.onnx_input_name: dummy})
        logger.info("Warm-up concluído: latência de cold-start eliminada")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gerencia o ciclo de vida e estado em memória do modelo ONNX e Grad-CAM (Fail-Fast)."""
    configure_logging(log_level=settings.LOG_LEVEL, json_format=settings.is_production)
    logger.info("Inicializando Positivo Vision MLOps Engine", env=settings.APP_ENV)

    load_model_engine(app.state)

    yield

    logger.info("Encerrando Positivo Vision MLOps Engine")


# Instância Principal do FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    description="Plataforma Industrial de Inspeção Visual de PCBs, Otimização de Borda & Governança de Modelos.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # Desativa Swagger clássico para servir Scalar obrigatório
    redoc_url=None,
)

# Inicialização de atributos padrão no app.state
app.state.onnx_session = None
app.state.onnx_input_name = None
app.state.pytorch_model = None
app.state.gradcam = None
app.state.anomaly_detector = None


# Montagem do diretório de assets estáticos (Logos e Favicon)
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """Serve o favicon oficial da Positivo Tecnologia."""
    favicon_file = STATIC_DIR / "favicon.png"
    if favicon_file.exists():
        return FileResponse(str(favicon_file), media_type="image/png")
    raise HTTPException(status_code=404, detail="Favicon não encontrado")


@app.get("/docs", include_in_schema=False)
async def scalar_docs() -> HTMLResponse:
    """Documentação Interativa de API servida via Scalar OpenAPI (Regra Normativa)."""
    html_content = """
    <!doctype html>
    <html>
      <head>
        <title>Positivo Vision MLOps — Scalar Docs</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body>
        <script id="api-reference" data-url="/openapi.json"></script>
        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/healthz", tags=["Orchestration"])
async def liveness_probe() -> dict[str, str]:
    """Liveness probe para Kubernetes e Google Cloud Run."""
    return {"status": "healthy"}


@app.get("/ready", tags=["Orchestration"])
async def readiness_probe() -> dict[str, str]:
    """Readiness probe confirmando que o modelo ONNX está carregado em memória."""
    if getattr(app.state, "onnx_session", None) is None:
        load_model_engine(app.state)
    if app.state.onnx_session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo ONNX de inferência ainda não aquecido na memória.",
        )
    return {"status": "ready"}


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo_interface() -> HTMLResponse:
    """Interface Web Executiva da Live Demo Industrial."""
    demo_file = Path("src/api/templates/demo.html")
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Template da Live Demo não encontrado.")
    return HTMLResponse(content=demo_file.read_text(encoding="utf-8"))


def _run_sliding_window_inference(
    pil_image: Image.Image,
    app_state: Any,
) -> dict[str, Any]:
    """Executa varredura por blocos ópticos (Sliding Window / Tiling) em imagens de alta resolução."""
    if getattr(app_state, "onnx_session", None) is None:
        load_model_engine(app_state)

    inference_id = f"inf_{uuid.uuid4().hex[:8]}"
    ood_result = check_out_of_distribution(pil_image)

    w, h = pil_image.size
    tile_size = 224
    # Stride adaptativo para manter acurácia e latência rápida
    stride = 180 if max(w, h) > 2200 else (140 if max(w, h) > 1000 else 112)

    x_coords = sorted(list(set(list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)])))
    y_coords = sorted(list(set(list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)])))

    transform = get_inference_transforms(img_size=tile_size)
    tiles_pil = []
    tiles_tensors = []
    coords = []

    for y in y_coords:
        for x in x_coords:
            crop = pil_image.crop((x, y, x + tile_size, y + tile_size))
            tiles_pil.append(crop)
            tiles_tensors.append(transform(crop))
            coords.append((x, y))

    batch = torch.stack(tiles_tensors).numpy().astype(np.float32)

    t0 = time.perf_counter()
    if app_state.onnx_session is not None:
        outputs = app_state.onnx_session.run(None, {app_state.onnx_input_name: batch})
        logits = outputs[0]
    else:
        with torch.no_grad():
            logits = app_state.pytorch_model(torch.tensor(batch)).numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    defect_candidates = []
    for i in range(len(tiles_pil)):
        p = probs[i]
        cls_idx = int(np.argmax(p))
        cls_name = CLASS_NAMES[cls_idx]
        if cls_name != "NORMAL" and p[cls_idx] >= 0.40:
            defect_candidates.append({
                "tile_idx": i,
                "class": cls_name,
                "confidence": float(p[cls_idx]),
                "probs": p,
                "coord": coords[i],
            })

    if defect_candidates:
        # Prioriza missing hole se detectado, ou o defeito com maior confiança geral
        missing_holes = [c for c in defect_candidates if c["class"] == "DEFECT_MISSING_HOLE"]
        if missing_holes:
            best = max(missing_holes, key=lambda c: c["confidence"])
        else:
            best = max(defect_candidates, key=lambda c: c["confidence"])

        pred_class = best["class"]
        confidence = best["confidence"]
        prob_dict = {CLASS_NAMES[k]: round(float(best["probs"][k]), 4) for k in range(len(CLASS_NAMES))}
        is_defective = True
        best_idx = best["tile_idx"]
        winning_tile = tiles_pil[best_idx]
        winning_coord = best["coord"]

        # 1. Grad-CAM focado no bloco onde o defeito físico reside
        winning_tensor = tiles_tensors[best_idx].unsqueeze(0)
        pred_idx = CLASS_NAMES.index(pred_class)
        try:
            heatmap = app_state.gradcam.generate_heatmap(winning_tensor, class_idx=pred_idx)
            overlay_pil = app_state.gradcam.overlay_on_image(winning_tile, heatmap, alpha=0.65, threshold=0.25)
            gradcam_base64 = GradCAM.pil_to_base64(overlay_pil, format="JPEG")
        except Exception as exc:
            logger.warning("Falha ao gerar Grad-CAM no tile", error=str(exc))
            gradcam_base64 = GradCAM.pil_to_base64(winning_tile, format="JPEG")

        # 2. Imagem macro anotada com mira/retículo vermelho no local do defeito
        macro_annotated = pil_image.copy()
        draw = ImageDraw.Draw(macro_annotated)
        bx, by = winning_coord
        draw.rectangle([bx, by, bx + tile_size, by + tile_size], outline="red", width=8)
        original_base64 = GradCAM.pil_to_base64(macro_annotated, format="JPEG")

    else:
        pred_class = "NORMAL"
        mean_probs = np.mean(probs, axis=0)
        confidence = float(mean_probs[0])
        prob_dict = {CLASS_NAMES[k]: round(float(mean_probs[k]), 4) for k in range(len(CLASS_NAMES))}
        is_defective = False
        original_base64 = GradCAM.pil_to_base64(pil_image, format="JPEG")

        center_idx = len(tiles_pil) // 2
        center_tile = tiles_pil[center_idx]
        center_tensor = tiles_tensors[center_idx].unsqueeze(0)
        try:
            heatmap = app_state.gradcam.generate_heatmap(center_tensor, class_idx=0)
            overlay_pil = app_state.gradcam.overlay_on_image(center_tile, heatmap, alpha=0.65, threshold=0.25)
            gradcam_base64 = GradCAM.pil_to_base64(overlay_pil, format="JPEG")
        except Exception:
            gradcam_base64 = GradCAM.pil_to_base64(center_tile, format="JPEG")

    return {
        "inference_id": inference_id,
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "is_defective": is_defective,
        "is_unknown_anomaly": False,
        "anomaly_score": 0.0,
        "probabilities": prob_dict,
        "inference_time_ms": round(latency_ms, 2),
        "engine": "onnxruntime-cpu (tiled)" if app_state.onnx_session else "pytorch-cpu",
        "model_version": "v1",
        "model_alias": settings.MODEL_ALIAS,
        "gradcam_base64": gradcam_base64,
        "original_image_base64": original_base64,
        "ood_flags": ood_result,
    }


def _run_core_inference(
    pil_image: Image.Image,
    app_state: Any,
) -> dict[str, Any]:
    """Executa a triagem OOD, inferência ONNX, Anomaly Head e Grad-CAM em uma imagem PIL."""
    if getattr(app_state, "onnx_session", None) is None:
        load_model_engine(app_state)

    # Para placas macro de alta resolução, ativa o Tiling Scanner
    w, h = pil_image.size
    if w > 320 or h > 320:
        return _run_sliding_window_inference(pil_image, app_state)

    inference_id = f"inf_{uuid.uuid4().hex[:8]}"

    # 1. Triagem OOD
    ood_result = check_out_of_distribution(pil_image)

    # 2. Pré-processamento
    transform = get_inference_transforms(img_size=224)
    input_tensor = transform(pil_image).unsqueeze(0)  # Shape (1, 3, 224, 224)
    input_numpy = input_tensor.numpy().astype(np.float32)

    # 3. Inferência ONNX Runtime (< 25ms)
    t0 = time.perf_counter()
    if app_state.onnx_session is not None:
        outputs = app_state.onnx_session.run(None, {app_state.onnx_input_name: input_numpy})
        logits = outputs[0][0]
    else:
        # Fallback com PyTorch caso o ONNX não tenha subido
        with torch.no_grad():
            logits = app_state.pytorch_model(input_tensor).squeeze(0).numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Softmax das probabilidades
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}
    is_defective = (pred_class != "NORMAL")

    # 4. Open-Set Anomaly Head
    anomaly_result = {"is_unknown_anomaly": False, "anomaly_score": 0.0}
    if hasattr(app_state, "anomaly_detector") and app_state.anomaly_detector is not None:
        with torch.no_grad():
            embedding = app_state.pytorch_model.extract_embedding(input_tensor)
        anomaly_result = app_state.anomaly_detector.score(embedding)

    is_unknown = anomaly_result.get("is_unknown_anomaly", False)
    anomaly_score = anomaly_result.get("anomaly_score", 0.0)

    # Se for detectada anomalia inédita, sinaliza defeito
    if is_unknown:
        is_defective = True

    # 5. Grad-CAM Explicabilidade Visual
    try:
        heatmap = app_state.gradcam.generate_heatmap(input_tensor, class_idx=pred_idx)
        overlay_pil = app_state.gradcam.overlay_on_image(pil_image, heatmap, alpha=0.65, threshold=0.25)
        gradcam_base64 = GradCAM.pil_to_base64(overlay_pil, format="JPEG")
    except Exception as exc:
        logger.warning("Falha ao gerar Grad-CAM", error=str(exc))
        gradcam_base64 = GradCAM.pil_to_base64(pil_image, format="JPEG")

    return {
        "inference_id": inference_id,
        "prediction": pred_class,
        "confidence": round(confidence, 4),
        "is_defective": is_defective,
        "is_unknown_anomaly": is_unknown,
        "anomaly_score": anomaly_score,
        "probabilities": prob_dict,
        "inference_time_ms": round(latency_ms, 2),
        "engine": "onnxruntime-cpu" if app_state.onnx_session else "pytorch-cpu",
        "model_version": "v1",
        "model_alias": settings.MODEL_ALIAS,
        "gradcam_base64": gradcam_base64,
        "original_image_base64": GradCAM.pil_to_base64(pil_image, format="JPEG"),
        "ood_flags": ood_result,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> PredictionResponse:
    """Executa a inspeção visual e explicabilidade com Grad-CAM via upload de imagem."""
    # Rate Limiting Guardrail
    rate_limiter.check_inference_limit(request)

    # Leitura e decodificação segura
    try:
        contents = await file.read()
        pil_img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo inválido. Envie uma imagem JPEG ou PNG.")

    result = _run_core_inference(pil_img, app.state)

    # Gravação assíncrona da telemetria em Parquet
    background_tasks.add_task(
        telemetry_manager.record_inference,
        inference_id=result["inference_id"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        is_defective=result["is_defective"],
        is_unknown_anomaly=result["is_unknown_anomaly"],
        anomaly_score=result["anomaly_score"],
        inference_time_ms=result["inference_time_ms"],
        is_ood=result["ood_flags"]["is_ood"],
    )

    return PredictionResponse(**result)


@app.get("/api/v1/predict-random", response_model=RandomSampleResponse, tags=["Inference"])
async def predict_random_sample(
    request: Request,
    background_tasks: BackgroundTasks,
    category: str = "any",
) -> RandomSampleResponse:
    """Sorteia instantaneamente uma amostra industrial do pool interno para a Live Demo em 1-clique.

    Categorias suportadas: 'any' (balanceado 50% normal / 50% defeito), 'normal', 'defect', 'unknown'.
    """
    # Rate Limiting Guardrail
    rate_limiter.check_inference_limit(request)

    pool_dir = settings.resolved_sample_pool_dir
    cat_lower = category.lower().strip()

    if cat_lower == "normal":
        candidate_dirs = [pool_dir / "normal"]
    elif (pool_dir / cat_lower).exists() and (pool_dir / cat_lower).is_dir():
        candidate_dirs = [pool_dir / cat_lower]
    elif cat_lower == "defect":
        candidate_dirs = [
            pool_dir / "defect_short",
            pool_dir / "defect_open",
            pool_dir / "defect_missing_hole",
            pool_dir / "defect_mousebite",
            pool_dir / "defect_spur",
            pool_dir / "defect_spurious_copper",
        ]
    elif cat_lower == "unknown":
        candidate_dirs = [pool_dir / "unknown"]
    else:
        # Sorteio balanceado: 50% chance de placa CONFORME e 50% chance de defeito/anomalia
        if random.random() < 0.5:
            candidate_dirs = [pool_dir / "normal"]
        else:
            candidate_dirs = [
                pool_dir / "defect_short",
                pool_dir / "defect_open",
                pool_dir / "defect_missing_hole",
                pool_dir / "defect_mousebite",
                pool_dir / "defect_spur",
                pool_dir / "defect_spurious_copper",
                pool_dir / "unknown",
            ]

    all_samples = []
    for d in candidate_dirs:
        if d.exists():
            all_samples.extend(list(d.glob("*.png")) + list(d.glob("*.jpg")))

    # Fallback geral se a categoria escolhida estiver vazia
    if not all_samples:
        all_samples = list(pool_dir.glob("*/*.png")) + list(pool_dir.glob("*/*.jpg"))

    if not all_samples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma amostra encontrada em data/sample_pool/.",
        )

    # Sorteio uniforme da esteira
    chosen_path = random.choice(all_samples)
    try:
        pil_img = Image.open(chosen_path).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao abrir amostra: {exc}")

    result = _run_core_inference(pil_img, app.state)
    original_base64 = GradCAM.pil_to_base64(pil_img, format="JPEG")

    # Telemetria assíncrona
    background_tasks.add_task(
        telemetry_manager.record_inference,
        inference_id=result["inference_id"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        is_defective=result["is_defective"],
        is_unknown_anomaly=result["is_unknown_anomaly"],
        anomaly_score=result["anomaly_score"],
        inference_time_ms=result["inference_time_ms"],
        is_ood=result["ood_flags"]["is_ood"],
    )

    result["original_image_base64"] = original_base64
    return RandomSampleResponse(
        sample_name=chosen_path.name,
        **result,
    )


@app.post("/api/v1/generate-report", response_model=TechnicalReportResponse, tags=["GenAI"])
async def generate_report_endpoint(
    request: Request,
    payload: TechnicalReportRequest,
) -> TechnicalReportResponse:
    """Gera o Laudo Técnico de Causa-Raiz SMT via LangGraph (StateGraph) & DeepSeek V4.1."""
    # Rate Limiting para rotas de LLM
    rate_limiter.check_report_limit(request)

    from src.models.smt_graph_agent import SMTGraphAgent

    graph_agent = SMTGraphAgent()
    graph_res = graph_agent.generate_report(
        inference_id=payload.inference_id,
        prediction=payload.prediction,
        confidence=payload.confidence,
        is_unknown_anomaly=payload.is_unknown_anomaly,
    )

    return TechnicalReportResponse(
        inference_id=payload.inference_id,
        source="LangGraph SMT StateGraph",
        model=graph_res["model_source"],
        ipc_standard="IPC-A-610 Class 3",
        report_markdown=graph_res["report_markdown"],
    )


@app.post("/api/v1/feedback", response_model=FeedbackResponse, tags=["Active Learning"])
async def record_hitl_feedback(
    payload: FeedbackRequest,
) -> FeedbackResponse:
    """Registra validação ou correção do técnico de bancada no stream Parquet (Active Learning)."""
    success = telemetry_manager.record_feedback(
        inference_id=payload.inference_id,
        is_correct=payload.is_correct,
        corrected_class=payload.corrected_class,
    )

    if not success:
        logger.warning("Feedback recebido para ID não persistido ou arquivo inexistente", id=payload.inference_id)

    return FeedbackResponse(
        inference_id=payload.inference_id,
        requires_retrain=not payload.is_correct,
    )


@app.get("/api/v1/telemetry/stats", response_model=TelemetryStatsResponse, tags=["Telemetry"])
async def get_telemetry_stats() -> TelemetryStatsResponse:
    """Retorna estatísticas operacionais agregadas da esteira SMT lidas do stream Parquet."""
    try:
        stats = telemetry_manager.get_stats()
        return TelemetryStatsResponse(**stats)
    except BaseException as exc:
        logger.warning("Falha ao obter telemetria, retornando fallback seguro", error=str(exc))
        return TelemetryStatsResponse(**telemetry_manager.get_default_stats())


@app.get("/api/v1/telemetry/audits", response_model=AuditsResponse, tags=["Telemetry"])
async def get_recent_audits(limit: int = 50) -> AuditsResponse:
    """Retorna o histórico detalhado de auditorias e validações humanas da esteira."""
    try:
        stats = telemetry_manager.get_stats()
        records = telemetry_manager.get_recent_audits(limit=limit)
        return AuditsResponse(
            total_returned=len(records),
            stats=TelemetryStatsResponse(**stats),
            records=[AuditRecord(**r) for r in records],
        )
    except BaseException as exc:
        logger.warning("Falha ao obter auditorias, retornando fallback seguro", error=str(exc))
        return AuditsResponse(
            total_returned=0,
            stats=TelemetryStatsResponse(**telemetry_manager.get_default_stats()),
            records=[],
        )


@app.get("/api/v1/telemetry/sql-metrics", tags=["Telemetry"])
async def get_telemetry_sql_metrics() -> dict[str, Any]:
    """Executa consultas analíticas ANSI SQL via DuckDB diretamente sobre o lago Parquet.

    Calcula métricas industriais de PPM (Partes Por Milhão), percentis de latência e auditoria HITL.
    """
    from src.monitoring.sql_analytics import IndustrialSQLAnalytics

    analytics = IndustrialSQLAnalytics()
    return analytics.get_consolidated_sql_summary()


@app.get("/api/v1/download-samples", tags=["Dataset"])
async def download_samples_zip() -> StreamingResponse:
    """Gera e faz download de um pacote ZIP com amostras da linha SMT para testes manuais de upload."""
    pool_dir = settings.resolved_sample_pool_dir
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        readme_content = (
            "POSITIVO TECNOLOGIA - PACOTE DE AMOSTRAS DE TESTE (AOI SMT)\n"
            "===========================================================\n\n"
            "Este arquivo contém amostras industriais balanceadas para teste do sistema Positivo Vision.\n"
            "Você pode fazer upload de qualquer uma destas imagens na interface Web (/demo):\n\n"
            "Pastas incluídas:\n"
            "  - normal/              : Placas conformes (sem defeito)\n"
            "  - defect_short/        : Placas com curto-circuito de solda\n"
            "  - defect_open/         : Placas com trilha rompida ou circuito aberto\n"
            "  - defect_missing_hole/ : Placas com furo de via/passagem ausente\n"
            "  - defect_spurious/     : Placas com rebarba/cobre espúrio\n"
            "  - unknown/             : Placas com defeitos anômalos inéditos (Open-Set Anomaly)\n\n"
            "Dataset de Referência Industrial:\n"
            "  DeepPCB Dataset (Peking University / PKU)\n"
            "  GitHub: https://github.com/Charmve/Surface-Defect-Detection/tree/master/DeepPCB\n"
            "  Kaggle: https://www.kaggle.com/datasets/akhatova/pcb-defects\n"
        )
        zf.writestr("LEIAME.txt", readme_content)

        categories = ["normal", "defect_short", "defect_open", "defect_missing_hole", "defect_spurious", "unknown"]
        for cat in categories:
            cat_dir = pool_dir / cat
            if cat_dir.exists():
                imgs = sorted(list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")))[:3]
                for img_p in imgs:
                    zf.write(img_p, arcname=f"{cat}/{img_p.name}")

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=positivo_vision_amostras_teste.zip"},
    )

