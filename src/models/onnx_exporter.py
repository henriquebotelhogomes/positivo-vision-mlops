"""Compilação, Quantização e Otimização de Borda via ONNX Runtime.

Exporta modelos treinados em PyTorch para o padrão aberto ONNX,
aplica quantização dinâmica INT8 e executa benchmarks comparativos de latência.
"""

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic


def export_to_onnx(
    model: nn.Module,
    output_path: Path | str,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 18,
) -> Path:
    """Exporta um modelo PyTorch para o formato padronizado ONNX."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape, requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path.resolve()),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    # Validação estrutural do grafo exportado
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    return output_path


def quantize_onnx_model(
    input_model_path: Path | str,
    output_model_path: Path | str,
) -> Path:
    """Aplica quantização dinâmica INT8 com fallback para FP32 otimizado."""
    input_model_path = Path(input_model_path)
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        quantize_dynamic(
            model_input=str(input_model_path),
            model_output=str(output_model_path),
            weight_type=QuantType.QUInt8,
        )
    except Exception:
        # Fallback gracioso para ONNX FP32
        shutil.copy2(input_model_path, output_model_path)

    return output_model_path


def benchmark_inference(
    pytorch_model: nn.Module,
    onnx_model_path: Path | str,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    num_runs: int = 40,
    warmup_runs: int = 10,
) -> dict[str, Any]:
    """Compara o tempo de resposta entre PyTorch e ONNX Runtime em CPU."""
    pytorch_model.eval()
    dummy_tensor = torch.randn(*input_shape)
    dummy_numpy = dummy_tensor.numpy().astype(np.float32)

    # 1. Benchmark PyTorch (CPU)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = pytorch_model(dummy_tensor)

        torch_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = pytorch_model(dummy_tensor)
            torch_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 2. Benchmark ONNX Runtime (CPU)
    ort_options = ort.SessionOptions()
    ort_options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_model_path), ort_options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_numpy})

    onnx_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_numpy})
        onnx_latencies.append((time.perf_counter() - t0) * 1000.0)

    torch_p50 = float(np.percentile(torch_latencies, 50))
    onnx_p50 = float(np.percentile(onnx_latencies, 50))
    speedup = round(torch_p50 / max(onnx_p50, 1e-4), 2)

    return {
        "pytorch_latency_p50_ms": round(torch_p50, 2),
        "pytorch_latency_p95_ms": round(float(np.percentile(torch_latencies, 95)), 2),
        "onnx_latency_p50_ms": round(onnx_p50, 2),
        "onnx_latency_p95_ms": round(float(np.percentile(onnx_latencies, 95)), 2),
        "speedup_factor": speedup,
        "onnx_throughput_fps": round(1000.0 / max(onnx_p50, 1e-4), 1),
    }
