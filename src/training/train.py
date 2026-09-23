"""Pipeline de Treinamento, Governança MLflow e Otimização de Borda.

Executa o ciclo de vida completo:
1. Ingestão e DataLoaders de imagens industriais.
2. Fine-tuning com Transfer Learning na CPU.
3. Avaliação em validação out-of-sample e cálculo de métricas industriais.
4. Matriz de confusão e calibração da Anomaly Head.
5. Otimização de limiar sensível a custo (RMA Shield).
6. Compilação e quantização dinâmica para ONNX Runtime.
7. Benchmark comparativo de latência (PyTorch vs ONNX).
8. Governança e promoção Champion vs. Challenger no Model Registry.
"""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import structlog
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.core.config import get_settings
from src.core.logging import configure_logging
from src.data.transforms import get_inference_transforms, get_train_transforms
from src.models.anomaly_head import OpenSetAnomalyDetector
from src.models.onnx_exporter import benchmark_inference, export_to_onnx, quantize_onnx_model
from src.models.vision_net import CLASS_NAMES, build_model
from src.registry.promote_model import evaluate_and_promote_model
from src.training.cost_optimizer import CostSensitiveThresholdOptimizer

# Força UTF-8 no stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = structlog.get_logger("positivo_vision.train")


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> Path:
    """Plota a matriz de confusão industrial e salva em PNG de alta resolução."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Matriz de Confusão Industrial (Inspeção PCB)",
        ylabel="Classe Real (Ground Truth)",
        xlabel="Predição do Modelo",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


INDUSTRIAL_FOLDER_ORDER = [
    "normal",
    "defect_short",
    "defect_open",
    "defect_missing_hole",
    "defect_spurious",
]


class IndustrialImageFolder(ImageFolder):
    """ImageFolder com ordenação canônica garantida para casar com CLASS_NAMES."""

    def find_classes(self, directory: str | Path) -> tuple[list[str], dict[str, int]]:
        classes = INDUSTRIAL_FOLDER_ORDER
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx


def run_training_pipeline(
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> dict:
    """Executa o pipeline completo de ponta a ponta com dados reais do DeepPCB."""
    settings = get_settings()
    configure_logging(log_level="INFO", json_format=False)
    logger.info("Iniciando pipeline de MLOps Industrial (DeepPCB Real Data)", epochs=epochs, batch_size=batch_size, lr=learning_rate)

    raw_dir = Path("data/raw")
    train_dir = raw_dir / "train"
    val_dir = raw_dir / "val"
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets e DataLoaders com mapeamento canônico
    train_dataset = IndustrialImageFolder(root=str(train_dir), transform=get_train_transforms())
    val_dataset = IndustrialImageFolder(root=str(val_dir), transform=get_inference_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Configuração do MLflow
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    device = torch.device("cpu")
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    with mlflow.start_run(run_name="deeppcb_real_production") as run:
        run_id = run.info.run_id
        logger.info("Executando MLflow Run (DeepPCB Real)", run_id=run_id)

        # Log de Parâmetros
        mlflow.log_params({
            "architecture": "MobileNetV3-Large",
            "dataset_source": "DeepPCB-Real-Peking-University",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss_function": "CrossEntropyLoss(label_smoothing=0.05)",
            "device": "cpu",
            "num_classes": len(CLASS_NAMES),
        })

        # 3. Loop de Treinamento
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            scheduler.step()
            epoch_loss = running_loss / max(total, 1)
            epoch_acc = correct / max(total, 1)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_acc", epoch_acc, step=epoch)
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
            logger.info("Epoca concluida", epoch=epoch, loss=round(epoch_loss, 4), acc=round(epoch_acc, 4))

        # 4. Avaliação em Validação Out-of-Sample
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        normal_embeddings = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

                # Extrai embeddings para a classe NORMAL
                embeddings = model.extract_embedding(images)
                for idx, label in enumerate(labels):
                    if label == 0:  # NORMAL
                        normal_embeddings.append(embeddings[idx].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 5. Métricas Industriais
        val_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
        val_precision = float(precision_score(all_labels, all_preds, average="weighted", zero_division=0))
        val_recall = float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))
        val_acc = float(np.mean(all_preds == all_labels))

        mlflow.log_metrics({
            "val_f1_weighted": val_f1,
            "val_precision_weighted": val_precision,
            "val_recall_weighted": val_recall,
            "val_accuracy": val_acc,
        })
        logger.info("Metricas de validacao out-of-sample", f1=val_f1, recall=val_recall, precision=val_precision)

        # 6. Matriz de Confusão
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = models_dir / "confusion_matrix.png"
        plot_and_save_confusion_matrix(cm, CLASS_NAMES, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

        # 7. Calibração da Anomaly Head (Open-Set)
        anomaly_detector = OpenSetAnomalyDetector()
        if len(normal_embeddings) > 0:
            anomaly_detector.fit_normal_samples(np.array(normal_embeddings))
            centroid_path = models_dir / "anomaly_centroid.npz"
            anomaly_detector.save(centroid_path)
            mlflow.log_artifact(str(centroid_path), artifact_path="anomaly_head")
            logger.info("Anomaly Head calibrada com sucesso", threshold=anomaly_detector.threshold)

        # 8. Otimização de Limiar Sensível a Custo (RMA Shield)
        y_binary = (all_labels > 0).astype(int)
        y_prob_defect = 1.0 - all_probs[:, 0]
        cost_optimizer = CostSensitiveThresholdOptimizer(
            cost_fp=settings.COST_FP_REWORK, cost_fn=settings.COST_FN_RMA
        )
        cost_report = cost_optimizer.optimize_threshold(y_binary, y_prob_defect)
        mlflow.log_params({
            "optimal_cost_threshold": cost_report["optimal_threshold"],
            "expected_financial_loss_brl": cost_report.get("total_financial_loss_brl", 0.0),
        })
        logger.info("Limiar de custo calibrado (RMA Shield)", **cost_report)

        # 9. Salvamento do Checkpoint PyTorch e Compilação ONNX
        torch.save(model.state_dict(), models_dir / "best_model.pt")
        candidate_onnx = models_dir / "candidate.onnx"
        quantized_onnx = models_dir / "candidate_quantized.onnx"
        export_to_onnx(model, candidate_onnx)
        quantize_onnx_model(candidate_onnx, quantized_onnx)
        champion_path = models_dir / "champion.onnx"
        shutil.copy2(quantized_onnx, champion_path)
        mlflow.log_artifact(str(quantized_onnx), artifact_path="edge_onnx")

        # 10. Benchmark de Latência (PyTorch vs ONNX Runtime)
        benchmark_results = benchmark_inference(model, quantized_onnx, num_runs=30, warmup_runs=5)
        mlflow.log_metrics(benchmark_results)
        logger.info("Benchmark de Latencia Concluido", **benchmark_results)

        # 11. Registro do Modelo PyTorch no MLflow
        dummy_sample = np.zeros((1, 3, 224, 224), dtype=np.float32)
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=dummy_sample,
        )

        # 12. Governança e Promoção Champion vs. Challenger
        promotion_result = evaluate_and_promote_model(
            model_name="positivo-pcb-vision",
            challenger_run_id=run_id,
            challenger_f1=val_f1,
            challenger_onnx_path=quantized_onnx,
            min_improvement=0.01,
        )
        mlflow.log_params({
            "is_champion": promotion_result["promoted"],
            "promotion_reason": promotion_result["reason"],
        })

    logger.info("Pipeline de Treinamento e MLOps finalizado com sucesso!")
    return {
        "run_id": run_id,
        "f1_score": val_f1,
        "latency_onnx_p50_ms": benchmark_results["onnx_latency_p50_ms"],
        "speedup": benchmark_results["speedup_factor"],
        "promoted": promotion_result["promoted"],
        "champion_path": promotion_result["active_champion_path"],
    }


if __name__ == "__main__":
    run_training_pipeline(epochs=10, batch_size=16, learning_rate=1e-3)
