"""Governança do Model Registry com Política Champion vs. Challenger.

Aplica promoção determinística baseada em Model Aliases (@champion e @challenger)
e ganho estatístico out-of-sample (F1-score >= champion + threshold).
"""

import shutil
from pathlib import Path
from typing import Any

import structlog
from mlflow.tracking import MlflowClient

from src.core.config import get_settings

logger = structlog.get_logger("positivo_vision.registry")


def evaluate_and_promote_model(
    model_name: str,
    challenger_run_id: str,
    challenger_f1: float,
    challenger_onnx_path: Path | str,
    min_improvement: float = 0.01,
) -> dict[str, Any]:
    """Avalia o modelo concorrente e promove a @champion se houver ganho estatístico comprovado."""
    settings = get_settings()
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    # 1. Registra a nova versão no catálogo
    model_uri = f"runs:/{challenger_run_id}/model"
    try:
        registered_model = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=challenger_run_id,
            description="Modelo candidato Challenger avaliado em conjunto cego de teste out-of-sample.",
        )
        challenger_version = registered_model.version
    except Exception:
        # Se o modelo registrado não existir ainda, cria
        client.create_registered_model(name=model_name)
        registered_model = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=challenger_run_id,
            description="Versão inicial registrada no MLflow Model Registry.",
        )
        challenger_version = registered_model.version

    logger.info("Nova versão registrada no Model Registry", model=model_name, version=challenger_version)

    # 2. Busca o atual @champion
    current_champion_f1 = -1.0
    has_champion = False

    try:
        current_champion = client.get_model_version_by_alias(name=model_name, alias="champion")
        has_champion = True
        champion_run = client.get_run(current_champion.run_id)
        current_champion_f1 = champion_run.data.metrics.get("val_f1_weighted", 0.0)
        logger.info(
            "Modelo Champion atual identificado",
            version=current_champion.version,
            champion_f1=current_champion_f1,
        )
    except Exception:
        logger.info("Nenhum modelo com alias @champion encontrado. Esta será a primeira versão em produção.")

    # 3. Regra de Promoção Determinística
    should_promote = False
    if not has_champion:
        should_promote = True
        reason = "Primeira versão homologada para produção."
    elif challenger_f1 >= (current_champion_f1 + min_improvement):
        should_promote = True
        reason = f"Ganho de performance comprovado: F1 {challenger_f1:.4f} >= {current_champion_f1:.4f} + {min_improvement}."
    else:
        reason = f"Desempenho insuficiente para promoção: F1 {challenger_f1:.4f} < {current_champion_f1:.4f} + {min_improvement}."

    # 4. Aplicação do Alias e Atualização de Borda
    target_onnx = settings.resolved_model_path
    target_onnx.parent.mkdir(parents=True, exist_ok=True)

    if should_promote:
        client.set_registered_model_alias(name=model_name, alias="champion", version=challenger_version)
        shutil.copy2(challenger_onnx_path, target_onnx)
        # Copia arquivo de dados associado se existir
        challenger_data = Path(str(challenger_onnx_path) + ".data")
        if challenger_data.exists():
            shutil.copy2(challenger_data, str(target_onnx) + ".data")
        logger.info("PROMOÇÃO CONCLUÍDA: Alias @champion atualizado", version=challenger_version, reason=reason)
    else:
        client.set_registered_model_alias(name=model_name, alias="challenger", version=challenger_version)
        logger.info("MODELO ARQUIVADO COMO @challenger", version=challenger_version, reason=reason)

    return {
        "promoted": should_promote,
        "challenger_version": challenger_version,
        "challenger_f1": challenger_f1,
        "current_champion_f1": current_champion_f1,
        "reason": reason,
        "active_champion_path": str(target_onnx),
    }
