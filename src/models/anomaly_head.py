"""Detector de Defeitos Inéditos (Open-Set Anomaly Head).

Mede a distância geométrica dos embeddings latentes em relação ao centroide
de normalidade da PCB de referência (Golden Sample). Sinaliza anomalias desconhecidas
que nunca estiveram no catálogo de treino fechado da fábrica.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch


class OpenSetAnomalyDetector:
    """Detecta anomalias fora do catálogo conhecido através de distâncias no espaço latente."""

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold
        self.normal_centroid: np.ndarray | None = None
        self.std_distance: float = 1.0

    def fit_normal_samples(self, embeddings: np.ndarray | torch.Tensor) -> None:
        """Calibra o centroide de normalidade com base em embeddings de PCBs conformes."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        # Normalização L2 para distância cosseno estrita
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        norm_embeddings = embeddings / norms

        self.normal_centroid = np.mean(norm_embeddings, axis=0)
        self.normal_centroid = self.normal_centroid / (np.linalg.norm(self.normal_centroid) + 1e-8)

        # Calcula a dispersão interna (distâncias do cluster normal)
        dists = 1.0 - np.dot(norm_embeddings, self.normal_centroid)
        self.std_distance = float(np.std(dists)) + 1e-6
        # Limiar adaptativo padrão: média + 3 sigmas
        self.threshold = float(np.mean(dists) + 3.0 * self.std_distance)

    def score(self, embedding: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Calcula o score de anomalia e flag de defeito inédito para um embedding."""
        if self.normal_centroid is None:
            # Fallback seguro caso não tenha sido calibrado
            return {"is_unknown_anomaly": False, "anomaly_score": 0.0, "distance": 0.0}

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Normaliza L2
        norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8
        norm_emb = embedding / norm

        # Distância Cosseno: 0 = idêntico, 1 = ortogonal, 2 = oposto
        cosine_sim = float(np.dot(norm_emb, self.normal_centroid)[0])
        cosine_dist = float(1.0 - cosine_sim)

        is_unknown = cosine_dist > self.threshold
        normalized_score = min(max(cosine_dist / (self.threshold + 1e-6), 0.0), 5.0)

        return {
            "is_unknown_anomaly": bool(is_unknown),
            "anomaly_score": round(normalized_score, 3),
            "cosine_distance": round(cosine_dist, 4),
            "threshold": round(self.threshold, 4),
        }

    def save(self, filepath: Path | str) -> None:
        """Salva o centroide calibrado em disco."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            centroid=self.normal_centroid,
            threshold=np.array(self.threshold),
            std_distance=np.array(self.std_distance),
        )

    def load(self, filepath: Path | str) -> None:
        """Carrega os pesos calibrados do detector."""
        filepath = Path(filepath)
        if not filepath.exists():
            return
        data = np.load(filepath)
        self.normal_centroid = data["centroid"]
        self.threshold = float(data["threshold"])
        self.std_distance = float(data["std_distance"])
