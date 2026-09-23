"""Testes unitários para a arquitetura neural, Anomaly Head e Grad-CAM."""

import numpy as np
import torch
from PIL import Image

from src.models.anomaly_head import OpenSetAnomalyDetector
from src.models.gradcam import GradCAM
from src.models.vision_net import build_model


def test_vision_net_forward_and_embeddings():
    """Valida o forward pass e a extração do vetor de embedding de 960 dimensões."""
    model = build_model(num_classes=5, pretrained=False)
    model.eval()

    dummy_tensor = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_tensor)
        embedding = model.extract_embedding(dummy_tensor)

    assert logits.shape == (2, 5)
    assert embedding.shape == (2, 960)


def test_anomaly_head_calibration_and_scoring():
    """Valida a calibração do centroide e cálculo do score de defeito inédito."""
    detector = OpenSetAnomalyDetector(threshold=0.30)

    # Simula embeddings de treino da classe NORMAL (cluster concentrado)
    np.random.seed(42)
    normal_centroid_true = np.ones(960) / np.sqrt(960)
    normal_samples = normal_centroid_true + np.random.normal(0, 0.02, (50, 960))

    detector.fit_normal_samples(normal_samples)
    assert detector.normal_centroid is not None
    assert detector.normal_centroid.shape == (960,)

    # Amostra dentro do cluster normal
    sample_normal = normal_centroid_true + np.random.normal(0, 0.01, (1, 960))
    res_normal = detector.score(sample_normal)
    assert res_normal["is_unknown_anomaly"] is False

    # Amostra anômala (vetor ortogonal/distante)
    sample_anomaly = np.random.normal(0, 1.0, (1, 960))
    res_anomaly = detector.score(sample_anomaly)
    assert res_anomaly["cosine_distance"] > res_normal["cosine_distance"]


def test_gradcam_heatmap_generation():
    """Valida a geração da matriz de calor Grad-CAM e a sobreposição em imagem PIL."""
    model = build_model(num_classes=5, pretrained=False)
    gradcam = GradCAM(model=model, target_layer=model.get_last_conv_layer())

    dummy_tensor = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(dummy_tensor, class_idx=1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0001

    orig_img = Image.new("RGB", (224, 224), color=(20, 75, 40))
    overlay = gradcam.overlay_on_image(orig_img, heatmap)
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (224, 224)

    b64_str = GradCAM.pil_to_base64(overlay)
    assert b64_str.startswith("data:image/jpeg;base64,")
