"""Pipeline de Transformações, Data Augmentation e Triagem OOD.

Inclui pré-processamento ImageNet, variações térmicas/luminosas industriais
e triagem de imagens fora de distribuição (OOD) via variância Laplaciana (blur)
e análise de luminância em espaço HSV.
"""

from typing import Any

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Parâmetros padrão ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Transformações com data augmentation para o ambiente de esteira industrial."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """Transformações determinísticas para validação e inferência."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def check_out_of_distribution(
    pil_image: Image.Image,
    min_laplacian_var: float = 35.0,
    min_luminance: float = 20.0,
    max_luminance: float = 240.0,
) -> dict[str, Any]:
    """Realiza triagem estatística rápida para detecção de anomalias OOD na imagem.

    Verifica:
    1. Variância Laplaciana (detecção de desfoque/blur mecânico de câmera).
    2. Nível de luminância média no canal V (detecção de oclusões ou falha de luz).
    """
    # Converte PIL para formato OpenCV (BGR)
    rgb_arr = np.array(pil_image.convert("RGB"))
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2GRAY)

    # 1. Variância Laplaciana
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = float(laplacian.var())
    is_blurred = laplacian_variance < min_laplacian_var

    # 2. Análise de Luminância no espaço HSV
    hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
    mean_luminance = float(np.mean(hsv[:, :, 2]))
    luminance_ok = (min_luminance <= mean_luminance <= max_luminance)

    # Status geral de aprovação OOD
    is_ood = is_blurred or (not luminance_ok)

    return {
        "is_ood": is_ood,
        "is_blurred": is_blurred,
        "laplacian_variance": round(laplacian_variance, 2),
        "mean_luminance": round(mean_luminance, 2),
        "luminance_ok": luminance_ok,
    }
