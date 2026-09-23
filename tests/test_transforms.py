"""Testes unitários para o módulo de transforms e triagem OOD."""

import numpy as np
import torch
from PIL import Image

from src.data.transforms import (
    check_out_of_distribution,
    get_inference_transforms,
    get_train_transforms,
)


def test_train_transforms_output_shape():
    """Valida o shape e formato do tensor gerado pelas transformações de treino."""
    img = Image.new("RGB", (300, 300), color=(20, 80, 40))
    transform = get_train_transforms(img_size=224)
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_inference_transforms_deterministic():
    """Valida que transformações de inferência são determinísticas."""
    img = Image.new("RGB", (250, 250), color=(15, 65, 35))
    transform = get_inference_transforms(img_size=224)
    t1 = transform(img)
    t2 = transform(img)

    assert torch.allclose(t1, t2)
    assert t1.shape == (3, 224, 224)


def test_ood_screening_sharp_image():
    """Valida que uma imagem nítida com iluminação controlada é aprovada."""
    # Gera imagem com bordas nítidas (alto gradiente / variância Laplaciana)
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :112] = [20, 75, 40]
    arr[:, 112:] = [210, 160, 60]
    img = Image.fromarray(arr)

    ood_result = check_out_of_distribution(img)
    assert ood_result["is_blurred"] is False
    assert ood_result["luminance_ok"] is True
    assert ood_result["is_ood"] is False


def test_ood_screening_flat_blur_image():
    """Valida que uma imagem completamente lisa (sem bordas/desfocada) é reprovada no blur."""
    img = Image.new("RGB", (224, 224), color=(50, 50, 50))
    ood_result = check_out_of_distribution(img, min_laplacian_var=10.0)

    assert ood_result["is_blurred"] is True
    assert ood_result["is_ood"] is True
