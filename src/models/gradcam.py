"""Mapeamento de Ativação de Classes Ponderado por Gradientes (Grad-CAM).

Implementa explicabilidade visual industrial (XAI) para destacar fisicamente
na PCB a localização exata do curto-circuito, trilha rompida ou anomalia.
"""

import base64
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Calcula e renderiza heatmaps de ativação Grad-CAM sobrepostos à imagem da PCB."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Hooks para captura de tensores intermediários
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Any, grad_output: tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Gera a matriz 2D normalizada (0.0 a 1.0) do mapa térmico de ativação."""
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback seguro: matriz zerada
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pooling dos gradientes
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Combinação linear ponderada
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Aplicação de ReLU (apenas características que ativam positivamente a classe)
        cam = F.relu(cam)

        cam_arr = cam.squeeze().cpu().numpy()
        cam_arr = cv2.resize(cam_arr, (input_tensor.shape[3], input_tensor.shape[2]))

        # Normalização Min-Max
        cam_min, cam_max = cam_arr.min(), cam_arr.max()
        if cam_max - cam_min > 1e-6:
            cam_arr = (cam_arr - cam_min) / (cam_max - cam_min)
        else:
            cam_arr = np.zeros_like(cam_arr)

        return cam_arr

    def overlay_on_image(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """Sobrepõe o mapa térmico colorido na imagem original da PCB com transparência."""
        # Redimensiona heatmap para o tamanho exato da imagem original
        w, h = original_image.size
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Converte para mapa térmico colorido de 8 bits
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        orig_arr = np.array(original_image.convert("RGB"))
        # Alpha blending
        overlay = cv2.addWeighted(orig_arr, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
        return Image.fromarray(overlay)

    @staticmethod
    def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        """Converte uma imagem PIL em string Base64 com data URI."""
        buffered = BytesIO()
        image.save(buffered, format=format, quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"
