"""Arquitetura Neural de Visão Computacional Industrial.

Utiliza MobileNetV3-Large com Transfer Learning pré-treinada na ImageNet,
otimizada para inferência em CPU industrial de esteira com extração de embeddings
latentes para a Anomaly Head (Open-Set).
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

CLASS_NAMES = [
    "NORMAL",
    "DEFECT_SHORT",
    "DEFECT_OPEN",
    "DEFECT_MISSING_HOLE",
    "DEFECT_MOUSEBITE",
    "DEFECT_SPUR",
    "DEFECT_SPURIOUS_COPPER",
]


class IndustrialVisionNet(nn.Module):
    """Rede convolucional híbrida para classificação de PCBs e extração de embeddings."""

    def __init__(
        self,
        num_classes: int = len(CLASS_NAMES),
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_large(weights=weights)

        # 1. Extrator Convolucional Profundo (Backbone)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Congela pesos iniciais do extrator e deixa as últimas camadas para adaptação industrial
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.features[-2:].parameters():
                param.requires_grad = True

        # Dimensão do espaço latente da MobileNetV3-Large
        in_features = 960

        # 2. Cabeçalho de Classificação Customizado
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai o vetor de embedding latente (960-D) antes do classificador."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem para frente retornando logits das classes."""
        embedding = self.extract_embedding(x)
        logits = self.classifier(embedding)
        return logits

    def get_last_conv_layer(self) -> nn.Module:
        """Retorna a última camada convolucional para acoplamento do Grad-CAM."""
        return self.features[-1]


def build_model(num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> IndustrialVisionNet:
    """Instancia a rede neural com as configurações industriais padrão."""
    return IndustrialVisionNet(num_classes=num_classes, pretrained=pretrained)
