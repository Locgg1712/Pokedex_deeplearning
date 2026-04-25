# src/model_dl.py
# CNN Model definition — MobileNetV2 with transfer learning.

import torch
import torch.nn as nn
from torchvision import models


class PokemonCNN(nn.Module):
    """
    MobileNetV2 backbone with a custom classifier head.
    Uses ImageNet pre-trained weights and fine-tunes the full network.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        # Load pre-trained MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Replace the final classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes=10, device="cpu"):
    """Factory function to instantiate the model on the desired device."""
    model = PokemonCNN(num_classes=num_classes)
    return model.to(device)
