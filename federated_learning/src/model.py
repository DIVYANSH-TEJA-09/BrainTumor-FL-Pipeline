"""
ResNet-18 model for 3-class brain tumour classification.
Pretrained on ImageNet; final FC layer replaced.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BrainTumorResNet(nn.Module):
    """
    ResNet-18 with the last FC layer replaced:
        512 → num_classes (default 3).
    """

    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(num_classes=3, device="cuda"):
    """Instantiate, move to device, and print a quick param summary."""
    model = BrainTumorResNet(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb     = total * 4 / (1024 ** 2)

    print(f"Model: ResNet-18  |  Params: {total:,} (trainable {train:,})  |  ~{mb:.1f} MB")
    return model
