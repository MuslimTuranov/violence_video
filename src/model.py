import os
from pathlib import Path
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18

def get_model(
    num_classes: int = 2,
    train_backbone: bool = False,
    pretrained: bool = True,
):
    cache_dir = Path(__file__).resolve().parents[1] / ".torch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_dir))
    weights = R3D_18_Weights.DEFAULT if pretrained else None
    model = r3d_18(weights=weights)

    if not train_backbone:
        for param in model.parameters():
            param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
