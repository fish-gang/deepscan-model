import torch.nn as nn
from torchvision import models

# List of pretrained models: https://docs.pytorch.org/vision/main/models.html#classification


def create_model(num_classes=12, backbone="efficientnet_b0", pretrained=True):
    weights = "IMAGENET1K_V1" if pretrained else None

    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # type: ignore

    elif backbone == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model
