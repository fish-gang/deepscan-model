import torch.nn as nn
from torchvision import models

# List of pretrained models: https://docs.pytorch.org/vision/main/models.html#classification


def create_model(num_classes=12, backbone="efficientnet_b0", pretrained=True):
    weights = "IMAGENET1K_V1" if pretrained else None

    # Lightweight model, very fast but less accurate
    if backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        assert isinstance(model.classifier[3], nn.Linear)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        assert isinstance(model.classifier[3], nn.Linear)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # Mid-range model, good balance of speed and accuracy
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif backbone == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif backbone == "resnet152":
        model = models.resnet152(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Heavier model, better accuracy but slower
    elif backbone == "efficientnet_b4":
        model = models.efficientnet_b4(weights=weights)
        assert isinstance(model.classifier[1], nn.Linear)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model
