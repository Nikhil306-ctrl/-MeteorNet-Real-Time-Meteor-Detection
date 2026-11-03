import torch.nn as nn
from torchvision import models

MODEL_NAMES = [
    "meteor",
    "resnet18",
    "mobilenet_v2",
]


class MeteorNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_resnet18(
    pretrained: bool = False,
    num_classes: int = 2,
    in_channels: int = 3,
):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_mobilenet_v2(
    pretrained: bool = False,
    num_classes: int = 2,
    in_channels: int = 3,
):
    weights = (
        models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    )
    model = models.mobilenet_v2(weights=weights)

    if in_channels != 3:
        model.features[0][0] = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def build_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = False,
    in_channels: int = 3,
    dropout: float = 0.4,
):
    normalized = name.lower().replace("-", "_")

    if normalized in {"meteor", "meteornet", "custom_cnn", "cnn"}:
        return MeteorNet(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    if normalized in {"resnet", "resnet18"}:
        return get_resnet18(
            pretrained=pretrained,
            num_classes=num_classes,
            in_channels=in_channels,
        )

    if normalized in {"mobilenet", "mobilenet_v2", "mobilenetv2"}:
        return get_mobilenet_v2(
            pretrained=pretrained,
            num_classes=num_classes,
            in_channels=in_channels,
        )

    if normalized in {"yolo", "yolov8", "yolov5"}:
        raise ValueError(
            "YOLO cannot be trained for this task because the dataset has "
            "no bounding-box annotations. Use a classification model, or "
            "create and validate a labelled detection dataset first."
        )

    raise ValueError(
        f"Unknown model {name!r}. Available models: {MODEL_NAMES}"
    )


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
