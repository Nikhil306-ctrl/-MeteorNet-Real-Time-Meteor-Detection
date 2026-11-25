import torch.nn as nn
from torchvision import models

MODEL_NAMES = [
    "meteor",
    "resnet18",
    "mobilenet_v2",
    "custom_cnn_groupnorm",
    "alexnet",
    "vgg16",
    "resnet34",
    "densenet121",
    "efficientnet_b0",
    "inception_v3",
    "shufflenet_v2_x1_0",
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

    if normalized == "custom_cnn_groupnorm":
        model = MeteorNet(in_channels, num_classes, dropout)
        for index, layer in enumerate(model.features):
            if isinstance(layer, nn.BatchNorm2d):
                model.features[index] = nn.GroupNorm(8, layer.num_features)
        return model

    if normalized in {"alexnet", "vgg16", "resnet34", "densenet121",
                      "efficientnet_b0", "inception_v3", "shufflenet_v2_x1_0"}:
        if in_channels != 3:
            raise ValueError("Additional backbones require RGB input")
        factories = {
            "alexnet": (models.alexnet, models.AlexNet_Weights),
            "vgg16": (models.vgg16, models.VGG16_Weights),
            "resnet34": (models.resnet34, models.ResNet34_Weights),
            "densenet121": (models.densenet121, models.DenseNet121_Weights),
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
            "inception_v3": (models.inception_v3, models.Inception_V3_Weights),
            "shufflenet_v2_x1_0": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights),
        }
        factory, weights_enum = factories[normalized]
        model = factory(weights=weights_enum.DEFAULT if pretrained else None)
        if normalized == "inception_v3":
            model.aux_logits = False
            model.AuxLogits = None
            model.transform_input = False
        if normalized in {"resnet34", "inception_v3", "shufflenet_v2_x1_0"}:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif normalized == "densenet121":
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

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
