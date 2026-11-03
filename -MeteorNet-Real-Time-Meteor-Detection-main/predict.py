import argparse

import torch
from PIL import Image

from src.config import ExperimentConfig
from src.dataset import build_transforms
from src.engine import outputs_to_probabilities
from src.models import build_model
from src.utils import get_device, load_checkpoint


def predict(image_path, checkpoint_path, device_name=None):
    device = get_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint has no config. Retrain with the current "
            "train.py before predicting."
        )
    config = ExperimentConfig.from_dict(checkpoint["config"])

    output_mode = checkpoint.get(
        "output_mode",
        config.model.output_mode,
    )
    in_channels = checkpoint.get(
        "in_channels",
        config.model.in_channels,
    )
    num_outputs = 1 if output_mode == "bce" else config.model.num_classes

    model = build_model(
        checkpoint["model_name"],
        num_classes=num_outputs,
        pretrained=False,
        in_channels=in_channels,
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = build_transforms(
        config.data,
        training=False,
        in_channels=in_channels,
    )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prediction, meteor_probability, non_meteor_probability = (
            outputs_to_probabilities(output, output_mode)
        )

    classes = checkpoint.get("classes", config.data.classes)
    predicted_index = int(prediction[0].item())
    meteor_probability = float(meteor_probability[0].item())
    non_meteor_probability = float(
        non_meteor_probability[0].item()
    )

    print(f"\nPrediction: {classes[predicted_index].upper()}")
    print(f"Meteor probability: {meteor_probability * 100:.2f}%")
    print(
        "Non-Meteor probability: "
        f"{non_meteor_probability * 100:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--checkpoint",
        default="models/meteornet_default_best.pth",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    predict(args.image, args.checkpoint, args.device)
