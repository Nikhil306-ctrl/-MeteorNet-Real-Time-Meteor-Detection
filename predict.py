
import argparse
import torch
from PIL import Image
from torchvision import transforms

from src.models import MeteorNet, get_resnet18


def predict(image_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model_name = checkpoint["model_name"]

    if model_name == "meteor":
        model = MeteorNet()
    else:
        model = get_resnet18(pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]

    classes = checkpoint["classes"]
    predicted_index = probabilities.argmax().item()

    print("\nPrediction:", classes[predicted_index].upper())
    print(f"Meteor probability: {probabilities[0].item() * 100:.2f}%")
    print(f"Non-Meteor probability: {probabilities[1].item() * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--checkpoint",
        default="models/meteor_best.pth"
    )

    args = parser.parse_args()

    predict(args.image, args.checkpoint)
