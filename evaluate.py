
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.dataset import MeteorDataset, get_transforms
from src.models import MeteorNet, get_resnet18
from src.utils import calculate_metrics, save_metrics


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MeteorDataset(args.data, get_transforms())

    if args.limit_per_class:
        counts = {0: 0, 1: 0}
        indices = []

        for i, (_, label) in enumerate(dataset.samples):
            if counts[label] < args.limit_per_class:
                indices.append(i)
                counts[label] += 1
    else:
        indices = list(range(len(dataset)))

    labels = [dataset.samples[i][1] for i in indices]

    _, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    if args.model == "meteor":
        model = MeteorNet()
    else:
        model = get_resnet18(pretrained=False)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().tolist()

            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions)

    metrics = calculate_metrics(true_labels, predicted_labels)

    print("\nEvaluation Results")
    print("------------------")

    for key, value in metrics.items():
        if key == "classification_report":
            print("\n", value)
        else:
            print(f"{key}: {value}")

    save_metrics(metrics, f"results/metrics/{args.model}_metrics.json")

    cm = metrics["confusion_matrix"]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks([0, 1], ["Meteor", "Non-Meteor"])
    plt.yticks([0, 1], ["Meteor", "Non-Meteor"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"results/graphs/{args.model}_confusion_matrix.png")
    plt.close()

    print("\nMetrics and confusion matrix saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="D_2 DATASET (METEORS)/sum_images"
    )
    parser.add_argument("--model", choices=["meteor", "resnet"], default="meteor")
    parser.add_argument("--checkpoint", default="models/meteor_best.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-class", type=int, default=None)

    evaluate(parser.parse_args())
