
import argparse
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import MeteorDataset, get_transforms
from src.models import MeteorNet, get_resnet18


def train_model(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    train_dataset = MeteorDataset(
        args.data,
        get_transforms(training=True)
    )

    val_dataset = MeteorDataset(
        args.data,
        get_transforms(training=False)
    )

    if args.limit_per_class:
        selected = []
        counts = {0: 0, 1: 0}

        for index, (_, label) in enumerate(train_dataset.samples):
            if counts[label] < args.limit_per_class:
                selected.append(index)
                counts[label] += 1

        indices = selected
        print(f"Limited dataset: {len(indices)} images")
    else:
        indices = list(range(len(train_dataset)))

    labels = [train_dataset.samples[i][1] for i in indices]

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")

    if args.model == "meteor":
        model = MeteorNet()
    else:
        model = get_resnet18(pretrained=args.pretrained)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    best_accuracy = 0.0

    Path("models").mkdir(exist_ok=True)
    Path("results/graphs").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        for images, batch_labels in progress:
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total += batch_labels.size(0)
            correct += (predictions == batch_labels).sum().item()

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        model.eval()

        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, batch_labels)

                val_loss_total += loss.item()

                predictions = outputs.argmax(dim=1)

                val_total += batch_labels.size(0)
                val_correct += (predictions == batch_labels).sum().item()

        val_loss = val_loss_total / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(f"\nTraining loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Training accuracy: {train_accuracy:.2f}%")
        print(f"Validation accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "classes": train_dataset.classes,
                "image_size": 128,
                "validation_accuracy": best_accuracy
            }, f"models/{args.model}_best.pth")

            print("Best model checkpoint saved.")

    history_path = f"results/metrics/{args.model}_history.json"

    with open(history_path, "w") as file:
        json.dump(history, file, indent=4)

    epochs = range(1, args.epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/graphs/{args.model}_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/graphs/{args.model}_loss.png")
    plt.close()

    print("\nTraining history saved.")
    print("Accuracy and loss graphs saved.")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="D_2 DATASET (METEORS)/sum_images"
    )

    parser.add_argument(
        "--model",
        choices=["meteor", "resnet"],
        default="meteor"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")

    train_model(parser.parse_args())
