
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MeteorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = ["meteor", "non_meteor"]

        self.class_to_idx = {
            "meteor": 0,
            "non_meteor": 1
        }

        self.samples = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name

            for image_path in sorted(class_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in [
                        ".jpg", ".jpeg", ".png", ".bmp"
                    ]
                ):
                    self.samples.append(
                        (image_path, self.class_to_idx[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(training=False):
    if training:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])

    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
