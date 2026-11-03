from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .config import DataConfig

Sample = Tuple[Path, int]


class ResizeWithPad:
    def __init__(self, size: int, fill: int = 0):
        if size <= 0:
            raise ValueError("ResizeWithPad size must be positive")
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Image must have positive width and height")

        scale = min(self.size / width, self.size / height)
        resized_width = max(
            1,
            min(self.size, int(round(width * scale))),
        )
        resized_height = max(
            1,
            min(self.size, int(round(height * scale))),
        )
        resized = image.resize(
            (resized_width, resized_height),
            Image.BILINEAR,
        )

        pad_width = self.size - resized_width
        pad_height = self.size - resized_height
        left = pad_width // 2
        top = pad_height // 2
        right = pad_width - left
        bottom = pad_height - top

        if pad_width or pad_height:
            resized = ImageOps.expand(
                resized,
                border=(left, top, right, bottom),
                fill=self.fill,
            )
        return resized


def discover_samples(
    root_dir: str,
    classes: Optional[Sequence[str]] = None,
    extensions: Optional[Sequence[str]] = None,
    include_checkpoints: bool = False,
) -> Tuple[List[Sample], List[str], dict]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_names = list(classes) if classes else ["meteor", "non_meteor"]
    valid_extensions = {
        extension.lower()
        for extension in (
            extensions
            if extensions
            else [".jpg", ".jpeg", ".png", ".bmp"]
        )
    }
    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    samples: List[Sample] = []
    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in valid_extensions:
                continue
            if (
                not include_checkpoints
                and ".ipynb_checkpoints" in image_path.parts
            ):
                continue
            samples.append(
                (image_path, class_to_idx[class_name])
            )

    return samples, class_names, class_to_idx


class MeteorDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform=None,
        samples: Optional[Sequence[Sample]] = None,
        classes: Optional[Sequence[str]] = None,
        include_checkpoints: bool = False,
    ):
        self.root_dir = Path(root_dir)
        if samples is None:
            samples, class_names, class_to_idx = discover_samples(
                root_dir,
                classes=classes,
                include_checkpoints=include_checkpoints,
            )
        else:
            samples = list(samples)
            class_names = (
                list(classes) if classes else ["meteor", "non_meteor"]
            )
            class_to_idx = {
                class_name: index
                for index, class_name in enumerate(class_names)
            }

        self.samples = samples
        self.classes = class_names
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TransformSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        transform=None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(
    data_config: DataConfig,
    training: bool = False,
    in_channels: int = 3,
):
    operations = []

    if training and data_config.random_resized_crop:
        operations.append(
            transforms.RandomResizedCrop(
                data_config.image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
            )
        )
    else:
        operations.append(
            ResizeWithPad(data_config.image_size)
        )

    if training:
        if data_config.horizontal_flip > 0:
            operations.append(
                transforms.RandomHorizontalFlip(
                    p=data_config.horizontal_flip
                )
            )
        if data_config.vertical_flip > 0:
            operations.append(
                transforms.RandomVerticalFlip(
                    p=data_config.vertical_flip
                )
            )
        if (
            data_config.rotation_degrees > 0
            or data_config.rotation_min < 0
        ):
            operations.append(
                transforms.RandomRotation(
                    degrees=(
                        data_config.rotation_min,
                        data_config.rotation_degrees,
                    )
                )
            )
        if data_config.color_jitter > 0:
            operations.append(
                transforms.ColorJitter(
                    brightness=data_config.color_jitter,
                    contrast=data_config.color_jitter,
                    saturation=data_config.color_jitter,
                    hue=0.0,
                )
            )

    if data_config.grayscale:
        operations.append(
            transforms.Grayscale(num_output_channels=in_channels)
        )

    operations.append(transforms.ToTensor())

    if data_config.normalize:
        if in_channels == 1:
            mean = [data_config.mean[0]]
            std = [data_config.std[0]]
        else:
            mean = data_config.mean
            std = data_config.std
        operations.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(operations)
