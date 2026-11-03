import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from src.config import resolve_data_root

CLASSES = ["meteor", "non_meteor"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(root: Path) -> dict:
    images = {}
    for class_name in CLASSES:
        folder = root / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Class directory not found: {folder}")
        images[class_name] = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file()
            and path.suffix.lower() in EXTENSIONS
            and ".ipynb_checkpoints" not in path.parts
        ]
    return images


def build_sheet(paths, output_path: Path, sample_count: int) -> None:
    selected = random.sample(paths, min(sample_count, len(paths)))
    sheet = Image.new("RGB", (750, 600), "white")
    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(selected):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((140, 105))

        x = (index % 5) * 150
        y = (index // 5) * 100
        sheet.paste(image, (x, y))
        draw.text((x + 3, y + 78), f"Image {index + 1}", fill="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Dataset root. Defaults to the workspace dataset location."
        ),
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    root = resolve_data_root(
        args.data,
        prefer_cwd=args.data is not None,
    )
    print(f"Dataset root: {root}")
    images = collect_images(root)
    output_dir = Path(args.output_dir)

    for class_name, paths in images.items():
        print(f"{class_name}: {len(paths)} images")
        build_sheet(
            paths,
            output_dir / f"{class_name}_samples.jpg",
            args.samples,
        )


if __name__ == "__main__":
    main()
