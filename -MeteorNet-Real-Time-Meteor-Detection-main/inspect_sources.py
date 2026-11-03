import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.config import resolve_data_root

CLASSES = ["meteor", "non_meteor"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_files(root: Path, class_name: str):
    folder = root / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Class directory not found: {folder}")
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
        and ".ipynb_checkpoints" not in path.parts
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=None,
        help=(
            "Dataset root. Defaults to the workspace dataset location."
        ),
    )
    args = parser.parse_args()
    root = resolve_data_root(
        args.data,
        prefer_cwd=args.data is not None,
    )
    print(f"Dataset root: {root}")

    hashes = {name: defaultdict(list) for name in CLASSES}
    years = {}
    numeric_ids = {}

    for class_name in CLASSES:
        files = collect_files(root, class_name)
        groups = []
        class_years = []
        class_ids = []

        print(f"\nScanning {class_name}: {len(files)} images")

        for file in files:
            groups.append(file.name.split("-trim")[0])

            year_match = re.search(r"(20\d{2})", file.name)
            if year_match:
                class_years.append(year_match.group(1))

            id_match = re.search(r"(?<!\d)(\d{6})(?!\d)", file.name)
            if id_match:
                class_ids.append(id_match.group(1))

            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            hashes[class_name][digest].append(file.name)

        counts = Counter(groups)
        years[class_name] = Counter(class_years)
        numeric_ids[class_name] = set(class_ids)

        repeated_groups = {
            group: count
            for group, count in counts.items()
            if count > 1
        }
        duplicate_images = sum(
            len(names) - 1
            for names in hashes[class_name].values()
        )

        print("Unique filename groups:", len(counts))
        print("Repeated filename groups:", len(repeated_groups))
        print(
            "Images beyond first image in each repeated group:",
            sum(count - 1 for count in repeated_groups.values()),
        )
        print("Exact duplicate images within class:", duplicate_images)
        print("Year distribution:", dict(sorted(years[class_name].items())))
        print("Unique numeric IDs:", len(numeric_ids[class_name]))

        print("Largest repeated groups:")
        for group, count in counts.most_common(5):
            print(count, "images:", group)

    meteor_hashes = set(hashes["meteor"])
    non_meteor_hashes = set(hashes["non_meteor"])
    cross_class_duplicates = meteor_hashes & non_meteor_hashes
    shared_ids = numeric_ids["meteor"] & numeric_ids["non_meteor"]

    print("\nCROSS-CLASS EXACT DUPLICATES:", len(cross_class_duplicates))
    for digest in list(cross_class_duplicates)[:10]:
        print("METEOR:", hashes["meteor"][digest][0])
        print("NON-METEOR:", hashes["non_meteor"][digest][0])

    print("SHARED NUMERIC IDS:", len(shared_ids))
    print("METEOR-ONLY NUMERIC IDS:", len(numeric_ids["meteor"] - shared_ids))
    print(
        "NON-METEOR-ONLY NUMERIC IDS:",
        len(numeric_ids["non_meteor"] - shared_ids),
    )


if __name__ == "__main__":
    main()
