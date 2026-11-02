
from pathlib import Path
from collections import Counter
import hashlib

root = Path("D_2 DATASET (METEORS)/sum_images")
classes = ["meteor", "non_meteor"]
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

hashes = {name: {} for name in classes}
group_counts = {}

for class_name in classes:
    folder = root / class_name

    files = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]

    groups = []
    
    print(f"\nScanning {class_name}: {len(files)} images")

    for file in files:
        prefix = file.name.split("-trim")[0]
        groups.append(prefix)

        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        hashes[class_name].setdefault(digest, []).append(file.name)

    counts = Counter(groups)
    group_counts[class_name] = counts

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
    print("Images beyond first image in each repeated group:",
          sum(count - 1 for count in repeated_groups.values()))
    print("Exact duplicate images within class:", duplicate_images)

    print("Largest repeated groups:")
    for group, count in counts.most_common(5):
        print(count, "images:", group)

meteor_hashes = set(hashes["meteor"])
non_meteor_hashes = set(hashes["non_meteor"])

cross_class_duplicates = meteor_hashes & non_meteor_hashes

print("\nCROSS-CLASS EXACT DUPLICATES:", len(cross_class_duplicates))

for digest in list(cross_class_duplicates)[:10]:
    print("METEOR:", hashes["meteor"][digest][0])
    print("NON-METEOR:", hashes["non_meteor"][digest][0])
