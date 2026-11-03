import json
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

Sample = Tuple[Path, int]


def extract_group_key(path: Path, group_key: str) -> str:
    name = Path(path).name
    if group_key == "prefix":
        return name.split("-trim")[0]
    if group_key == "numeric_id":
        match = re.search(r"(?<!\d)(\d{6})(?!\d)", name)
        if match:
            return match.group(1)
        return name.split("-trim")[0]
    if group_key == "filename":
        return Path(path).stem
    raise ValueError(f"Unsupported group key: {group_key}")


def limit_samples_per_class(
    samples: Sequence[Sample],
    max_per_class: int,
    seed: int,
) -> List[Sample]:
    if max_per_class <= 0:
        return list(samples)

    by_class: Dict[int, List[int]] = defaultdict(list)
    for index, (_, label) in enumerate(samples):
        by_class[label].append(index)

    rng = random.Random(seed)
    selected: List[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        selected.extend(indices[:max_per_class])

    selected.sort()
    return [samples[index] for index in selected]


def split_samples(
    samples: Sequence[Sample],
    ratios: Sequence[float],
    seed: int,
    split_unit: str = "prefix",
    group_key: str = "prefix",
) -> dict:
    if split_unit not in {"image", "prefix", "group"}:
        raise ValueError(
            "split_unit must be 'image', 'prefix', or 'group'"
        )

    effective_group_key = "prefix" if split_unit == "prefix" else group_key
    normalized = _normalize_ratios(ratios)
    split_names = ["train", "val", "test"]
    splits: Dict[str, List[int]] = {name: [] for name in split_names}
    group_sets: Dict[str, set] = {name: set() for name in split_names}

    group_members: Dict[str, List[int]] = defaultdict(list)
    for index, (path, _) in enumerate(samples):
        if split_unit == "image":
            key = f"image:{index}"
        else:
            key = extract_group_key(path, effective_group_key)
        group_members[key].append(index)

    if split_unit == "image":
        groups_by_class: Dict[int, List[str]] = defaultdict(list)
        for key, indices in group_members.items():
            label = int(samples[indices[0]][1])
            groups_by_class[label].append(key)

        rng = random.Random(seed)
        for label in sorted(groups_by_class):
            keys = sorted(groups_by_class[label])
            rng.shuffle(keys)
            counts = _allocate_counts(len(keys), normalized)

            start = 0
            for split_name, count in zip(split_names, counts):
                for key in keys[start:start + count]:
                    splits[split_name].extend(group_members[key])
                    group_sets[split_name].add(key)
                start += count
    else:
        _assign_groups_by_composition(
            group_members,
            samples,
            normalized,
            split_names,
            splits,
            group_sets,
            seed,
        )

    for split_name in split_names:
        splits[split_name] = sorted(set(splits[split_name]))

    if split_unit in {"group", "prefix"}:
        _validate_group_split(group_sets)
    _validate_class_coverage(splits, samples, split_names)

    class_counts = {
        name: _class_counts(splits[name], samples)
        for name in split_names
    }

    return {
        "splits": splits,
        "groups": {
            name: sorted(group_sets[name]) for name in split_names
        },
        "group_counts": {
            name: len(group_sets[name]) for name in split_names
        },
        "counts": {
            name: len(splits[name]) for name in split_names
        },
        "class_counts": class_counts,
        "ratios": normalized,
        "split_unit": split_unit,
        "group_key": effective_group_key,
        "seed": seed,
    }


def _normalize_ratios(ratios: Sequence[float]) -> List[float]:
    values = [max(0.0, float(value)) for value in ratios]
    total = sum(values)
    if total <= 0:
        raise ValueError("Split ratios must have a positive sum")
    return [value / total for value in values]


def _allocate_counts(size: int, ratios: Sequence[float]) -> List[int]:
    if size <= 0:
        return [0 for _ in ratios]

    raw = [size * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = size - sum(counts)

    order = sorted(
        range(len(raw)),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    nonzero = [
        index for index, ratio in enumerate(ratios) if ratio > 0
    ]
    if size >= len(nonzero):
        for index in nonzero:
            if counts[index] > 0:
                continue
            donor = max(range(len(counts)), key=lambda item: counts[item])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1

    return counts


def _group_assignment_cost(
    group_counts: Dict[int, int],
    current_counts: Dict[int, int],
    target_counts: Dict[int, int],
) -> float:
    cost = 0.0
    for label, count in group_counts.items():
        target = target_counts.get(label, 0)
        if target <= 0:
            return float("inf")
        remaining = target - current_counts.get(label, 0)
        cost -= count * (remaining / target)
    return cost


def _assign_groups_by_composition(
    group_members: Dict[str, List[int]],
    samples: Sequence[Sample],
    ratios: Sequence[float],
    split_names: Sequence[str],
    splits: Dict[str, List[int]],
    group_sets: Dict[str, set],
    seed: int,
) -> None:
    group_composition: Dict[str, Dict[int, int]] = {}
    class_totals: Dict[int, int] = defaultdict(int)

    for key, indices in group_members.items():
        counts = _class_counts(indices, samples)
        group_composition[key] = counts
        for label, count in counts.items():
            class_totals[label] += count

    targets: Dict[str, Dict[int, int]] = {
        name: {} for name in split_names
    }
    for label, total in class_totals.items():
        allocations = _allocate_counts(total, ratios)
        for name, count in zip(split_names, allocations):
            targets[name][label] = count

    current: Dict[str, Dict[int, int]] = {
        name: defaultdict(int) for name in split_names
    }
    split_order = {
        name: index for index, name in enumerate(split_names)
    }

    rng = random.Random(seed)
    keys = list(group_members)
    rng.shuffle(keys)
    keys.sort(
        key=lambda key: -sum(group_composition[key].values())
    )

    for key in keys:
        split_name = min(
            split_names,
            key=lambda name: (
                _group_assignment_cost(
                    group_composition[key],
                    current[name],
                    targets[name],
                ),
                split_order[name],
            ),
        )
        splits[split_name].extend(group_members[key])
        group_sets[split_name].add(key)
        for label, count in group_composition[key].items():
            current[split_name][label] += count


def _validate_group_split(
    group_sets: Dict[str, set],
) -> None:
    seen: Dict[str, str] = {}
    for split_name, groups in group_sets.items():
        for group in groups:
            previous = seen.get(group)
            if previous is not None and previous != split_name:
                raise ValueError(
                    f"Group {group!r} appears in both {previous!r} and "
                    f"{split_name!r}"
                )
            seen[group] = split_name


def _class_counts(
    indices: Sequence[int],
    samples: Sequence[Sample],
) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for index in indices:
        label = int(samples[index][1])
        counts[label] = counts.get(label, 0) + 1
    return counts


def _validate_class_coverage(
    splits: Dict[str, List[int]],
    samples: Sequence[Sample],
    split_names: Sequence[str],
) -> None:
    labels = sorted({int(label) for _, label in samples})
    if len(labels) < 2:
        return

    for split_name in split_names:
        indices = splits[split_name]
        if not indices:
            continue

        present = set(_class_counts(indices, samples))
        missing = [label for label in labels if label not in present]
        if not missing:
            continue

        warnings.warn(
            f"Split {split_name!r} has {len(indices)} sample(s) but is "
            f"missing class label(s) {missing}; this can happen when a class "
            "has fewer groups than requested partitions",
            stacklevel=2,
        )


def build_split_manifest(
    samples: Sequence[Sample],
    split_result: dict,
    root_dir: str,
) -> dict:
    root = Path(root_dir).resolve()
    records = []

    for split_name, indices in split_result["splits"].items():
        for index in indices:
            path, label = samples[index]
            resolved = Path(path).resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                relative = Path(path).name
            group = None
            if split_result["split_unit"] in {"group", "prefix"}:
                group = extract_group_key(
                    path,
                    split_result["group_key"],
                )
            records.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "label": int(label),
                    "group": group,
                    "split": split_name,
                }
            )

    records.sort(key=lambda item: item["path"])

    return {
        "version": 1,
        "split_unit": split_result["split_unit"],
        "group_key": split_result["group_key"],
        "ratios": split_result["ratios"],
        "seed": split_result.get("seed"),
        "counts": split_result["counts"],
        "class_counts": split_result.get("class_counts", {}),
        "group_counts": split_result.get("group_counts", {}),
        "samples": records,
    }


def save_split_manifest(manifest: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)


def load_split_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def manifest_to_samples(
    manifest: dict,
    root_dir: str,
) -> Tuple[List[Sample], Dict[str, List[int]]]:
    root = Path(root_dir)
    samples: List[Sample] = []
    indices: Dict[str, List[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for record in manifest["samples"]:
        relative = Path(record["path"])
        image_path = relative if relative.is_absolute() else root / relative
        index = len(samples)
        samples.append((image_path, int(record["label"])))
        split_name = record["split"]
        if split_name not in indices:
            indices[split_name] = []
        indices[split_name].append(index)

    return samples, indices
