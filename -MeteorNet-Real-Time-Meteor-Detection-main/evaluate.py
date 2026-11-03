import argparse
import warnings
from pathlib import Path

from torch.utils.data import DataLoader

from src.config import (
    ExperimentConfig,
    resolve_data_root,
    validate_config,
)
from src.dataset import (
    MeteorDataset,
    TransformSubset,
    build_transforms,
    discover_samples,
)
from src.engine import evaluate_model, make_criterion
from src.models import build_model
from src.splitting import (
    build_split_manifest,
    limit_samples_per_class,
    load_split_manifest,
    manifest_to_samples,
    save_split_manifest,
    split_samples,
)
from src.utils import (
    calculate_metrics,
    get_device,
    load_checkpoint,
    plot_confusion_matrix,
    plot_roc_curve,
    sanitize_name,
    save_metrics,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a MeteorNet checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=None,
    )
    parser.add_argument(
        "--split-unit",
        choices=["image", "prefix", "group"],
        default=None,
    )
    parser.add_argument(
        "--group-key",
        choices=["prefix", "numeric_id", "filename"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--rebuild-split", action="store_true")
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args) -> None:
    if args.data is not None:
        config.data.root = args.data
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.limit_per_class is not None:
        config.data.max_per_class = (
            None
            if args.limit_per_class == 0
            else args.limit_per_class
        )
    if args.split_ratios is not None:
        config.data.split_ratios = list(args.split_ratios)
    if args.split_unit is not None:
        config.data.split_unit = args.split_unit
    if args.group_key is not None:
        config.data.group_key = args.group_key
        if args.split_unit is None:
            config.data.split_unit = "group"
    if args.seed is not None:
        config.data.seed = args.seed
        config.train.seed = args.seed
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers


def load_or_rebuild_split(
    config: ExperimentConfig,
    args,
    checkpoint,
    data_root,
):
    experiment_name = sanitize_name(config.experiment_name)
    default_path = (
        Path("results")
        / "splits"
        / f"{experiment_name}_split.json"
    )

    if args.split_manifest:
        candidates = [args.split_manifest]
    else:
        candidates = [
            checkpoint.get("split_manifest"),
            str(default_path),
        ]

    if not args.rebuild_split:
        manifest_file = None
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                manifest_file = Path(candidate)
                break

        if manifest_file is not None:
            manifest = load_split_manifest(str(manifest_file))
            samples, indices = manifest_to_samples(
                manifest,
                str(data_root),
            )
            print(f"Loaded split manifest: {manifest_file}")
            return samples, indices, manifest, manifest_file

        checked = ", ".join(
            str(candidate)
            for candidate in candidates
            if candidate
        )
        raise FileNotFoundError(
            f"Split manifest not found. Checked: {checked}. "
            "Re-run training or pass --split-manifest. Use "
            "--rebuild-split only if you accept rebuilding the split."
        )

    samples, _, _ = discover_samples(
        str(data_root),
        classes=config.data.classes,
        extensions=config.data.extensions,
        include_checkpoints=config.data.include_checkpoints,
    )
    if config.data.max_per_class:
        samples = limit_samples_per_class(
            samples,
            config.data.max_per_class,
            config.data.seed,
        )

    split_result = split_samples(
        samples,
        ratios=config.data.split_ratios,
        seed=config.data.seed,
        split_unit=config.data.split_unit,
        group_key=config.data.group_key,
    )
    manifest = build_split_manifest(
        samples,
        split_result,
        str(data_root),
    )
    rebuild_path = (
        Path(args.split_manifest)
        if args.split_manifest
        else default_path
    )
    save_split_manifest(manifest, str(rebuild_path))
    _, indices = manifest_to_samples(manifest, str(data_root))
    print(f"Rebuilt split manifest: {rebuild_path}")
    return samples, indices, manifest, rebuild_path


def main():
    args = parse_args()
    device = get_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint has no config. Retrain with the current "
            "train.py before evaluating."
        )

    config = ExperimentConfig.from_dict(checkpoint["config"])
    apply_overrides(config, args)
    validate_config(config)
    data_root = resolve_data_root(
        config.data.root,
        prefer_cwd=args.data is not None,
    )
    set_seed(config.train.seed)
    print(f"Dataset root: {data_root}")

    samples, indices, manifest, manifest_path = load_or_rebuild_split(
        config,
        args,
        checkpoint,
        data_root,
    )

    split_name = args.split
    split_indices = indices.get(split_name, [])
    if not split_indices:
        available = manifest.get("counts", {})
        # Give an actionable message when the saved manifest pre-dates
        # the 70/15/15 ratio change (test=0 in old 50/50 manifests).
        hint = ""
        if split_name == "test" and available.get("test", 0) == 0:
            hint = (
                " The stored manifest has test=0, which means it was "
                "created with an old split ratio (e.g. 50/50/0). "
                "Re-run training with the updated configs to get a "
                "manifest that includes a test partition, or pass "
                "--split val to evaluate on the validation partition."
            )
        raise ValueError(
            f"Split {split_name!r} is empty. Available counts: "
            f"{available}.{hint}"
        )

    base_dataset = MeteorDataset(
        str(data_root),
        transform=None,
        samples=samples,
        classes=checkpoint.get("classes", config.data.classes),
    )
    eval_transform = build_transforms(
        config.data,
        training=False,
        in_channels=checkpoint.get(
            "in_channels",
            config.model.in_channels,
        ),
    )
    dataset = TransformSubset(
        base_dataset,
        split_indices,
        transform=eval_transform,
    )

    pin_memory = (
        config.data.pin_memory and device.type == "cuda"
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.data.num_workers > 0,
    )

    output_mode = checkpoint.get(
        "output_mode",
        config.model.output_mode,
    )
    num_outputs = 1 if output_mode == "bce" else config.model.num_classes
    model = build_model(
        checkpoint["model_name"],
        num_classes=num_outputs,
        pretrained=False,
        in_channels=checkpoint.get(
            "in_channels",
            config.model.in_channels,
        ),
        dropout=config.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    criterion = make_criterion(output_mode)
    result = evaluate_model(
        model,
        loader,
        criterion,
        device,
        output_mode,
        amp=config.train.amp,
    )
    metrics = calculate_metrics(
        result["labels"],
        result["predictions"],
        result["meteor_probabilities"],
    )
    metrics["loss"] = result["loss"]
    metrics["split"] = split_name
    metrics["split_manifest"] = str(manifest_path)
    metrics["checkpoint"] = str(args.checkpoint)

    experiment_name = sanitize_name(config.experiment_name)
    prefix = args.output_prefix or f"{experiment_name}_{split_name}"
    metrics_path = (
        Path("results") / "metrics" / f"{prefix}_metrics.json"
    )
    graphs_dir = Path("results") / "graphs"
    save_metrics(metrics, str(metrics_path))

    class_names = checkpoint.get("classes", config.data.classes)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        str(graphs_dir / f"{prefix}_confusion_matrix.png"),
        class_names=class_names,
    )
    if metrics.get("roc_curve"):
        plot_roc_curve(
            metrics["roc_curve"],
            metrics["roc_auc"],
            str(graphs_dir / f"{prefix}_roc.png"),
        )

    print("\nEvaluation Results")
    print("------------------")
    print(f"Split: {split_name}")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print()
    print("Meteor (positive class) metrics")
    print(f"  Precision : {metrics['meteor_precision']:.4f}")
    print(f"  Recall    : {metrics['meteor_recall']:.4f}")
    print(f"  F1        : {metrics['meteor_f1']:.4f}")
    print()
    print(f"Macro F1   : {metrics['f1_macro']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC    : {metrics['roc_auc']:.4f}")
    else:
        warnings.warn(
            "ROC-AUC could not be computed for this evaluation "
            "(single class in split or missing probability scores).",
        )
        print("ROC AUC    : N/A (see warning above)")
    print(f"Loss       : {metrics['loss']:.4f}")
    print("\nPer-class metrics")
    for name, values in metrics["per_class"].items():
        print(
            f"{name}: precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} "
            f"f1={values['f1']:.4f} "
            f"support={values['support']}"
        )
    print("\nConfusion matrix")
    for row in metrics["confusion_matrix"]:
        print(row)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
