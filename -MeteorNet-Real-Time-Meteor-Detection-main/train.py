import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    ExperimentConfig,
    load_config,
    resolve_data_root,
    validate_config,
)
from src.dataset import (
    MeteorDataset,
    TransformSubset,
    build_transforms,
    discover_samples,
)
from src.engine import (
    evaluate_model,
    make_criterion,
    make_grad_scaler,
    make_optimizer,
    make_scheduler,
    train_one_epoch,
)
from src.models import MODEL_NAMES, build_model
from src.splitting import (
    build_split_manifest,
    limit_samples_per_class,
    load_split_manifest,
    manifest_to_samples,
    save_split_manifest,
    split_samples,
)
from src.utils import (
    EarlyStopping,
    calculate_metrics,
    get_device,
    load_checkpoint,
    plot_confusion_matrix,
    plot_history,
    plot_roc_curve,
    sanitize_name,
    save_checkpoint,
    save_json,
    save_metrics,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MeteorNet classification model."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES + ["yolo"],
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
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
    parser.add_argument(
        "--output-mode",
        choices=["ce", "bce"],
        default=None,
    )
    parser.add_argument(
        "--monitor",
        choices=["val_accuracy", "val_loss", "val_f1_macro"],
        default=None,
    )
    parser.add_argument("--rotation-degrees", type=float, default=None)
    parser.add_argument("--horizontal-flip", type=float, default=None)
    parser.add_argument("--vertical-flip", type=float, default=None)
    parser.add_argument("--color-jitter", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
    )
    return parser.parse_args()


def apply_overrides(config: ExperimentConfig, args) -> None:
    if args.data is not None:
        config.data.root = args.data
    if args.model is not None:
        config.model.name = args.model
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    if args.image_size is not None:
        config.data.image_size = args.image_size
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
    if args.output_mode is not None:
        config.model.output_mode = args.output_mode
    if args.monitor is not None:
        config.train.monitor = args.monitor
    if args.rotation_degrees is not None:
        config.data.rotation_degrees = args.rotation_degrees
    if args.horizontal_flip is not None:
        config.data.horizontal_flip = args.horizontal_flip
    if args.vertical_flip is not None:
        config.data.vertical_flip = args.vertical_flip
    if args.color_jitter is not None:
        config.data.color_jitter = args.color_jitter
    if args.early_stopping_patience is not None:
        config.train.early_stopping_patience = (
            args.early_stopping_patience
        )
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.seed is not None:
        config.data.seed = args.seed
        config.train.seed = args.seed
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.pretrained is not None:
        config.model.pretrained = args.pretrained
    if args.amp is not None:
        config.train.amp = args.amp
    if args.deterministic is not None:
        config.train.deterministic = args.deterministic
    if args.evaluate_test is not None:
        config.train.evaluate_test = args.evaluate_test


def build_data(config: ExperimentConfig, data_root):
    samples, class_names, class_to_idx = discover_samples(
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

    if not split_result["splits"]["train"]:
        raise ValueError("Training split is empty")
    if not split_result["splits"]["val"]:
        raise ValueError("Validation split is empty")

    return (
        samples,
        class_names,
        class_to_idx,
        split_result,
    )


def build_loaders(
    config: ExperimentConfig,
    samples,
    class_names,
    split_result,
    device,
    data_root,
):
    base_dataset = MeteorDataset(
        str(data_root),
        transform=None,
        samples=samples,
        classes=class_names,
    )

    train_transform = build_transforms(
        config.data,
        training=True,
        in_channels=config.model.in_channels,
    )
    eval_transform = build_transforms(
        config.data,
        training=False,
        in_channels=config.model.in_channels,
    )

    train_dataset = TransformSubset(
        base_dataset,
        split_result["splits"]["train"],
        transform=train_transform,
    )
    val_dataset = TransformSubset(
        base_dataset,
        split_result["splits"]["val"],
        transform=eval_transform,
    )

    test_dataset = None
    if split_result["splits"]["test"]:
        test_dataset = TransformSubset(
            base_dataset,
            split_result["splits"]["test"],
            transform=eval_transform,
        )

    pin_memory = (
        config.data.pin_memory and device.type == "cuda"
    )
    persistent_workers = config.data.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    return train_loader, val_loader, test_loader


def checkpoint_state(
    config: ExperimentConfig,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_score,
    class_names,
    class_to_idx,
    history,
    split_manifest_path,
):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "model_name": config.model.name,
        "output_mode": config.model.output_mode,
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "image_size": config.data.image_size,
        "mean": config.data.mean,
        "std": config.data.std,
        "grayscale": config.data.grayscale,
        "in_channels": config.model.in_channels,
        "config": config.to_dict(),
        "history": history,
        "split_manifest": str(split_manifest_path),
    }


def main():
    args = parse_args()
    device = get_device(args.device)

    models_dir = Path("models")
    results_dir = Path("results")
    metrics_dir = results_dir / "metrics"
    graphs_dir = results_dir / "graphs"
    splits_dir = results_dir / "splits"
    for directory in [
        models_dir,
        metrics_dir,
        graphs_dir,
        splits_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Determine initial config / experiment name for auto-resume resolution
    base_config = load_config(args.config)
    apply_overrides(base_config, args)
    experiment_name = sanitize_name(base_config.experiment_name)
    best_path = models_dir / f"{experiment_name}_best.pth"
    last_path = models_dir / f"{experiment_name}_last.pth"

    resume_path = args.resume
    if resume_path == "auto":
        resume_path = str(last_path)

    checkpoint = None
    if resume_path:
        if not Path(resume_path).is_file():
            if args.resume == "auto":
                print(
                    f"Info: No checkpoint found at {resume_path!r}; "
                    "starting from scratch."
                )
                resume_path = None
            else:
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_path!r}. "
                    "Pass --resume without a path to auto-resume from the "
                    "last checkpoint, or omit --resume to start fresh."
                )
        else:
            checkpoint = load_checkpoint(resume_path, device)
            print(f"Loaded checkpoint for resume: {resume_path}")

    if checkpoint is not None:
        if "config" in checkpoint and args.config is None:
            config = ExperimentConfig.from_dict(checkpoint["config"])
        else:
            config = base_config
        apply_overrides(config, args)
        if args.model is None and "model_name" in checkpoint:
            config.model.name = checkpoint["model_name"]
        if args.output_mode is None and "output_mode" in checkpoint:
            config.model.output_mode = checkpoint["output_mode"]
    else:
        config = base_config

    validate_config(config)
    data_root = resolve_data_root(
        config.data.root,
        prefer_cwd=args.data is not None,
    )

    set_seed(
        config.train.seed,
        deterministic=config.train.deterministic,
    )
    print(f"Using device: {device}")
    if config.train.amp and device.type != "cuda":
        print(
            "Info: AMP (mixed precision) is configured but the selected "
            "device is not CUDA. AMP will be disabled for this run. "
            "On the GPU machine, AMP will activate automatically."
        )
    print(f"Dataset root: {data_root}")

    # Re-use existing split manifest when resuming, if available
    split_loaded = False
    if checkpoint is not None:
        manifest_candidates = [
            checkpoint.get("split_manifest"),
            str(splits_dir / f"{experiment_name}_split.json"),
        ]
        for candidate in manifest_candidates:
            if candidate and Path(candidate).is_file():
                try:
                    manifest = load_split_manifest(str(candidate))
                    samples, indices = manifest_to_samples(
                        manifest,
                        str(data_root),
                    )
                    split_result = {
                        "splits": indices,
                        "counts": {k: len(v) for k, v in indices.items()},
                        "split_unit": manifest.get(
                            "split_unit",
                            config.data.split_unit,
                        ),
                        "group_key": manifest.get(
                            "group_key",
                            config.data.group_key,
                        ),
                        "ratios": manifest.get(
                            "ratios",
                            config.data.split_ratios,
                        ),
                        "seed": manifest.get("seed", config.data.seed),
                    }
                    class_names = checkpoint.get("classes", config.data.classes)
                    class_to_idx = checkpoint.get(
                        "class_to_idx",
                        {c: i for i, c in enumerate(class_names)},
                    )
                    split_manifest_path = Path(candidate)
                    split_loaded = True
                    print(
                        f"Loaded existing split manifest for resume: {split_manifest_path}"
                    )
                    break
                except Exception as exc:
                    warnings.warn(
                        f"Could not load manifest from {candidate}: {exc}"
                    )

    if not split_loaded:
        (
            samples,
            class_names,
            class_to_idx,
            split_result,
        ) = build_data(config, data_root)
        split_manifest = build_split_manifest(
            samples,
            split_result,
            str(data_root),
        )
        split_manifest_path = (
            splits_dir / f"{experiment_name}_split.json"
        )
        save_split_manifest(split_manifest, str(split_manifest_path))
        print(f"Split manifest: {split_manifest_path}")

    print(f"Classes: {class_to_idx}")
    if (
        config.data.split_unit == "group"
        and config.data.group_key == "numeric_id"
    ):
        print(
            "Warning: numeric_id groups are shared across classes; "
            "this split can leak source IDs across splits."
        )
    print(
        "Split counts: "
        f"train={split_result['counts']['train']}, "
        f"val={split_result['counts']['val']}, "
        f"test={split_result['counts']['test']}"
    )

    save_json(
        config.to_dict(),
        str(metrics_dir / f"{experiment_name}_config.json"),
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = build_loaders(
        config,
        samples,
        class_names,
        split_result,
        device,
        data_root,
    )

    num_outputs = (
        1
        if config.model.output_mode == "bce"
        else config.model.num_classes
    )
    model = build_model(
        config.model.name,
        num_classes=num_outputs,
        pretrained=config.model.pretrained,
        in_channels=config.model.in_channels,
        dropout=config.model.dropout,
    )
    model = model.to(device)

    criterion = make_criterion(
        config.model.output_mode,
        label_smoothing=config.train.label_smoothing,
    )
    optimizer = make_optimizer(model, config.train)
    scheduler = make_scheduler(optimizer, config.train)
    scaler = make_grad_scaler(device, config.train.amp)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1_macro": [],
        "val_f1_macro": [],
        "val_roc_auc": [],
        "learning_rate": [],
    }

    monitor_metric = {
        "val_accuracy": "accuracy",
        "val_loss": "loss",
        "val_f1_macro": "f1_macro",
    }[config.train.monitor]

    if config.train.monitor == "val_loss":
        best_score = float("inf")
        early_mode = "min"
    else:
        best_score = float("-inf")
        early_mode = "max"

    start_epoch = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if (
            scheduler is not None
            and checkpoint.get("scheduler_state_dict") is not None
        ):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        history = checkpoint.get("history", history)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    early_stopping = EarlyStopping(
        patience=config.train.early_stopping_patience,
        mode=early_mode,
    )
    if start_epoch > 0 and early_stopping.patience > 0:
        early_stopping.best = best_score

    for epoch in range(start_epoch, config.train.epochs):
        train_progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1}/{config.train.epochs}"
            ),
            leave=False,
        )
        train_result = train_one_epoch(
            model,
            train_progress,
            criterion,
            optimizer,
            device,
            config.model.output_mode,
            scaler,
            amp=config.train.amp,
            grad_clip=config.train.grad_clip,
        )
        train_progress.close()
        train_metrics = calculate_metrics(
            train_result["labels"],
            train_result["predictions"],
            train_result["meteor_probabilities"],
        )
        train_metrics["loss"] = train_result["loss"]

        val_result = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            config.model.output_mode,
            amp=config.train.amp,
        )
        val_metrics = calculate_metrics(
            val_result["labels"],
            val_result["predictions"],
            val_result["meteor_probabilities"],
        )
        val_metrics["loss"] = val_result["loss"]

        learning_rate = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_result["loss"])
        history["val_loss"].append(val_result["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_f1_macro"].append(train_metrics["f1_macro"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["learning_rate"].append(learning_rate)

        if scheduler is not None:
            scheduler.step()

        score = val_metrics[monitor_metric]
        if config.train.monitor == "val_loss":
            improved = score < best_score
        else:
            improved = score > best_score

        if improved:
            best_score = score
            save_checkpoint(
                checkpoint_state(
                    config,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_score,
                    class_names,
                    class_to_idx,
                    history,
                    split_manifest_path,
                ),
                str(best_path),
            )

        save_checkpoint(
            checkpoint_state(
                config,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_score,
                class_names,
                class_to_idx,
                history,
                split_manifest_path,
            ),
            str(last_path),
        )

        auc_display = (
            f"{val_metrics['roc_auc']:.4f}"
            if val_metrics["roc_auc"] is not None
            else "N/A"
        )
        print(
            f"Epoch {epoch + 1}/{config.train.epochs} "
            f"train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"val_auc={auc_display}"
        )

        if early_stopping.step(score):
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"after {early_stopping.counter} epochs without "
                "improvement."
            )
            break

    history_path = metrics_dir / f"{experiment_name}_history.json"
    save_json(history, str(history_path))
    graph_paths = plot_history(
        history,
        str(graphs_dir),
        experiment_name,
    )

    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"History: {history_path}")
    for path in graph_paths:
        print(f"Graph: {path}")

    if (
        config.train.evaluate_test
        and test_loader is not None
        and best_path.is_file()
    ):
        checkpoint = load_checkpoint(str(best_path), device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_result = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            config.model.output_mode,
            amp=config.train.amp,
        )
        test_metrics = calculate_metrics(
            test_result["labels"],
            test_result["predictions"],
            test_result["meteor_probabilities"],
        )
        test_metrics["loss"] = test_result["loss"]
        test_metrics_path = (
            metrics_dir / f"{experiment_name}_test_metrics.json"
        )
        save_metrics(test_metrics, str(test_metrics_path))
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            str(
                graphs_dir
                / f"{experiment_name}_test_confusion_matrix.png"
            ),
            class_names=class_names,
        )
        if test_metrics.get("roc_curve"):
            plot_roc_curve(
                test_metrics["roc_curve"],
                test_metrics["roc_auc"],
                str(graphs_dir / f"{experiment_name}_test_roc.png"),
            )
        print(f"Test metrics: {test_metrics_path}")


if __name__ == "__main__":
    main()
