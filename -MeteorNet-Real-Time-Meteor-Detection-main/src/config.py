import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = (
    PROJECT_ROOT.parent / "D_2 DATASET (METEORS)" / "sum_images"
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_data_root(
    root: Optional[str] = None,
    *,
    base_dir: Optional[Path] = None,
    prefer_cwd: bool = False,
) -> Path:
    """Resolve the dataset root independently of the terminal CWD."""
    project_root = (
        Path(base_dir).expanduser().resolve()
        if base_dir is not None
        else PROJECT_ROOT
    )
    raw_root = None if root is None else str(root).strip()

    if not raw_root:
        candidates = [
            DEFAULT_DATA_ROOT,
            project_root
            / "D_2 DATASET (METEORS)"
            / "sum_images",
        ]
    else:
        candidate = Path(raw_root).expanduser()
        if candidate.is_absolute():
            candidates = [candidate]
        else:
            cwd_candidate = (Path.cwd() / candidate).resolve()
            project_candidate = (
                project_root / candidate
            ).resolve()
            if prefer_cwd:
                candidates = [cwd_candidate, project_candidate]
            else:
                candidates = [project_candidate, cwd_candidate]

    checked = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(resolved)
        if resolved.is_dir():
            return resolved

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Dataset root not found. Checked:\n"
        f"{checked_text}\n"
        "Pass --data with an absolute path or update data.root in the "
        "configuration."
    )


@dataclass
class DataConfig:
    root: str = str(DEFAULT_DATA_ROOT)
    classes: List[str] = field(default_factory=lambda: ["meteor", "non_meteor"])
    extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"]
    )
    image_size: int = 224
    grayscale: bool = False
    normalize: bool = True
    mean: List[float] = field(
        default_factory=lambda: list(IMAGENET_MEAN)
    )
    std: List[float] = field(
        default_factory=lambda: list(IMAGENET_STD)
    )
    rotation_min: float = 0.0
    rotation_degrees: float = 0.0
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    color_jitter: float = 0.0
    random_resized_crop: bool = False
    max_per_class: Optional[int] = None
    include_checkpoints: bool = False
    split_unit: str = "prefix"
    group_key: str = "prefix"
    split_ratios: List[float] = field(
        default_factory=lambda: [0.7, 0.15, 0.15]
    )
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class ModelConfig:
    name: str = "resnet18"
    pretrained: bool = False
    output_mode: str = "ce"
    num_classes: int = 2
    dropout: float = 0.4
    in_channels: int = 3


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9
    scheduler: str = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    scheduler_eta_min: float = 0.0
    label_smoothing: float = 0.0
    amp: bool = True
    grad_clip: float = 0.0
    monitor: str = "val_accuracy"
    early_stopping_patience: int = 0
    evaluate_test: bool = False
    deterministic: bool = False
    seed: int = 42


@dataclass
class ExperimentConfig:
    experiment_name: str = "meteornet"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        config = cls()
        _merge_dataclass(config, data)
        validate_config(config)
        return config

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_dataclass(target: Any, data: dict) -> None:
    for item in fields(target):
        if item.name not in data:
            continue
        current = getattr(target, item.name)
        value = data[item.name]
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(target, item.name, value)


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    if path is None:
        config = ExperimentConfig()
    else:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        config = ExperimentConfig.from_dict(data)
    validate_config(config)
    return config


def save_config(config: ExperimentConfig, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=4)


def validate_config(config: ExperimentConfig) -> None:
    data = config.data
    model = config.model
    train = config.train

    if data.image_size <= 0:
        raise ValueError("data.image_size must be positive")
    if len(data.classes) != 2:
        raise ValueError("data.classes must contain exactly 2 classes")
    if not data.extensions:
        raise ValueError("data.extensions cannot be empty")
    if not 0.0 <= data.horizontal_flip <= 1.0:
        raise ValueError("data.horizontal_flip must be between 0 and 1")
    if not 0.0 <= data.vertical_flip <= 1.0:
        raise ValueError("data.vertical_flip must be between 0 and 1")
    if data.rotation_degrees < 0:
        raise ValueError("data.rotation_degrees cannot be negative")
    if data.rotation_min > data.rotation_degrees:
        raise ValueError(
            "data.rotation_min cannot be greater than data.rotation_degrees"
        )
    if data.color_jitter < 0:
        raise ValueError("data.color_jitter cannot be negative")
    if len(data.mean) != 3 or len(data.std) != 3:
        raise ValueError("data.mean and data.std must contain 3 values")
    if any(value <= 0 for value in data.std):
        raise ValueError("data.std values must be positive")
    if data.split_unit not in {"image", "prefix", "group"}:
        raise ValueError(
            "data.split_unit must be 'image', 'prefix', or 'group'"
        )
    if data.group_key not in {"prefix", "numeric_id", "filename"}:
        raise ValueError(
            "data.group_key must be 'prefix', 'numeric_id', or 'filename'"
        )
    if len(data.split_ratios) != 3:
        raise ValueError("data.split_ratios must contain 3 values")
    if any(value < 0 for value in data.split_ratios):
        raise ValueError("data.split_ratios cannot be negative")
    if sum(data.split_ratios) <= 0:
        raise ValueError("data.split_ratios must have a positive sum")
    if data.max_per_class is not None and data.max_per_class <= 0:
        raise ValueError("data.max_per_class must be positive or null")
    if data.num_workers < 0:
        raise ValueError("data.num_workers cannot be negative")

    if model.output_mode not in {"ce", "bce"}:
        raise ValueError("model.output_mode must be 'ce' or 'bce'")
    if model.output_mode == "ce" and model.num_classes != 2:
        raise ValueError(
            "model.num_classes must be 2 when model.output_mode is 'ce'"
        )
    if model.in_channels not in {1, 3}:
        raise ValueError("model.in_channels must be 1 or 3")
    if not data.grayscale and model.in_channels != 3:
        raise ValueError(
            "model.in_channels must be 3 when data.grayscale is false"
        )
    if not 0.0 <= model.dropout <= 1.0:
        raise ValueError("model.dropout must be between 0 and 1")

    if train.epochs <= 0:
        raise ValueError("train.epochs must be positive")
    if train.batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    if train.learning_rate <= 0:
        raise ValueError("train.learning_rate must be positive")
    if not 0.0 <= train.label_smoothing < 1.0:
        raise ValueError("train.label_smoothing must be in [0, 1)")
    if train.grad_clip < 0:
        raise ValueError("train.grad_clip cannot be negative")
    if train.scheduler_step_size <= 0:
        raise ValueError("train.scheduler_step_size must be positive")
    if train.scheduler_gamma <= 0:
        raise ValueError("train.scheduler_gamma must be positive")
    if train.scheduler_eta_min < 0:
        raise ValueError("train.scheduler_eta_min cannot be negative")
    if train.optimizer not in {"adam", "adamw", "sgd"}:
        raise ValueError("train.optimizer must be 'adam', 'adamw', or 'sgd'")
    if train.scheduler not in {"none", "step", "cosine"}:
        raise ValueError(
            "train.scheduler must be 'none', 'step', or 'cosine'"
        )
    if train.monitor not in {
        "val_accuracy",
        "val_loss",
        "val_f1_macro",
    }:
        raise ValueError(
            "train.monitor must be 'val_accuracy', 'val_loss', or "
            "'val_f1_macro'"
        )
    if train.early_stopping_patience < 0:
        raise ValueError(
            "train.early_stopping_patience cannot be negative"
        )
