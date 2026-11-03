from contextlib import nullcontext

import torch
import torch.nn as nn

from .config import TrainConfig
from .utils import AverageMeter


def make_criterion(
    output_mode: str,
    label_smoothing: float = 0.0,
):
    if output_mode == "bce":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def make_optimizer(
    model: nn.Module,
    train_config: TrainConfig,
):
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if train_config.optimizer == "adam":
        return torch.optim.Adam(
            parameters,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

    if train_config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )

    if train_config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=train_config.learning_rate,
            momentum=train_config.momentum,
            weight_decay=train_config.weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {train_config.optimizer}"
    )


def make_scheduler(
    optimizer,
    train_config: TrainConfig,
):
    if train_config.scheduler == "none":
        return None

    if train_config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.scheduler_step_size,
            gamma=train_config.scheduler_gamma,
        )

    if train_config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.epochs,
            eta_min=train_config.scheduler_eta_min,
        )

    raise ValueError(
        f"Unsupported scheduler: {train_config.scheduler}"
    )


def make_grad_scaler(
    device: torch.device,
    enabled: bool,
):
    enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler(
            device.type,
            enabled=enabled,
        )
    except (AttributeError, TypeError, ValueError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(
    device: torch.device,
    enabled: bool,
):
    enabled = enabled and device.type == "cuda"
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(
            device_type=device.type,
            enabled=True,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def outputs_to_probabilities(
    outputs: torch.Tensor,
    output_mode: str,
):
    if output_mode == "bce":
        non_meteor_probability = torch.sigmoid(outputs).reshape(-1)
        meteor_probability = 1.0 - non_meteor_probability
        predictions = (
            non_meteor_probability >= 0.5
        ).long()
        return predictions, meteor_probability, non_meteor_probability

    probabilities = torch.softmax(outputs, dim=1)
    predictions = probabilities.argmax(dim=1)
    meteor_probability = probabilities[:, 0]
    if probabilities.shape[1] > 1:
        non_meteor_probability = probabilities[:, 1]
    else:
        non_meteor_probability = 1.0 - meteor_probability
    return predictions, meteor_probability, non_meteor_probability


def _targets_for_loss(
    labels: torch.Tensor,
    output_mode: str,
):
    if output_mode == "bce":
        return labels.float().unsqueeze(1)
    return labels.long()


def _raise_if_oom(exc: RuntimeError, device: torch.device) -> None:
    """Re-raise a CUDA out-of-memory RuntimeError with a clear message.

    Detects both ``torch.cuda.OutOfMemoryError`` (PyTorch >= 1.12) and
    older ``RuntimeError`` whose message contains 'out of memory'.
    If the exception is not OOM-related the function returns without
    raising, allowing the original ``raise`` in the caller to propagate.
    """
    is_oom = isinstance(
        exc,
        getattr(torch.cuda, "OutOfMemoryError", type(None)),
    ) or "out of memory" in str(exc).lower()

    if is_oom and device.type == "cuda":
        raise RuntimeError(
            "CUDA ran out of GPU memory during training or evaluation.\n"
            "Suggestions (do NOT silently change other settings):\n"
            "  1. Reduce --batch-size (e.g. halve it).\n"
            "  2. Enable mixed-precision if not already on: --amp.\n"
            "  3. Reduce --image-size.\n"
            "  4. Use a smaller model (e.g. --model meteor).\n"
            f"Original error: {exc}"
        ) from exc


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    output_mode: str,
    scaler,
    amp: bool = True,
    grad_clip: float = 0.0,
) -> dict:
    model.train()
    loss_meter = AverageMeter()
    labels_list = []
    predictions_list = []
    probabilities_list = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = _targets_for_loss(labels, output_mode)

        optimizer.zero_grad(set_to_none=True)

        try:
            with autocast_context(device, amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

        except RuntimeError as exc:
            _raise_if_oom(exc, device)
            raise

        loss_meter.update(loss.item(), images.size(0))
        if hasattr(loader, "set_postfix"):
            loader.set_postfix(loss=f"{loss.item():.4f}")

        predictions, meteor_probability, _ = outputs_to_probabilities(
            outputs.detach(),
            output_mode,
        )
        labels_list.append(labels.detach().cpu())
        predictions_list.append(predictions.detach().cpu())
        probabilities_list.append(
            meteor_probability.detach().cpu()
        )

    return _pack_epoch_result(
        loss_meter.average,
        labels_list,
        predictions_list,
        probabilities_list,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    output_mode: str,
    amp: bool = True,
) -> dict:
    model.eval()
    loss_meter = AverageMeter()
    labels_list = []
    predictions_list = []
    probabilities_list = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = _targets_for_loss(labels, output_mode)

        try:
            with autocast_context(device, amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
        except RuntimeError as exc:
            _raise_if_oom(exc, device)
            raise

        loss_meter.update(loss.item(), images.size(0))

        predictions, meteor_probability, _ = outputs_to_probabilities(
            outputs.detach(),
            output_mode,
        )
        labels_list.append(labels.detach().cpu())
        predictions_list.append(predictions.detach().cpu())
        probabilities_list.append(
            meteor_probability.detach().cpu()
        )

    return _pack_epoch_result(
        loss_meter.average,
        labels_list,
        predictions_list,
        probabilities_list,
    )


def _pack_epoch_result(
    loss: float,
    labels_list,
    predictions_list,
    probabilities_list,
) -> dict:
    if labels_list:
        labels = torch.cat(labels_list).tolist()
        predictions = torch.cat(predictions_list).tolist()
        probabilities = torch.cat(probabilities_list).tolist()
    else:
        labels = []
        predictions = []
        probabilities = []

    return {
        "loss": float(loss),
        "labels": labels,
        "predictions": predictions,
        "meteor_probabilities": probabilities,
    }
