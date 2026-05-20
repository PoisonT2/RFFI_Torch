from __future__ import annotations

import copy
import json
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from dataset_preparation import ChannelIndSpectrogram, LoadDataset, awgn


@dataclass
class TrainConfig:
    epochs: int = 400
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 60
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.2
    min_delta: float = 2e-3
    min_lr: float = 1e-6
    monitor: str = "val_accuracy"
    monitor_mode: str = "max"
    optimizer: str = "rmsprop"
    seed: int = 42
    device: str = "auto"
    use_amp: bool = True
    log_interval: int = 1


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    class_values: list[int]
    train_stats: dict[str, float]

    @property
    def num_classes(self) -> int:
        return len(self.class_values)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def preserve_rng_state():
    """Temporarily allow deterministic child work without disturbing caller RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    return requested


def _sequence(values: Iterable[int] | Sequence[int]) -> list[int]:
    return [int(v) for v in values]


def _to_file_list(file_paths) -> list[Path]:
    if isinstance(file_paths, (str, Path)):
        return [Path(file_paths)]
    return [Path(p) for p in file_paths]


def _load_iq_from_files(file_paths, dev_range, pkt_range):
    loader = LoadDataset()
    data_parts = []
    label_parts = []
    for file_path in _to_file_list(file_paths):
        data, labels = loader.load_iq_samples(file_path, dev_range, pkt_range)
        data_parts.append(data)
        label_parts.append(labels)
    return np.concatenate(data_parts, axis=0), np.concatenate(label_parts, axis=0)


def map_labels(labels: np.ndarray, class_values: Sequence[int]) -> np.ndarray:
    mapping = {int(value): idx for idx, value in enumerate(class_values)}
    mapped = np.empty(labels.shape[0], dtype=np.int64)
    for idx, label in enumerate(labels.astype(int)):
        if label not in mapping:
            raise ValueError(f"Label {label} is not in class mapping {list(mapping)}")
        mapped[idx] = mapping[label]
    return mapped


def stratified_train_val_split(labels: np.ndarray, val_split: float, seed: int):
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        if cls_indices.size == 1:
            train_indices.extend(cls_indices.tolist())
            continue
        n_val = max(1, int(round(cls_indices.size * val_split)))
        n_val = min(n_val, cls_indices.size - 1)
        val_indices.extend(cls_indices[:n_val].tolist())
        train_indices.extend(cls_indices[n_val:].tolist())

    if len(val_indices) == 0:
        raise ValueError("Validation split is empty; increase samples per class")

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def _compute_train_stats(x_train: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x_train)),
        "std": float(np.std(x_train) + 1e-6),
    }


def _apply_stats(x: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return ((x - stats["mean"]) / stats["std"]).astype(np.float32, copy=False)


def _to_nchw(spec: np.ndarray) -> np.ndarray:
    return spec.transpose(0, 3, 1, 2).astype(np.float32, copy=False)


def prepare_dataset(
    file_paths,
    dev_range,
    pkt_range,
    val_split: float = 0.1,
    seed: int = 42,
    train_snr_range=(20, 80),
    val_snr_range=None,
    snr_distribution: str = "uniform",
    normalize_spectrogram: bool = False,
) -> PreparedData:
    """Load, split, augment, transform, and normalize a training dataset."""
    class_values = _sequence(dev_range)
    iq_data, raw_labels = _load_iq_from_files(file_paths, class_values, pkt_range)
    labels = map_labels(raw_labels, class_values)
    train_idx, val_idx = stratified_train_val_split(labels, val_split, seed)

    rng = np.random.default_rng(seed)
    iq_train = iq_data[train_idx]
    iq_val = iq_data[val_idx]
    if train_snr_range is not None:
        iq_train = awgn(iq_train, train_snr_range, distribution=snr_distribution, rng=rng, copy=True)
    if val_snr_range is not None:
        iq_val = awgn(iq_val, val_snr_range, distribution=snr_distribution, rng=rng, copy=True)

    spec_maker = ChannelIndSpectrogram()
    x_train = _to_nchw(spec_maker.channel_ind_spectrogram(iq_train))
    x_val = _to_nchw(spec_maker.channel_ind_spectrogram(iq_val))
    if normalize_spectrogram:
        stats = _compute_train_stats(x_train)
        x_train = _apply_stats(x_train, stats)
        x_val = _apply_stats(x_val, stats)
    else:
        stats = {"mean": 0.0, "std": 1.0}

    return PreparedData(
        x_train=x_train,
        y_train=labels[train_idx],
        x_val=x_val,
        y_val=labels[val_idx],
        class_values=class_values,
        train_stats=stats,
    )


def prepare_test_dataset(file_path, class_values, pkt_range, train_stats):
    iq_data, raw_labels = _load_iq_from_files(file_path, class_values, pkt_range)
    labels = map_labels(raw_labels, class_values)
    spec = _to_nchw(ChannelIndSpectrogram().channel_ind_spectrogram(iq_data))
    return _apply_stats(spec, train_stats), labels


def _make_loader(x, y, batch_size, shuffle, seed):
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _make_optimizer(model: nn.Module, config: TrainConfig):
    name = config.optimizer.lower()
    if name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _monitor_is_better(current, best, mode, min_delta):
    if mode == "max":
        return current > best + min_delta
    if mode == "min":
        return current < best - min_delta
    raise ValueError(f"Unsupported monitor_mode: {mode}")


def _initial_monitor_value(mode):
    if mode == "max":
        return -float("inf")
    if mode == "min":
        return float("inf")
    raise ValueError(f"Unsupported monitor_mode: {mode}")


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * target.size(0)
        all_targets.append(target.detach().cpu().numpy())
        all_preds.append(output.detach().argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return {
        "loss": total_loss / max(1, len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_model_from_arrays(
    model: nn.Module,
    x_train,
    y_train,
    x_val,
    y_val,
    config: TrainConfig | None = None,
    quiet: bool = False,
):
    config = TrainConfig() if config is None else config
    set_seed(config.seed)
    device = resolve_device(config.device)
    model = model.to(device)
    train_loader = _make_loader(x_train, y_train, config.batch_size, True, config.seed)
    val_loader = _make_loader(x_val, y_val, config.batch_size, False, config.seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, config)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.reduce_lr_factor,
        patience=config.reduce_lr_patience,
        min_lr=config.min_lr,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = copy.deepcopy(model.state_dict())
    best_monitor = _initial_monitor_value(config.monitor_mode)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, use_amp=amp_enabled
        )
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, criterion, device, use_amp=amp_enabled)

        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        if config.monitor not in row:
            raise ValueError(f"Monitor {config.monitor} is not available; choose one of {list(row)}")
        monitor_value = float(row[config.monitor])
        improved = _monitor_is_better(
            monitor_value,
            best_monitor,
            config.monitor_mode,
            config.min_delta,
        )
        same_accuracy_better_loss = (
            config.monitor == "val_accuracy"
            and abs(monitor_value - best_monitor) <= config.min_delta
            and val_metrics["loss"] < best_val_loss - 1e-4
        )
        if improved:
            best_monitor = monitor_value
            best_val_loss = float(val_metrics["loss"])
            best_val_acc = float(val_metrics["accuracy"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        elif same_accuracy_better_loss:
            best_val_loss = float(val_metrics["loss"])
            best_val_acc = float(val_metrics["accuracy"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            patience_counter += 1

        if not quiet and (epoch == 1 or epoch % config.log_interval == 0):
            print(
                f"Epoch {epoch:03d}/{config.epochs}: "
                f"train_loss={row['train_loss']:.4f}, train_acc={row['train_accuracy']:.4f}, "
                f"val_loss={row['val_loss']:.4f}, val_acc={row['val_accuracy']:.4f}, "
                f"lr={row['lr']:.6g}, best_{config.monitor}={best_monitor:.4f}"
            )

        if patience_counter >= config.patience:
            if not quiet:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "monitor": config.monitor,
        "best_monitor": float(best_monitor),
        "epochs_ran": len(history),
        "history": history,
    }


def predict_model(model: nn.Module, x, batch_size=128, device="auto"):
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    loader = DataLoader(torch.as_tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    preds = []
    probs = []
    with torch.no_grad():
        for data in loader:
            output = model(data.to(device))
            prob = torch.softmax(output, dim=1)
            probs.append(prob.cpu().numpy())
            preds.append(prob.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def evaluate_model_from_arrays(model: nn.Module, x, y, batch_size=128, device="auto"):
    y_pred, _ = predict_model(model, x, batch_size=batch_size, device=device)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def save_confusion_matrix(conf_mat, class_values, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(int(v) + 1) for v in class_values]
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        np.asarray(conf_mat),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 7},
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def parameter_summary(model: nn.Module):
    total_params = 0
    trainable_params = 0
    sq_sum = 0.0
    abs_sum = 0.0
    tensor_count = 0
    for param in model.parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        data = param.detach().float()
        sq_sum += float(torch.sum(data ** 2).cpu())
        abs_sum += float(torch.sum(torch.abs(data)).cpu())
        tensor_count += 1
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "parameter_tensors": int(tensor_count),
        "l2_norm": float(np.sqrt(sq_sum)),
        "mean_abs": float(abs_sum / max(1, total_params)),
    }


def compare_model_parameters(reference_name, reference_model, candidate_name, candidate_model):
    reference = parameter_summary(reference_model)
    candidate = parameter_summary(candidate_model)
    return {
        reference_name: reference,
        candidate_name: candidate,
        "candidate_to_reference_param_ratio": float(
            candidate["total_params"] / max(1, reference["total_params"])
        ),
        "parameter_delta": int(candidate["total_params"] - reference["total_params"]),
    }


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_checkpoint(
    output_path,
    model: nn.Module,
    model_name: str,
    class_values,
    train_stats,
    train_config: TrainConfig,
    train_result,
    architecture=None,
    extra=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "class_values": [int(v) for v in class_values],
        "train_stats": dict(train_stats),
        "train_config": asdict(train_config),
        "train_result": train_result,
        "architecture": architecture,
        "parameter_summary": parameter_summary(model),
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, output_path)
