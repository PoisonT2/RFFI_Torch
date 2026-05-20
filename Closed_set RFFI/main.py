from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deep_learning_models import ClassificationNet
from training_utils import (
    TrainConfig,
    evaluate_model_from_arrays,
    prepare_dataset,
    prepare_test_dataset,
    save_checkpoint,
    save_confusion_matrix,
    save_json,
    set_seed,
)


DATA_ROOT = Path("./LoRa_RFF") / "dataset"
CHANNEL_DIR = DATA_ROOT / "Test" / "channel_problem"


def _default_test_files():
    return [CHANNEL_DIR / name for name in ["B.h5", "C.h5", "D.h5", "E.h5", "F.h5"]]


def _channel_training_files(train_file, use_channel_augmentations=True):
    train_file = Path(train_file)
    if not use_channel_augmentations or train_file.name != "A.h5":
        return train_file
    aug_names = ["A_aug_0hz.h5", "A_aug_10hz.h5", "A_aug_30hz.h5", "A_aug_50hz.h5", "A_aug_100hz.h5"]
    files = [train_file]
    files.extend(train_file.parent / name for name in aug_names if (train_file.parent / name).exists())
    return files


def _range(start, stop):
    return range(int(start), int(stop))


def build_train_config(args) -> TrainConfig:
    monitor_mode = args.monitor_mode
    if monitor_mode == "auto":
        monitor_mode = "min" if args.monitor == "val_loss" else "max"
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        min_delta=args.min_delta,
        min_lr=args.min_lr,
        monitor=args.monitor,
        monitor_mode=monitor_mode,
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
    )


def train_cnn(
    train_file,
    dev_range=range(30, 40),
    pkt_range=range(0, 200),
    val_split=0.1,
    train_config: TrainConfig | None = None,
    model_out="cnn.pth",
    train_snr_range=(20, 80),
    results_dir="results/cnn",
    normalize_spectrogram=False,
):
    train_config = TrainConfig() if train_config is None else train_config
    set_seed(train_config.seed)
    data = prepare_dataset(
        train_file,
        dev_range=dev_range,
        pkt_range=pkt_range,
        val_split=val_split,
        seed=train_config.seed,
        train_snr_range=train_snr_range,
        normalize_spectrogram=normalize_spectrogram,
    )

    model = ClassificationNet(data.num_classes)
    model, train_result = train_model_and_save(
        model=model,
        data=data,
        train_config=train_config,
        model_out=model_out,
        results_dir=results_dir,
    )
    return model, data, train_result


def train_model_and_save(model, data, train_config, model_out, results_dir):
    from training_utils import train_model_from_arrays

    model, train_result = train_model_from_arrays(
        model,
        data.x_train,
        data.y_train,
        data.x_val,
        data.y_val,
        config=train_config,
    )
    model.class_values = data.class_values
    model.train_stats = data.train_stats
    model.train_config = train_config

    results_dir = Path(results_dir)
    save_checkpoint(
        model_out,
        model,
        model_name="cnn_baseline",
        class_values=data.class_values,
        train_stats=data.train_stats,
        train_config=train_config,
        train_result=train_result,
    )
    save_json(train_result, results_dir / "train_history.json")
    print(f"Saved CNN checkpoint to {model_out}")
    return model, train_result


def evaluate_files(model, test_files, class_values, pkt_range, train_stats, results_dir, batch_size=128, device="auto"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for file_path in test_files:
        file_path = Path(file_path)
        print(f"\nTesting on {file_path}")
        x_test, y_test = prepare_test_dataset(file_path, class_values, pkt_range, train_stats)
        metrics = evaluate_model_from_arrays(model, x_test, y_test, batch_size=batch_size, device=device)
        results[file_path.name] = {
            key: value for key, value in metrics.items() if key != "confusion_matrix"
        }
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_values,
            results_dir / f"confusion_matrix_{file_path.stem}.pdf",
        )
        print(
            f"{file_path.name}: acc={metrics['accuracy']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
        )
    save_json(results, results_dir / "test_metrics.json")
    return results


def _load_checkpoint(checkpoint_path, device="auto"):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_values = checkpoint["class_values"]
    model = ClassificationNet(len(class_values))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_values = class_values
    model.train_stats = checkpoint["train_stats"]
    return model, checkpoint


def run_channel_experiment(args):
    dev_range = _range(args.dev_start, args.dev_stop)
    pkt_range = _range(args.pkt_start, args.pkt_stop)
    config = build_train_config(args)
    train_snr_range = None if args.no_train_noise else (args.train_snr_min, args.train_snr_max)
    train_files = _channel_training_files(args.train_file, args.use_channel_augmentations)
    print(f"Training files: {[str(p) for p in train_files] if isinstance(train_files, list) else [str(train_files)]}")

    model, data, train_result = train_cnn(
        train_files,
        dev_range=dev_range,
        pkt_range=pkt_range,
        val_split=args.val_split,
        train_config=config,
        model_out=args.model_out,
        train_snr_range=train_snr_range,
        results_dir=args.results_dir,
        normalize_spectrogram=args.normalize_spectrogram,
    )
    print(f"Best validation accuracy: {train_result['best_val_accuracy']:.4f}")
    return evaluate_files(
        model,
        args.test_files,
        data.class_values,
        pkt_range,
        data.train_stats,
        args.results_dir,
        batch_size=args.eval_batch_size,
        device=args.device,
    )


def train(file_path_in, dev_range=range(30, 40), pkt_range=range(0, 200)):
    """Compatibility wrapper for older imports."""
    model, _, _ = train_cnn(file_path_in, dev_range=dev_range, pkt_range=pkt_range)
    return model


def test(file_path_in, model, dev_range=range(30, 40), pkt_range=range(0, 200)):
    """Compatibility wrapper for older imports."""
    class_values = getattr(model, "class_values", list(dev_range))
    train_stats = getattr(model, "train_stats", {"mean": 0.0, "std": 1.0})
    x_test, y_test = prepare_test_dataset(file_path_in, class_values, pkt_range, train_stats)
    metrics = evaluate_model_from_arrays(model, x_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics["accuracy"]


def parse_args():
    parser = argparse.ArgumentParser(description="LoRa RFFI CNN training and evaluation")
    parser.add_argument("--mode", choices=["channel", "train", "test", "smoke"], default="channel")
    parser.add_argument("--train-file", type=Path, default=CHANNEL_DIR / "A.h5")
    parser.add_argument("--test-files", type=Path, nargs="*", default=_default_test_files())
    parser.add_argument("--checkpoint", type=Path, default=Path("cnn.pth"))
    parser.add_argument("--model-out", type=Path, default=Path("cnn.pth"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "cnn")
    parser.add_argument("--dev-start", type=int, default=30)
    parser.add_argument("--dev-stop", type=int, default=40)
    parser.add_argument("--pkt-start", type=int, default=0)
    parser.add_argument("--pkt-stop", type=int, default=200)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.2)
    parser.add_argument("--min-delta", type=float, default=2e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--monitor", choices=["val_loss", "val_accuracy"], default="val_accuracy")
    parser.add_argument("--monitor-mode", choices=["auto", "min", "max"], default="auto")
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="rmsprop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--train-snr-min", type=float, default=20.0)
    parser.add_argument("--train-snr-max", type=float, default=80.0)
    parser.add_argument("--no-train-noise", action="store_true")
    parser.add_argument("--use-channel-augmentations", dest="use_channel_augmentations", action="store_true", default=True)
    parser.add_argument("--no-channel-augmentations", dest="use_channel_augmentations", action="store_false")
    parser.add_argument("--normalize-spectrogram", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "smoke":
        args.epochs = 1
        args.batch_size = 8
        args.eval_batch_size = 16
        args.pkt_stop = min(args.pkt_stop, args.pkt_start + 2)
        args.test_files = args.test_files[:1]
        args.results_dir = Path("results") / "smoke_cnn"
        args.model_out = Path("results") / "smoke_cnn.pth"
        args.mode = "channel"

    if args.mode == "channel":
        run_channel_experiment(args)
    elif args.mode == "train":
        dev_range = _range(args.dev_start, args.dev_stop)
        pkt_range = _range(args.pkt_start, args.pkt_stop)
        train_snr_range = None if args.no_train_noise else (args.train_snr_min, args.train_snr_max)
        train_files = _channel_training_files(args.train_file, args.use_channel_augmentations)
        train_cnn(
            train_files,
            dev_range=dev_range,
            pkt_range=pkt_range,
            val_split=args.val_split,
            train_config=build_train_config(args),
            model_out=args.model_out,
            train_snr_range=train_snr_range,
            results_dir=args.results_dir,
            normalize_spectrogram=args.normalize_spectrogram,
        )
    elif args.mode == "test":
        model, checkpoint = _load_checkpoint(args.checkpoint, device=args.device)
        evaluate_files(
            model,
            args.test_files,
            checkpoint["class_values"],
            _range(args.pkt_start, args.pkt_stop),
            checkpoint["train_stats"],
            args.results_dir,
            batch_size=args.eval_batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
