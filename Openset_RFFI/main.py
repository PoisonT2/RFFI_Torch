import argparse
import copy
import tempfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from dataset_preparation import ChannelIndSpectrogram, LoadDataset, awgn
from deep_learning_models import (
    FeatureExtractor,
    convert_keras_h5_to_pth,
    extract_features,
    load_feature_extractor,
    save_feature_extractor,
    triplet_loss,
)


DATA_ROOT = Path("./LoRa_RFF")
DATASET_ROOT = DATA_ROOT / "dataset"
SOURCE_MODEL_ROOT = DATA_ROOT / "models"
LOCAL_MODEL_ROOT = Path("./Openset_RFFI") / "models"


def parse_range(text):
    if isinstance(text, np.ndarray):
        return text.astype(int)
    if isinstance(text, range):
        return np.asarray(list(text), dtype=int)
    if isinstance(text, (list, tuple)):
        return np.asarray(text, dtype=int)

    text = str(text).strip()
    if ":" in text:
        values = [int(v) for v in text.split(":") if v != ""]
        if len(values) == 2:
            return np.arange(values[0], values[1], dtype=int)
        if len(values) == 3:
            return np.arange(values[0], values[1], values[2], dtype=int)
        raise ValueError("Range must be start:end or start:end:step.")

    return np.asarray([int(v) for v in text.split(",") if v.strip() != ""], dtype=int)


def select_device(device):
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; using CPU instead.")
        return "cpu"
    return device


def _label_index(labels):
    labels = np.asarray(labels).reshape(-1)
    return {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}


def _sample_triplet_batch(data, labels, dev_range, batch_size):
    labels = np.asarray(labels).reshape(-1)
    index_by_label = _label_index(labels)
    classes = np.asarray(
        [int(dev) for dev in dev_range if int(dev) in index_by_label], dtype=int
    )

    if len(classes) < 2:
        raise ValueError("Triplet training needs at least two classes in the split.")

    anchors, positives, negatives = [], [], []
    for _ in range(batch_size):
        anchor_label = int(np.random.choice(classes))
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = int(np.random.choice(classes))

        anchor_idx = int(np.random.choice(index_by_label[anchor_label]))
        positive_idx = int(np.random.choice(index_by_label[anchor_label]))
        negative_idx = int(np.random.choice(index_by_label[negative_label]))

        anchors.append(data[anchor_idx])
        positives.append(data[positive_idx])
        negatives.append(data[negative_idx])

    return (
        np.asarray(anchors, dtype=np.float32),
        np.asarray(positives, dtype=np.float32),
        np.asarray(negatives, dtype=np.float32),
    )


def _to_torch(batch, device):
    return torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)


def train_feature_extractor(
    file_path=DATASET_ROOT / "Train" / "dataset_training_aug.h5",
    dev_range=np.arange(0, 30, dtype=int),
    pkt_range=np.arange(0, 1000, dtype=int),
    snr_range=np.arange(20, 80),
    save_path=None,
    device="auto",
    epochs=1000,
    batch_size=32,
    patience=20,
):
    """
    Train the RFF extractor using the original triplet-loss method.
    """
    device = select_device(device)
    loader = LoadDataset()
    data, label = loader.load_iq_samples(file_path, dev_range, pkt_range)
    data = awgn(data, snr_range)

    spectrogram = ChannelIndSpectrogram().channel_ind_spectrogram(data)
    del data

    data_train, data_valid, label_train, label_valid = train_test_split(
        spectrogram, label, test_size=0.1, shuffle=True
    )
    del spectrogram, label

    margin = 0.1
    train_steps = max(1, data_train.shape[0] // batch_size)
    valid_steps = max(1, data_valid.shape[0] // batch_size)

    model = FeatureExtractor(data_train.shape[1:]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=10
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for _ in range(train_steps):
            anchor, positive, negative = _sample_triplet_batch(
                data_train, label_train, dev_range, batch_size
            )
            anchor = _to_torch(anchor, device)
            positive = _to_torch(positive, device)
            negative = _to_torch(negative, device)

            optimizer.zero_grad()
            loss = triplet_loss(model(anchor), model(positive), model(negative), margin)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= train_steps

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for _ in range(valid_steps):
                anchor, positive, negative = _sample_triplet_batch(
                    data_valid, label_valid, dev_range, batch_size
                )
                anchor = _to_torch(anchor, device)
                positive = _to_torch(positive, device)
                negative = _to_torch(negative, device)
                loss = triplet_loss(
                    model(anchor), model(positive), model(negative), margin
                )
                valid_loss += loss.item()

        valid_loss /= valid_steps
        scheduler.step(valid_loss)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {train_loss:.6f} - val_loss: {valid_loss:.6f}"
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    model.load_state_dict(best_state)
    if save_path is not None:
        save_feature_extractor(model, save_path)
        print(f"Saved PyTorch extractor to {save_path}")

    return model


def test_classification(
    file_path_enrol,
    file_path_clf,
    feature_extractor_name,
    dev_range_enrol=np.arange(30, 40, dtype=int),
    pkt_range_enrol=np.arange(0, 100, dtype=int),
    dev_range_clf=np.arange(30, 40, dtype=int),
    pkt_range_clf=np.arange(100, 200, dtype=int),
    device="auto",
    batch_size=128,
):
    device = select_device(device)
    feature_extractor = load_feature_extractor(feature_extractor_name, device=device)
    loader = LoadDataset()
    spec = ChannelIndSpectrogram()

    data_enrol, label_enrol = loader.load_iq_samples(
        file_path_enrol, dev_range_enrol, pkt_range_enrol
    )
    data_enrol = spec.channel_ind_spectrogram(data_enrol)
    feature_enrol = extract_features(feature_extractor, data_enrol, device, batch_size)
    del data_enrol

    n_neighbors = min(15, len(feature_enrol))
    knnclf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
    knnclf.fit(feature_enrol, np.ravel(label_enrol))

    data_clf, true_label = loader.load_iq_samples(
        file_path_clf, dev_range_clf, pkt_range_clf
    )
    data_clf = spec.channel_ind_spectrogram(data_clf)
    feature_clf = extract_features(feature_extractor, data_clf, device, batch_size)
    del data_clf

    pred_label = knnclf.predict(feature_clf)
    true_label = np.ravel(true_label)
    acc = accuracy_score(true_label, pred_label)
    print("Overall accuracy = %.4f" % acc)

    return pred_label, true_label, acc


def test_rogue_device_detection(
    feature_extractor_name,
    file_path_enrol=DATASET_ROOT / "Test" / "dataset_residential.h5",
    dev_range_enrol=np.arange(30, 40, dtype=int),
    pkt_range_enrol=np.arange(0, 100, dtype=int),
    file_path_legitimate=DATASET_ROOT / "Test" / "dataset_residential.h5",
    dev_range_legitimate=np.arange(30, 40, dtype=int),
    pkt_range_legitimate=np.arange(100, 200, dtype=int),
    file_path_rogue=DATASET_ROOT / "Test" / "dataset_rogue.h5",
    dev_range_rogue=np.arange(40, 45, dtype=int),
    pkt_range_rogue=np.arange(0, 100, dtype=int),
    device="auto",
    batch_size=128,
):
    def _compute_eer(fpr, tpr, thresholds):
        fnr = 1 - tpr
        min_index = np.argmin(np.abs(fpr - fnr))
        eer = np.mean((fpr[min_index], fnr[min_index]))
        return eer, thresholds[min_index]

    device = select_device(device)
    feature_extractor = load_feature_extractor(feature_extractor_name, device=device)
    loader = LoadDataset()
    spec = ChannelIndSpectrogram()

    data_enrol, label_enrol = loader.load_iq_samples(
        file_path_enrol, dev_range_enrol, pkt_range_enrol
    )
    data_enrol = spec.channel_ind_spectrogram(data_enrol)
    feature_enrol = extract_features(feature_extractor, data_enrol, device, batch_size)
    del data_enrol

    n_neighbors = min(15, len(feature_enrol))
    knnclf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
    knnclf.fit(feature_enrol, np.ravel(label_enrol))

    data_legitimate, label_legitimate = loader.load_iq_samples(
        file_path_legitimate, dev_range_legitimate, pkt_range_legitimate
    )
    data_rogue, label_rogue = loader.load_iq_samples(
        file_path_rogue, dev_range_rogue, pkt_range_rogue
    )

    data_test = np.concatenate([data_legitimate, data_rogue])
    label_test = np.ravel(np.concatenate([label_legitimate, label_rogue]))
    del data_legitimate, data_rogue

    data_test = spec.channel_ind_spectrogram(data_test)
    feature_test = extract_features(feature_extractor, data_test, device, batch_size)
    del data_test

    distances, _ = knnclf.kneighbors(feature_test)
    detection_score = distances.mean(axis=1)

    true_label = np.zeros(len(label_test), dtype=int)
    true_label[
        (label_test <= dev_range_legitimate[-1])
        & (label_test >= dev_range_legitimate[0])
    ] = 1

    fpr, tpr, thresholds = roc_curve(true_label, detection_score, pos_label=1)
    fpr = 1 - fpr
    tpr = 1 - tpr
    eer, _ = _compute_eer(fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, eer


def run_smoke_test():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "smoke.h5"
        num_dev = 3
        packets_per_dev = 20
        iq_len = 1024
        labels = np.repeat(np.arange(1, num_dev + 1), packets_per_dev)[None, :]
        iq = rng.standard_normal((num_dev * packets_per_dev, iq_len * 2)).astype(
            np.float32
        )

        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("data", data=iq)
            h5_file.create_dataset("label", data=labels)

        loader = LoadDataset()
        data, label = loader.load_iq_samples(
            h5_path, np.arange(0, num_dev), np.arange(0, packets_per_dev)
        )
        spec = ChannelIndSpectrogram().channel_ind_spectrogram(data)
        model = FeatureExtractor(spec.shape[1:])
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        anchor, positive, negative = _sample_triplet_batch(
            spec, label, np.arange(0, num_dev), batch_size=8
        )
        optimizer.zero_grad()
        loss = triplet_loss(
            model(_to_torch(anchor, "cpu")),
            model(_to_torch(positive, "cpu")),
            model(_to_torch(negative, "cpu")),
        )
        loss.backward()
        optimizer.step()

        pth_path = Path(tmp_dir) / "smoke.pth"
        save_feature_extractor(model, pth_path)
        reloaded = load_feature_extractor(pth_path, device="cpu", input_shape=spec.shape[1:])
        features = extract_features(reloaded, spec[:10], device="cpu", batch_size=4)

        assert spec.shape == (num_dev * packets_per_dev, 102, 6, 1)
        assert features.shape == (10, 512)
        assert np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)

    print("Smoke test passed.")


def build_argparser():
    parser = argparse.ArgumentParser(description="PyTorch open-set LoRa RFFI")
    parser.add_argument(
        "--task",
        default="classification",
        choices=["train", "classification", "rogue", "convert", "smoke-test"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)

    parser.add_argument(
        "--model", default=str(SOURCE_MODEL_ROOT / "Extractor_1.h5")
    )
    parser.add_argument(
        "--output-model", default=str(LOCAL_MODEL_ROOT / "Extractor.pth")
    )

    parser.add_argument(
        "--train-file", default=str(DATASET_ROOT / "Train" / "dataset_training_aug.h5")
    )
    parser.add_argument("--train-devices", default="0:30")
    parser.add_argument("--train-packets", default="0:1000")
    parser.add_argument("--snr", default="20:80")

    parser.add_argument(
        "--enrol-file", default=str(DATASET_ROOT / "Test" / "dataset_residential.h5")
    )
    parser.add_argument(
        "--clf-file", default=str(DATASET_ROOT / "Test" / "channel_problem" / "A.h5")
    )
    parser.add_argument("--enrol-devices", default="30:40")
    parser.add_argument("--enrol-packets", default="0:100")
    parser.add_argument("--test-devices", default="30:40")
    parser.add_argument("--test-packets", default="100:200")

    parser.add_argument(
        "--legitimate-file",
        default=str(DATASET_ROOT / "Test" / "dataset_residential.h5"),
    )
    parser.add_argument(
        "--rogue-file", default=str(DATASET_ROOT / "Test" / "dataset_rogue.h5")
    )
    parser.add_argument("--rogue-devices", default="40:45")
    parser.add_argument("--rogue-packets", default="0:100")
    parser.add_argument("--plot", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()

    if args.task == "smoke-test":
        run_smoke_test()
        return

    if args.task == "convert":
        convert_keras_h5_to_pth(args.model, args.output_model)
        print(f"Converted {args.model} -> {args.output_model}")
        return

    if args.task == "train":
        train_feature_extractor(
            file_path=args.train_file,
            dev_range=parse_range(args.train_devices),
            pkt_range=parse_range(args.train_packets),
            snr_range=parse_range(args.snr),
            save_path=args.output_model,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        return

    if args.task == "classification":
        pred_label, true_label, acc = test_classification(
            file_path_enrol=args.enrol_file,
            file_path_clf=args.clf_file,
            feature_extractor_name=args.model,
            dev_range_enrol=parse_range(args.enrol_devices),
            pkt_range_enrol=parse_range(args.enrol_packets),
            dev_range_clf=parse_range(args.test_devices),
            pkt_range_clf=parse_range(args.test_packets),
            device=args.device,
            batch_size=args.batch_size,
        )
        if args.plot:
            conf_mat = confusion_matrix(true_label, pred_label)
            classes = parse_range(args.test_devices) + 1
            plt.figure()
            sns.heatmap(
                conf_mat,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=classes,
                yticklabels=classes,
            )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.show()
        print(f"Classification accuracy: {acc:.4f}")
        return

    fpr, tpr, roc_auc, eer = test_rogue_device_detection(
        feature_extractor_name=args.model,
        file_path_enrol=args.enrol_file,
        dev_range_enrol=parse_range(args.enrol_devices),
        pkt_range_enrol=parse_range(args.enrol_packets),
        file_path_legitimate=args.legitimate_file,
        dev_range_legitimate=parse_range(args.test_devices),
        pkt_range_legitimate=parse_range(args.test_packets),
        file_path_rogue=args.rogue_file,
        dev_range_rogue=parse_range(args.rogue_devices),
        pkt_range_rogue=parse_range(args.rogue_packets),
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"Rogue detection AUC: {roc_auc:.4f}, EER: {eer:.4f}")

    if args.plot:
        plt.figure(figsize=(4.8, 2.8))
        plt.xlim(-0.01, 1.02)
        plt.ylim(-0.01, 1.02)
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}, EER = {eer:.3f}", c="r")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc=4)
        plt.show()


if __name__ == "__main__":
    main()
