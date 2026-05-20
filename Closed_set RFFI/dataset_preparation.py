from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from scipy import signal


def _as_sequence(values: Iterable[int] | Sequence[int]) -> list[int]:
    return [int(v) for v in values]


def awgn(data, snr_range, distribution="uniform", rng=None, copy=True):
    """Add additive white Gaussian noise to complex IQ samples."""
    rng = np.random.default_rng() if rng is None else rng
    noisy = np.array(data, dtype=np.complex64, copy=copy)
    snr_values = list(snr_range)
    if len(snr_values) == 0:
        raise ValueError("snr_range must not be empty")
    if len(snr_values) == 2:
        snr_min, snr_max = float(snr_values[0]), float(snr_values[1])
    else:
        snr_min, snr_max = float(min(snr_values)), float(max(snr_values))

    pkt_num = noisy.shape[0]
    if distribution == "normal":
        mean = (snr_min + snr_max) / 2.0
        std = max((snr_max - snr_min) / 6.0, 1e-12)
        snr_db = rng.normal(mean, std, pkt_num)
        snr_db = np.clip(snr_db, snr_min, snr_max)
    elif distribution == "uniform":
        snr_db = rng.uniform(snr_min, snr_max, pkt_num)
    else:
        raise ValueError(f"Unsupported SNR distribution: {distribution}")

    for idx in range(pkt_num):
        sample = noisy[idx]
        power = np.mean(np.abs(sample) ** 2)
        noise_power = power / (10 ** (snr_db[idx] / 10.0))
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(sample.shape[0]) + 1j * rng.standard_normal(sample.shape[0])
        )
        noisy[idx] = sample + noise.astype(np.complex64)
    return noisy


class LoadDataset:
    """Loader for LoRa-RFF HDF5 files.

    Labels returned by this class are zero-based device ids. For example,
    devices stored as 31..40 in the HDF5 file are returned as 30..39.
    """

    def __init__(self, dataset_name="data", labelset_name="label"):
        self.dataset_name = dataset_name
        self.labelset_name = labelset_name

    def _convert_to_complex(self, data):
        """Convert HDF5 [I branch, Q branch] rows to complex IQ samples."""
        if data.ndim != 2 or data.shape[1] % 2 != 0:
            raise ValueError(f"Expected a 2D array with an even feature count, got {data.shape}")
        half = data.shape[1] // 2
        real = data[:, :half].astype(np.float32, copy=False)
        imag = data[:, half:].astype(np.float32, copy=False)
        return (real + 1j * imag).astype(np.complex64, copy=False)

    def _read_labels(self, h5_file):
        labels = h5_file[self.labelset_name][:].astype(np.int64).reshape(-1) - 1
        if labels.size == 0:
            raise ValueError("The HDF5 label dataset is empty")
        return labels

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        """Load selected IQ samples and zero-based labels from one HDF5 file."""
        file_path = Path(file_path)
        dev_values = _as_sequence(dev_range)
        pkt_values = _as_sequence(pkt_range)
        if len(dev_values) == 0 or len(pkt_values) == 0:
            raise ValueError("dev_range and pkt_range must both be non-empty")

        with h5py.File(file_path, "r") as h5_file:
            labels = self._read_labels(h5_file)
            unique_labels = np.unique(labels)
            num_pkt_per_dev = {
                int(dev): int(np.sum(labels == dev))
                for dev in unique_labels
            }

            sample_index_list = []
            for dev_idx in dev_values:
                dev_indices = np.where(labels == dev_idx)[0]
                if dev_indices.size == 0:
                    raise ValueError(
                        f"Device {dev_idx} (zero-based) is not present in {file_path}. "
                        f"Available zero-based labels: {unique_labels.tolist()}"
                    )
                max_pkt = max(pkt_values)
                if max_pkt >= dev_indices.size:
                    raise ValueError(
                        f"pkt_range requests packet {max_pkt} for device {dev_idx}, "
                        f"but only {dev_indices.size} packets are available"
                    )
                sample_index_list.extend(dev_indices[pkt_values].tolist())

            data = h5_file[self.dataset_name][sample_index_list]
            labels = labels[sample_index_list]

        first_dev = int(unique_labels.min() + 1)
        last_dev = int(unique_labels.max() + 1)
        min_packets = min(num_pkt_per_dev.values())
        max_packets = max(num_pkt_per_dev.values())
        print(
            f"Dataset {file_path.name}: devices {first_dev}..{last_dev}, "
            f"packets/device {min_packets}..{max_packets}, loaded {len(labels)} samples."
        )
        return self._convert_to_complex(data), labels.astype(np.int64, copy=False)


class ChannelIndSpectrogram:
    def __init__(self, win_len=256, overlap=128, crop_ratio=(0.3, 0.7), eps=1e-12):
        self.win_len = int(win_len)
        self.overlap = int(overlap)
        self.crop_ratio = crop_ratio
        self.eps = float(eps)

    def _normalization(self, data):
        s_norm = np.zeros(data.shape, dtype=np.complex64)
        for idx in range(data.shape[0]):
            rms = np.sqrt(np.mean(np.abs(data[idx]) ** 2))
            s_norm[idx] = data[idx] / max(float(rms), self.eps)
        return s_norm

    def _spec_crop(self, x):
        num_row = x.shape[0]
        start = int(round(num_row * self.crop_ratio[0]))
        end = int(round(num_row * self.crop_ratio[1]))
        if start >= end:
            raise ValueError(f"Invalid crop_ratio {self.crop_ratio} for {num_row} rows")
        return x[start:end]

    def _gen_single_channel_ind_spectrogram(self, sig):
        _, _, spec = signal.stft(
            sig,
            window="boxcar",
            nperseg=self.win_len,
            noverlap=self.overlap,
            nfft=self.win_len,
            return_onesided=False,
            padded=False,
            boundary=None,
        )
        spec = np.fft.fftshift(spec, axes=0)
        numerator = np.abs(spec[:, 1:]) ** 2
        denominator = np.abs(spec[:, :-1]) ** 2 + self.eps
        chan_ind_spec_amp = np.log10(np.maximum(numerator / denominator, self.eps))
        return chan_ind_spec_amp.astype(np.float32, copy=False)

    def channel_ind_spectrogram(self, data):
        """Convert complex IQ samples to channel independent spectrograms."""
        if data.ndim != 2:
            raise ValueError(f"Expected complex IQ data with shape (N, L), got {data.shape}")
        data = self._normalization(data.astype(np.complex64, copy=False))
        first = self._spec_crop(self._gen_single_channel_ind_spectrogram(data[0]))
        output = np.empty((data.shape[0], first.shape[0], first.shape[1], 1), dtype=np.float32)
        output[0, :, :, 0] = first
        for idx in range(1, data.shape[0]):
            output[idx, :, :, 0] = self._spec_crop(self._gen_single_channel_ind_spectrogram(data[idx]))
        return output


def load_data_to_tensor(
    file_path,
    dev_range,
    pkt_range,
    snr_range=None,
    snr_distribution="uniform",
    seed=None,
):
    """Load one HDF5 file and convert it to PyTorch tensors.

    Returns:
        X: torch.float32 tensor with shape (N, 1, H, W)
        y: torch.long tensor with zero-based device labels
    """
    loader = LoadDataset()
    data, labels = loader.load_iq_samples(file_path, dev_range, pkt_range)
    if snr_range is not None:
        data = awgn(data, snr_range, distribution=snr_distribution, rng=np.random.default_rng(seed))

    spec_data = ChannelIndSpectrogram().channel_ind_spectrogram(data)
    X = torch.from_numpy(spec_data).permute(0, 3, 1, 2).contiguous().float()
    y = torch.from_numpy(labels).long()
    return X, y
