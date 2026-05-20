import numpy as np
import h5py
from numpy import sqrt, sum
from numpy.random import standard_normal, uniform
from scipy import signal


def awgn(data, snr_range):
    """Add complex AWGN to IQ samples using the original implementation."""
    pkt_num = data.shape[0]
    snr_db = uniform(snr_range[0], snr_range[-1], pkt_num)

    for pkt_idx in range(pkt_num):
        s = data[pkt_idx]
        snr_linear = 10 ** (snr_db[pkt_idx] / 10)
        power = sum(abs(s) ** 2) / len(s)
        noise_power = power / snr_linear
        noise = sqrt(noise_power / 2) * (
            standard_normal(len(s)) + 1j * standard_normal(len(s))
        )
        data[pkt_idx] = s + noise

    return data


class LoadDataset:
    """Load LoRa RFFI HDF5 datasets."""

    def __init__(self):
        self.dataset_name = "data"
        self.labelset_name = "label"

    @staticmethod
    def _convert_to_complex(data):
        """Convert concatenated I/Q branches to complex IQ samples."""
        num_col = data.shape[1]
        half = round(num_col / 2)
        return data[:, :half] + 1j * data[:, half:]

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        """
        Load IQ samples and zero-based labels from a LoRa RFFI HDF5 dataset.

        dev_range is zero-based, matching the original code after label -= 1.
        pkt_range selects packet indexes inside each selected device.
        """
        with h5py.File(file_path, "r") as h5_file:
            label = np.asarray(h5_file[self.labelset_name][:]).astype(int).reshape(-1) - 1

            label_start = int(label[0]) + 1
            label_end = int(label[-1]) + 1
            num_dev = label_end - label_start + 1
            num_pkt_per_dev = int(len(label) / num_dev)
            print(
                "Dataset information: Dev "
                + str(label_start)
                + " to Dev "
                + str(label_end)
                + ", "
                + str(num_pkt_per_dev)
                + " packets per device."
            )

            sample_index_list = []
            pkt_range = np.asarray(list(pkt_range), dtype=int)
            for dev_idx in dev_range:
                sample_index_dev = np.where(label == int(dev_idx))[0][pkt_range].tolist()
                sample_index_list.extend(sample_index_dev)

            data = h5_file[self.dataset_name][sample_index_list]
            data = self._convert_to_complex(data)
            label = label[sample_index_list]

        return data, label


class ChannelIndSpectrogram:
    """Generate channel-independent spectrograms from complex IQ samples."""

    @staticmethod
    def _normalization(data):
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i] / rms

        return s_norm

    @staticmethod
    def _spec_crop(x):
        num_row = x.shape[0]
        return x[round(num_row * 0.3) : round(num_row * 0.7)]

    @staticmethod
    def _gen_single_channel_ind_spectrogram(sig, win_len=256, overlap=128):
        _, _, spec = signal.stft(
            sig,
            window="boxcar",
            nperseg=win_len,
            noverlap=overlap,
            nfft=win_len,
            return_onesided=False,
            padded=False,
            boundary=None,
        )

        spec = np.fft.fftshift(spec, axes=0)
        chan_ind_spec = spec[:, 1:] / spec[:, :-1]
        return np.log10(np.abs(chan_ind_spec) ** 2)

    def channel_ind_spectrogram(self, data):
        data = self._normalization(data)

        num_sample = data.shape[0]
        num_row = int(256 * 0.4)
        num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])

        for i in range(num_sample):
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i, :, :, 0] = chan_ind_spec_amp

        return data_channel_ind_spec.astype(np.float32)
