from pathlib import Path
import math

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def normalize_input_shape(input_shape):
    """Return an (H, W, C) input shape from NHWC, NCHW, HWC, or CHW."""
    shape = tuple(int(v) for v in input_shape)

    if len(shape) == 4:
        shape = shape[1:]

    if len(shape) != 3:
        raise ValueError("input_shape must be HWC, CHW, NHWC, or NCHW.")

    if shape[0] <= 4 and shape[-1] > 4:
        channels, height, width = shape
    else:
        height, width, channels = shape

    return height, width, channels


class SamePadConv2d(nn.Module):
    """TensorFlow/Keras-compatible Conv2D(padding='same') for PyTorch."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        in_h, in_w = x.shape[-2:]
        out_h = math.ceil(in_h / self.stride[0])
        out_w = math.ceil(in_w / self.stride[1])

        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - in_h, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - in_w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, first_layer=False):
        super().__init__()
        self.first_layer = first_layer
        self.conv1 = SamePadConv2d(in_channels, out_channels, kernel_size)
        self.conv2 = SamePadConv2d(out_channels, out_channels, kernel_size)
        self.shortcut = (
            SamePadConv2d(in_channels, out_channels, 1) if first_layer else None
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return F.relu(residual + out)


class FeatureExtractor(nn.Module):
    """RFF extractor matching the original Keras model."""

    def __init__(self, input_shape=(102, 62, 1), embedding_dim=512):
        super().__init__()
        self.input_shape = normalize_input_shape(input_shape)
        self.embedding_dim = embedding_dim
        height, width, channels = self.input_shape

        self.conv1 = SamePadConv2d(channels, 32, 7, stride=2)
        self.resblock1 = ResBlock(32, 32)
        self.resblock2 = ResBlock(32, 32)
        self.resblock3 = ResBlock(32, 64, first_layer=True)
        self.resblock4 = ResBlock(64, 64)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        flattened_size = self._infer_flattened_size(height, width, channels)
        self.fc = nn.Linear(flattened_size, embedding_dim)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        return self.avgpool(x)

    def _infer_flattened_size(self, height, width, channels):
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            return int(self._forward_conv(dummy).reshape(1, -1).shape[1])

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("FeatureExtractor expects a 4D tensor.")

        if x.shape[1] != self.input_shape[2] and x.shape[-1] == self.input_shape[2]:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self._forward_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1, eps=1e-12)


def triplet_loss(anchor, positive, negative, margin=0.1, reduction="mean"):
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = F.relu(pos_dist - neg_dist + margin)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def identity_loss(_, y_pred):
    return y_pred.mean()


class TripletLossModel(nn.Module):
    def __init__(self, embedding_net, margin=0.1):
        super().__init__()
        self.embedding_net = embedding_net
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = self.embedding_net(anchor)
        positive = self.embedding_net(positive)
        negative = self.embedding_net(negative)
        return triplet_loss(anchor, positive, negative, self.margin, reduction="none")


class TripletNet:
    """Compatibility wrapper for the original Keras TripletNet API."""

    def __init__(self):
        self.datashape = None

    def feature_extractor(self, datashape):
        self.datashape = datashape
        return FeatureExtractor(datashape[1:] if len(datashape) == 4 else datashape)

    @staticmethod
    def create_triplet_net(embedding_net, alpha):
        return TripletLossModel(embedding_net, margin=alpha)


def extract_features(model, data, device="cpu", batch_size=128):
    """Extract L2-normalized embeddings from NHWC numpy arrays."""
    model = model.to(device)
    model.eval()
    features = []

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = torch.from_numpy(data[start : start + batch_size]).float()
            batch = batch.permute(0, 3, 1, 2).to(device)
            features.append(model(batch).cpu().numpy())

    return np.concatenate(features, axis=0)


def save_feature_extractor(model, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "input_shape": model.input_shape,
            "embedding_dim": model.embedding_dim,
        },
        file_path,
    )


def load_feature_extractor(file_path, device="cpu", input_shape=(102, 62, 1)):
    file_path = Path(file_path)

    if file_path.suffix.lower() in {".h5", ".hdf5"}:
        model = FeatureExtractor(input_shape=input_shape)
        load_keras_h5_weights(model, file_path)
        return model.to(device)

    checkpoint = torch.load(file_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        input_shape = checkpoint.get("input_shape", input_shape)
        embedding_dim = checkpoint.get("embedding_dim", 512)
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        embedding_dim = 512
    else:
        state_dict = checkpoint
        embedding_dim = 512

    model = FeatureExtractor(input_shape=input_shape, embedding_dim=embedding_dim)
    model.load_state_dict(state_dict)
    return model.to(device)


def _copy_keras_conv(h5_file, layer_name, module):
    layer = h5_file["model_weights"][layer_name][layer_name]
    kernel = np.asarray(layer["kernel:0"])
    bias = np.asarray(layer["bias:0"])
    module.conv.weight.data.copy_(torch.from_numpy(kernel).permute(3, 2, 0, 1))
    module.conv.bias.data.copy_(torch.from_numpy(bias))


def _copy_keras_dense(h5_file, layer_name, module):
    layer = h5_file["model_weights"][layer_name][layer_name]
    kernel = np.asarray(layer["kernel:0"])
    bias = np.asarray(layer["bias:0"])
    module.weight.data.copy_(torch.from_numpy(kernel).t())
    module.bias.data.copy_(torch.from_numpy(bias))


def load_keras_h5_weights(model, h5_path):
    """Load the original Keras extractor H5 weights into the PyTorch model."""
    with h5py.File(h5_path, "r") as h5_file:
        _copy_keras_conv(h5_file, "conv2d_1", model.conv1)
        _copy_keras_conv(h5_file, "conv2d_2", model.resblock1.conv1)
        _copy_keras_conv(h5_file, "conv2d_3", model.resblock1.conv2)
        _copy_keras_conv(h5_file, "conv2d_4", model.resblock2.conv1)
        _copy_keras_conv(h5_file, "conv2d_5", model.resblock2.conv2)
        _copy_keras_conv(h5_file, "conv2d_6", model.resblock3.conv1)
        _copy_keras_conv(h5_file, "conv2d_7", model.resblock3.conv2)
        _copy_keras_conv(h5_file, "conv2d_8", model.resblock3.shortcut)
        _copy_keras_conv(h5_file, "conv2d_9", model.resblock4.conv1)
        _copy_keras_conv(h5_file, "conv2d_10", model.resblock4.conv2)
        _copy_keras_dense(h5_file, "dense_1", model.fc)

    return model


def convert_keras_h5_to_pth(h5_path, pth_path, input_shape=(102, 62, 1)):
    model = FeatureExtractor(input_shape=input_shape)
    load_keras_h5_weights(model, h5_path)
    save_feature_extractor(model, pth_path)
    return model
