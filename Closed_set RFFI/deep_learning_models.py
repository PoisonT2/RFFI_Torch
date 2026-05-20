from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block aligned with Closed_set_RFFI/deep_learning_models.py."""

    def __init__(self, in_channels, out_channels, kernel_size=3, first_layer=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if first_layer or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class ClassificationNet(nn.Module):
    """Closed-set CNN baseline ported from the Keras implementation.

    Shape flow for the default spectrogram is the same as Closed_set_RFFI:
    102x62 -> Conv stride 2 -> 51x31 -> AvgPool2d(2) -> 25x15.
    LazyLinear keeps this robust if the spectrogram shape changes.
    """

    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3)
        self.res1 = ResBlock(32, 32, kernel_size=3)
        self.res2 = ResBlock(32, 32, kernel_size=3)
        self.res3 = ResBlock(32, 64, kernel_size=3, first_layer=True)
        self.res4 = ResBlock(64, 64, kernel_size=3)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.LazyLinear(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_features(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))
