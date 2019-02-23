import torch
import torch.nn as nn


class ResNetEncoder(nn.Module):

    def __init__(self, in_channels, out_dimension, depth=64, num_blocks=4):
        super(ResNetEncoder, self).__init__()

        layers = [
            nn.Conv2d(in_channels, depth, kernel_size=3, stride=2, padding=1)
        ]

        for n in range(1, num_blocks + 1):
            in_channels = depth * min(4, n)
            out_channels = depth * min(4, n + 1)
            layers.append(BasicBlock(in_channels, out_channels))

        # so, after all these layers the
        # input is downsampled by 2**(1 + num_blocks)

        layers.extend([
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        ])

        self.layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(out_channels, out_dimension)
        self.fc2 = nn.Linear(out_channels, out_dimension)

    def forward(self, x):
        """
        I assume that h and w are
        divisible by 2**(1 + num_blocks).

        The input tensor represents
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            two float tensors with shape [b, out_dimension].
        """
        x = 2.0 * x - 1.0
        x = self.layers(x)  # shape [b, out_channels, 1, 1]
        x = x.view(x.size(0), -1)

        mean = self.fc1(x)
        logvar = self.fc2(x)
        return mean, logvar


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)
