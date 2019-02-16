import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=64, downsample=8):
        super(UNet, self).__init__()
        assert downsample > 3

        self.beginning = nn.Sequential(
            nn.Conv2d(
                in_channels, depth,
                kernel_size=4, stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(depth)
        )
        down_path = [
            UNetBlock(depth, 2 * depth),
            UNetBlock(2 * depth, 4 * depth),
            UNetBlock(4 * depth, 8 * depth),
        ]
        down_path += [
            UNetBlock(8 * depth, 8 * depth)
            for _ in range(downsample - 4)
        ]
        self.down_path = nn.ModuleList(down_path)

        up_path = [UNetUpsamplingBlock(8 * depth, 8 * depth)]
        up_path += [
            UNetUpsamplingBlock(2 * 8 * depth, 8 * depth)
            for _ in range(downsample - 4)
        ]
        up_path += [
            UNetUpsamplingBlock(2 * 8 * depth, 4 * depth),
            UNetUpsamplingBlock(2 * 4 * depth, 2 * depth),
            UNetUpsamplingBlock(2 * 2 * depth, depth),
            nn.Sequential(
                nn.ConvTranspose2d(
                    depth, out_channels,
                    kernel_size=4, stride=2,
                    padding=1
                ),
                nn.Tanh()
            )
        ]
        self.up_path = nn.ModuleList(up_path)

    def forward(self, x):
        """
        I assume that h and w are divisible by 2**(4 + num_blocks).

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, out_channels, h, w].
        """
        x = 2.0 * x - 1.0
        x = self.beginning(x)

        outputs = [x]
        for i, b in enumerate(self.down_path, 2):
            x = b(x)  # it has stride 2**i
            outputs.append(x)

        # now `x` has shape [b, 8 * depth, h/stride, w/stride],
        # where stride = 2**(4 + num_blocks)

        for i, b in enumerate(self.up_path, 1):

            if i == 1:
                x = b(x)
                continue

            y = outputs[-i]
            x = b(x, y)

        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class UNetUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpsamplingBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.ReLU(inplace=True)
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x, y=None):

        if y is not None:
            x = torch.cat([x, y], dim=1)

        return self.layers(x)
