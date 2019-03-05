import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, depth=64, downsample=6):
        super(UNet, self).__init__()

        # START

        params = {
            'kernel_size': 4, 'stride': 2,
            'padding': 1, 'bias': False
        }

        self.beginning = nn.Sequential(
            nn.Conv2d(in_channels, depth, **params),
            nn.InstanceNorm2d(depth, affine=True)
        )

        # DOWNSAMPLE

        down_path = []
        num_features = 0  # number of weights for all adains
        for i in range(1, downsample):

            in_depth = min(2**(i - 1), 8) * depth
            out_depth = min(2**i, 8) * depth

            down_path.append(UNetBlock(in_depth, out_depth))
            num_features += 2 * out_depth

        self.down_path = nn.ModuleList(down_path)

        # NOISE TO STYLE MAPPING

        z_dimension = 8

        self.mapping = nn.Sequential(
            nn.Linear(z_dimension, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_features)
        )

        # UPSAMPLE

        up_path = [UNetUpsamplingBlock(8 * depth, 8 * depth)]
        for i in reversed(range(1, downsample - 1)):

            in_depth = 2 * min(2**i, 8) * depth
            out_depth = min(2**(i - 1), 8) * depth

            up_path.append(UNetUpsamplingBlock(in_depth, out_depth))

        self.up_path = nn.ModuleList(up_path)

        # IMAGE PREDICTOR

        params = {
            'kernel_size': 4, 'stride': 2,
            'padding': 1, 'bias': True
        }

        self.end = nn.Sequential(
            nn.ConvTranspose2d(2 * depth, out_channels, **params),
            nn.Tanh()
        )

    def forward(self, x, z):
        """
        I assume that h and w are divisible by 2**downsample.

        Input and output tensors represent
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
            z: a float tensor with shape [b, z_dimension].
        Returns:
            a float tensor with shape [b, out_channels, h, w].
        """

        weights = self.mapping(z).unsqueeze(2).unsqueeze(3)
        # it has shape [b, num_features, 1, 1]

        x = 2.0 * x - 1.0
        x = self.beginning(x)

        s = 0  # start
        outputs = [x]
        for i, b in enumerate(self.down_path, 2):

            d = 2 * b.adain.in_channels
            w = weights[:, s:(s + d)]
            s += d

            x = b(x, w)  # it has stride 2**i
            outputs.append(x)

        # now `x` has shape [b, 8 * depth, h/stride, w/stride],
        # where stride = 2**downsample

        for i, b in enumerate(self.up_path, 1):

            if i == 1:
                x = b(x)
                continue

            y = outputs[-i]
            x = torch.cat([x, y], dim=1)
            x = b(x)

        y = outputs[0]
        x = torch.cat([x, y], dim=1)
        x = self.end(x)
        return 0.5 * x + 0.5


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, d):
        super(AdaptiveInstanceNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(d)
        self.in_channels = d

    def forward(self, x, weights):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            weights: a long tensor with shape [b, 2 * d, 1, 1].
        Returns:
            a float tensor with shape [b, d, h, w].
        """
        d = x.size(1)
        gamma, beta = torch.split(weights, [d, d], dim=1)
        return (gamma + 1.0) * self.normalize(x) + beta


class UNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()

        params = {
            'kernel_size': 4, 'stride': 2,
            'padding': 1, 'bias': False
        }

        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, **params)
        )
        self.adain = AdaptiveInstanceNorm(out_channels)

    def forward(self, x, weights):
        x = self.layers(x)
        x = self.adain(x, weights)
        return x


class UNetUpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNetUpsamplingBlock, self).__init__()

        params = {
            'kernel_size': 4, 'stride': 2,
            'padding': 1, 'bias': False
        }

        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, **params),
            nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        return self.layers(x)
