import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels, depth=48, downsample=4, num_blocks=4):
        """
        Arguments:
            in_channels: an integer.
            out_channels: an integer.
            depth: an integer.
            downsample: an integer, the input will
                be downsampled in `2**downsample` times
                before applying resnet blocks.
            num_blocks: an integer, number of resnet blocks.
        """
        super(Generator, self).__init__()

        # BEGINNING

        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, depth, kernel_size=7, bias=False),
            nn.InstanceNorm2d(depth, affine=True),
            nn.ReLU(inplace=True)
        )

        # DOWNSAMPLING

        num_weights = 0
        # number of weights (gammas and betas)
        # needed for all adain layers

        down_path = []
        for i in range(downsample):
            m = 2**i  # multiplier
            down_path.append(Downsample(depth * m))
            num_weights += 4 * depth * m

        # MIDDLE BLOCKS

        blocks = []
        m = 2**downsample

        for _ in range(num_blocks):
            blocks.append(ResnetBlock(depth * m))
            num_weights += 4 * depth * m

        # UPSAMPLING

        up_path = []
        for i in range(downsample):
            m = 2**(downsample - 1 - i)
            up_path.append(Upsample(depth * m * 2))
            num_weights += 2 * depth * m

        # END

        self.end = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(depth, out_channels, kernel_size=7),
            nn.Tanh()
        )

        # NOISE TO STYLE MAPPING

        z_dimension = 8

        self.mapping = nn.Sequential(
            nn.Linear(z_dimension, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_weights)
        )

        self.down_path = nn.ModuleList(down_path)
        self.blocks = nn.ModuleList(blocks)
        self.up_path = nn.ModuleList(up_path)

    def forward(self, x, z):
        """
        I assume that h and w are
        divisible by 2**downsample.

        Input and output tensors represent
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
            z: a float tensor with shape [b, z_dimension].
        Returns:
            a float tensor with shape [b, out_channels, h, w].
        """

        weights = self.mapping(z).unsqueeze(2).unsqueeze(3)
        # it has shape [b, num_weights, 1, 1]

        x = 2.0 * x - 1.0
        x = self.start(x)

        # start
        s = 0

        for m in self.down_path:

            d = 4 * x.size(1)
            w = weights[:, s:(s + d)]
            s += d
            x = m(x, w)

        for m in self.blocks:

            d = 4 * x.size(1)
            w = weights[:, s:(s + d)]
            s += d
            x = m(x, w)

        for m in self.up_path:

            d = x.size(1)
            w = weights[:, s:(s + d)]
            s += d
            x = m(x, w)

        x = self.end(x)
        return 0.5 * x + 0.5


class Downsample(nn.Module):

    def __init__(self, d):
        super(Downsample, self).__init__()

        params = {
            'kernel_size': 3, 'stride': 2,
            'padding': 0, 'bias': False
        }

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(d, 2 * d, **params)
        self.adain = AdaptiveInstanceNorm(2 * d)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, w):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            w: a float tensor with shape [b, 4 * d, 1, 1].
        Returns:
            a float tensor with shape [b, 2 * d, h // 2, w // 2].
        """
        twice_d = 2 * x.size(1)
        gamma, beta = torch.split(w, [twice_d, twice_d], dim=1)

        x = self.pad(x)
        x = self.conv(x)
        x = self.adain(x, gamma + 1.0, beta)
        x = self.relu(x)

        return x


class Upsample(nn.Module):

    def __init__(self, d):
        super(Upsample, self).__init__()

        params = {
            'kernel_size': 3, 'stride': 1,
            'padding': 0, 'bias': False
        }

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(d, d // 2, **params)
        self.adain = AdaptiveInstanceNorm(d // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, w):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            w: a float tensor with shape [b, d, 1, 1].
        Returns:
            a float tensor with shape [b, d // 2, 2 * h, 2 * w].
        """
        half_d = x.size(1) // 2
        gamma, beta = torch.split(w, [half_d, half_d], dim=1)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.pad(x)
        x = self.conv(x)
        x = self.adain(x, gamma + 1.0, beta)
        x = self.relu(x)

        return x


class ResnetBlock(nn.Module):

    def __init__(self, d):
        super(ResnetBlock, self).__init__()

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(d, d, kernel_size=3, bias=False)
        self.adain1 = AdaptiveInstanceNorm(d)
        self.relu1 = nn.ReLU(inplace=True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(d, d, kernel_size=3, bias=False)
        self.adain2 = AdaptiveInstanceNorm(d)

    def forward(self, x, w):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            w: a float tensor with shape [b, 4 * d, 1, 1].
        Returns:
            a float tensor with shape [b, d, h, w].
        """

        d = x.size(1)
        split = torch.split(w, [d, d, d, d], dim=1)
        gamma1, beta1, gamma2, beta2 = split

        y = self.pad1(x)
        y = self.conv1(y)
        y = self.adain1(y, gamma1, beta1)
        y = self.relu1(y)

        y = self.pad2(y)
        y = self.conv2(y)
        y = self.adain2(y, gamma2, beta2)

        return x + y


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, d):
        super(AdaptiveInstanceNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(d)
        self.in_channels = d

    def forward(self, x, gamma, beta):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            gamma, beta: float tensors with shape [b, d, 1, 1].
        Returns:
            a float tensor with shape [b, d, h, w].
        """
        return gamma * self.normalize(x) + beta
