import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels, depth=32):
        """
        Arguments:
            in_channels: an integer.
            out_channels: an integer.
            depth: an integer.
            downsample: an integer.
            num_blocks: an integer, number of resnet blocks.
        """
        super(Generator, self).__init__()

        # NOISE INJECTION

        downsample = 6
        z_dimension = 8
        m = 2**downsample

        self.mapping = nn.Sequential(
            nn.Linear(z_dimension, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 4 * 4 * depth * m)
        )

        # JUST BLOCKS

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResnetBlock(depth * m))

        # UPSAMPLING

        params = {
            'kernel_size': 3, 'stride': 2,
            'padding': 1, 'bias': False,
            'output_padding': 1
        }
        up_path = []

        for i in range(downsample):

            m = 2**(downsample - 1 - i)

            up_path.append(ResnetBlock(depth * m))

        # END

        up_path.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(2 * depth, out_channels, kernel_size=7),
            nn.Tanh()
        ))



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
        x = 2.0 * x - 1.0

        weights = self.mapping(z).unsqueeze(2).unsqueeze(3)
        # it has shape [b, num_features, 1, 1]

        outputs = []
        for m in self.down_path:
            x = m(x)
            outputs.append(x)

        s = 0  # start
        d = 4 * x.size(1)

        for m in self.blocks:

            w = weights[:, s:(s + d)]
            s += d
            x = m(x, w)

        for i, m in enumerate(self.up_path, 1):

            if i > 1:
                y = outputs[-i]
                x = torch.cat([x, y], dim=1)

            if i == len(outputs):
                x = m(x)
                continue

            d = 2 * m[1].in_channels
            w = weights[:, s:(s + d)]
            s += d

            x = m[0](x)
            x = m[1](x, w)
            x = m[2](x)

        return 0.5 * x + 0.5


class Upsample(nn.Module):

    def __init__(self, d):
        super(Upsample, self).__init__()

        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(d, d // 2, kernel_size=3)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
        Returns:
            a float tensor with shape [b, d // 2, 2 * h, 2 * w].
        """
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.layers(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, d, c):
        super(ResnetBlock, self).__init__()

        self.start = nn.Sequential(
            SPADE(d),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(d, d, kernel_size=3, bias=False)
        )
        self.end = nn.Sequential(
            SPADE(d),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(d, d, kernel_size=3, bias=False)
        )
        self.get_weights = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 4 * d, kernel_size=3)
        )

    def forward(self, x, y):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            y: a float tensor with shape [b, c, h, w].
        Returns:
            a float tensor with shape [b, d, h, w].
        """
        d = x.size(1)

        w = self.get_weights(y)  # shape [b, 4 * d, h, w]
        split = torch.split(w, [d, d, d, d], dim=1)
        gamma1, beta1, gamma2, beta2 = split

        y = self.start(x, gamma1, beta1)
        y = self.end(y, gamma2, beta2)

        return x + y


class SPADE(nn.Module):

    def __init__(self, d):
        super(SPADE, self).__init__()

        # like in the original paper:
        # self.normalize = nn.BatchNorm2d(d, affine=False)

        self.normalize = nn.InstanceNorm2d(d)
        self.in_channels = d

    def forward(self, x, gamma, beta):
        """
        Arguments:
            x, gamma, beta: float tensors with shape [b, d, h, w].
        Returns:
            a float tensor with shape [b, d, h, w].
        """
        return (gamma + 1.0) * self.normalize(x) + beta
