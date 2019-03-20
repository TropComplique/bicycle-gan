import torch
import torch.nn as nn
from .unet import AdaptiveInstanceNorm


class ResnetGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, depth=32, downsample=3, num_blocks=3):
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
        super(ResnetGenerator, self).__init__()

        # DOWNSAMPLING

        down_path = []
        down_path.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, depth, kernel_size=7, bias=False),
            nn.InstanceNorm2d(depth, affine=True),
            nn.ReLU(inplace=True)
        ))

        params = {
            'kernel_size': 3, 'stride': 2,
            'padding': 0, 'bias': False
        }

        for i in range(downsample):
            m = 2**i  # multiplier
            down_path.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(depth * m, depth * m * 2, **params),
                nn.InstanceNorm2d(depth * m * 2, affine=True),
                nn.ReLU(inplace=True)
            ))

        # MIDDLE BLOCKS

        # number of weights used by adains
        num_features = 0

        blocks = []
        m = 2**downsample

        for _ in range(num_blocks):
            blocks.append(ResnetBlock(depth * m))
            num_features += 4 * depth * m

        # UPSAMPLING

        params = {
            'kernel_size': 3, 'stride': 2,
            'padding': 1, 'bias': False,
            'output_padding': 1
        }
        up_path = []

        for i in range(downsample):

            m = 2**(downsample - 1 - i)
            k = 1 if i == 0 else 2

            up_path.append(nn.ModuleList([
                nn.ConvTranspose2d(depth * m * 2 * k, depth * m, **params),
                AdaptiveInstanceNorm(depth * m),
                nn.ReLU(inplace=True)
            ]))
            num_features += 2 * depth * m

        # END

        up_path.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(2 * depth, out_channels, kernel_size=7),
            nn.Tanh()
        ))

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


class ResnetBlock(nn.Module):

    def __init__(self, d):
        super(ResnetBlock, self).__init__()

        self.layers = nn.ModuleList([
            nn.ReflectionPad2d(1),
            nn.Conv2d(d, d, kernel_size=3, bias=False),
            AdaptiveInstanceNorm(d),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(d, d, kernel_size=3, bias=False),
            AdaptiveInstanceNorm(d)
        ])

    def forward(self, x, w):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
            w: a float tensor with shape [b, 4 * d, 1, 1].
        Returns:
            a float tensor with shape [b, d, h, w].
        """
        d = x.size(1)
        w1, w2 = torch.split(w, [2 * d, 2 * d], dim=1)

        y = self.layers[0](x)
        y = self.layers[1](y)
        y = self.layers[2](y, w1)
        y = self.layers[3](y)
        y = self.layers[4](y)
        y = self.layers[5](y)
        y = self.layers[6](y, w2)

        return x + y
