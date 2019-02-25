import torch
import torch.nn as nn


class ResnetGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, depth, downsample, num_blocks):
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

        # BEGINNING

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, depth, kernel_size=7, bias=False),
            nn.InstanceNorm2d(depth),
            nn.ReLU(inplace=True)
        ]

        # DOWNSAMPLING

        params = {
            'kernel_size': 3, 'stride': 2,
            'padding': 1, 'bias': False
        }

        for i in range(downsample):
            m = 2**i  # multiplier
            layers.extend([
                nn.Conv2d(depth * m, depth * m * 2, **params),
                nn.InstanceNorm2d(depth * m * 2),
                nn.ReLU(inplace=True)
            ])

        # MIDDLE BLOCKS

        m = 2**downsample
        for _ in range(num_blocks):
            layers.append(ResnetBlock(depth * m))

        # UPSAMPLING

        params = {
            'kernel_size': 3, 'stride': 2,
            'padding': 1, 'bias': False
            'output_padding': 1
        }

        for i in range(downsample):
            m = 2**(downsample - 1 - i)
            layers.extend([
                nn.ConvTranspose2d(depth * m * 2, depth * m, **params),
                nn.InstanceNorm2d(depth * m),
                nn.ReLU(inplace=True)
            ])

        # END

        layers.extend([
            nn.ReflectionPad2d(3),
            nn.Conv2d(depth, out_channels, kernel_size=7),
            nn.Tanh()
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        I assume that h and w are
        divisible by 2**downsample.

        Input and output tensors represent
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, out_channels, h, w].
        """
        x = 2.0 * x - 1.0
        x = self.layers(x)
        x = 0.5 * x + 0.5
        return x


class ResnetBlock(nn.Module):

    def __init__(self, depth):
        super(ResnetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(depth, depth, kernel_size=3, bias=False),
            nn.InstanceNorm2d(depth),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(depth, depth, kernel_size=3, bias=False),
            nn.InstanceNorm2d(depth)
        )

    def forward(self, x):
        return x + self.layers(x)
