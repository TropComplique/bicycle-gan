import torch
import torch.nn as nn


class MultiScaleDiscriminator(nn.Module):
    """
    This discriminator looks on
    patches of different scales.
    """
    def __init__(self, in_channels, depth=64, downsample=3):
        super(MultiScaleDiscriminator, self).__init__()

        self.subnetwork1 = get_layers(in_channels, depth, downsample)
        self.subnetwork2 = get_layers(in_channels, depth // 2, downsample)
        self.downsampler = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        I assume that h and w are
        divisible by 2**(downsample + 1).

        The input tensor represents
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            scores1: a float tensor with shape [b, 1, h/s, w/s].
            scores2: a float tensor with shape [b, 1, (h/s)/2, (w/s)/2],
                where s = 2**downsample.
        """

        x = 2.0 * x - 1.0
        scores1 = self.subnetwork1(x)

        x = self.downsampler(x)  # [b, in_channels, h/2, w/2]
        scores2 = self.subnetwork2(x)

        return scores1, scores2


class GlobalDiscriminator(nn.Module):
    """
    This discriminator looks
    on the entire input images.
    """
    def __init__(self, in_channels, depth=64, downsample=4):
        super(GlobalDiscriminator, self).__init__()

        self.layers = get_layers(in_channels, depth, downsample)
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)

        n = downsample - 1
        out_channels = depth * min(2**n, 8)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, input):
        """
        I assume that h and w are
        divisible by 2**downsample.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b].
        """
        x = 2.0 * x - 1.0
        x = self.layers(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        scores = self.fc(x).squeeze(1)
        return scores


class PixelDiscriminator(nn.Module):
    """
    This discriminator looks
    only on pixel values.
    """
    def __init__(self, in_channels, depth=64):
        super(PixelDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=1, bias=False),
            nn.InstanceNorm2d(depth, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth, depth * 2, kernel_size=1, bias=False),
            nn.InstanceNorm2d(depth * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(depth * 2, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, 1, h, w].
        """
        return self.layers(x)


def get_layers(in_channels, depth=64, downsample=3):
    """
    This set of layers downsamples in `2**downsample` times.
    """
    out_channels = in_channels
    sequence = []

    params = {
        'kernel_size': 4, 'stride': 2,
        'padding': 1, 'bias': False
    }

    for n in range(downsample):

        in_channels = out_channels
        out_channels = depth * min(2**n, 8)

        sequence.extend([
            nn.Conv2d(in_channels, out_channels, **params),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    """
    Right now receptive field is
    22 if downsample = 3,
    46 if downsample = 4,
    94 if downsample = 5.
    """

    # add the final score predictor
    sequence.extend([
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)
    ])

    """
    Right now receptive field is
    54 if downsample = 3,
    110 if downsample = 4,
    222 if downsample = 5.
    See https://fomoro.com/projects/project/receptive-field-calculator
    """

    return nn.Sequential(*sequence)
