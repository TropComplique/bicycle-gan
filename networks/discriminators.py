import torch
import torch.nn as nn


class MultiScaleDiscriminator(nn.Module):
    """
    This discriminator looks on
    patches of different scales.
    """
    def __init__(self, in_channels, depth=64, num_layers=4):
        super(MultiScaleDiscriminator, self).__init__()

        self.subnetwork1 = get_layers(in_channels, depth, num_layers)
        self.subnetwork2 = get_layers(in_channels, depth // 2, num_layers)
        self.downsampler = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        I assume that h and w are
        divisible by 2**(num_layers + 1).

        The input tensor represents
        images with pixel values in [0, 1] range.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            scores1: a float tensor with shape [b, 1, h/s, w/s], where s = 2**num_layers.
            scores2: a float tensor with shape [b, 1, h/s, w/s], where s = 2**(num_layers + 1).
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
    def __init__(self, in_channels, depth=64, num_layers=4):
        super(GlobalDiscriminator, self).__init__()

        self.layers = get_layers(in_channels, depth, num_layers)
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)

        n = num_layers - 1
        out_channels = depth * min(2**n, 8)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, input):
        """
        I assume that h and w are
        divisible by 2**num_layers.

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
            nn.Conv2d(in_channels, depth, kernel_size=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(depth, depth * 2, kernel_size=1, bias=False),
            nn.InstanceNorm2d(depth * 2),
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


def get_layers(in_channels, depth=64, num_layers=4):
    """
    This set of layers downsamples in `2**num_layers` times.
    """
    out_channels = in_channels
    sequence = []

    for n in range(num_layers):

        in_channels = out_channels
        out_channels = depth * min(2**n, 8)

        sequence.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    # add the final score predictor
    sequence.append(
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, 1, kernel_size=3, padding=1),
    )
    return nn.Sequential(*sequence)
