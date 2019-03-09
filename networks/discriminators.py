import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDiscriminator(nn.Module):
    """
    This discriminator looks on
    patches of different scales.
    """
    def __init__(self, in_channels, depth=64, downsample=3):
        super(MultiScaleDiscriminator, self).__init__()

        self.subnetwork1 = get_layers(in_channels, depth, downsample)
        self.subnetwork2 = get_layers(in_channels, depth // 2, downsample)
        self.subnetwork3 = get_layers(in_channels, depth // 2, downsample)

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
            scores3: a float tensor.
        """
        h, w = x.size()[2:]

        x = 2.0 * x - 1.0
        scores1 = self.subnetwork1(x)

        x = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear')
        scores2 = self.subnetwork2(x)

        x = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear')
        scores3 = self.subnetwork3(x)

        return scores1, scores2, scores3


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
    26 if downsample = 2,
    54 if downsample = 3,
    110 if downsample = 4,
    222 if downsample = 5.
    See https://fomoro.com/projects/project/receptive-field-calculator
    """

    return nn.Sequential(*sequence)
