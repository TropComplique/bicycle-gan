import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F


class ResnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_blocks):
        super(ResnetGenerator, self).__init__()

        # number of times the network
        # downsamples the input:
        downsample = 2

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, depth, kernel_size=7),
            nn.InstanceNorm2d(depth),
            nn.ReLU(inplace=True)
        ]

        for i in range(downsample):
            m = 2**i  # multiplier
            layers += [
                nn.Conv2d(depth * m, depth * m * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(depth * m * 2),
                nn.ReLU(inplace=True)
            ]

        m = 2**downsample
        for _ in range(num_blocks):
            layers += [ResnetBlock(depth * m)]

        for i in range(downsample):
            m = 2**(downsample - i)
            layers += [
                nn.ConvTranspose2d(
                    depth * m, (depth * m) // 2,
                    kernel_size=3, stride=2,
                    padding=1, output_padding=1
                ),
                nn.InstanceNorm2d((depth * m) // 2),
                nn.ReLU(inplace=True)
            ]

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(depth, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        I assume that h and w are divisible by 2**downsample.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, out_channels, h, w].
        """
        return self.layers(x)


class ResnetBlock(nn.Module):
    def __init__(self, depth):
        super(ResnetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(depth, depth, kernel_size=3),
            nn.InstanceNorm2d(depth),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(depth, depth, kernel_size=3),
            nn.InstanceNorm2d(depth)
        )

    def forward(self, x):
        return x + self.layers(x)


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output

net = E_ResNet(input_nc, output_nc, 64, n_blocks=5, norm_layer=norm_layer,
                       nl_layer='lrelu', vaeLike=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out
