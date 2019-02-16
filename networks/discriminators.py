import torch
import torch.nn as nn
import torch.nn.init


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(D_NLayersMulti, self).__init__()

        self.layers1 = self.get_layers(input_nc, ndf, n_layers, norm_layer)
        self.down = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

        self.layers2 = self.get_layers(input_nc, ndf // 2, n_layers, norm_layer)

    def get_layers(self, in_channels, depth=64, num_layers=3):

        sequence = [
            nn.Conv2d(in_channels, depth, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(depth * m),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        m, m_previous = 1, 1
        for n in range(1, num_layers + 1):

            m_previous = m
            m = min(2**n, 8)

            sequence += [
                nn.Conv2d(depth * m_previous, depth * m, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(depth * m),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        sequence += [
            nn.Conv2d(depth * m, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        ]


        if global:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=0)]
        sequence += [nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=0)]

        def weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(weights_init)
        return nn.Sequential(*sequence)

    def forward(self, input):
        """
        I assume that h and w are divisible by 2**downsample.

        Arguments:
            x: a float tensor with shape [b, in_channels, h, w].
        Returns:
            a float tensor with shape [b, 1]???
        """

        result = []
        result.append(self.layers1(x))
        x = self.down(x)
        result.append(self.layers2(x))
        return result



class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
