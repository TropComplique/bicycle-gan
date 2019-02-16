import torch
import torch.nn as nn
import torch.nn.init


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth=64, num_blocks=5):
        super(ResNetEncoder, self).__init__()

        layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, num_blocks):
            layers += [BasicBlock(depth * min(4, n), depth * min(4, n + 1))]

        layers += [nl_layer(), nn.AvgPool2d(8)]

        self.layers = nn.Sequential(*layers)
        self.fc1 = nn.Linear(output_ndf, output_nc)
        self.fc2 = nn.Linear(output_ndf, output_nc)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        mean = self.fc1(x)
        var = self.fc2(x)
        return mean, var


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)]

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)
