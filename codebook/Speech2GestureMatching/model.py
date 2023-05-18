import numpy as np
import torch
from torch import nn
from constant import NUM_MFCC_FEAT, NUM_JOINTS, num_frames

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


"""
ResyncNet is based on the 2D-UNet implementation by Naoto Usuyama.

https://github.com/usuyama/pytorch-unet

"""


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def double_conv_instancenorm(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm1d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm1d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True)
    )


class FlattenConv1D(nn.Module):
  def forward(self, x):
    N, C, D = x.size()
    return x.view(N, -1)


class ResyncNet(nn.Module):
    def __init__(self):
        super(ResyncNet, self).__init__()

        self.dconv_down1 = double_conv(NUM_MFCC_FEAT+NUM_JOINTS, 128)
        self.dconv_down2 = double_conv(128, 256)
        self.dconv_down3 = double_conv(256, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.avgpool = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(256 + 256, 256)
        self.dconv_up1 = double_conv(256 + 128, 256)
        self.conv_last = nn.Conv1d(256, NUM_JOINTS, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.avgpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.avgpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.avgpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        out = self.conv_last(x)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dconv_down1 = double_conv_instancenorm(NUM_MFCC_FEAT+NUM_JOINTS, 512)
        self.dconv_down2 = double_conv_instancenorm(512, 256)
        self.dconv_down3 = double_conv_instancenorm(256, 128)
        self.avgpool = nn.AvgPool1d(2)

        self.disc = nn.Sequential(
            self.dconv_down1,
            self.avgpool,
            self.dconv_down2,
            self.avgpool,
            self.dconv_down3,
            self.avgpool,
            FlattenConv1D(),
            nn.Linear(8 * 2 * num_frames, 1, bias=False)
        )

    def forward(self, x):
        return self.disc(x)