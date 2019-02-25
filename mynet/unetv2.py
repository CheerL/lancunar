from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasicNet


class conv_block(nn.Module):
    '''(conv => BN => PReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    '''conv_block'''

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    '''maxpool => conv_block'''

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    '''(upsample/transconv + skip) => conv_block'''

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = partial(F.interpolate, scale_factor=2, mode='bilinear', align_corners=True)
            # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # input is CHW
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    '''conv => tranpose => flatten'''

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self._out_ch = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(-1, self._out_ch)


class UNetV2(BasicNet):
    def __init__(self, n_channels=1, n_classes=2, loss_type='nll', *args, **kwargs):
        super().__init__()
        if loss_type == 'nll':
            self.softmax = F.log_softmax
            self.loss = self.nll_loss
        elif loss_type == 'dice':
            self.softmax = F.softmax
            self.loss = self.dice_loss

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.net_name = 'U-Net-v2'

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return self.softmax(x, dim=1)
