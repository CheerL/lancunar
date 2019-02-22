import torch
import torch.nn as nn
import torch.nn.functional as F
from net import BasicNet
from functools import partial

class Indentity(nn.Module):
    def forward(self, x):
        return x

class ConvBlock(nn.Module):
    '''(conv => BN => PReLU [=> dropout]) * 2'''
    def __init__(self, in_ch, out_ch, dropout=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Dropout2d(dropout) if dropout is not 1 else Indentity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
            nn.Dropout2d(dropout) if dropout is not 1 else Indentity()
        )

    def forward(self, x):
        return self.conv(x)

class DownConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=1):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class UpConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=1):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.conv = ConvBlock(in_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    '''conv => tranpose => flatten'''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self._out_ch = out_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(-1, self._out_ch)

class UNet(BasicNet):
    def __init__(self, n_channels=1, n_classes=2, dropout=1, loss_type='nll'):
        super(UNet, self).__init__()
        if loss_type == 'nll':
            self.softmax = F.log_softmax
            self.loss = self.nll_loss
        elif loss_type == 'dice':
            self.softmax = F.softmax
            self.loss = self.dice_loss

        self.down1 = DownConvBlock(n_channels, 64, dropout)
        self.down2 = DownConvBlock(64, 128, dropout)
        self.down3 = DownConvBlock(128, 256, dropout)
        self.down4 = DownConvBlock(256, 512, dropout)
        self.mid = ConvBlock(512, 1024, dropout)
        self.up4 = UpConvBlock(1024, 512, dropout)
        self.up3 = UpConvBlock(512, 256, dropout)
        self.up2 = UpConvBlock(256, 128, dropout)
        self.up1 = UpConvBlock(128, 64, dropout)
        self.outc = OutConv(64, n_classes)
        self.net_name = 'U-Net'

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.mid(x)
        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        x = self.outc(x)
        return self.softmax(x, dim=1)
