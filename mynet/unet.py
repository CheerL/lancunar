from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BasicNet


class ConvBlock(nn.Module):
    '''
    => (conv => BN => ReLU) * 2 =>
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class ResConvBlock(nn.Module):
    '''
    => (BN => ReLU => conv) * 2 =>
      \=>   conv   =>   BN   =>/ 
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = self.residual(x)
        return self.conv(x) + residual


class ResBottleneckBlock(nn.Module):
    '''
    => BN => ReLU => conv1 => BN => ReLU => conv3 => BN => ReLU => conv1 =>
      \=>                           conv                               =>/ 
    '''
    def __init__(self, in_ch, out_ch, rate=4):
        super().__init__()
        mid_ch = int(out_ch / rate)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(in_ch),
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(mid_ch),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(mid_ch),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            # nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = self.residual(x)
        return self.conv(x) + residual

class DownConvBlock(nn.Module):
    '''
    => convblock [=> dropout] => maxpool =>
                              \=>  skip  =>
    '''
    def __init__(self, Block, in_ch, out_ch, dropout=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = Block(in_ch, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout is not 1 else None
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.pool(x), x


class UpConvBlock(nn.Module):
    '''
    => upsample => cat [=> dropout] => convblock =>
    =>  skip  =>/
    '''
    def __init__(self, Block, in_ch, out_ch, dropout=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(in_ch, out_ch, 3, padding=1)
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.dropout = nn.Dropout2d(dropout) if dropout is not 1 else None
        self.conv = Block(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv(x)


class BasicUNet(BasicNet):
    def __init__(self, Block, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(loss_type=loss_type)
        down_blocks_channel = [(n_channels, 64)] + [(2 ** (i + 6), 2 ** (i + 7)) for i in range(block_num - 1)]
        up_blocks_channel = [(2 ** (i + 7), 2 ** (i + 6)) for i in reversed(range(block_num))]
        self.block_num = block_num
        self.down_blocks = nn.ModuleList([
            DownConvBlock(Block, *block_channel, dropout)
            for block_channel in down_blocks_channel
        ])
        self.up_blocks = nn.ModuleList([
            UpConvBlock(Block, *block_channel, dropout)
            for block_channel in up_blocks_channel
        ])
        self.mid = Block(2 ** (block_num + 5), 2 ** (block_num + 6))
        self.out = nn.Conv2d(64, n_classes, 1)
        self.net_name = 'BasicUNet'

    def forward(self, x):
        skip_list = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_list.append(skip)

        x = self.mid(x)

        for skip, up_block in zip(reversed(skip_list), self.up_blocks):
            x = up_block(x, skip)

        return self.softmax(self.out(x), dim=1)

class UNet(BasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ConvBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'UNet'


class ResUNet(BasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ResConvBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'ResUNet'

class ResBottleneckUNet(BasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ResBottleneckBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'ResBottleneckUNet'
