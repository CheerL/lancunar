import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import ConvBlock, ResConvBlock, BasicUNet


def get_deep_supvised_loss(loss_function, block_num, weight=None):
    def deep_supvised_loss(logits, labels):
        labels = [F.max_pool2d(labels, 2 ** i) if i else labels for i in reversed(range(block_num))]
        losses = torch.Tensor([loss_function(logit, label) for logit, label in zip(logits, labels)])
        weight = torch.Tensor(weight) if weight is not None else torch.ones(block_num).float()
        weight /= weight.sum()
        return (losses * weight).sum()
    return deep_supvised_loss

class DSBasicUNet(BasicUNet):
    def __init__(self, Block, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(Block, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.out = nn.ModuleList([
            nn.Conv2d(up_block.out_ch, n_classes, 1)
            for up_block in self.up_blocks
        ])
        self.loss = get_deep_supvised_loss(self.loss, self.block_num)
        self.net_name = 'DSBasicUNet'

    def forward(self, x):
        skip_list = []
        out_list = []

        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_list.append(skip)

        x = self.mid(x)

        for skip, up_block, out in zip(reversed(skip_list), self.up_blocks, self.out):
            x = up_block(x, skip)
            out_list.append(self.softmax(out(x), dim=1))
        return out_list


class DSUNet(DSBasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ConvBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'DSNet'


class DSResUNet(DSBasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ResConvBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'DSResUNet'
