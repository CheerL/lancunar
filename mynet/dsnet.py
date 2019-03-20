import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import ConvBlock, ResConvBlock, ResBottleneckBlock, BasicUNet


def get_deep_supvised_loss(loss_function, block_num, weight=None):
    if weight is None:
        weight = torch.ones(block_num).float()
    else:
        weight = torch.Tensor(weight)
    loss_weight = weight.cuda()

    def deep_supvised_loss(logits, labels):
        labels = labels.double()
        labels = [F.max_pool2d(labels, 2 ** i) if i else labels for i in reversed(range(block_num))]
        losses = torch.stack([loss_function(logit, label) for logit, label in zip(logits, labels)])
        return (losses * loss_weight).sum()
    return deep_supvised_loss

class DSBasicUNet(BasicUNet):
    def __init__(self, Block, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', ds_weight=None, *args, **kwargs):
        super().__init__(Block, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.out = nn.ModuleList([
            nn.Conv2d(up_block.out_ch, n_classes, 1)
            for up_block in self.up_blocks
        ])
        self.loss = get_deep_supvised_loss(self.loss, self.block_num, ds_weight)
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

    def iou(self, logits, labels):
        logits = logits[-1]
        return super().iou(logits, labels)

    def get_pred(self, logits, size):
        logits = logits[-1]
        return super().get_pred(logits, size)

    def img(self, vis, data, labels, logits, size):
        block_num = len(logits)
        img_dict = {
            'input': data,
            'gt': labels.view(-1, 1, size, size),
            'pred': self.get_pred(logits, size).view(-1, 1, size, size)
        }
        for num in range(1, block_num):
            sub_size = int(size / (2 ** num))
            img_dict['sub_pred{}'.format(num)] = logits[-num-1].max(1)[1].view(-1, 1, sub_size, sub_size)
            img_dict['sub_gt{}'.format(num)] = F.max_pool2d(labels.float(), 2 ** num).view(-1, 1, sub_size, sub_size)
        vis.img_many(img_dict)


class DSUNet(DSBasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ConvBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'DSUNet'


class DSResUNet(DSBasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ResConvBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'DSResUNet'


class DSResBottleneckUNet(DSBasicUNet):
    def __init__(self, n_channels=1, n_classes=2, block_num=4, dropout=1, loss_type='dice', *args, **kwargs):
        super().__init__(ResBottleneckBlock, n_channels, n_classes, block_num, dropout, loss_type, *args, **kwargs)
        self.net_name = 'DSResBottleneckUNet'
