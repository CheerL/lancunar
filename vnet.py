import torch
import torch.nn as nn
import torch.nn.functional as F
from net import BasicNet


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, inChans,  outChans, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, outChans)
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(outChans)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(inChans, outChans, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(inChans, outChans, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(outChans)
        self.relu2 = ELUCons(elu, outChans)

    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.ops = _make_nConv(outChans, outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.ops(down)
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)

        self.ops1 = LUConv(outChans*2, outChans, elu)
        self.ops2 = _make_nConv(outChans, outChans, nConvs-1, elu)

    def forward(self, x, skipx):
        out = self.relu1(self.bn1(self.up_conv(x)))
        xcat = torch.cat((out, skipx), 1)
        out = self.ops1(xcat)
        out = self.ops2(out)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=1, padding=0)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv1(x)

        # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(-1, 2)
        out = self.softmax(out, dim=1)
        # treat channel 0 as the predicted output
        return out


class VNet(BasicNet):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_channels=1, n_classes=2, loss_type='nll', elu=False):
        super(VNet, self).__init__()
        if loss_type == 'nll':
            nll = True
            self.loss = self.nll_loss
        elif loss_type == 'dice':
            nll = False
            self.loss = self.dice_loss

        self.in_tr = InputTransition(n_channels, 32, elu)
        self.down_tr64 = DownTransition(32, 64, 2, elu)
        self.down_tr128 = DownTransition(64, 128, 2, elu)
        self.up_tr64 = UpTransition(128, 64, 2, elu)
        self.up_tr32 = UpTransition(64, 32, 2, elu)
        self.out_tr = OutputTransition(32, n_classes, elu, nll)
        self.net_name = 'V-Net'

    def forward(self, x):
        out32 = self.in_tr(x)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)

        out = self.up_tr64(out128, out64)
        out = self.up_tr32(out, out32)

        out = self.out_tr(out)
        return out
