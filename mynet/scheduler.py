import math
from torch.optim.lr_scheduler import *


class NoWorkLR(LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        lr_lambda = [lambda _: 1] * len(self.optimizer.param_groups)
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class RewarmCosineAnnealingLR(CosineAnnealingLR):
    def get_lr(self):
        x = math.pi * (self.last_epoch % self.T_max) / self.T_max
        return [self.eta_min + (lr - self.eta_min) * (1 + math.cos(x)) / 2 for lr in self.base_lrs]


class SomeCosineAnnealingLR(CosineAnnealingLR):
    def get_period(self):
        return int(math.log2(self.last_epoch / self.T_max + 1))

    def get_lr(self):
        period = self.get_period()
        T_period = self.T_max ** period
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - T_period + 1) / T_period)) / 2
                for base_lr in self.base_lrs]

class RewarmLongCosineAnnealingLR(CosineAnnealingLR):
    def get_period(self):
        return int(math.log2(self.last_epoch / self.T_max + 1))

    def get_lr(self):
        period = self.get_period()
        period_T = self.T_max * 2 ** period
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * (self.last_epoch - period_T + self.T_max) / period_T)) / 2
                for base_lr in self.base_lrs]
