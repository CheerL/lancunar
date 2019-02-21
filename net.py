import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    @staticmethod
    def dice_loss(output, target):
        smooth = 0.001
        pred = output[:, 1]
        target = target.float()
        loss = 1 - (2 * (pred * target).sum() + smooth) / (pred.sum() + target.sum() + smooth)
        return loss

    @staticmethod
    def nll_loss(output, target):
        return F.nll_loss(output, target)

    @staticmethod
    def iou(pred, target):
        smooth = 0.001
        true_pred = pred[:, 1].float()
        intersection = true_pred[target == 1].sum()
        union = true_pred.sum() + target.float().sum() - intersection
        # print(intersection + smooth, union + smooth, (intersection + smooth) / (union + smooth))
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def dice_similarity_coefficient(pred, target):
        pass

    @staticmethod
    def sensitivity(pred, target):
        pass
