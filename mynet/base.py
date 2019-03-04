import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss = getattr(self, loss_type)

    @staticmethod
    def dice_loss(logits: torch.Tensor, labels: torch.Tensor):
        """
        logits: shape is (batch_size, 2, width, height), value belongs [0, 1]
        labels: shape is (batch_size, width, height), value belongs {0, 1}
        """
        batch_size = logits.shape[0]
        smooth = 0.01                    # avoid dividing by 0
        labels = labels.float()
        probs = logits[:, 1]             # the probability of being classified into positive class
        numerator = 2 * (probs * labels).sum((1, 2))
        denominator = (probs + labels).sum((1, 2))
        dice = (numerator + smooth) / (denominator + smooth)
        loss = 1 - dice.sum() / batch_size
        return loss

    @staticmethod
    def dice_loss_with_background(logits: torch.Tensor, labels: torch.Tensor):
        """
        logits: shape is (batch_size, 2, width, height), value belongs [0, 1]
        labels: shape is (batch_size, width, height), value belongs {0, 1}
        """
        batch_size = logits.shape[0]
        smooth = 0.01                     # avoid dividing by 0
        true_labels = labels.float()
        true_probs = logits[:, 1]         # the probability of being classified into positive class
        true_numerator = (true_probs * true_labels).sum((1, 2))
        true_denominator = (true_probs + true_labels).sum((1, 2))
        true_dice = (true_numerator + smooth) / (true_denominator + smooth)

        false_labels = 1 - true_labels
        false_probs = logits[:, 0]        # the probability of being classified into negative class
        false_numerator = (false_probs * false_labels).sum((1, 2))
        false_denominator = (false_probs + false_labels).sum((1, 2))
        false_dice = (false_numerator + smooth) / (false_denominator + smooth)

        loss = 1 - (true_dice + false_dice).sum() / batch_size
        return loss


    @staticmethod
    def weighted_dice_loss(logits: torch.Tensor, labels: torch.Tensor, weight_1: float=0.7, weight_2: float=0.3):
        """
        Tversky loss: https://arxiv.org/pdf/1706.05721.pdf
        logits: shape is (batch_size, 2, width, height), value belongs [0, 1]
        labels: shape is (batch_size, width, height), value belongs {0, 1}
        weight_1: weight for false negative(True -> False)
        weight_2: weight for false positive(False -> True)
        """
        batch_size = logits.shape[0]
        smooth = 0.01                     # avoid dividing by 0
        labels = labels.float()
        true_probs = logits[:, 1]         # the probability of being classified into positive class
        false_probs = logits[:, 0]        # the probability of being classified into negative class
        true_pos = true_probs * labels
        false_neg = true_probs * (1 - labels)
        false_pos = false_probs * labels
        numerator = true_pos.sum((1, 2))
        denominator = (true_pos + weight_1 * false_neg + weight_2 * false_pos).sum((1, 2))
        dice = (numerator + smooth) / (denominator + smooth)
        loss = 1 - dice.sum() / batch_size
        return loss

    @staticmethod
    def flatten_dice_loss(logits: torch.Tensor, labels: torch.Tensor):
        """
        logits: shape is (batch_size, 2, width, height), value belongs [0, 1]
        labels: shape is (batch_size, width, height), value belongs {0, 1}
        """
        smooth = 0.01                     # avoid dividing by 0
        probs = logits[:, 1]             # the probability of being classified into positive class
        probs = probs.view(-1)              # shape is (batch_size * width * height)
        labels = labels.float()
        labels = labels.view(-1)            # shape is (batch_size * width * height)
        intersection = probs * labels
        dice = (2 * intersection.sum() + smooth) / (probs.sum() + labels.sum() + smooth)
        loss = 1 - dice
        return loss
    
    @staticmethod
    def nll_loss(logits, labels):
        return F.nll_loss(logits, labels)

    @staticmethod
    def iou(logits, labels):
        smooth = 0.01
        probs = logits[:, 1]
        intersection = probs[labels == 1].sum()
        union = probs.sum() + labels.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    @staticmethod
    def dice_similarity_coefficient(pred, target):
        pass

    @staticmethod
    def sensitivity(pred, target):
        pass
