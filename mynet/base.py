import torch.nn as nn
import torch.nn.functional as F
from .loss import get_loss_function


class BasicNet(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        self.loss = get_loss_function(loss_type)

        if 'cross_entropy' in loss_type:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def iou(self, logits, labels):
        smooth = 0.01
        probs = logits[:, 1]
        intersection = probs[labels == 1].sum()
        union = probs.sum() + labels.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def dice_similarity_coefficient(self,logits, labels):
        pass

    def sensitivity(self, logits, labels):
        pass
