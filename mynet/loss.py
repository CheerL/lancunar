import sys
import torch

def dice_score(logits: torch.Tensor, labels: torch.Tensor, axes=(1, 2)):
    """
    logits: shape is (batch_size, 2, width, height), value belongs [0, 1]
    labels: shape is (batch_size, width, height), value belongs {0, 1}
    """
    smooth = 0.01                    # avoid dividing by 0
    labels = labels.float()
    probs = logits[:, 1]             # the probability of being classified into positive class
    numerator = 2 * (probs * labels).sum(axes)
    denominator = (probs + labels).sum(axes)
    return (numerator + smooth) / (denominator + smooth)

def tversky_score(logits: torch.Tensor, labels: torch.Tensor,
            weight_1: float=0.5, weight_2: float=0.5):
    """
    logits: shape is (batch_size, 2, width, height), value belongs [0, 1]
    labels: shape is (batch_size, width, height), value belongs {0, 1}
    weight_1: weight for false negative(True -> False)
    weight_2: weight for false positive(False -> True)
    """
    smooth = 0.01                     # avoid dividing by 0
    labels = labels.float()
    true_probs = logits[:, 1]         # the probability of being classified into positive class
    false_probs = logits[:, 0]        # the probability of being classified into negative class
    true_pos = true_probs * labels
    false_neg = true_probs * (1 - labels)
    false_pos = false_probs * labels
    numerator = true_pos.sum((1, 2))
    denominator = 2 * (true_pos + weight_1 * false_neg + weight_2 * false_pos).sum((1, 2))
    return (numerator + smooth) / (denominator + smooth)

def dice_loss(logits: torch.Tensor, labels: torch.Tensor):
    batch_size = logits.shape[0]
    dice = dice_score(logits, labels)
    return 1 - dice.sum() / batch_size

def flatten_dice_loss(logits: torch.Tensor, labels: torch.Tensor):
    dice = dice_score(logits, labels, axes=(1, 2, 3))
    return 1- dice

def tversky_loss(logits: torch.Tensor, labels: torch.Tensor,
                 weight_1: float=0.5, weight_2: float=0.5):
    """
    Tversky loss: https://arxiv.org/pdf/1706.05721.pdf
    """
    batch_size = logits.shape[0]
    tversky = tversky_score(logits, labels, weight_1, weight_2)
    return 1 - tversky.sum() / batch_size

def focal_tversky_loss(logits: torch.Tensor, labels: torch.Tensor,
                       weight_1: float=0.5, weight_2: float=0.5, gamma: float=2):
    """
    focal Tversky loss: https://arxiv.org/pdf/1810.07842.pdf
    """
    batch_size = logits.shape[0]
    tversky = tversky_score(logits, labels, weight_1, weight_2)
    return 1 - tversky.pow(gamma).sum() / batch_size

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

    return 1 - (true_dice + false_dice).sum() / batch_size

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    return torch.nn.functional.nll_loss(logits, labels)

def get_loss_function(loss_type: str):
    this = sys.modules[__name__]
    return getattr(this, '{}_loss'.format(loss_type))
