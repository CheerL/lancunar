import torch
import torchvision

from .base import BasicNet

class ResNet(BasicNet):
    def __init__(self, num_classes=2, dropout=0.5, loss_type='cross_entropy'):
        super().__init__(loss_type)
        self.resnet = torchvision.models.resnet50(num_classes=num_classes)
        self.resnet.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net_name = 'ResNet50'
        self.dropout = torch.nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

#         x = self.resnet.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return self.softmax(x, dim=1)

    def iou(self, pred, gt):
        smooth = 0.01
        gt = gt.float()
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        return (intersection + smooth) / (union + smooth)

    def get_prob(self, logits):
        return torch.exp(logits)

    def get_pred(self, logits, data, hard=False):
        seg = data[:, 1].float()
        prob = self.get_prob(logits)
        if hard:
            return seg * prob.max(1)[1].float().view(-1, 1, 1)
        else:
            return seg * prob[:, 1].view(-1, 1, 1)
    
    def img(self, vis, data, gt, logits, size):
        input_ = data[:, 0].view(-1, 1, size, size)
        seg = data[:, 1].view(-1, 1, size, size)
        gt = gt.view(-1, 1, size, size)
        prob = self.get_prob(logits)
        vis.img_many({
            'input': input_,
            'seg': seg,
            'gt': gt,
            'pred': seg * prob[:, 1].view(-1, 1, 1, 1).detach().float(),
            'hard_pred': seg * prob.max(1)[1].view(-1, 1, 1, 1).float()
        })