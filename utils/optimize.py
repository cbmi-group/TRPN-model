   
import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

   
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']  

def configure_optimizers(parameters, lr, weight_decay, gamma, lr_decay_every_x_epochs):
    optimizer = optim.SGD(params=parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_decay_every_x_epochs, gamma=gamma)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=lr_decay_every_x_epochs)
    return optimizer, scheduler


def soft_iou_loss(pred, label):
    b = pred.size()[0]
    pred = pred.view(b, -1)
    label = label.view(b, -1)
    inter = torch.sum(torch.mul(pred, label), dim=-1, keepdim=False)
    unit = torch.sum(torch.mul(pred, pred) + label, dim=-1, keepdim=False) - inter
    return torch.mean(1 - inter / unit)

class iou_loss(nn.Module):
    def __init__(self):
        super(iou_loss, self).__init__()

    def forward(self, pred, label):
        b = pred.size()[0]
        pred = pred.view(b, -1)
        label = label.view(b, -1)
        inter = torch.sum(pred*label, dim=-1, keepdim=False)
        union = torch.sum(pred*pred + label, dim=-1, keepdim=False) - inter
        return torch.mean(1 - inter / union)
        

def weighted_edge_loss(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = nn.functional.binary_cross_entropy(
        prediction.squeeze().float(),label.float(), weight=mask, reduction='sum') / (num_negative + num_positive)
    
    return cost


def BCEDiceLoss(inputs, target):
    bce = F.binary_cross_entropy(inputs, target)
    smooth = 1e-10
    num = target.size(0)
    inputs = inputs.view(num, -1)
    target = target.view(num, -1)
    intersection = (inputs * target)
    dice = (2. * intersection.sum(1) + smooth) / (inputs.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return 0.5 * bce + dice
    

class CEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=3):
        super(CEDiceLoss, self).__init__()
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):    
        
        logits = F.softmax(inputs, dim=1)

        dice = 0.0
        for c in range(self.num_classes):
            pred = logits[:,c,:,:]
            label = (targets==c).float()
            intersection = torch.sum(pred * label)
            union = torch.sum(pred) + torch.sum(label)
            dice += (1 - 2 * intersection / (union + 1e-10))
        dice_loss = dice / inputs.size()[0]

        return self.alpha * F.cross_entropy(inputs, targets) + dice_loss


class iou_loss_multi_class(nn.Module):
    def __init__(self, num_classes=3):
        super(iou_loss_multi_class, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        logits = F.softmax(inputs, dim=1)
        b = inputs.size()[0]
        for c in range(self.num_classes):
            # import ipdb; ipdb.set_trace()
            pred = logits[:,c,:,:].view(b,-1)
            label = (targets==c).float().view(b,-1)
            intersection = torch.sum(pred * label, dim=-1, keepdim=False)
            union = torch.sum(pred * pred + label, dim=-1, keepdim=False) - intersection
            if c == 0:
                loss = 1 - intersection / union
            else:
                loss += (1 - intersection / union)

        return torch.mean(loss)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice
      
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, is_binary = True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.is_binary = is_binary
        self.bce_fn = nn.BCELoss(weight=self.weight)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        if self.is_binary:
            logpt = - self.bce_fn(input, target)
        else:
            logpt = -F.cross_entropy(input, target)

        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, ignore_index=None, reduction='mean',**kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6 # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, prob, target):
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -self.alpha * (pos_weight * torch.log(prob))
        
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -(1-self.alpha) * (neg_weight * torch.log(1-prob))

        loss = (pos_loss + neg_loss)

        return loss.mean()

## test
if __name__ == "__main__":
    
    a = torch.rand((2,1,8,8))
    b = torch.ones((2,1,8,8))
    l_new = BinaryFocalLoss()
    l_old = FocalLoss2d()
    loss1 = l_old(a,b)
    loss2 = l_new(a,b)
    print(loss1)
    print(loss2)
