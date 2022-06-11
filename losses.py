"""Calculating the loss
You can build the loss function of BsiNet by combining multiple losses
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss





class log_cosh_dice_loss(nn.Module):
    def __init__(self, num_classes=1, smooth=1, alpha=0.7):
        super(log_cosh_dice_loss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        x = self.dice_loss(outputs, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def dice_loss(self, y_pred, y_true):
        """[function to compute dice loss]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        intersection = torch.sum((y_true * y_pred))
        coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
        return (1. - coeff)


def focal_loss(predict, label, alpha=0.6, beta=2):
    probs = torch.sigmoid(predict)
    # 交叉熵Loss
    ce_loss = nn.BCELoss()
    ce_loss = ce_loss(probs,label)
    alpha_ = torch.ones_like(predict) * alpha
    # 正label 为alpha, 负label为1-alpha
    alpha_ = torch.where(label > 0, alpha_, 1.0 - alpha_)
    probs_ = torch.where(label > 0, probs, 1.0 - probs)
    # loss weight matrix
    loss_matrix = alpha_ * torch.pow((1.0 - probs_), beta)
    # 最终loss 矩阵，为对应的权重与loss值相乘，控制预测越不准的产生更大的loss
    loss = loss_matrix * ce_loss
    loss = torch.sum(loss)
    return loss



class Loss:
    def __init__(self, dice_weight=0.0, class_weights=None, num_classes=1, device=None):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-7
            cls_weight = self.dice_weight / self.num_classes
            for cls in range(self.num_classes):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss


class LossMulti:
    def __init__(
            self, jaccard_weight=0.0, class_weights=None, num_classes=1, device=None
    ):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):

        targets = targets.squeeze(1)

        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-7  # 原先是1e-7
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (
                        torch.log((intersection + eps) / (union - intersection + eps))
                        * self.jaccard_weight
                )
        return loss


class LossBsiNet:
    def __init__(self, weights=[1, 1, 1]):
        self.criterion1 = LossMulti(num_classes=2)   #mask_loss
        self.criterion2 = LossMulti(num_classes=2)   #contour_loss
        self.criterion3 = nn.MSELoss()               ##distance_loss
        self.weights = weights

    def __call__(self, outputs1, outputs2, outputs3, targets1, targets2, targets3):
        #
        criterion = (
                self.weights[0] * self.criterion1(outputs1, targets1)
                + self.weights[1] * self.criterion2(outputs2, targets2)
                + self.weights[2] * self.criterion3(outputs3, targets3)
        )

        return criterion








