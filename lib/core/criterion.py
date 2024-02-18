# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config
import pdb


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])




class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_label=-1, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_label = ignore_label
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight, ignore_index=self.ignore_label)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class CrossEntropyWithFocalLoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, alpha=0.50, gamma=1.0):
        super(CrossEntropyWithFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            weight=weight,
            ignore_label=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])



class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        functions = [self._ce_forward] * \
            (len(weights) - 1) + [self._ohem_forward]
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])
