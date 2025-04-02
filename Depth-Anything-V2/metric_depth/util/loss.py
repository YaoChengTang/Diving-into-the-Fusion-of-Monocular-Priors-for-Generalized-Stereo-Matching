import torch
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class AffineInvariantLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        # Affine-invariant (d-t) / s
        with torch.no_grad():
            pred_median   = pred[valid_mask].median()
            target_median = target[valid_mask].median()
            pred_scale    = (pred-pred_median).mean()
            target_scale  = (target - target_median).mean()
            target = (target - target_median) / target_scale * pred_scale + pred_median  # avoid unstable opt for pred when pred_scale is too small
        # pred   = (pred - pred_median) / pred_scale
        # target = (target - target_median) / target_scale
        
        valid_mask = valid_mask.detach()
        loss = F.l1_loss(target[valid_mask], pred[valid_mask])

        return loss