import torch
import torch.nn as nn
import torch.nn.functional as F

class OSullivan_Loss(nn.Module):
    def __init__(self):
        super(OSullivan_Loss, self).__init__()

    def forward(self, seg_pred, seg_target, offset_pred, offset_target):
        # 1. Segmentation Loss (랜드마크 영역에 가중치 5배 부여)
        weights = torch.ones(41).to(seg_pred.device) 
        weights[1:] = 5.0  

        seg_pred_flat = seg_pred.contiguous().view(-1, 41)
        seg_target_flat = seg_target.view(-1)
        loss_seg = F.nll_loss(seg_pred_flat, seg_target_flat, weight=weights)

        # 2. Offset Loss (랜드마크 영역인 점들만 계산)
        mask = seg_target_flat > 0 
        if torch.sum(mask) > 0:
            offset_pred_flat = offset_pred.contiguous().view(-1, 3)
            offset_target_flat = offset_target.contiguous().view(-1, 3)
            
            valid_pred = offset_pred_flat[mask]
            valid_target = offset_target_flat[mask]
            loss_offset = F.l1_loss(valid_pred, valid_target)
        else:
            loss_offset = 0.0

        # 3. Total Loss
        total_loss = loss_seg + loss_offset
        return total_loss