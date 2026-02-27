import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes=41):
        super(get_model, self).__init__()
        
        # 1. Feature Extraction (PointNet++ Encoder & Decoder)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128, 128])

        # ---------------------------------------------------------
        # 여기서부터 논문의 Figure 2와 정확히 일치하도록 두 갈래로 완벽히 분리됩니다.
        # ---------------------------------------------------------

        # 2. Segmentation Stage (3 stacked MLPs & Dropout in final two layers)
        self.seg_conv1 = nn.Conv1d(128, 128, 1)
        self.seg_bn1 = nn.BatchNorm1d(128)
        
        self.seg_drop1 = nn.Dropout(0.5)
        self.seg_conv2 = nn.Conv1d(128, 128, 1)
        self.seg_bn2 = nn.BatchNorm1d(128)
        
        self.seg_drop2 = nn.Dropout(0.5) 
        self.seg_conv3 = nn.Conv1d(128, num_classes, 1)

        # 3. Fully Convolutional Block (3 PointConv layers for Offsets)
        self.off_conv1 = nn.Conv1d(128, 128, 1)
        self.off_bn1 = nn.BatchNorm1d(128)
        
        self.off_conv2 = nn.Conv1d(128, 128, 1)
        self.off_bn2 = nn.BatchNorm1d(128)
        
        self.off_conv3 = nn.Conv1d(128, 3, 1)

    def forward(self, xyz):
        B, C, N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        
        # --- Feature Extraction ---
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)

        # --- Segmentation Stage ---
        s = F.relu(self.seg_bn1(self.seg_conv1(l0_points)))
        s = self.seg_drop1(s)
        s = F.relu(self.seg_bn2(self.seg_conv2(s)))
        s = self.seg_drop2(s)
        seg_pred = self.seg_conv3(s)
        seg_pred = F.log_softmax(seg_pred, dim=1)
        seg_pred = seg_pred.permute(0, 2, 1) 

        # --- Offset Stage ---
        o = F.relu(self.off_bn1(self.off_conv1(l0_points)))
        o = F.relu(self.off_bn2(self.off_conv2(o)))
        offset_pred = self.off_conv3(o)
        offset_pred = offset_pred.permute(0, 2, 1) 

        return seg_pred, offset_pred