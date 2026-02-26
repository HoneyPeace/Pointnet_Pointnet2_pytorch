"""
@Author: Yuan Wang (Modified for O'Sullivan PointNet++)
@File: augmentations.py
"""

import torch
import numpy as np

'''data normalization'''
def normalize_data(batch_data):
    # batch_data : batch_size * num_points * num_dims
    B, N, C = batch_data.shape
    centroid = torch.mean(batch_data, axis=1)
    batch_data = batch_data - centroid.unsqueeze(1).repeat(1, N, 1)
    m = torch.max(torch.sqrt(torch.sum(batch_data ** 2, axis=2)),axis=1)[0]
    batch_data = batch_data / m.view(-1, 1, 1)
    return batch_data

'''data Scale and Translate'''
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        # 현재 텐서가 있는 장치(CPU)를 파악
        current_device = pc.device 
        
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            # pc의 장치와 동일하게 텐서 생성 (디바이스 충돌 방지)
            scale_tensor = torch.from_numpy(xyz1).float().to(current_device)
            trans_tensor = torch.from_numpy(xyz2).float().to(current_device)
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scale_tensor) + trans_tensor
        return pc