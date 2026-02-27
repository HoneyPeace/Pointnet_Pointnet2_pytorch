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

'''data Rotate and Translate (오슐리번 논문 100% 반영)'''
class PointcloudRotateAndTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        current_device = pc.device 
        
        for i in range(bsize):
            # 1. 무작위 3차원 회전 (x, y, z 축 기준 0~360도)
            angles = np.random.uniform(low=0.0, high=2 * np.pi, size=[3])
            cx, cy, cz = np.cos(angles)
            sx, sy, sz = np.sin(angles)

            # X축, Y축, Z축 회전 행렬
            Rx = np.array([[1, 0, 0],
                           [0, cx, -sx],
                           [0, sx, cx]])
            Ry = np.array([[cy, 0, sy],
                           [0, 1, 0],
                           [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0],
                           [sz, cx, 0],
                           [0, 0, 1]])
            
            # 최종 회전 행렬 (R = Rz * Ry * Rx)
            R = np.dot(Rz, np.dot(Ry, Rx))
            R_tensor = torch.from_numpy(R).float().to(current_device)
            
            # 점들에 회전 적용
            pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], R_tensor.T)
            
            # 2. 무작위 이동 (Translation)
            xyz_trans = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            trans_tensor = torch.from_numpy(xyz_trans).float().to(current_device)
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + trans_tensor
            
        return pc