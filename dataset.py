import os
import torch
import numpy as np
from torch.utils.data import Dataset
import augmentations as aug 

class EarLandmarkDataset(Dataset):
    def __init__(self, npy_dir, partition='train'):
        self.npy_dir = npy_dir
        self.partition = partition
        
        shape_path = os.path.join(self.npy_dir, f'shape_{self.partition}.npy')
        land_path  = os.path.join(self.npy_dir, f'landmark_{self.partition}.npy')
        
        if not os.path.exists(shape_path):
            raise FileNotFoundError(f"NPY not found in {self.npy_dir}")

        self.points = np.load(shape_path, allow_pickle=True)
        self.landmarks = np.load(land_path, allow_pickle=True)
        
        if self.partition == 'train':
            self.augmenter = aug.PointcloudRotateAndTranslate()

    def __getitem__(self, item):
        points = self.points[item].copy()      
        landmarks = self.landmarks[item].copy()

        N = points.shape[0]
        M = landmarks.shape[0] 
        
        # [완벽 수정됨] 순수 points 만으로 중심점과 스케일을 계산하여 정규화
        centroid = np.mean(points, axis=0)
        points = points - centroid
        landmarks = landmarks - centroid # 랜드마크도 동일하게 이동
        
        m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / m
        landmarks = landmarks / m # 랜드마크도 동일하게 축소
        
        # 증강(Augmentation)을 위해 잠시 합침
        if self.partition == 'train':
            points_t = torch.from_numpy(points).float().unsqueeze(0)
            landmarks_t = torch.from_numpy(landmarks).float().unsqueeze(0)
            combined = torch.cat([points_t, landmarks_t], dim=1) 
            
            combined = self.augmenter(combined)
            
            combined = combined.squeeze(0).numpy()
            points = combined[:N, :]
            landmarks = combined[N:, :]

        # Segmentation & Offset 정답지 생성 (O'Sullivan 반경 로직)
        bbox_lengths = np.max(points, axis=0) - np.min(points, axis=0)
        ear_height = np.max(bbox_lengths) 
        
        min_lm_dist = float('inf')
        for i in range(M):
            for j in range(i + 1, M):
                dist = np.linalg.norm(landmarks[i] - landmarks[j])
                if dist < min_lm_dist:
                    min_lm_dist = dist
                    
        radius = min(0.025 * ear_height, 0.5 * min_lm_dist)
        
        seg_target = np.zeros(N, dtype=np.int64) 
        offset_target = np.zeros((N, 3), dtype=np.float32)
        
        for k in range(M):
            lm = landmarks[k]
            dists_to_lm = np.linalg.norm(points - lm, axis=1)
            mask = dists_to_lm <= radius
            seg_target[mask] = k + 1 
            offset_target[mask] = lm - points[mask] 

        return torch.from_numpy(points).float(), \
               torch.from_numpy(seg_target).long(), \
               torch.from_numpy(offset_target).float(), \
               torch.from_numpy(landmarks).float()

    def __len__(self):
        return self.points.shape[0]