import os
import torch
import numpy as np
from torch.utils.data import Dataset
import augmentations as aug 

class EarLandmarkDataset(Dataset):
    def __init__(self, npy_dir, partition='train'):
        self.npy_dir = npy_dir
        self.partition = partition
        
        # Result 폴더 내의 npy_dir에서 데이터 로드
        shape_path = os.path.join(self.npy_dir, f'shape_{self.partition}.npy')
        land_path  = os.path.join(self.npy_dir, f'landmark_{self.partition}.npy')
        
        if not os.path.exists(shape_path):
            raise FileNotFoundError(f"NPY not found in {self.npy_dir}. Run util.py first.")

        self.points = np.load(shape_path, allow_pickle=True)
        self.landmarks = np.load(land_path, allow_pickle=True)
        
        if self.partition == 'train':
            self.scaler = aug.PointcloudScaleAndTranslate()

    def __getitem__(self, item):
        points = self.points[item]      
        landmarks = self.landmarks[item] 

        N = points.shape[0]
        M = landmarks.shape[0] 
        
        # Augmentation (점과 랜드마크 묶어서 변형)
        points_t = torch.from_numpy(points).float().unsqueeze(0)
        landmarks_t = torch.from_numpy(landmarks).float().unsqueeze(0)
        combined = torch.cat([points_t, landmarks_t], dim=1) 
        
        combined = aug.normalize_data(combined)
        if self.partition == 'train':
            combined = self.scaler(combined)
            
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