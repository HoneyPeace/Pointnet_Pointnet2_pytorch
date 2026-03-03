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

        # Segmentation & Offset 정답지 생성 (보로노이 다이어그램 기반 공유 로직)
        bbox_lengths = np.max(points, axis=0) - np.min(points, axis=0)
        ear_height = np.max(bbox_lengths) 
        
        # 1. 랜드마크가 가질 수 있는 최대 온전한 반경 설정 (귀 높이의 2.5%)
        max_radius = 0.025 * ear_height
        
        seg_target = np.zeros(N, dtype=np.int64) 
        offset_target = np.zeros((N, 3), dtype=np.float32)
        
        # 2. 모든 점(N)과 모든 랜드마크(M) 사이의 거리 행렬(N x M)을 한 번에 계산
        diff = points[:, np.newaxis, :] - landmarks[np.newaxis, :, :] 
        dists = np.linalg.norm(diff, axis=2) # (N, M) 거리 행렬
        
        # 3. 각 점의 입장에서 가장 가까운 랜드마크의 '거리'와 '번호'를 찾음 (핵심!)
        min_dists = np.min(dists, axis=1)         # (N,) 가장 짧은 거리
        closest_lm_idx = np.argmin(dists, axis=1) # (N,) 가장 가까운 랜드마크 번호 (0~39)
        
        # 4. '가장 가까운 거리'가 최대 반경(max_radius) 안에 들어오는 점들만 마스킹
        # 반경 밖이면 영원히 배경(Class 0)으로 남음
        valid_mask = min_dists <= max_radius
        
        # 5. 마스킹된 점들에 한해, 가장 가까운 랜드마크의 번호를 정답지로 부여 (클래스 1~40)
        seg_target[valid_mask] = closest_lm_idx[valid_mask] + 1
        
        # 6. 오프셋: (자신이 소속된 랜드마크의 위치) - (현재 점의 위치)
        assigned_landmarks = landmarks[closest_lm_idx[valid_mask]] 
        offset_target[valid_mask] = assigned_landmarks - points[valid_mask]

        return torch.from_numpy(points).float(), \
               torch.from_numpy(seg_target).long(), \
               torch.from_numpy(offset_target).float(), \
               torch.from_numpy(landmarks).float()

    def __len__(self):
        return self.points.shape[0]