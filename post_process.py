import numpy as np
import torch

def compute_pred_landmarks(points, seg_pred, offset_pred, num_landmarks=40):
    """
    엄격한 입력 규격을 요구하는 오슐리번 후처리(Voting) 함수.
    규격이 다르면 즉시 에러를 발생시켜 디버깅을 돕습니다.
    
    Args:
        points (torch.Tensor): [Batch, N, 3]
        seg_pred (torch.Tensor): [Batch, N, 41] (배경 1 + 랜드마크 40)
        offset_pred (torch.Tensor): [Batch, N, 3]
        num_landmarks (int): 찾고자 하는 랜드마크의 개수 (기본값: 40)
        
    Returns:
        batch_landmarks (list): 각 배치별 예측된 랜드마크 좌표 딕셔너리 리스트
    """
    
    # 1. 형태(Shape) 강제 검증: 다르면 무자비하게 에러를 발생시킵니다 (연구자의 시간 절약)
    assert points.dim() == 3 and points.size(2) == 3, \
        f"[Shape Error] points는 [B, N, 3]이어야 합니다. 현재: {points.shape}"
    assert seg_pred.dim() == 3 and seg_pred.size(2) == (num_landmarks + 1), \
        f"[Shape Error] seg_pred는 [B, N, {num_landmarks + 1}]이어야 합니다. 현재: {seg_pred.shape}"
    assert offset_pred.dim() == 3 and offset_pred.size(2) == 3, \
        f"[Shape Error] offset_pred는 [B, N, 3]이어야 합니다. 현재: {offset_pred.shape}"

    # 2. 안전하게 CPU Numpy로 변환
    points_np = points.detach().cpu().numpy()
    offset_np = offset_pred.detach().cpu().numpy()
    
    # 3. [B, N, 41] 형태의 예측 확률에서 argmax를 통해 클래스 라벨 [B, N] 추출
    seg_labels = torch.argmax(seg_pred, dim=2).detach().cpu().numpy()

    batch_size = points_np.shape[0]
    batch_landmarks = []

    # 4. 배치별 연산 진행
    for b in range(batch_size):
        landmarks = {}
        
        # 0번부터 39번 랜드마크 순회
        for k in range(num_landmarks): 
            target_class = k + 1 # 클래스 0은 배경
            
            # 현재 랜드마크 영역으로 예측된 점들의 인덱스 찾기
            indices = np.where(seg_labels[b] == target_class)[0]

            # 검출 실패 (Undetected)
            if len(indices) == 0:
                landmarks[k] = None 
                continue
            
            # 투표(Voting) 연산: 위치(P) + 오프셋(O)
            pts_xyz = points_np[b][indices]
            pts_offset = offset_np[b][indices]
            votes = pts_xyz + pts_offset 

            # 투표 결과 평균 계산
            final_pos = np.mean(votes, axis=0)
            landmarks[k] = final_pos

        batch_landmarks.append(landmarks)

    return batch_landmarks