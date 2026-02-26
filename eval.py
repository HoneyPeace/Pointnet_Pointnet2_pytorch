import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from My_args import parser
from models.model import get_model
from dataset import EarLandmarkDataset
from post_process import compute_pred_landmarks

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 입력된 명령어를 바탕으로 정확한 백업 폴더 경로 역추적
    project_dir = os.path.join(args.output_root, args.exp_name)
    base_str = f"Pts{args.num_points}_batch{args.batch_size}_train{args.train_len}"
    setting_str = f"{base_str}_{args.user_tag}" if args.user_tag else base_str
    target_folder_name = f"{setting_str}_{args.run_id}"
    run_root = os.path.join(project_dir, target_folder_name)

    if not os.path.exists(run_root):
        print(f"[Error] 지정한 실험 폴더를 찾을 수 없습니다: {run_root}")
        print("명령어(train_len, batch_size 등)가 학습 때와 정확히 일치하는지 확인하세요.")
        sys.exit(1)

    model_path = os.path.join(run_root, 'Models', 'best_model.pth')
    npy_dir = os.path.join(run_root, 'npy_data')

    # 2. 데이터 및 모델 로드
    print(f">> Loading Test Dataset from: {npy_dir}")
    TEST_DATASET = EarLandmarkDataset(npy_dir=npy_dir, partition=args.Eval_DataType)
    # 평가는 개별 확인을 위해 batch_size=1 강제 적용
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)

    print(f">> Loading Model from {model_path}...")
    classifier = get_model(num_classes=args.num_classes, normal_channel=args.normal).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # 3. 평가 루프
    all_me_list = []      
    per_lm_errors = {i: [] for i in range(args.landmark_num)} 
    failure_count = 0     

    print("\n>> Starting Evaluation...")
    with torch.no_grad():
        for points, _, _, gt_landmarks in tqdm(testDataLoader):
            points = points.to(device)
            points_input = points.transpose(2, 1) 
            
            seg_pred, offset_pred = classifier(points_input)
            pred_landmarks_batch = compute_pred_landmarks(points, seg_pred, offset_pred, num_landmarks=args.landmark_num)
            
            pred_lm_dict = pred_landmarks_batch[0] 
            gt_lm_array = gt_landmarks[0].numpy()  

            sample_errors = []
            sample_failed = False
            
            for k in range(args.landmark_num):
                pred_pos = pred_lm_dict[k]
                gt_pos = gt_lm_array[k]
                
                if pred_pos is None:
                    sample_failed = True
                    break
                
                dist = np.linalg.norm(pred_pos - gt_pos)
                sample_errors.append(dist)
                per_lm_errors[k].append(dist)
                
            if sample_failed:
                failure_count += 1
            else:
                sample_me = np.mean(sample_errors)
                all_me_list.append(sample_me)

    # 4. 결과 출력 및 텍스트 저장
    print("\n" + "="*50)
    print(f"       Evaluation Result: {target_folder_name}")
    print("="*50)
    
    if len(all_me_list) > 0:
        total_mean_error = np.mean(all_me_list)
        total_std_error = np.std(all_me_list)
        failure_rate = (failure_count / len(testDataLoader)) * 100
        
        print(f" Mean Error : {total_mean_error:.4f} mm")
        print(f" Std Dev    : {total_std_error:.4f} mm")
        print(f" Failure    : {failure_rate:.2f} %")
        print("-" * 50)
        
        eval_txt_path = os.path.join(run_root, f"Eval_ME_{total_mean_error:.4f}.txt")
        with open(eval_txt_path, 'w') as f:
            f.write(f"Run Directory: {run_root}\n")
            f.write(f"Mean Error: {total_mean_error:.4f} mm\n")
            f.write(f"Std Deviation: {total_std_error:.4f} mm\n")
            f.write(f"Failure Rate: {failure_rate:.2f} %\n")
            f.write("-" * 50 + "\n[Per-Landmark Errors]\n")
            for k in range(args.landmark_num):
                if len(per_lm_errors[k]) > 0:
                    lm_mean = np.mean(per_lm_errors[k])
                    lm_std = np.std(per_lm_errors[k])
                    f.write(f"  - LM {k:02d}: {lm_mean:.4f} ± {lm_std:.4f} mm\n")
                    
        print(f" [완료] 평가 결과 저장됨: {eval_txt_path}")
    else:
        print(" [실패] 검출된 랜드마크가 없습니다.")
    print("="*50)

if __name__ == '__main__':
    main()