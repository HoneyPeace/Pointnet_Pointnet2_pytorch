import os
import sys
import torch
import numpy as np
import warnings
from tqdm import tqdm

from My_args import parser
from models.model import get_model
from dataset import EarLandmarkDataset
from post_process import compute_pred_landmarks

# RuntimeWarning (nan 관련) 숨기기
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 경로 설정
    project_dir = os.path.join(args.output_root, args.exp_name)
    base_str = f"Pts{args.num_points}_batch{args.batch_size}_train{args.train_len}"
    setting_str = f"{base_str}_{args.user_tag}" if args.user_tag else base_str
    target_folder_name = f"{setting_str}_{args.run_id}"
    run_root = os.path.join(project_dir, target_folder_name)

    model_path = os.path.join(run_root, 'Models', 'best_model.pth')
    npy_dir = os.path.join(run_root, 'npy_data')

    if not os.path.exists(model_path):
        print(f"[Error] 모델을 찾을 수 없습니다: {model_path}")
        sys.exit(1)

    print(f">> Loading Test Dataset from: {npy_dir}")
    TEST_DATASET = EarLandmarkDataset(npy_dir=npy_dir, partition=args.Eval_DataType)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)

    # [스케일 복원용 원본 데이터 로드]
    raw_shape_path = os.path.join(npy_dir, f'shape_{args.Eval_DataType}.npy')
    raw_land_path = os.path.join(npy_dir, f'landmark_{args.Eval_DataType}.npy')
    raw_shapes = np.load(raw_shape_path, allow_pickle=True)
    raw_lands = np.load(raw_land_path, allow_pickle=True)

    print(f">> Loading Model from {model_path}...")
    classifier = get_model(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    per_landmark_me_list = [] 

    print("\n>> Starting Evaluation...")
    with torch.no_grad():
        for i, (points, _, _, gt_landmarks) in enumerate(tqdm(testDataLoader)):
            points = points.to(device)
            points_input = points.transpose(2, 1) 
            
            seg_pred, offset_pred = classifier(points_input)
            pred_landmarks_batch = compute_pred_landmarks(points, seg_pred, offset_pred, num_landmarks=args.landmark_num)
            
            pred_lm_dict = pred_landmarks_batch[0] 
            gt_lm_array = gt_landmarks[0].numpy()  

            raw_pts = raw_shapes[i]
            raw_lms = raw_lands[i]
            combined = np.concatenate([raw_pts, raw_lms], axis=0)
            centroid = np.mean(combined, axis=0)
            m_scale = np.max(np.sqrt(np.sum((combined - centroid) ** 2, axis=1)))

            sample_errors = []
            for k in range(args.landmark_num):
                pred_pos = pred_lm_dict[k]
                gt_pos = gt_lm_array[k]
                
                # [수정] 랜드마크 누락(None) 시 에러 방지 -> nan(결측치)으로 안전하게 대체
                if pred_pos is None:
                    sample_errors.append(np.nan)
                    continue
                
                dist_norm = np.linalg.norm(pred_pos - gt_pos)
                dist_mm = dist_norm * m_scale 
                sample_errors.append(dist_mm)
                
            per_landmark_me_list.append(sample_errors)

    # -----------------------------------------------------------------------------
    # 5. 결과 집계 및 텍스트 저장 (결측치 nan 무시하고 계산)
    # -----------------------------------------------------------------------------
    per_landmark_me_array = np.array(per_landmark_me_list) 
    
    # np.nanmean과 np.nanstd를 사용하여 누락된 점은 수학적으로 안전하게 무시합니다.
    lm_means = np.nanmean(per_landmark_me_array, axis=0) 
    lm_stds = np.nanstd(per_landmark_me_array, axis=0)  
    
    average_me = np.nanmean(lm_means)
    std_me = np.nanmean(lm_stds)

    sample_mes = np.nanmean(per_landmark_me_array, axis=1) 
    sr_10 = np.sum(sample_mes < 10.0) / len(sample_mes) * 100
    sr_5  = np.sum(sample_mes < 5.0) / len(sample_mes) * 100

    report_lines = []
    report_lines.append("==========================================")
    report_lines.append(f"   Evaluation Result: {args.exp_name}")
    report_lines.append("==========================================")
    report_lines.append(f" Run ID      : {target_folder_name}")
    report_lines.append(f" Data Type   : {args.Eval_DataType}")
    report_lines.append(f" User Comment: {args.user_tag if args.user_tag else 'None'}")
    report_lines.append(f" Train Data  : {args.train_len} samples")
    report_lines.append(f" Num Points  : {args.num_points}")
    report_lines.append("------------------------------------------")
    report_lines.append(f" Average ME : {average_me:.4f} mm")
    report_lines.append(f" Average Std: {std_me:.4f} mm (Mean of Landmark Stds)") 
    report_lines.append(f" SR @ 10mm  : {sr_10:.2f} %")
    report_lines.append(f" SR @ 5mm   : {sr_5:.2f} %")
    report_lines.append("==========================================")
    
    # Top 5 Hardest
    report_lines.append("\n>>> Top 5 Hardest Landmarks:")
    worst_indices = np.argsort(lm_means)[::-1][:5]
    for i in worst_indices:
        report_lines.append(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm")
    
    # Top 5 Easiest
    report_lines.append("\n>>> Top 5 Easiest Landmarks:")
    best_indices = np.argsort(lm_means)[:5]
    for i in best_indices:
        report_lines.append(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm")
    
    # All Landmarks
    report_lines.append("\n>>> Per-landmark ME (Mean ± Std):")
    for i in range(args.landmark_num):
        report_lines.append(f"    LM {i:02d}: {lm_means[i]:.3f} ± {lm_stds[i]:.3f} mm")
        
    report_lines.append("    ------------------------------------")
    report_lines.append(f"    All  : {average_me:.3f} ± {std_me:.3f} mm")

    filename = f"ME{average_me:.4f}_std{std_me:.4f}.txt"
    result_txt_path = os.path.join(run_root, filename)
    
    with open(result_txt_path, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n" + "\n".join(report_lines))
    print(f"\n[Done] Results saved to: {run_root}")
    print(f"      Filename: {filename}")
    print(f"Average ME: {average_me:.4f} ± {std_me:.4f}\n")

if __name__ == '__main__':
    main()