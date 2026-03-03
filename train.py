import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 커스텀 모듈 임포트
from My_args import parser
from init import _init_
from util import process_and_save_npy
from dataset import EarLandmarkDataset
from models.model import get_model
from loss import OSullivan_Loss

def main():
    args = parser.parse_args()
    args = _init_(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 원본 데이터를 NPY로 변환
    print(f"\n>> Processing and Saving Raw Data to NPY...")
    process_and_save_npy(args.data_root, args.npy_dir, args.train_dataset_name, args.num_points)
    process_and_save_npy(args.data_root, args.npy_dir, args.test_dataset_name, args.num_points)

    # 2. 데이터셋 로드
    print(f"\n>> Loading Datasets...")
    TRAIN_DATASET = EarLandmarkDataset(npy_dir=args.npy_dir, partition=args.train_dataset_name)
    TEST_DATASET = EarLandmarkDataset(npy_dir=args.npy_dir, partition=args.test_dataset_name)

    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # [데이터 검증] 정답지(GT) 내 클래스별 포인트 할당량 확인
    print("\n" + "="*60)
    print("📊 [데이터 검증] 정답지(GT) 내 클래스별 포인트 할당량 확인")
    sample_points, sample_seg, _, _ = TRAIN_DATASET[0]
    
    unique_classes, counts = torch.unique(sample_seg, return_counts=True)
    counts_dict = dict(zip(unique_classes.numpy(), counts.numpy()))
    
    bg_points = counts_dict.get(0, 0)
    lm_points_list = [counts_dict.get(k, 0) for k in range(1, args.landmark_num + 1)]
    
    print(f"   👉 총 포인트 수: {sample_points.shape[0]}개")
    print(f"   👉 배경(Class 0) 포인트: {bg_points}개 ({(bg_points/sample_points.shape[0])*100:.1f}%)")
    print(f"   👉 랜드마크 1개당 평균 할당 포인트: {np.mean(lm_points_list):.1f}개")
    print(f"      (최소 {np.min(lm_points_list)}개 ~ 최대 {np.max(lm_points_list)}개)")
    print("="*60 + "\n")

    # 텍스트 로그 파일 경로 설정 및 헤더 작성
    log_file_path = os.path.join(args.log_dir, 'train_log.txt')
    with open(log_file_path, "w", encoding='utf-8') as f:
        f.write(f"=== Training Log: {args.exp_name} ({args.user_tag}) ===\n")
        f.write(f"Total Epochs: {args.epoch}, Batch Size: {args.batch_size}, Points: {args.num_points}\n")
        f.write("="*60 + "\n\n")
    print(f"📝 [로그 저장] 매 에포크의 학습 기록이 다음 파일에 자동 저장됩니다:\n   👉 {log_file_path}\n")

    # 3. 모델 및 Loss 초기화
    print(f">> Loading Model...")
    classifier = get_model(num_classes=args.num_classes).to(device)
    criterion = OSullivan_Loss().to(device)
    
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # [추가] 최종 리포팅을 위한 변수 추적기
    best_test_loss = float('inf')
    best_epoch = 0
    best_model_path = ""

    # 4. 본격적인 학습 루프
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch+1}/{args.epoch}:')
        
        # --- TRAIN ---
        classifier.train()
        train_loss = 0.0
        train_correct_all, train_total_all = 0, 0
        train_correct_lm, train_total_lm = 0, 0 
        
        optimizer.zero_grad()
        
        for i, (points, target_seg, target_offset, _) in enumerate(tqdm(trainDataLoader, desc='Training')):
            points, target_seg, target_offset = points.to(device), target_seg.to(device), target_offset.to(device)
            points_input = points.transpose(2, 1)
            
            seg_pred, offset_pred = classifier(points_input)
            loss = criterion(seg_pred, target_seg, offset_pred, target_offset)
            
            loss = loss / args.accumulation_steps
            loss.backward()
            
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * args.accumulation_steps
            
            pred_choice = seg_pred.argmax(dim=2)
            correct = (pred_choice == target_seg)
            
            train_correct_all += correct.sum().item()
            train_total_all += target_seg.numel()
            
            lm_mask = target_seg > 0
            train_correct_lm += correct[lm_mask].sum().item()
            train_total_lm += lm_mask.sum().item()

        # --- EVAL (Validation) ---
        classifier.eval()
        test_loss = 0.0
        test_correct_all, test_total_all = 0, 0
        test_correct_lm, test_total_lm = 0, 0
        zero_pred_warning = 0 
        
        with torch.no_grad():
            for points, target_seg, target_offset, _ in tqdm(testDataLoader, desc='Evaluating'):
                points, target_seg, target_offset = points.to(device), target_seg.to(device), target_offset.to(device)
                points_input = points.transpose(2, 1)
                
                seg_pred, offset_pred = classifier(points_input)
                loss = criterion(seg_pred, target_seg, offset_pred, target_offset)
                test_loss += loss.item()
                
                pred_choice = seg_pred.argmax(dim=2)
                correct = (pred_choice == target_seg)
                
                test_correct_all += correct.sum().item()
                test_total_all += target_seg.numel()
                
                lm_mask = target_seg > 0
                test_correct_lm += correct[lm_mask].sum().item()
                test_total_lm += lm_mask.sum().item()
                
                for b in range(points.shape[0]):
                    pred_classes = torch.unique(pred_choice[b])
                    if len(pred_classes) < (args.landmark_num + 1):
                        zero_pred_warning += 1

        # 5. 결과 집계 및 출력
        t_loss_avg = train_loss / len(trainDataLoader)
        v_loss_avg = test_loss / len(testDataLoader)
        
        t_acc_all = (train_correct_all / train_total_all) * 100
        t_acc_lm  = (train_correct_lm / train_total_lm) * 100 if train_total_lm > 0 else 0
        
        v_acc_all = (test_correct_all / test_total_all) * 100
        v_acc_lm  = (test_correct_lm / test_total_lm) * 100 if test_total_lm > 0 else 0

        log_str =  f"[Epoch {epoch+1:03d}/{args.epoch}] \n"
        log_str += f"  - [Train] Loss: {t_loss_avg:.4f} | 전체 Acc: {t_acc_all:.2f}% | 랜드마크 Acc: {t_acc_lm:.2f}%\n"
        log_str += f"  - [Val]   Loss: {v_loss_avg:.4f} | 전체 Acc: {v_acc_all:.2f}% | 랜드마크 Acc: {v_acc_lm:.2f}%\n"
        
        if zero_pred_warning > 0:
            log_str += f"  ⚠️ [주의] 평가 중 {zero_pred_warning}개의 샘플에서 랜드마크 점을 0개로 예측했습니다. (nan 발생 가능성)\n"

        print(f"[Train] Loss: {t_loss_avg:.4f} | 전체 Acc: {t_acc_all:.2f}% | 랜드마크 Acc: {t_acc_lm:.2f}%")
        print(f"[Val]   Loss: {v_loss_avg:.4f} | 전체 Acc: {v_acc_all:.2f}% | 랜드마크 Acc: {v_acc_lm:.2f}%")
        if zero_pred_warning > 0:
            print(f"   ⚠️ [주의] 평가 중 {zero_pred_warning}개의 샘플에서 랜드마크 점을 0개로 예측했습니다. (nan 발생 가능성)")

        # 6. 베스트 모델 저장
        if v_loss_avg < best_test_loss:
            best_test_loss = v_loss_avg
            best_epoch = epoch + 1  # 최고 성능 에포크 업데이트
            best_model_path = os.path.abspath(os.path.join(args.ckpt_dir, 'best_model.pth')) # 절대 경로로 추출
            
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_test_loss,
            }, best_model_path)
            
            best_msg = f"⭐️ 베스트 모델 갱신 (Val Loss: {best_test_loss:.4f})"
            print(best_msg)
            log_str += f"  {best_msg}\n"
            
        log_str += "-" * 60 + "\n"
        
        with open(log_file_path, "a", encoding='utf-8') as f:
            f.write(log_str)

    # ---------------------------------------------------------
    # 7. [추가] 학습 종료 후 최종 요약 리포트 출력
    # ---------------------------------------------------------
    final_msg = "\n" + "🚀"*30 + "\n"
    final_msg += "🎉 [학습 종료] 지정된 모든 에포크가 완료되었습니다!\n"
    final_msg += "🚀"*30 + "\n"
    if best_epoch > 0:
        final_msg += f"🏆 최고 성능 모델 (Best Epoch) : {best_epoch} 번째 Epoch\n"
        final_msg += f"📉 갱신된 최저 Val Loss      : {best_test_loss:.4f}\n"
        final_msg += f"💾 베스트 가중치 저장 경로   : \n   👉 {best_model_path}\n"
    else:
        final_msg += "⚠️ 베스트 모델이 한 번도 저장되지 않았습니다. (Loss 계산 확인 필요)\n"
    final_msg += "="*60 + "\n"

    print(final_msg)

    # 최종 요약도 로그 텍스트 마지막에 기록
    with open(log_file_path, "a", encoding='utf-8') as f:
        f.write(final_msg)

if __name__ == '__main__':
    main()