import os
import torch
import logging
from tqdm import tqdm

from My_args import parser
from init import _init_
from util import process_and_save_npy
from dataset import EarLandmarkDataset
from models.model import get_model
from loss import OSullivan_Loss

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # init을 통해 폴더 생성 및 args 갱신
    args = _init_(args) 

    '''LOGGING SETUP'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(args.log_dir, 'train.txt'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(args)

    # 파티션명에 맞춰 NPY 생성 (train, test)
    process_and_save_npy(args.data_root, args.npy_dir, args.train_dataset_name, args.num_points)
    process_and_save_npy(args.data_root, args.npy_dir, args.test_dataset_name, args.num_points)

    logger.info(f">> Loading Datasets from {args.npy_dir}...")
    TRAIN_DATASET = EarLandmarkDataset(npy_dir=args.npy_dir, partition=args.train_dataset_name)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    TEST_DATASET = EarLandmarkDataset(npy_dir=args.npy_dir, partition=args.test_dataset_name)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    classifier = get_model(num_classes=args.num_classes, normal_channel=args.normal).cuda()
    criterion = OSullivan_Loss().cuda()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_acc = 0.0 

    for epoch in range(args.epoch):
        logger.info(f'\nEpoch {epoch+1}/{args.epoch}')
        classifier.train()
        total_loss_sum, correct_seg = 0, 0
        
        optimizer.zero_grad() # Accumulation 시작 전 초기화
        
        with tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), desc="Train") as tepoch:
            for i, (points, seg_target, offset_target, _) in tepoch:
                points = points.float().cuda().transpose(2, 1)
                seg_target = seg_target.long().cuda()
                offset_target = offset_target.float().cuda()

                seg_pred, offset_pred = classifier(points)
                loss = criterion(seg_pred, seg_target, offset_pred, offset_target)
                
                # Gradient Accumulation 로직
                loss = loss / args.accumulation_steps
                loss.backward()
                
                if (i + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                pred_choice = seg_pred.data.max(2)[1]
                correct_seg += pred_choice.eq(seg_target.data).cpu().sum().item()
                total_loss_sum += (loss.item() * args.accumulation_steps)
                
                tepoch.set_postfix(loss=(loss.item() * args.accumulation_steps))

        train_acc = correct_seg / (len(trainDataLoader) * args.batch_size * args.num_points)
        logger.info(f'Train Loss: {total_loss_sum/len(trainDataLoader):.4f} | Seg Acc: {train_acc:.4f}')

        # Validation 루프
        with torch.no_grad():
            classifier.eval()
            val_correct_seg, val_total_points = 0, 0
            
            for points, seg_target, _, _ in testDataLoader:
                points = points.float().cuda().transpose(2, 1)
                seg_target = seg_target.long().cuda()
                
                seg_pred, _ = classifier(points)
                pred_choice = seg_pred.data.max(2)[1]
                val_correct_seg += pred_choice.eq(seg_target.data).cpu().sum().item()
                val_total_points += (args.batch_size * args.num_points)
            
            test_acc = val_correct_seg / val_total_points
            logger.info(f'Validation Seg Acc: {test_acc:.4f}')

            if test_acc >= best_acc:
                best_acc = test_acc
                savepath = os.path.join(args.ckpt_dir, 'best_model.pth')
                torch.save({'model_state_dict': classifier.state_dict()}, savepath)
                logger.info(f'*** Best Model Saved to {savepath} ***')

if __name__ == '__main__':
    main()