import argparse

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='O\'Sullivan 3D Ear Landmark Detection')

# [1] 실험 및 경로 설정 (기존 명령어 스타일 반영)
parser.add_argument('--exp_name', type=str, default='Ear_10', help='실험 최상위 폴더명')
parser.add_argument('--user_tag', type=str, default='', help='폴더명 뒤에 붙일 커스텀 태그 (예: 증강12)')
parser.add_argument('--run_id', type=int, default=1, help='평가 시 불러올 특정 실험 번호')
parser.add_argument('--data_root', type=str, default='../Data', help='원본 데이터 경로')
parser.add_argument('--output_root', type=str, default='../Result', help='결과 저장 경로')
parser.add_argument('--gpu', type=str, default='0', help='사용할 GPU 번호')

# [2] 데이터셋 및 샘플링 설정
parser.add_argument('--train_dataset_name', type=str, default='train', help='학습 데이터 폴더명')
parser.add_argument('--test_dataset_name', type=str, default='test', help='테스트 데이터 폴더명')
parser.add_argument('--Eval_DataType', type=str, default='test', help='평가 시 사용할 데이터 파티션')
parser.add_argument('--train_len', type=int, default=0, help='학습 데이터 개수 (평가 폴더명 추적용)')

# [3] 학습 하이퍼파라미터
parser.add_argument('--num_points', type=int, default=4096, help='입력 포인트 개수')
parser.add_argument('--batch_size', type=int, default=32, help='배치 사이즈')
parser.add_argument('--accumulation_steps', type=int, default=2, help='그래디언트 누적 스텝 (VRAM 절약용)')
parser.add_argument('--epoch', type=int, default=250, help='총 에포크 수')
parser.add_argument('--learning_rate', type=float, default=0.001, help='초기 학습률')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='가중치 감소')

# [4] 모델 구조 설정 (PointNet++)
parser.add_argument('--landmark_num', type=int, default=44, help='랜드마크 총 개수')
parser.add_argument('--num_classes', type=int, default=45, help='Segmentation 클래스 (배경 1 + 랜드마크 40)')
