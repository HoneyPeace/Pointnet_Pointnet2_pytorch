import os
import shutil
import glob

def get_train_len(data_root, train_dir_name):
    """학습 데이터 폴더(.ply)의 개수를 자동으로 세어줍니다."""
    train_path = os.path.join(data_root, train_dir_name)
    if os.path.exists(train_path):
        return len(glob.glob(os.path.join(train_path, "*.ply"))) + len(glob.glob(os.path.join(train_path, "*.obj")))
    return 0

def _init_(args):
    # 1. Train 데이터 개수 파악
    args.train_len = get_train_len(args.data_root, args.train_dataset_name)
    
    # 2. 박사님 스타일의 폴더명 조합
    project_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(project_dir, exist_ok=True)
    
    base_str = f"Pts{args.num_points}_batch{args.batch_size}_train{args.train_len}"
    setting_str = f"{base_str}_{args.user_tag}" if args.user_tag else base_str
    
    # 3. 중복되지 않는 run_id 자동 탐색
    run_id = 1
    while True:
        target_folder_name = f"{setting_str}_{run_id}"
        run_root = os.path.join(project_dir, target_folder_name)
        if not os.path.exists(run_root):
            break
        run_id += 1
        
    args.run_id = run_id
    args.run_root = run_root
    
    # 4. 하위 폴더 생성
    args.ckpt_dir = os.path.join(run_root, 'Models')
    args.npy_dir = os.path.join(run_root, 'npy_data')
    args.code_dir = os.path.join(run_root, 'Code_Backup')
    args.log_dir = os.path.join(run_root, 'logs')
    
    for d in [args.ckpt_dir, args.npy_dir, args.code_dir, args.log_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 5. 소스코드 백업
    src_files = ['train.py', 'eval.py', 'dataset.py', 'models/model.py', 'util.py', 'loss.py', 'My_args.py', 'augmentations.py', 'post_process.py']
    for f in src_files:
        if os.path.exists(f):
            dest = os.path.join(args.code_dir, os.path.basename(f))
            shutil.copy(f, dest)
            
    print(f"\n>> [Init] Folder created: {run_root}")
    return args