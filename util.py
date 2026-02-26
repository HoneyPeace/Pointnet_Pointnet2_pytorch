import os 
import glob
import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm

def read_ply_and_asc(data_folder):
    """지정된 폴더에서 .ply와 .asc를 읽어옴"""
    ply_files = sorted(glob.glob(os.path.join(data_folder, "*.ply")))
    asc_files = sorted(glob.glob(os.path.join(data_folder, "*.asc")))
    
    if len(ply_files) == 0 or len(ply_files) != len(asc_files):
        print(f"   [Warning] Data mismatch or empty in {data_folder}. PLY: {len(ply_files)}, ASC: {len(asc_files)}")
        return [], [], []

    shape_list, lm_list, name_list = [], [], []

    for ply_f, asc_f in tqdm(zip(ply_files, asc_files), total=len(ply_files), desc=f"Loading {os.path.basename(data_folder)}"):
        # Name
        basename = os.path.splitext(os.path.basename(ply_f))[0]
        name_list.append(basename)
        
        # PLY
        plydata = PlyData.read(ply_f)
        vertex = plydata['vertex']
        points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        shape_list.append(points)
        
        # ASC
        try: lms = np.loadtxt(asc_f, delimiter=',')
        except ValueError: lms = np.loadtxt(asc_f)
        lm_list.append(lms)
        
    return shape_list, lm_list, name_list

def random_sample_points(shape_list, num_points):
    """각 Point Cloud에서 num_points만큼 무작위 샘플링 (FPS 속도 문제 대체)"""
    sampled_shapes = []
    for shape in shape_list:
        if shape.shape[0] > num_points:
            indices = np.random.choice(shape.shape[0], num_points, replace=False)
            sampled_shapes.append(shape[indices, :])
        else:
            # 점이 부족하면 중복 허용 샘플링
            indices = np.random.choice(shape.shape[0], num_points, replace=True)
            sampled_shapes.append(shape[indices, :])
    return sampled_shapes

def process_and_save_npy(data_root, npy_dir, partition, num_points):
    """Data 폴더에서 읽어와 NPY로 변환 후 Result/npy_data에 저장"""
    target_folder = os.path.join(data_root, partition) # ex: ../Data/train
    print(f"\n--- Processing Raw Data: {partition} ---")
    
    if not os.path.exists(target_folder):
        print(f"   [Error] Folder not found: {target_folder}")
        return False

    shape_list, lm_list, name_list = read_ply_and_asc(target_folder)
    if not shape_list: return False

    # 샘플링 (2048개)
    print("   Sampling points...")
    sampled_shapes = random_sample_points(shape_list, num_points)
    
    # NPY 저장 (Result 폴더 내부의 npy_dir에 저장)
    np.save(os.path.join(npy_dir, f'shape_{partition}.npy'), np.array(sampled_shapes))
    np.save(os.path.join(npy_dir, f'landmark_{partition}.npy'), np.array(lm_list))
    np.save(os.path.join(npy_dir, f'name_{partition}.npy'), np.array(name_list))
    
    print(f"   Saved to {npy_dir} (Suffix: _{partition})")
    return True