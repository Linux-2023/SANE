#!/usr/bin/env python3
"""
检查HDF5文件中夹爪数据的值范围

读取文件夹下所有HDF5文件，统计并打印夹爪数据的值范围（最小值和最大值）。

Example usage:
    uv run examples/flexiv/utils/data_process/inspect_gripper_range.py --input-dir /path/to/hdf5/files
"""

from pathlib import Path
import h5py
import numpy as np
import tyro


def inspect_gripper_range(
    input_dir: Path,
    left_gripper_idx: int = 7,
    right_gripper_idx: int = 15,
):
    """
    检查文件夹下所有HDF5文件中夹爪数据的值范围
    
    Args:
        input_dir: 输入文件夹路径
        left_gripper_idx: 左夹爪索引（默认7）
        right_gripper_idx: 右夹爪索引（默认15）
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # 获取所有HDF5文件
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files\n")
    
    # 统计所有夹爪值
    all_gripper_values = []
    processed_files = 0
    
    for hdf5_file in hdf5_files:
        try:
            with h5py.File(hdf5_file, "r") as f:
                # 检查可用的数据集
                datasets = []
                if "/observations/qpos" in f:
                    datasets.append(("/observations/qpos", f["/observations/qpos"][:]))
                if "/observations/pose" in f:
                    datasets.append(("/observations/pose", f["/observations/pose"][:]))
                if "/action" in f:
                    datasets.append(("/action", f["/action"][:]))
                if "/action_pose" in f:
                    datasets.append(("/action_pose", f["/action_pose"][:]))
                
                if not datasets:
                    continue
                
                # 提取夹爪值
                for dataset_name, data in datasets:
                    if data.ndim == 2 and data.shape[1] > max(left_gripper_idx, right_gripper_idx):
                        # 跳过14维数据（旋转向量格式，不包含夹爪）
                        if data.shape[1] == 14:
                            continue
                        all_gripper_values.extend(data[:, left_gripper_idx].tolist())
                        all_gripper_values.extend(data[:, right_gripper_idx].tolist())
                
                processed_files += 1
        except Exception as e:
            print(f"Warning: Failed to process {hdf5_file.name}: {e}")
    
    if not all_gripper_values:
        print("No gripper data found in any files")
        return
    
    all_gripper_values = np.array(all_gripper_values)
    
    # 打印统计信息
    print(f"Processed {processed_files}/{len(hdf5_files)} files")
    print(f"Total gripper values: {len(all_gripper_values)}")
    print(f"\nGripper value range:")
    print(f"  Min: {all_gripper_values.min():.6f}")
    print(f"  Max: {all_gripper_values.max():.6f}")
    print(f"  Mean: {all_gripper_values.mean():.6f}")
    print(f"  Std: {all_gripper_values.std():.6f}")


if __name__ == "__main__":
    tyro.cli(inspect_gripper_range)

