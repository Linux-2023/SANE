#!/usr/bin/env python3
"""
HDF5文件夹爪数据归一化工具

该脚本用于读取文件夹下所有HDF5文件，将其中的夹爪数据按照设定的上下限归一化。

夹爪数据位置：
- qpos 数据：左夹爪索引 7，右夹爪索引 15
- pose 数据（16维）：左夹爪索引 7，右夹爪索引 15
  - pose 格式：左臂位姿(7) + 左夹爪(1) + 右臂位姿(7) + 右夹爪(1) = 16维
  - 注意：14维 pose 数据（旋转向量格式）不包含夹爪，会跳过处理

支持的数据集：
- /observations/qpos - 关节位置
- /action - 动作数据
- /observations/pose - 位姿数据（16维）
- /action_pose - 位姿动作数据（16维）
- /observations/prev_pose - 上一帧位姿数据（16维，如果存在）

归一化公式：normalized = (value - min) / (max - min) * (target_max - target_min) + target_min

Example usage:
    # 归一化到 [0, 1] 范围，夹爪原始范围 [0, 1]
    uv run examples/flexiv/utils/data_process/normalize_gripper_in_hdf5.py \
        --input-dir /path/to/hdf5/files \
        --gripper-min 0.0 --gripper-max 1.0 \
        --target-min 0.0 --target-max 1.0
    
    # 归一化到 [-1, 1] 范围，夹爪原始范围 [0, 1]
    uv run examples/flexiv/utils/data_process/normalize_gripper_in_hdf5.py \
        --input-dir /path/to/hdf5/files \
        --gripper-min 0.0 --gripper-max 1.0 \
        --target-min -1.0 --target-max 1.0
    
    # 指定输出目录（不覆盖原文件）
    uv run examples/flexiv/utils/data_process/normalize_gripper_in_hdf5.py \
        --input-dir /path/to/hdf5/files \
        --output-dir /path/to/output \
        --gripper-min 0.0 --gripper-max 1.0
"""

import argparse
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import tqdm
import tyro
import shutil


def normalize_gripper_data(
    value: float,
    gripper_min: float,
    gripper_max: float,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> float:
    """
    归一化夹爪数据
    
    Args:
        value: 原始夹爪值
        gripper_min: 夹爪原始最小值
        gripper_max: 夹爪原始最大值
        target_min: 目标范围最小值（默认0.0）
        target_max: 目标范围最大值（默认1.0）
    
    Returns:
        归一化后的值
    """
    if gripper_max == gripper_min:
        return target_min
    
    # 归一化到 [0, 1]
    normalized = (value - gripper_min) / (gripper_max - gripper_min)
    
    # 缩放到目标范围
    return normalized * (target_max - target_min) + target_min


def normalize_grippers_in_dataset(
    data: np.ndarray,
    gripper_min: float,
    gripper_max: float,
    target_min: float,
    target_max: float,
    left_gripper_idx: int,
    right_gripper_idx: int,
) -> np.ndarray | None:
    """
    归一化数据集中的夹爪数据（通用函数）
    
    Args:
        data: 数据集数组，shape (num_frames, num_dims)
        gripper_min: 夹爪原始最小值
        gripper_max: 夹爪原始最大值
        target_min: 目标范围最小值
        target_max: 目标范围最大值
        left_gripper_idx: 左夹爪索引
        right_gripper_idx: 右夹爪索引
    
    Returns:
        归一化后的数据，如果无法处理则返回None
    """
    if data.ndim != 2:
        return None
    
    num_frames, num_dims = data.shape
    
    # 检查维度是否足够
    if num_dims <= max(left_gripper_idx, right_gripper_idx):
        return None
    
    # 14维格式（旋转向量）不包含夹爪，跳过
    if num_dims == 14:
        return None
    
    # 归一化夹爪数据
    normalized = data.copy()
    normalized[:, left_gripper_idx] = np.array([
        normalize_gripper_data(val, gripper_min, gripper_max, target_min, target_max)
        for val in data[:, left_gripper_idx]
    ])
    normalized[:, right_gripper_idx] = np.array([
        normalize_gripper_data(val, gripper_min, gripper_max, target_min, target_max)
        for val in data[:, right_gripper_idx]
    ])
    
    return normalized


def process_hdf5_file(
    input_path: Path,
    output_path: Path | None,
    gripper_min: float,
    gripper_max: float,
    target_min: float,
    target_max: float,
    left_gripper_idx: int = 7,
    right_gripper_idx: int = 15,
    overwrite: bool = False,
) -> bool:
    """
    处理单个HDF5文件，归一化夹爪数据
    
    Args:
        input_path: 输入HDF5文件路径
        output_path: 输出HDF5文件路径，如果为None则覆盖原文件
        gripper_min: 夹爪原始最小值
        gripper_max: 夹爪原始最大值
        target_min: 目标范围最小值
        target_max: 目标范围最大值
        left_gripper_idx: 左夹爪在qpos中的索引（默认7）
        right_gripper_idx: 右夹爪在qpos中的索引（默认15）
        overwrite: 是否覆盖原文件
    
    Returns:
        是否成功处理
    """
    try:
        # 确定输出路径
        if output_path is None:
            if overwrite:
                output_path = input_path
            else:
                # 创建备份文件
                backup_path = input_path.with_suffix('.hdf5.backup')
                shutil.copy2(input_path, backup_path)
                output_path = input_path
        print(f"Output path: {output_path}")
        
        # 使用临时文件处理，避免覆盖原文件时出现问题
        use_temp = (output_path == input_path)
        if use_temp:
            temp_path = input_path.with_suffix('.hdf5.tmp')
            final_output = temp_path
        else:
            final_output = output_path
        
        # 读取原文件并处理所有数据
        with h5py.File(input_path, "r") as f_in:
            # 检查是否至少有一个可处理的数据集
            has_qpos = "/observations/qpos" in f_in
            has_pose = "/observations/pose" in f_in
            
            if not has_qpos and not has_pose:
                print(f"Warning: Neither /observations/qpos nor /observations/pose found in {input_path}")
                return False
            
            # 读取并归一化所有需要处理的数据
            normalized_data = {}
            
            # 需要处理的数据集列表
            datasets_to_process = []
            if has_qpos:
                datasets_to_process.append("/observations/qpos")
            if "/action" in f_in:
                datasets_to_process.append("/action")
            if has_pose:
                datasets_to_process.append("/observations/pose")
            if "/action_pose" in f_in:
                datasets_to_process.append("/action_pose")
            if "/observations/prev_pose" in f_in:
                datasets_to_process.append("/observations/prev_pose")
            
            # 统一处理所有数据集
            for dataset_name in datasets_to_process:
                if dataset_name in f_in:
                    data = f_in[dataset_name][:]
                    normalized = normalize_grippers_in_dataset(
                        data, gripper_min, gripper_max, target_min, target_max,
                        left_gripper_idx, right_gripper_idx
                    )
                    if normalized is not None:
                        normalized_data[dataset_name] = normalized
                    elif data.shape[1] == 14:
                        # 14维格式不包含夹爪，静默跳过
                        pass
                    else:
                        print(f"Warning: {dataset_name} cannot be normalized (dim={data.shape[1]})")
            
            # 写入所有数据到输出文件
            with h5py.File(final_output, "w") as f_out:
                # 复制所有数据，使用归一化后的数据替换需要更新的数据集
                def copy_dataset(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if name in normalized_data:
                            # 写入归一化后的数据
                            f_out.create_dataset(name, data=normalized_data[name], dtype=obj.dtype)
                        else:
                            # 复制其他数据集
                            f_out.create_dataset(name, data=obj[:], dtype=obj.dtype)
                    elif isinstance(obj, h5py.Group):
                        if name not in f_out:
                            f_out.create_group(name)
                
                f_in.visititems(copy_dataset)
        
        # 如果使用临时文件，替换原文件
        if use_temp:
            final_output.replace(input_path)
        
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def normalize_gripper_in_hdf5(
    input_dir: Path,
    output_dir: Path | None = None,
    gripper_min: float = 0.0,
    gripper_max: float = 1.0,
    target_min: float = 0.0,
    target_max: float = 1.0,
    left_gripper_idx: int = 7,
    right_gripper_idx: int = 15,
    overwrite: bool = False,
):
    """
    归一化文件夹下所有HDF5文件中的夹爪数据
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径，如果为None则覆盖原文件
        gripper_min: 夹爪原始最小值
        gripper_max: 夹爪原始最大值
        target_min: 目标范围最小值（默认0.0）
        target_max: 目标范围最大值（默认1.0）
        left_gripper_idx: 左夹爪在qpos中的索引（默认7）
        right_gripper_idx: 右夹爪在qpos中的索引（默认15）
        overwrite: 是否覆盖原文件（仅在output_dir为None时有效）
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # 获取所有HDF5文件
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    print(f"Gripper normalization: [{gripper_min}, {gripper_max}] -> [{target_min}, {target_max}]")
    print(f"Left gripper index: {left_gripper_idx}, Right gripper index: {right_gripper_idx}")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        if overwrite:
            print("Warning: Will overwrite original files!")
        else:
            print("Will create backup files (.hdf5.backup)")
    
    # 处理每个文件
    success_count = 0
    for hdf5_file in tqdm.tqdm(hdf5_files, desc="Processing files"):
        if output_dir is not None:
            output_path = output_dir / hdf5_file.name
        else:
            output_path = None
        
        if process_hdf5_file(
            hdf5_file,
            output_path,
            gripper_min,
            gripper_max,
            target_min,
            target_max,
            left_gripper_idx,
            right_gripper_idx,
            overwrite,
        ):
            success_count += 1
    
    print(f"\nSuccessfully processed {success_count}/{len(hdf5_files)} files")


if __name__ == "__main__":
    tyro.cli(normalize_gripper_in_hdf5)

