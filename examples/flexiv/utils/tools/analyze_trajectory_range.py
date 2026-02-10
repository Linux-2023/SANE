#!/usr/bin/env python3
"""
分析 HDF5 文件中记录的运动轨迹范围

用法:
    python analyze_trajectory_range.py <hdf5_file_path>
"""

import sys
from pathlib import Path
import h5py
import numpy as np


def analyze_trajectory_range(hdf5_file: Path):
    """分析 HDF5 文件中轨迹的位置和旋转范围"""
    print(f"正在分析文件: {hdf5_file}\n")
    
    with h5py.File(hdf5_file, "r") as f:
        # 检查有哪些数据可用
        print("文件中的键:")
        def print_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name} (shape={obj.shape}, dtype={obj.dtype})")
            else:
                print(f"  {name}/")
        f.visititems(print_keys)
        print()
        
        # 尝试读取不同的数据
        data_keys = [
            ("action", "/action"),
            ("observations/pose", "/observations/pose"),
            ("observations/qpos", "/observations/qpos"),
        ]
        
        for key_name, key_path in data_keys:
            if key_path in f:
                data = f[key_path][:]  # shape: (num_steps, dim)
                num_steps = data.shape[0]
                dim = data.shape[1]
                
                print(f"{'='*80}")
                print(f"分析: {key_name}")
                print(f"{'='*80}")
                print(f"数据形状: {data.shape} ({num_steps} 步, {dim} 维)\n")
                
                # 双臂数据：每臂 7 维（位置3 + 旋转4/3 + 夹爪1）
                if dim == 16:
                    print("检测到双臂 16 维数据格式:")
                    print("  [左臂位置(3) + 左臂旋转(4) + 左臂夹爪(1) + 右臂位置(3) + 右臂旋转(4) + 右臂夹爪(1)]\n")
                    
                    # 左臂
                    left_pos = data[:, 0:3]
                    left_quat = data[:, 3:7]
                    left_gripper = data[:, 7]
                    
                    print("左臂位置范围 (x, y, z):")
                    print(f"  X: [{left_pos[:, 0].min():.4f}, {left_pos[:, 0].max():.4f}] | "
                          f"范围: {left_pos[:, 0].max() - left_pos[:, 0].min():.4f}")
                    print(f"  Y: [{left_pos[:, 1].min():.4f}, {left_pos[:, 1].max():.4f}] | "
                          f"范围: {left_pos[:, 1].max() - left_pos[:, 1].min():.4f}")
                    print(f"  Z: [{left_pos[:, 2].min():.4f}, {left_pos[:, 2].max():.4f}] | "
                          f"范围: {left_pos[:, 2].max() - left_pos[:, 2].min():.4f}")
                    print(f"  中心: ({left_pos[:, 0].mean():.4f}, {left_pos[:, 1].mean():.4f}, {left_pos[:, 2].mean():.4f})")
                    
                    print("\n左臂四元数范围 (qw, qx, qy, qz):")
                    print(f"  Qw: [{left_quat[:, 0].min():.4f}, {left_quat[:, 0].max():.4f}]")
                    print(f"  Qx: [{left_quat[:, 1].min():.4f}, {left_quat[:, 1].max():.4f}]")
                    print(f"  Qy: [{left_quat[:, 2].min():.4f}, {left_quat[:, 2].max():.4f}]")
                    print(f"  Qz: [{left_quat[:, 3].min():.4f}, {left_quat[:, 3].max():.4f}]")
                    
                    print(f"\n左臂夹爪范围: [{left_gripper.min():.4f}, {left_gripper.max():.4f}]")
                    
                    print("\n" + "-"*80 + "\n")
                    
                    # 右臂
                    right_pos = data[:, 8:11]
                    right_quat = data[:, 11:15]
                    right_gripper = data[:, 15]
                    
                    print("右臂位置范围 (x, y, z):")
                    print(f"  X: [{right_pos[:, 0].min():.4f}, {right_pos[:, 0].max():.4f}] | "
                          f"范围: {right_pos[:, 0].max() - right_pos[:, 0].min():.4f}")
                    print(f"  Y: [{right_pos[:, 1].min():.4f}, {right_pos[:, 1].max():.4f}] | "
                          f"范围: {right_pos[:, 1].max() - right_pos[:, 1].min():.4f}")
                    print(f"  Z: [{right_pos[:, 2].min():.4f}, {right_pos[:, 2].max():.4f}] | "
                          f"范围: {right_pos[:, 2].max() - right_pos[:, 2].min():.4f}")
                    print(f"  中心: ({right_pos[:, 0].mean():.4f}, {right_pos[:, 1].mean():.4f}, {right_pos[:, 2].mean():.4f})")
                    
                    print("\n右臂四元数范围 (qw, qx, qy, qz):")
                    print(f"  Qw: [{right_quat[:, 0].min():.4f}, {right_quat[:, 0].max():.4f}]")
                    print(f"  Qx: [{right_quat[:, 1].min():.4f}, {right_quat[:, 1].max():.4f}]")
                    print(f"  Qy: [{right_quat[:, 2].min():.4f}, {right_quat[:, 2].max():.4f}]")
                    print(f"  Qz: [{right_quat[:, 3].min():.4f}, {right_quat[:, 3].max():.4f}]")
                    
                    print(f"\n右臂夹爪范围: [{right_gripper.min():.4f}, {right_gripper.max():.4f}]")
                    
                    # 计算位置变化范围（相邻帧之间的差值）
                    print("\n" + "-"*80)
                    print("相邻帧之间的位置变化范围:")
                    left_pos_diff = np.diff(left_pos, axis=0)
                    right_pos_diff = np.diff(right_pos, axis=0)
                    
                    print("左臂位置变化:")
                    print(f"  ΔX: 最大={np.abs(left_pos_diff[:, 0]).max():.4f}, 平均={np.abs(left_pos_diff[:, 0]).mean():.4f}")
                    print(f"  ΔY: 最大={np.abs(left_pos_diff[:, 1]).max():.4f}, 平均={np.abs(left_pos_diff[:, 1]).mean():.4f}")
                    print(f"  ΔZ: 最大={np.abs(left_pos_diff[:, 2]).max():.4f}, 平均={np.abs(left_pos_diff[:, 2]).mean():.4f}")
                    
                    print("右臂位置变化:")
                    print(f"  ΔX: 最大={np.abs(right_pos_diff[:, 0]).max():.4f}, 平均={np.abs(right_pos_diff[:, 0]).mean():.4f}")
                    print(f"  ΔY: 最大={np.abs(right_pos_diff[:, 1]).max():.4f}, 平均={np.abs(right_pos_diff[:, 1]).mean():.4f}")
                    print(f"  ΔZ: 最大={np.abs(right_pos_diff[:, 2]).max():.4f}, 平均={np.abs(right_pos_diff[:, 2]).mean():.4f}")
                    
                    print()
                elif dim == 14:
                    print("检测到双臂 14 维数据格式（旋转向量）:")
                    print("  [左臂位置(3) + 左臂旋转向量(3) + 左臂夹爪(1) + 右臂位置(3) + 右臂旋转向量(3) + 右臂夹爪(1)]\n")
                    
                    # 左臂
                    left_pos = data[:, 0:3]
                    left_rotvec = data[:, 3:6]
                    left_gripper = data[:, 6]
                    
                    print("左臂位置范围 (x, y, z):")
                    print(f"  X: [{left_pos[:, 0].min():.4f}, {left_pos[:, 0].max():.4f}] | "
                          f"范围: {left_pos[:, 0].max() - left_pos[:, 0].min():.4f}")
                    print(f"  Y: [{left_pos[:, 1].min():.4f}, {left_pos[:, 1].max():.4f}] | "
                          f"范围: {left_pos[:, 1].max() - left_pos[:, 1].min():.4f}")
                    print(f"  Z: [{left_pos[:, 2].min():.4f}, {left_pos[:, 2].max():.4f}] | "
                          f"范围: {left_pos[:, 2].max() - left_pos[:, 2].min():.4f}")
                    print(f"  中心: ({left_pos[:, 0].mean():.4f}, {left_pos[:, 1].mean():.4f}, {left_pos[:, 2].mean():.4f})")
                    
                    print("\n左臂旋转向量范围 (rx, ry, rz):")
                    print(f"  Rx: [{left_rotvec[:, 0].min():.4f}, {left_rotvec[:, 0].max():.4f}]")
                    print(f"  Ry: [{left_rotvec[:, 1].min():.4f}, {left_rotvec[:, 1].max():.4f}]")
                    print(f"  Rz: [{left_rotvec[:, 2].min():.4f}, {left_rotvec[:, 2].max():.4f}]")
                    
                    print(f"\n左臂夹爪范围: [{left_gripper.min():.4f}, {left_gripper.max():.4f}]")
                    
                    print("\n" + "-"*80 + "\n")
                    
                    # 右臂
                    right_pos = data[:, 7:10]
                    right_rotvec = data[:, 10:13]
                    right_gripper = data[:, 13]
                    
                    print("右臂位置范围 (x, y, z):")
                    print(f"  X: [{right_pos[:, 0].min():.4f}, {right_pos[:, 0].max():.4f}] | "
                          f"范围: {right_pos[:, 0].max() - right_pos[:, 0].min():.4f}")
                    print(f"  Y: [{right_pos[:, 1].min():.4f}, {right_pos[:, 1].max():.4f}] | "
                          f"范围: {right_pos[:, 1].max() - right_pos[:, 1].min():.4f}")
                    print(f"  Z: [{right_pos[:, 2].min():.4f}, {right_pos[:, 2].max():.4f}] | "
                          f"范围: {right_pos[:, 2].max() - right_pos[:, 2].min():.4f}")
                    print(f"  中心: ({right_pos[:, 0].mean():.4f}, {right_pos[:, 1].mean():.4f}, {right_pos[:, 2].mean():.4f})")
                    
                    print("\n右臂旋转向量范围 (rx, ry, rz):")
                    print(f"  Rx: [{right_rotvec[:, 0].min():.4f}, {right_rotvec[:, 0].max():.4f}]")
                    print(f"  Ry: [{right_rotvec[:, 1].min():.4f}, {right_rotvec[:, 1].max():.4f}]")
                    print(f"  Rz: [{right_rotvec[:, 2].min():.4f}, {right_rotvec[:, 2].max():.4f}]")
                    
                    print(f"\n右臂夹爪范围: [{right_gripper.min():.4f}, {right_gripper.max():.4f}]")
                    
                    print()
                else:
                    print(f"未知的数据维度: {dim}\n")
                    # 显示所有维度的范围
                    print("各维度范围:")
                    for i in range(dim):
                        print(f"  维度 {i}: [{data[:, i].min():.4f}, {data[:, i].max():.4f}] | "
                              f"范围: {data[:, i].max() - data[:, i].min():.4f}")
                    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python analyze_trajectory_range.py <hdf5_file_path>")
        sys.exit(1)
    
    hdf5_file = Path(sys.argv[1])
    if not hdf5_file.exists():
        print(f"错误: 文件不存在: {hdf5_file}")
        sys.exit(1)
    
    analyze_trajectory_range(hdf5_file)

