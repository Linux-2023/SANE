#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 HDF5 文件中的 pose 从四元数格式转换为旋转向量格式

pose 格式转换：
- 输入: [x, y, z, qw, qx, qy, qz] (7维)
- 输出: [x, y, z, rx, ry, rz] (6维)

处理的字段：
- /observations/pose
- /action_pose
- /observations/prev_pose (如果存在)

用法:
    python convert_pose_quat_to_rotvec.py --input-dir /path/to/hdf5/files [--backup]
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# def quat_to_rotvec(pose_quat: np.ndarray) -> np.ndarray:
#     """
#     将位姿从四元数格式转换为旋转向量格式
    
#     Args:
#         pose_quat: [x, y, z, qw, qx, qy, qz] (N, 7) 或 (7,)
    
#     Returns:
#         pose_rotvec: [x, y, z, rx, ry, rz] (N, 6) 或 (6,)
#     """
#     pose_quat = np.asarray(pose_quat)
#     is_1d = pose_quat.ndim == 1
    
#     if is_1d:
#         pose_quat = pose_quat.reshape(1, -1)
    
#     num_samples = pose_quat.shape[0]
#     pose_rotvec = np.zeros((num_samples, 6), dtype=np.float32)
    
#     # 位置部分 (前3维)
#     pose_rotvec[:, :3] = pose_quat[:, :3]
    
#     # 四元数部分 (后4维: qw, qx, qy, qz) -> 旋转向量
#     quat = pose_quat[:, 3:7]  # [qw, qx, qy, qz]
#     # scipy 使用 [qx, qy, qz, qw] 格式
#     quat_scipy = np.column_stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]])
    
#     # 转换为旋转向量
#     rotations = R.from_quat(quat_scipy)
#     rotvec = rotations.as_rotvec()
#     pose_rotvec[:, 3:6] = rotvec
    
#     if is_1d:
#         return pose_rotvec[0]
#     return pose_rotvec


def dual_arm_quat_to_rotvec(dual_arm_quat: np.ndarray) -> np.ndarray:
    """
    将双臂位姿从四元数格式转换为旋转向量格式
    
    输入格式（16维）：
    - 左臂位姿（3维: x, y, z）
    - 左臂四元数（4维: qw, qx, qy, qz）
    - 左臂夹爪（1维）
    - 右臂位姿（3维: x, y, z）
    - 右臂四元数（4维: qw, qx, qy, qz）
    - 右臂夹爪（1维）
    
    输出格式（14维）：
    - 左臂位姿（3维: x, y, z）
    - 左臂旋转向量（3维: rx, ry, rz）
    - 左臂夹爪（1维）
    - 右臂位姿（3维: x, y, z）
    - 右臂旋转向量（3维: rx, ry, rz）
    - 右臂夹爪（1维）
    
    Args:
        dual_arm_quat: 双臂位姿向量 (N, 16) 或 (16,)
    
    Returns:
        dual_arm_rotvec: 双臂位姿向量 (N, 14) 或 (14,)
    """
    dual_arm_quat = np.asarray(dual_arm_quat, dtype=np.float32)
    is_1d = dual_arm_quat.ndim == 1
    
    if is_1d:
        dual_arm_quat = dual_arm_quat.reshape(1, -1)
    
    if dual_arm_quat.shape[1] != 16:
        raise ValueError(f"输入向量维度应为16，实际为 {dual_arm_quat.shape[1]}")
    
    num_samples = dual_arm_quat.shape[0]
    dual_arm_rotvec = np.zeros((num_samples, 14), dtype=np.float32)
    
    # 左臂部分 (索引 0-7)
    # 左臂位姿 (0:3)
    dual_arm_rotvec[:, 0:3] = dual_arm_quat[:, 0:3]
    # 左臂四元数 (3:7) -> 旋转向量
    left_quat = dual_arm_quat[:, 3:7]  # [qw, qx, qy, qz]
    left_quat_scipy = np.column_stack([left_quat[:, 1], left_quat[:, 2], left_quat[:, 3], left_quat[:, 0]])
    left_rotations = R.from_quat(left_quat_scipy)
    left_rotvec = left_rotations.as_rotvec()
    dual_arm_rotvec[:, 3:6] = left_rotvec
    # 左臂夹爪 (7)
    dual_arm_rotvec[:, 6] = dual_arm_quat[:, 7]
    
    # 右臂部分 (索引 8-15)
    # 右臂位姿 (8:11)
    dual_arm_rotvec[:, 7:10] = dual_arm_quat[:, 8:11]
    # 右臂四元数 (11:15) -> 旋转向量
    right_quat = dual_arm_quat[:, 11:15]  # [qw, qx, qy, qz]
    right_quat_scipy = np.column_stack([right_quat[:, 1], right_quat[:, 2], right_quat[:, 3], right_quat[:, 0]])
    right_rotations = R.from_quat(right_quat_scipy)
    right_rotvec = right_rotations.as_rotvec()
    dual_arm_rotvec[:, 10:13] = right_rotvec
    # 右臂夹爪 (15)
    dual_arm_rotvec[:, 13] = dual_arm_quat[:, 15]
    
    if is_1d:
        return dual_arm_rotvec[0]
    return dual_arm_rotvec


def dual_arm_rotvec_to_quat(dual_arm_rotvec: np.ndarray) -> np.ndarray:
    """
    将双臂位姿从旋转向量格式转换为四元数格式（dual_arm_quat_to_rotvec 的反向过程）
    
    输入格式（14维）：
    - 左臂位姿（3维: x, y, z）
    - 左臂旋转向量（3维: rx, ry, rz）
    - 左臂夹爪（1维）
    - 右臂位姿（3维: x, y, z）
    - 右臂旋转向量（3维: rx, ry, rz）
    - 右臂夹爪（1维）
    
    输出格式（16维）：
    - 左臂位姿（3维: x, y, z）
    - 左臂四元数（4维: qw, qx, qy, qz）
    - 左臂夹爪（1维）
    - 右臂位姿（3维: x, y, z）
    - 右臂四元数（4维: qw, qx, qy, qz）
    - 右臂夹爪（1维）
    
    Args:
        dual_arm_rotvec: 双臂位姿向量 (N, 14) 或 (14,)
    
    Returns:
        dual_arm_quat: 双臂位姿向量 (N, 16) 或 (16,)
    """
    dual_arm_rotvec = np.asarray(dual_arm_rotvec, dtype=np.float32)
    is_1d = dual_arm_rotvec.ndim == 1
    
    if is_1d:
        dual_arm_rotvec = dual_arm_rotvec.reshape(1, -1)
    
    if dual_arm_rotvec.shape[1] != 14:
        raise ValueError(f"输入向量维度应为14，实际为 {dual_arm_rotvec.shape[1]}")
    
    num_samples = dual_arm_rotvec.shape[0]
    dual_arm_quat = np.zeros((num_samples, 16), dtype=np.float32)
    
    # 左臂部分 (索引 0-6)
    # 左臂位姿 (0:3)
    dual_arm_quat[:, 0:3] = dual_arm_rotvec[:, 0:3]
    # 左臂旋转向量 (3:6) -> 四元数
    left_rotvec = dual_arm_rotvec[:, 3:6]
    left_rotations = R.from_rotvec(left_rotvec)
    left_quat_scipy = left_rotations.as_quat()  # [qx, qy, qz, qw]
    # 转换为 [qw, qx, qy, qz] 格式
    dual_arm_quat[:, 3] = left_quat_scipy[:, 3]  # qw
    dual_arm_quat[:, 4] = left_quat_scipy[:, 0]  # qx
    dual_arm_quat[:, 5] = left_quat_scipy[:, 1]  # qy
    dual_arm_quat[:, 6] = left_quat_scipy[:, 2]  # qz
    # 左臂夹爪 (6 -> 7)
    dual_arm_quat[:, 7] = dual_arm_rotvec[:, 6]
    
    # 右臂部分 (索引 7-13)
    # 右臂位姿 (7:10)
    dual_arm_quat[:, 8:11] = dual_arm_rotvec[:, 7:10]
    # 右臂旋转向量 (10:13) -> 四元数
    right_rotvec = dual_arm_rotvec[:, 10:13]
    right_rotations = R.from_rotvec(right_rotvec)
    right_quat_scipy = right_rotations.as_quat()  # [qx, qy, qz, qw]
    # 转换为 [qw, qx, qy, qz] 格式
    dual_arm_quat[:, 11] = right_quat_scipy[:, 3]  # qw
    dual_arm_quat[:, 12] = right_quat_scipy[:, 0]  # qx
    dual_arm_quat[:, 13] = right_quat_scipy[:, 1]  # qy
    dual_arm_quat[:, 14] = right_quat_scipy[:, 2]  # qz
    # 右臂夹爪 (13 -> 15)
    dual_arm_quat[:, 15] = dual_arm_rotvec[:, 13]
    
    if is_1d:
        return dual_arm_quat[0]
    return dual_arm_quat


def fix_quaternion_discontinuity(last: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    修复两个16维向量之间四元数的数值跳变
    
    四元数在表示同一个旋转时有两种等价表示：q 和 -q。这可能导致相邻帧之间的
    四元数数值发生跳变（例如从 [1, 0, 0, 0] 跳到 [-1, 0, 0, 0]）。
    此函数检测并修复这种跳变，确保 current 中的四元数与 last 保持一致的方向。
    
    输入格式（16维）：
    - 左臂位姿（3维: x, y, z）
    - 左臂四元数（4维: qw, qx, qy, qz）- 索引 3-6
    - 左臂夹爪（1维）- 索引 7
    - 右臂位姿（3维: x, y, z）- 索引 8-10
    - 右臂四元数（4维: qw, qx, qy, qz）- 索引 11-14
    - 右臂夹爪（1维）- 索引 15
    
    Args:
        last: 上一帧的双臂位姿向量 (16,)
        current: 当前帧的双臂位姿向量 (16,)，可能会被修改
    
    Returns:
        fixed_current: 修复后的当前帧向量 (16,)
    """
    last = np.asarray(last, dtype=np.float32).reshape(16)
    current = np.asarray(current, dtype=np.float32).reshape(16).copy()
    
    # 提取左臂四元数（索引 3-6: qw, qx, qy, qz）
    last_left_quat = last[3:7]
    current_left_quat = current[3:7]
    
    # 计算四元数点积，如果为负说明需要取反
    left_quat_dot = np.dot(last_left_quat, current_left_quat)
    if left_quat_dot < 0:
        # 取反以保持一致的方向
        current[3:7] = -current_left_quat
    
    # 提取右臂四元数（索引 11-14: qw, qx, qy, qz）
    last_right_quat = last[11:15]
    current_right_quat = current[11:15]
    
    # 计算四元数点积，如果为负说明需要取反
    right_quat_dot = np.dot(last_right_quat, current_right_quat)
    if right_quat_dot < 0:
        # 取反以保持一致的方向
        current[11:15] = -current_right_quat
    
    return current


def convert_hdf5_file(hdf5_path: Path, backup: bool = False):
    """
    转换单个 HDF5 文件中的 pose 数据
    
    强制使用 dual_arm_quat_to_rotvec 函数进行转换，输入应为16维双臂位姿向量。
    
    Args:
        hdf5_path: HDF5 文件路径
        backup: 是否创建备份文件
    """
    if backup:
        backup_path = hdf5_path.with_suffix('.hdf5.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy2(hdf5_path, backup_path)
            print(f"  备份文件: {backup_path}")
    
    with h5py.File(hdf5_path, 'r+') as f:
        # 处理 /observations/pose (强制使用 dual_arm_quat_to_rotvec)
        if '/observations/pose' in f:
            pose_quat = f['/observations/pose'][:]
            # 检查维度，如果不是16维则报错
            if pose_quat.shape[-1] != 16:
                raise ValueError(
                    f"/observations/pose 维度应为16（双臂位姿），实际为 {pose_quat.shape[-1]}。"
                    f"请使用 dual_arm_quat_to_rotvec 函数进行转换。"
                )
            pose_rotvec = dual_arm_quat_to_rotvec(pose_quat)
            del f['/observations/pose']
            f.create_dataset('/observations/pose', data=pose_rotvec, dtype=np.float32)
            print(f"  ✅ 转换 /observations/pose: {pose_quat.shape} -> {pose_rotvec.shape} (使用 dual_arm_quat_to_rotvec)")
        
        # 处理 /action_pose (强制使用 dual_arm_quat_to_rotvec)
        if '/action_pose' in f:
            action_pose_quat = f['/action_pose'][:]
            # 检查维度，如果不是16维则报错
            if action_pose_quat.shape[-1] != 16:
                raise ValueError(
                    f"/action_pose 维度应为16（双臂位姿），实际为 {action_pose_quat.shape[-1]}。"
                    f"请使用 dual_arm_quat_to_rotvec 函数进行转换。"
                )
            action_pose_rotvec = dual_arm_quat_to_rotvec(action_pose_quat)
            del f['/action_pose']
            f.create_dataset('/action_pose', data=action_pose_rotvec, dtype=np.float32)
            print(f"  ✅ 转换 /action_pose: {action_pose_quat.shape} -> {action_pose_rotvec.shape} (使用 dual_arm_quat_to_rotvec)")
        
        # 处理 /observations/prev_pose (如果存在，强制使用 dual_arm_quat_to_rotvec)
        if '/observations/prev_pose' in f:
            prev_pose_quat = f['/observations/prev_pose'][:]
            # 检查维度，如果不是16维则报错
            if prev_pose_quat.shape[-1] != 16:
                raise ValueError(
                    f"/observations/prev_pose 维度应为16（双臂位姿），实际为 {prev_pose_quat.shape[-1]}。"
                    f"请使用 dual_arm_quat_to_rotvec 函数进行转换。"
                )
            prev_pose_rotvec = dual_arm_quat_to_rotvec(prev_pose_quat)
            del f['/observations/prev_pose']
            f.create_dataset('/observations/prev_pose', data=prev_pose_rotvec, dtype=np.float32)
            print(f"  ✅ 转换 /observations/prev_pose: {prev_pose_quat.shape} -> {prev_pose_rotvec.shape} (使用 dual_arm_quat_to_rotvec)")


def main():
    parser = argparse.ArgumentParser(description="将 HDF5 文件中的 pose 从四元数转换为旋转向量")
    parser.add_argument('--input-dir', type=str, required=True,
                       help='包含 HDF5 文件的目录路径')
    parser.add_argument('--backup', action='store_true',
                       help='转换前创建备份文件')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"目录不存在: {input_dir}")
    
    # 查找所有 HDF5 文件
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"在 {input_dir} 中未找到 HDF5 文件")
        return
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    print("=" * 60)
    
    for hdf5_path in tqdm(hdf5_files, desc="转换文件"):
        print(f"\n处理: {hdf5_path.name}")
        try:
            convert_hdf5_file(hdf5_path, backup=args.backup)
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("✅ 转换完成")


if __name__ == "__main__":
    main()

