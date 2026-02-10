"""
将 hdf5 文件中的关节角转换为位姿

读取文件夹下所有 hdf5 文件，将 qpos 和 action 中的关节角通过前向运动学转换为位姿，
并保存到新的 hdf5 文件中。

转换内容:
- observations/qpos: 将关节角转换为位姿
- observations/prev_qpos: 将关节角转换为位姿（如果存在），否则创建并使用上一帧的位姿
- action: 将关节角转换为位姿（如果存在）

用法:
    python modify_hdf5_joints_to_poses.py --input-dir /path/to/hdf5/files
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import h5py
import numpy as np
import tqdm
import tyro

from forward_kinematics import ForwardKinematics


def convert_joints_array_to_poses(
    joints_array: np.ndarray,
    fk_left: ForwardKinematics,
    fk_right: ForwardKinematics,
) -> np.ndarray:
    """
    将关节角数组转换为位姿数组
    
    Args:
        joints_array: 关节角数组，shape (num_steps, 16)
                     格式: [左臂关节角(7) + 左臂夹爪(1) + 右臂关节角(7) + 右臂夹爪(1)]
        fk_left: 左臂前向运动学解算器
        fk_right: 右臂前向运动学解算器
    
    Returns:
        位姿数组，shape (num_steps, 16)
        格式: [左臂位姿(7) + 左臂夹爪(1) + 右臂位姿(7) + 右臂夹爪(1)]
    """
    num_steps = joints_array.shape[0]
    
    # 提取左右臂关节角和夹爪值
    left_joints = joints_array[:, 0:7]  # (num_steps, 7)
    left_gripper = joints_array[:, 7:8]  # (num_steps, 1)
    right_joints = joints_array[:, 8:15]  # (num_steps, 7)
    right_gripper = joints_array[:, 15:16]  # (num_steps, 1)
    
    # 通过前向运动学计算位姿
    left_poses = np.zeros((num_steps, 7), dtype=np.float32)
    right_poses = np.zeros((num_steps, 7), dtype=np.float32)
    
    for i in range(num_steps):
        left_poses[i] = fk_left.compute(left_joints[i])
        right_poses[i] = fk_right.compute(right_joints[i])
    
    # 组合位姿: [left_pose(7), left_gripper(1), right_pose(7), right_gripper(1)]
    return np.concatenate([
        left_poses, left_gripper, right_poses, right_gripper
    ], axis=1)


def validate_dataset(f_in: h5py.File, dataset_name: str, num_steps: int, min_dims: int = 16) -> Tuple[bool, np.ndarray]:
    """
    验证并读取数据集
    
    Args:
        f_in: HDF5 文件对象
        dataset_name: 数据集名称
        num_steps: 期望的步数
        min_dims: 最小维度要求
    
    Returns:
        (is_valid, data): 是否有效和数据数组
    """
    if dataset_name not in f_in:
        return False, None
    
    data = f_in[dataset_name][:]
    steps, dims = data.shape
    
    if steps != num_steps:
        return False, None
    if dims < min_dims:
        return False, None
    
    return True, data


def copy_dataset_attrs(f_in: h5py.File, f_out: h5py.File, dataset_path: str):
    """复制数据集的属性（不复制数据）"""
    if dataset_path in f_in and dataset_path in f_out:
        for attr_name, attr_value in f_in[dataset_path].attrs.items():
            f_out[dataset_path].attrs[attr_name] = attr_value


def convert_joints_to_poses(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
):
    """
    将 hdf5 文件中的关节角转换为位姿
    
    转换 qpos、prev_qpos 和 action 中的关节角为位姿：
    - qpos: [左臂关节角(7) + 左臂夹爪(1) + 右臂关节角(7) + 右臂夹爪(1)] 
           -> [左臂位姿(7) + 左臂夹爪(1) + 右臂位姿(7) + 右臂夹爪(1)]
    - prev_qpos: 如果存在则转换，否则创建并使用上一帧的位姿
    - action: 同样的转换（如果存在）
    
    Args:
        input_dir: 输入文件夹路径，包含 hdf5 文件
        output_dir: 输出文件夹路径。如果为 None，使用输入文件夹
        overwrite: 是否覆盖已存在的输出文件
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise ValueError(f"输入文件夹不存在: {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化前向运动学解算器
    print("初始化前向运动学解算器...")
    fk_left = ForwardKinematics(arm_side='left')
    fk_right = ForwardKinematics(arm_side='right')
    print("✅ 前向运动学解算器初始化完成")
    
    # 查找所有 hdf5 文件
    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if len(hdf5_files) == 0:
        print(f"⚠️  在 {input_dir} 中未找到 hdf5 文件")
        return
    
    print(f"找到 {len(hdf5_files)} 个 hdf5 文件")
    
    # 处理每个文件
    for hdf5_file in tqdm.tqdm(hdf5_files, desc="处理文件"):
        # 生成输出文件名
        output_filename = hdf5_file.stem + "_pose.hdf5"
        output_path = output_dir / output_filename
        
        # 检查输出文件是否已存在
        if output_path.exists() and not overwrite:
            print(f"⚠️  跳过已存在的文件: {output_path}")
            continue
        
        try:
            # 读取原始文件
            with h5py.File(hdf5_file, "r") as f_in:
                # 读取 qpos 数据
                if "/observations/qpos" not in f_in:
                    print(f"⚠️  文件 {hdf5_file} 中未找到 /observations/qpos，跳过")
                    continue
                
                qpos = f_in["/observations/qpos"][:]  # shape: (num_steps, num_joints)
                num_steps, num_joints = qpos.shape
                
                # 检查 qpos 维度
                if num_joints < 16:
                    print(f"⚠️  文件 {hdf5_file} 的 qpos 维度不足 16，当前为 {num_joints}，跳过")
                    continue
                
                # 转换 qpos
                new_qpos = convert_joints_array_to_poses(qpos, fk_left, fk_right)
                
                # 读取并转换 prev_qpos（如果存在）
                has_prev_qpos, prev_qpos = validate_dataset(f_in, "/observations/prev_qpos", num_steps)
                if has_prev_qpos:
                    new_prev_qpos = convert_joints_array_to_poses(prev_qpos, fk_left, fk_right)
                else:
                    # 如果不存在，创建新的，使用上一帧的位姿
                    new_prev_qpos = np.zeros((num_steps, 16), dtype=np.float32)
                    if num_steps > 1:
                        new_prev_qpos[1:] = new_qpos[:-1]  # 后续帧使用上一帧的 qpos
                
                # 读取并转换 action（如果存在）
                has_action, action = validate_dataset(f_in, "action", num_steps)
                new_action = None
                if has_action:
                    new_action = convert_joints_array_to_poses(action, fk_left, fk_right)
                
                # 创建新的 hdf5 文件
                with h5py.File(output_path, "w") as f_out:
                    # 复制所有其他数据（跳过需要转换的数据集）
                    skip_datasets = {"/observations/qpos", "/observations/prev_qpos"}
                    if has_action:
                        skip_datasets.add("action")
                    
                    def copy_dataset(name, obj):
                        if name in skip_datasets:
                            return
                        if isinstance(obj, h5py.Dataset):
                            if name not in f_out:
                                f_out.create_dataset(name, data=obj[:], compression=obj.compression)
                        elif isinstance(obj, h5py.Group):
                            if name not in f_out:
                                f_out.create_group(name)
                    
                    f_in.visititems(copy_dataset)
                    
                    # 确保 observations 组存在
                    if "/observations" not in f_out:
                        f_out.create_group("observations")
                    
                    # 保存转换后的 qpos
                    if "/observations/qpos" in f_out:
                        del f_out["/observations/qpos"]
                    f_out["/observations/qpos"] = new_qpos
                    copy_dataset_attrs(f_in, f_out, "/observations/qpos")
                    
                    # 保存转换后的 prev_qpos
                    if "/observations/prev_qpos" in f_out:
                        del f_out["/observations/prev_qpos"]
                    f_out["/observations/prev_qpos"] = new_prev_qpos
                    if has_prev_qpos:
                        copy_dataset_attrs(f_in, f_out, "/observations/prev_qpos")
                    
                    # 保存转换后的 action（如果存在）
                    if has_action and new_action is not None:
                        if "action" in f_out:
                            del f_out["action"]
                        f_out["action"] = new_action
                        copy_dataset_attrs(f_in, f_out, "action")
                
                print(f"✅ 已处理: {hdf5_file.name} -> {output_filename}")
                
        except Exception as e:
            print(f"❌ 处理文件 {hdf5_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ 处理完成！输出文件保存在: {output_dir}")


if __name__ == "__main__":
    tyro.cli(convert_joints_to_poses)

