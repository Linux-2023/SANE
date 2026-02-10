#!/usr/bin/env python3
"""
使用相对位姿重放 hdf5 文件中的轨迹

读取 hdf5 文件中的 action 数据（世界坐标系），计算相邻帧之间的相对位姿变化，
然后从机器人当前位姿开始逐步应用这些相对变化。

关键点：
- action 数据在世界坐标系中
- 计算相邻帧之间的相对位姿变化（delta pose）
- 从机器人当前位姿开始，逐步应用相对变化

用法:
    python replay_trajectory_relative_pose.py \
        --hdf5-file path/to/episode.hdf5 \
        --config config.yaml \
        [--speed 1.0] \
        [--start-step 0] \
        [--end-step -1] \
        [--scale 1.0]
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import time
import sys
import os

# 添加路径以导入 env_dual_feetech
current_dir = os.path.dirname(__file__)
flexiv_examples_dir = os.path.abspath(os.path.join(current_dir, '../../'))
if flexiv_examples_dir not in sys.path:
    sys.path.insert(0, flexiv_examples_dir)

try:
    from env_dual_feetech import DualArmEnvironmentFeetech, create_dual_arm_feetech_env_from_config
except ImportError as e:
    print(f"Warning: Could not import DualArmEnvironmentFeetech: {e}")
    print("Falling back to DualArmEnvironment")
    from env_dual import DualArmEnvironment, create_dual_arm_env_from_config
    DualArmEnvironmentFeetech = DualArmEnvironment
    create_dual_arm_feetech_env_from_config = create_dual_arm_env_from_config


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法 (qw, qx, qy, qz) 格式
    
    Args:
        q1: 第一个四元数 [qw, qx, qy, qz]
        q2: 第二个四元数 [qw, qx, qy, qz]
    
    Returns:
        乘积四元数 [qw, qx, qy, qz]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭 (qw, qx, qy, qz) 格式"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """归一化四元数"""
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    将四元数转换为旋转矩阵
    
    Args:
        q: 四元数 [qw, qx, qy, qz]
    
    Returns:
        3x3 旋转矩阵
    """
    q = quat_normalize(q)
    w, x, y, z = q
    
    # 计算旋转矩阵
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    
    return R


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    球形线性插值 (Slerp) 用于四元数插值
    q1, q2: 四元数，格式 [qw, qx, qy, qz]
    t: 插值比例，0~1
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    dot = np.dot(q1, q2)
    # 取最短路径
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # 太接近便直接线性插值
        result = q1 + t*(q2 - q1)
        return quat_normalize(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize((s0 * q1) + (s1 * q2))


def compute_relative_pose(prev_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
    """
    计算两个位姿之间的相对变化
    
    Args:
        prev_pose: 前一帧位姿 [x, y, z, qw, qx, qy, qz]
        current_pose: 当前帧位姿 [x, y, z, qw, qx, qy, qz]
    
    Returns:
        相对位姿变化 [delta_x, delta_y, delta_z, delta_qw, delta_qx, delta_qy, delta_qz]
    """
    # 位置差
    delta_pos = current_pose[:3] - prev_pose[:3]
    # # 取反y分量
    # delta_pos[1] = -delta_pos[1]
    # # z变化量设置为0
    # delta_pos[2] = 0.0
    
    # 旋转差：relative_quat = current_quat * conjugate(prev_quat)
    prev_quat = quat_normalize(prev_pose[3:7])
    current_quat = quat_normalize(current_pose[3:7])
    prev_quat_conj = quat_conjugate(prev_quat)
    delta_quat = quat_multiply(current_quat, prev_quat_conj)
    delta_quat = quat_normalize(delta_quat)
    
    return np.concatenate([delta_pos, delta_quat])


def apply_relative_pose(
    base_pose: np.ndarray, 
    delta_pose: np.ndarray, 
    scale: float = 1.0,
    delta_in_flange_frame: bool = True
) -> np.ndarray:
    """
    将相对位姿变化应用到基础位姿，并可用 scale 缩放变化量
    
    关键：如果相对变化是在法兰坐标系下描述的，需要转换到基座坐标系
    
    Args:
        base_pose: 基础位姿（基座坐标系）[x, y, z, qw, qx, qy, qz]
        delta_pose: 相对位姿变化 [delta_x, delta_y, delta_z, delta_qw, delta_qx, delta_qy, delta_qz]
        scale: 缩放因子（0~1），控制应用多少比例的变化量
        delta_in_flange_frame: 如果为True，表示delta_pose是在法兰坐标系下描述的，需要转换到基座坐标系
    
    Returns:
        新的位姿（基座坐标系）[x, y, z, qw, qx, qy, qz]
    """
    delta_pose = delta_pose.copy()
    
    # 如果相对变化是在法兰坐标系下描述的，需要转换到基座坐标系
    if delta_in_flange_frame:
        # 获取当前末端位姿的旋转矩阵（从基座到法兰）
        base_quat = quat_normalize(base_pose[3:7])
        R_flange_to_base = quat_to_rotation_matrix(base_quat)
        
        # 将位置变化从法兰坐标系转换到基座坐标系
        delta_pos_flange = delta_pose[:3] * scale
        delta_pos_base = R_flange_to_base @ delta_pos_flange
        
        # 对于旋转，相对旋转在法兰坐标系和基座坐标系下是一样的（因为是相对旋转）
        # 但我们需要确保旋转是相对于当前姿态的
        delta_quat = quat_normalize(delta_pose[3:7])
        
        # 应用旋转：new_quat = base_quat * delta_quat
        # 如果scale != 1.0，需要对旋转进行插值
        if scale == 1.0:
            new_quat = quat_multiply(base_quat, delta_quat)
            new_quat = quat_normalize(new_quat)
        else:
            # 目标旋转
            target_quat = quat_multiply(base_quat, delta_quat)
            target_quat = quat_normalize(target_quat)
            # 用 slerp 在 base_quat 和 target_quat 之间插值
            new_quat = slerp(base_quat, target_quat, scale)
        
        # 位置：基座坐标系下的位置 + 转换后的位置变化
        new_pos = base_pose[:3] + delta_pos_base
        
    else:
        # 如果相对变化已经在基座坐标系下，直接应用
        if scale == 1.0:
            new_pos = base_pose[:3] + delta_pose[:3] * scale
            base_quat = quat_normalize(base_pose[3:7])
            delta_quat = quat_normalize(delta_pose[3:7])
            new_quat = quat_multiply(base_quat, delta_quat)
            new_quat = quat_normalize(new_quat)
        else:
            new_pos = base_pose[:3] + scale * delta_pose[:3]
            base_quat = quat_normalize(base_pose[3:7])
            delta_quat = quat_normalize(delta_pose[3:7])
            target_quat = quat_multiply(base_quat, delta_quat)
            target_quat = quat_normalize(target_quat)
            new_quat = slerp(base_quat, target_quat, scale)
    
    return np.concatenate([new_pos, new_quat])


def load_trajectory(hdf5_file: Path) -> tuple:
    """
    从 hdf5 文件加载轨迹数据并计算相对位姿变化
    
    Args:
        hdf5_file: HDF5 文件路径
    
    Returns:
        (relative_poses, num_steps, initial_pose)
        relative_poses: 相对位姿变化数组 (num_steps-1, 16) - 第一帧没有相对变化
        num_steps: 步数
        initial_pose: 第一帧的绝对位姿（用于初始化）
    """
    with h5py.File(hdf5_file, "r") as f:
        # 读取动作数据（世界坐标系中的绝对位姿）
        if "action" not in f:
            raise ValueError(f"文件 {hdf5_file} 中未找到 action 数据")
        
        actions = f["action"][:]  # shape: (num_steps, 16)
        num_steps = actions.shape[0]
        
        print(f"读取到 {num_steps} 个时间步的动作数据（世界坐标系）")
        
        # 计算相对位姿变化
        relative_poses = []
        initial_pose = actions[0].copy()  # 保存第一帧作为参考
        
        for i in range(1, num_steps):
            prev_pose = actions[i-1]
            current_pose = actions[i]
            
            # 计算相对位姿变化（只对位姿部分，夹爪直接使用当前值）
            delta_pose = np.zeros(16, dtype=np.float32)
            
            # 左臂相对位姿
            left_prev = prev_pose[0:7]
            left_curr = current_pose[0:7]
            left_delta = compute_relative_pose(left_prev, left_curr)
            delta_pose[0:7] = left_delta
            delta_pose[7] = current_pose[7]  # 左夹爪直接使用当前值
            
            # 右臂相对位姿
            right_prev = prev_pose[8:15]
            right_curr = current_pose[8:15]
            right_delta = compute_relative_pose(right_prev, right_curr)
            delta_pose[8:15] = right_delta
            delta_pose[15] = current_pose[15]  # 右夹爪直接使用当前值
            
            relative_poses.append(delta_pose)
        
        relative_poses = np.array(relative_poses)
        print(f"计算了 {len(relative_poses)} 个相对位姿变化")
        
        return relative_poses, num_steps, initial_pose


def replay_trajectory_relative(
    hdf5_file: Path,
    config_file: str = None,
    speed: float = 1.0,
    start_step: int = 0,
    end_step: int = -1,
    safety_check: bool = True,
    delay_between_steps: float = None,
    scale: float = 1.0,
):
    """
    使用相对位姿在真实机械臂上回放轨迹，并支持缩放应用变化量
    
    Args:
        hdf5_file: HDF5 文件路径
        config_file: 环境配置文件路径（可选）
        speed: 回放速度倍数（1.0 = 原始速度）
        start_step: 起始步数（从0开始）
        end_step: 结束步数（-1 表示到最后）
        safety_check: 是否启用安全检测
        delay_between_steps: 每步之间的延迟（秒），如果为 None 则根据原始时间戳计算
        scale: 只应用百分之多少的变化量（0~1，默认1.0），既应用 100%（注意对旋转为slerp插值）
    """
    # 加载轨迹数据
    relative_poses, num_steps, initial_pose = load_trajectory(hdf5_file)
    
    # 确定实际使用的步数范围
    if end_step < 0:
        end_step = num_steps
    end_step = min(end_step, num_steps)
    start_step = max(0, start_step)
    
    if start_step >= end_step:
        raise ValueError(f"起始步数 {start_step} 必须小于结束步数 {end_step}")
    
    actual_steps = end_step - start_step
    print(f"将回放第 {start_step} 到 {end_step-1} 步，共 {actual_steps} 步")
    
    # 读取时间戳（如果存在）
    timestamps = None
    with h5py.File(hdf5_file, "r") as f:
        if "/observations/timestamps" in f:
            timestamps_group = f["/observations/timestamps"]
            if "robot" in timestamps_group:
                timestamps = timestamps_group["robot"][:]
                print(f"检测到时间戳数据，将根据时间戳控制回放速度")
    
    # 创建环境
    print("\n正在初始化机械臂环境...")
    if config_file:
        try:
            env = create_dual_arm_feetech_env_from_config(config_file)
        except:
            from env_dual import create_dual_arm_env_from_config
            env = create_dual_arm_env_from_config(config_file, verbose=True)
    else:
        print("⚠️  警告：未指定配置文件，将使用默认配置")
        env = DualArmEnvironmentFeetech(
            left_robot_sn="",
            right_robot_sn="",
            use_pose_control=True,
            collision_check_enabled=safety_check,
            verbose=True
        )
    
    try:
        # 重置环境
        print("\n正在重置环境到初始位置...")
        env.reset()
        time.sleep(2.0)  # 等待机器人到达初始位置
        
        # 获取机器人当前位姿作为起始点
        obs = env.get_observation()
        current_robot_pose = obs["state"].copy()
        
        print("\n机器人当前位姿（将从此开始应用相对变化）:")
        print(f"  左臂: pos=[{current_robot_pose[0]:.4f}, {current_robot_pose[1]:.4f}, {current_robot_pose[2]:.4f}], "
              f"gripper={current_robot_pose[7]:.3f}")
        print(f"  右臂: pos=[{current_robot_pose[8]:.4f}, {current_robot_pose[9]:.4f}, {current_robot_pose[10]:.4f}], "
              f"gripper={current_robot_pose[15]:.3f}")
        
        # 准备回放
        print(f"\n开始回放轨迹（速度: {speed}x，使用相对位姿, 缩放: {scale}）...")
        print("按 Ctrl+C 可以中断回放")
        print("-" * 80)
        
        # 回放循环
        step_count = 0
        start_time = time.time()
        last_action_time = start_time
        
        for i in range(start_step, end_step - 1):  # -1 因为相对位姿只有 num_steps-1 个
            try:
                # 获取相对位姿变化
                delta_pose = relative_poses[i]
                
                # # 对左、右臂y分量分别取反再应用
                # delta_pose = delta_pose.copy()
                # # 左臂y分量
                # delta_pose[1] = -delta_pose[1]
                # # 右臂y分量
                # delta_pose[9] = -delta_pose[9]
                # # z变化量设置为0（只影响增量，不修改当前值）
                # delta_pose[2] = 0.0
                # delta_pose[10] = 0.0
                
                # 应用到当前机器人位姿
                target_pose = current_robot_pose.copy()
                
                # 左臂：应用相对位姿变化
                left_base = target_pose[0:7]
                left_delta = delta_pose[0:7]  # 使用左臂的delta
                # delta_in_flange_frame=True 表示相对变化是在法兰坐标系下描述的
                left_new = apply_relative_pose(left_base, left_delta, scale=scale, delta_in_flange_frame=True)
                target_pose[0:7] = left_new
                target_pose[7] = delta_pose[7]  # 使用左臂夹爪值
                
                # 右臂：应用相对位姿变化
                right_base = target_pose[8:15]
                right_delta = delta_pose[8:15]  # 使用右臂的delta
                # delta_in_flange_frame=True 表示相对变化是在法兰坐标系下描述的
                right_new = apply_relative_pose(right_base, right_delta, scale=scale, delta_in_flange_frame=True)
                target_pose[8:15] = right_new
                target_pose[15] = delta_pose[15]  # 使用右臂夹爪值

                # 此处不再强制target_pose的z分量为0，而是保持当前位置的z
                # target_pose[2] = 0.0
                # target_pose[10] = 0.0
                
                # 更新当前位姿（用于下一次计算）
                current_robot_pose = target_pose.copy()
                
                # 构建动作字典
                action_dict = {
                    "actions": target_pose.astype(np.float32)
                }
                
                # 应用动作
                env.apply_action(action_dict)
                
                step_count += 1
                
                # 计算延迟
                if delay_between_steps is not None:
                    step_delay = delay_between_steps / speed
                elif timestamps is not None and i < len(timestamps) - 1:
                    original_dt = timestamps[i + 1] - timestamps[i]
                    step_delay = original_dt / speed
                else:
                    # 默认延迟（30fps，即 1/30 ≈ 0.0333 秒）
                    step_delay = (1.0 / 30.0) / speed
                
                # 等待
                current_time = time.time()
                elapsed = current_time - last_action_time
                sleep_time = max(0, step_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_action_time = time.time()
                
                # 每 10 步打印一次进度
                if step_count % 10 == 0:
                    elapsed_total = time.time() - start_time
                    progress = (step_count / actual_steps) * 100
                    print(f"进度: {step_count}/{actual_steps} ({progress:.1f}%) | "
                          f"已用时间: {elapsed_total:.1f}s | "
                          f"预计剩余: {(elapsed_total / step_count * (actual_steps - step_count)):.1f}s")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  回放被用户中断")
                break
            except Exception as e:
                print(f"\n⚠️  执行第 {i} 步时出错: {e}")
                import traceback
                traceback.print_exc()
                print("继续执行下一步...")
                continue
        
        # 回放完成
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"✅ 回放完成！")
        print(f"   总步数: {step_count}")
        print(f"   总时间: {total_time:.2f} 秒")
        print(f"   平均速度: {step_count / total_time:.2f} 步/秒")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 回放过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在关闭环境...")
        env.close()
        print("环境已关闭")


def main():
    parser = argparse.ArgumentParser(
        description='使用相对位姿在真实机械臂上回放 hdf5 文件中记录的轨迹'
    )
    parser.add_argument(
        '--hdf5-file',
        type=str,
        required=True,
        help='HDF5 文件路径'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='环境配置文件路径（YAML 格式）'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='回放速度倍数（默认: 1.0，即原始速度）'
    )
    parser.add_argument(
        '--start-step',
        type=int,
        default=0,
        help='起始步数（默认: 0）'
    )
    parser.add_argument(
        '--end-step',
        type=int,
        default=-1,
        help='结束步数（默认: -1，表示到最后）'
    )
    parser.add_argument(
        '--no-safety-check',
        action='store_true',
        help='禁用安全检测（碰撞检测等）'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=None,
        help='每步之间的固定延迟（秒），如果指定则忽略时间戳'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='只应用百分之多少的变化量 (0~1, 默认1.0即全部应用)'
    )
    
    args = parser.parse_args()
    
    hdf5_file = Path(args.hdf5_file)
    if not hdf5_file.exists():
        raise FileNotFoundError(f"文件不存在: {hdf5_file}")
    if args.scale < 0.0 or args.scale > 1.0:
        raise ValueError(f"--scale 参数应在 0~1 之间, 当前为: {args.scale}")
    
    replay_trajectory_relative(
        hdf5_file=hdf5_file,
        config_file=args.config,
        speed=args.speed,
        start_step=args.start_step,
        end_step=args.end_step,
        safety_check=not args.no_safety_check,
        delay_between_steps=args.delay,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()

