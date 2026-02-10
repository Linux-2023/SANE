"""
在真实机械臂上回放 hdf5 文件中记录的轨迹

读取 hdf5 文件中的轨迹数据（qpos 和 action），并在真实机械臂上执行回放。

支持两种数据格式：
1. 关节角数据：16维 [左臂关节角(7) + 左臂夹爪(1) + 右臂关节角(7) + 右臂夹爪(1)]
2. 位姿数据：16维 [左臂位姿(7: x,y,z,qw,qx,qy,qz) + 左臂夹爪(1) + 右臂位姿(7) + 右臂夹爪(1)]

用法:
    python replay_trajectory.py --hdf5-file path/to/episode.hdf5 [--config config.yaml] [--speed 1.0] [--start-step 0] [--end-step -1]
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import time
import sys
import os

# 添加路径以导入 env_dual
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from env_dual import DualArmEnvironment, create_dual_arm_env_from_config


def detect_data_format(hdf5_file: Path) -> str:
    """
    检测 hdf5 文件中的数据格式（关节角或位姿）
    
    Args:
        hdf5_file: HDF5 文件路径
    
    Returns:
        "joint" 或 "pose"
    """
    with h5py.File(hdf5_file, "r") as f:
        if "/observations/qpos" not in f:
            raise ValueError(f"文件 {hdf5_file} 中未找到 /observations/qpos")
        
        qpos = f["/observations/qpos"][:]
        if len(qpos) == 0:
            raise ValueError(f"文件 {hdf5_file} 中 qpos 数据为空")
        
        # 检查第一帧数据
        first_qpos = qpos[0]
        
        # 如果数据是位姿格式，位置值通常在合理范围内（例如 -1 到 1 米）
        # 关节角通常在 -π 到 π 之间
        left_pos = first_qpos[0:3]  # 前三个值
        
        # 简单启发式：如果位置值在合理范围内（-2 到 2 米），可能是位姿
        # 如果值在 -π 到 π 之间，可能是关节角
        if np.all(np.abs(left_pos) < 2.0) and np.any(np.abs(left_pos) > 0.1):
            # 检查是否有四元数（后4个值应该在 -1 到 1 之间，且接近单位四元数）
            left_quat = first_qpos[3:7]
            quat_norm = np.linalg.norm(left_quat)
            if 0.9 < quat_norm < 1.1:  # 接近单位四元数
                return "pose"
        
        # 默认认为是关节角
        return "joint"


def load_trajectory(hdf5_file: Path) -> tuple:
    """
    从 hdf5 文件加载轨迹数据
    
    Args:
        hdf5_file: HDF5 文件路径
    
    Returns:
        (actions, num_steps, data_format)
        actions: 动作数组 (num_steps, 16)
        num_steps: 步数
        data_format: "joint" 或 "pose"
    """
    with h5py.File(hdf5_file, "r") as f:
        # 读取动作数据
        if "action" not in f:
            raise ValueError(f"文件 {hdf5_file} 中未找到 action 数据")
        
        actions = f["action"][:]  # shape: (num_steps, 16)
        num_steps = actions.shape[0]
        
        print(f"读取到 {num_steps} 个时间步的动作数据")
        
        # 检测数据格式
        data_format = detect_data_format(hdf5_file)
        print(f"检测到数据格式: {'位姿控制' if data_format == 'pose' else '关节角控制'}")
        
        return actions, num_steps, data_format


def replay_trajectory(
    hdf5_file: Path,
    config_file: str = None,
    speed: float = 1.0,
    start_step: int = 0,
    end_step: int = -1,
    use_action: bool = True,
    safety_check: bool = True,
    delay_between_steps: float = None,
):
    """
    在真实机械臂上回放轨迹
    
    Args:
        hdf5_file: HDF5 文件路径
        config_file: 环境配置文件路径（可选）
        speed: 回放速度倍数（1.0 = 原始速度，0.5 = 一半速度，2.0 = 两倍速度）
        start_step: 起始步数（从0开始）
        end_step: 结束步数（-1 表示到最后）
        use_action: 是否使用 action 数据（True）还是 qpos 数据（False）
        safety_check: 是否启用安全检测（碰撞检测等）
        delay_between_steps: 每步之间的延迟（秒），如果为 None 则根据原始时间戳计算
    """
    # 加载轨迹数据
    actions, num_steps, data_format = load_trajectory(hdf5_file)
    
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
        env = create_dual_arm_env_from_config(config_file, verbose=True)
    else:
        # 使用默认配置（需要手动指定机器人序列号等）
        print("⚠️  警告：未指定配置文件，将使用默认配置")
        print("   请确保在代码中正确设置机器人序列号等参数")
        env = DualArmEnvironment(
            left_robot_sn="",  # 需要手动设置
            right_robot_sn="",  # 需要手动设置
            use_pose_control=(data_format == "pose"),
            collision_check_enabled=safety_check,
            verbose=True
        )
    
    # 设置控制模式
    if env._use_pose_control != (data_format == "pose"):
        print(f"⚠️  警告：环境控制模式 ({'位姿' if env._use_pose_control else '关节角'}) "
              f"与数据格式 ({'位姿' if data_format == 'pose' else '关节角'}) 不匹配")
        print(f"   将使用环境当前的控制模式")
    
    try:
        # 重置环境
        print("\n正在重置环境到初始位置...")
        env.reset()
        time.sleep(2.0)  # 等待机器人到达初始位置
        
        # 准备回放
        print(f"\n开始回放轨迹（速度: {speed}x）...")
        print("按 Ctrl+C 可以中断回放")
        print("-" * 80)
        
        # 如果使用 qpos 而不是 action
        if not use_action:
            with h5py.File(hdf5_file, "r") as f:
                qpos_data = f["/observations/qpos"][:]
                actions = qpos_data  # 使用 qpos 作为目标位置
        
        # 回放循环
        step_count = 0
        start_time = time.time()
        last_action_time = start_time
        
        for i in range(start_step, end_step):
            try:
                action = actions[i]
                
                # 构建动作字典
                action_dict = {
                    "actions": np.array(action, dtype=np.float32)
                }
                
                # 应用动作
                env.apply_action(action_dict)
                
                step_count += 1
                
                # 计算延迟
                if delay_between_steps is not None:
                    # 使用指定的延迟
                    step_delay = delay_between_steps / speed
                elif timestamps is not None and i < len(timestamps) - 1:
                    # 根据原始时间戳计算延迟
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
        description='在真实机械臂上回放 hdf5 文件中记录的轨迹'
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
        '--use-qpos',
        action='store_true',
        help='使用 qpos 数据而不是 action 数据（默认使用 action）'
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
    
    args = parser.parse_args()
    
    hdf5_file = Path(args.hdf5_file)
    if not hdf5_file.exists():
        raise FileNotFoundError(f"文件不存在: {hdf5_file}")
    
    replay_trajectory(
        hdf5_file=hdf5_file,
        config_file=args.config,
        speed=args.speed,
        start_step=args.start_step,
        end_step=args.end_step,
        use_action=not args.use_qpos,
        safety_check=not args.no_safety_check,
        delay_between_steps=args.delay,
    )


if __name__ == "__main__":
    main()

