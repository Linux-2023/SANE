#!/usr/bin/env python3
"""
双臂可视化交互脚本

用法:
    python visualize_dual_arm.py

功能:
    - 在浏览器中显示双臂 3D 模型
    - 支持交互式输入关节角度
    - 支持预设姿态切换
    - 实时更新显示
"""

import numpy as np
import sys
import time

from dual_arm_visualizer import DualArmVisualizer


# 预设姿态
PRESET_POSES = {
    "zero": {
        "name": "零位（初始姿态）",
        "q_left": np.zeros(7),
        "q_right": np.zeros(7),
    },
    "ready": {
        "name": "准备姿态",
        "q_left": np.array([0, 0.3, 0, -0.6, 0, 0.5, 0]),
        "q_right": np.array([0, 0.3, 0, -0.6, 0, 0.5, 0]),
    },
    "reach": {
        "name": "伸展姿态",
        "q_left": np.array([0, 0.8, 0, -1.2, 0, 0.8, 0]),
        "q_right": np.array([0, 0.8, 0, -1.2, 0, 0.8, 0]),
    },
    "close": {
        "name": "靠近姿态（可能碰撞）",
        "q_left": np.array([0.5, 1.0, 0, -1.5, 0, 1.0, 0]),
        "q_right": np.array([-0.5, 1.0, 0, -1.5, 0, 1.0, 0]),
    },
    "fold": {
        "name": "折叠姿态",
        "q_left": np.array([0, -0.5, 0, 1.5, 0, -0.5, 0]),
        "q_right": np.array([0, -0.5, 0, 1.5, 0, -0.5, 0]),
    },
}


def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 60)
    print("命令帮助")
    print("=" * 60)
    print("  预设姿态:")
    for key, pose in PRESET_POSES.items():
        print(f"    {key:10s} - {pose['name']}")
    print()
    print("  关节角设置（弧度）:")
    print("    left <j1> ... <j7>       - 设置左臂关节角（弧度）")
    print("    right <j1> ... <j7>      - 设置右臂关节角（弧度）")
    print("    both <j1> ... <j7>       - 同时设置双臂关节角（弧度）")
    print()
    print("  关节角设置（角度）:")
    print("    leftd <j1> ... <j7>      - 设置左臂关节角（角度）")
    print("    rightd <j1> ... <j7>     - 设置右臂关节角（角度）")
    print("    bothd <j1> ... <j7>      - 同时设置双臂关节角（角度）")
    print()
    print("  其他命令:")
    print("    deg <值>                  - 转换角度到弧度")
    print("    rad <值>                  - 转换弧度到角度")
    print("    anim                      - 播放演示动画")
    print("    info                      - 显示当前状态")
    print("    help                      - 显示此帮助")
    print("    quit/exit                 - 退出程序")
    print("=" * 60)


def parse_angles(args: list) -> np.ndarray:
    """解析角度参数"""
    if len(args) != 7:
        raise ValueError(f"需要 7 个关节角，当前有 {len(args)} 个")
    return np.array([float(x) for x in args])


def animation_demo(viz: DualArmVisualizer):
    """播放演示动画"""
    print("播放演示动画...")
    
    n_steps = 100
    for i in range(n_steps):
        t = i / n_steps * 2 * np.pi
        
        q_left = np.array([
            0.5 * np.sin(t),
            0.3 + 0.4 * np.sin(t),
            0.3 * np.sin(t * 2),
            -0.5 - 0.5 * np.sin(t),
            0.2 * np.sin(t * 3),
            0.5 + 0.3 * np.sin(t),
            0.3 * np.sin(t * 2)
        ])
        
        q_right = np.array([
            -0.5 * np.sin(t),
            0.3 + 0.4 * np.sin(t),
            -0.3 * np.sin(t * 2),
            -0.5 - 0.5 * np.sin(t),
            -0.2 * np.sin(t * 3),
            0.5 + 0.3 * np.sin(t),
            -0.3 * np.sin(t * 2)
        ])
        
        viz.display(q_left, q_right)
        time.sleep(0.03)
    
    print("动画播放完成")


def main():
    print("=" * 60)
    print("双臂可视化器")
    print("=" * 60)
    
    # 创建可视化器
    print("\n正在启动可视化器...")
    viz = DualArmVisualizer(open_browser=True)
    print(f"浏览器已打开: {viz.url}")
    
    # 初始化状态
    q_left = np.zeros(7)
    q_right = np.zeros(7)
    viz.display(q_left, q_right)
    
    print_help()
    
    # 尝试导入碰撞检测器（可选）
    try:
        from dual_arm_collision_checker import create_checker
        checker = create_checker(mode="mesh")
        has_collision_checker = True
        print("\n碰撞检测器已启用")
    except ImportError:
        has_collision_checker = False
        print("\n碰撞检测器未启用（可选）")
    
    # 主循环
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0]
            
            # 退出
            if action in ["quit", "exit", "q"]:
                print("退出程序")
                break
            
            # 帮助
            elif action == "help":
                print_help()
            
            # 预设姿态
            elif action in PRESET_POSES:
                pose = PRESET_POSES[action]
                q_left = pose["q_left"].copy()
                q_right = pose["q_right"].copy()
                viz.display(q_left, q_right)
                print(f"已切换到: {pose['name']}")
            
            # 设置左臂（弧度）
            elif action == "left":
                q_left = parse_angles(parts[1:])
                viz.display(q_left, q_right)
                print(f"左臂 (rad): {q_left}")
            
            # 设置右臂（弧度）
            elif action == "right":
                q_right = parse_angles(parts[1:])
                viz.display(q_left, q_right)
                print(f"右臂 (rad): {q_right}")
            
            # 同时设置双臂（弧度）
            elif action == "both":
                q = parse_angles(parts[1:])
                q_left = q.copy()
                q_right = q.copy()
                viz.display(q_left, q_right)
                print(f"双臂 (rad): {q}")
            
            # 设置左臂（角度）
            elif action == "leftd":
                q_deg = parse_angles(parts[1:])
                q_left = np.deg2rad(q_deg)
                viz.display(q_left, q_right)
                print(f"左臂 (deg): {q_deg} -> (rad): {q_left}")
            
            # 设置右臂（角度）
            elif action == "rightd":
                q_deg = parse_angles(parts[1:])
                q_right = np.deg2rad(q_deg)
                viz.display(q_left, q_right)
                print(f"右臂 (deg): {q_deg} -> (rad): {q_right}")
            
            # 同时设置双臂（角度）
            elif action == "bothd":
                q_deg = parse_angles(parts[1:])
                q_left = np.deg2rad(q_deg)
                q_right = q_left.copy()
                viz.display(q_left, q_right)
                print(f"双臂 (deg): {q_deg} -> (rad): {q_left}")
            
            # 角度转换
            elif action == "deg":
                deg = float(parts[1])
                rad = np.deg2rad(deg)
                print(f"{deg}° = {rad:.6f} rad")
            
            # 弧度转角度
            elif action == "rad":
                rad = float(parts[1])
                deg = np.rad2deg(rad)
                print(f"{rad} rad = {deg:.4f}°")
            
            # 演示动画
            elif action == "anim":
                animation_demo(viz)
            
            # 显示当前状态
            elif action == "info":
                print(f"左臂关节角: {q_left}")
                print(f"右臂关节角: {q_right}")
                
                # 获取法兰盘位置
                left_pose, right_pose = viz.get_flange_poses(q_left, q_right)
                print(f"左法兰盘位置: {left_pose[:3, 3]}")
                print(f"右法兰盘位置: {right_pose[:3, 3]}")
                
                # 碰撞检测
                if has_collision_checker:
                    collision = checker.check_collision(q_left, q_right)
                    min_dist = checker.compute_min_distance(q_left, q_right)
                    status = "⚠️ 碰撞!" if collision else "✓ 安全"
                    print(f"碰撞状态: {status}, 最小距离: {min_dist:.4f}m")
            
            else:
                print(f"未知命令: {action}，输入 'help' 查看帮助")
        
        except ValueError as e:
            print(f"错误: {e}")
        except KeyboardInterrupt:
            print("\n退出程序")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()

