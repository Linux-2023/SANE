"""
测试 Model.reachable() 接口
检查笛卡尔位姿是否可达，并返回逆运动学解
"""

import sys
import os
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flexivrdk


def test_reachable():
    """测试 reachable 接口"""
    
    # 连接机器人
    robot_sn = "Rizon4s-063239"
    print(f"连接机器人: {robot_sn}")
    robot = flexivrdk.Robot(robot_sn)
    
    # 等待机器人就绪
    print("等待机器人就绪...")
    while not robot.operational():
        pass
    print("机器人已就绪")
    
    # 创建 Model 实例
    model = flexivrdk.Model(robot)
    
    # 获取当前状态作为种子位置
    states = robot.states()
    current_joints = list(states.q)
    current_pose = list(states.tcp_pose)
    
    print(f"\n当前关节位置: {np.round(np.degrees(current_joints), 2)} deg")
    print(f"当前TCP位姿: {np.round(current_pose, 4)}")
    print(f"  位置 (xyz): {np.round(current_pose[:3], 4)} m")
    print(f"  姿态 (qwxyz): {np.round(current_pose[3:], 4)}")
    
    # 测试1: 当前位姿应该是可达的
    print("\n" + "="*50)
    print("测试1: 检查当前位姿是否可达")
    is_reachable, ik_solution = model.reachable(current_pose, current_joints, False)
    print(f"  可达: {is_reachable}")
    if is_reachable:
        print(f"  IK解: {np.round(np.degrees(ik_solution), 2)} deg")
    
    # 测试2: 在当前位置附近的位姿（应该可达）
    print("\n" + "="*50)
    print("测试2: 检查附近位姿是否可达 (z+5cm)")
    nearby_pose = current_pose.copy()
    nearby_pose[2] += 0.05  # Z方向上移5cm
    is_reachable, ik_solution = model.reachable(nearby_pose, current_joints, False)
    print(f"  目标位姿: {np.round(nearby_pose, 4)}")
    print(f"  可达: {is_reachable}")
    if is_reachable:
        print(f"  IK解: {np.round(np.degrees(ik_solution), 2)} deg")
    
    # 测试3: 远离的位姿（可能不可达）
    print("\n" + "="*50)
    print("测试3: 检查远处位姿是否可达 (x=2m)")
    far_pose = current_pose.copy()
    far_pose[0] = 2.0  # X方向2米（超出工作空间）
    is_reachable, ik_solution = model.reachable(far_pose, current_joints, False)
    print(f"  目标位姿: {np.round(far_pose, 4)}")
    print(f"  可达: {is_reachable}")
    if is_reachable:
        print(f"  IK解: {np.round(np.degrees(ik_solution), 2)} deg")
    
    # 测试4: 使用 free_orientation=True（只约束位置）
    print("\n" + "="*50)
    print("测试4: 检查位姿可达性 (free_orientation=True)")
    test_pose = current_pose.copy()
    test_pose[0] += 0.1  # X方向移动10cm
    test_pose[1] += 0.1  # Y方向移动10cm
    is_reachable, ik_solution = model.reachable(test_pose, current_joints, True)
    print(f"  目标位姿: {np.round(test_pose, 4)}")
    print(f"  可达 (仅约束位置): {is_reachable}")
    if is_reachable:
        print(f"  IK解: {np.round(np.degrees(ik_solution), 2)} deg")
    
    # 测试5: 对比 free_orientation 的影响
    print("\n" + "="*50)
    print("测试5: 对比 free_orientation 参数的影响")
    edge_pose = current_pose.copy()
    edge_pose[0] += 0.2  # X方向移动20cm
    
    is_reachable_strict, ik_strict = model.reachable(edge_pose, current_joints, False)
    is_reachable_free, ik_free = model.reachable(edge_pose, current_joints, True)
    
    print(f"  目标位姿: {np.round(edge_pose, 4)}")
    print(f"  严格约束 (free_orientation=False): {is_reachable_strict}")
    print(f"  仅约束位置 (free_orientation=True): {is_reachable_free}")
    
    print("\n" + "="*50)
    print("测试完成!")


if __name__ == "__main__":
    test_reachable()

