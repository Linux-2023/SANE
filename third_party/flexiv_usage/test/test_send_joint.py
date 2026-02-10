#!/usr/bin/env python
"""
test_send_joint.py

测试 send_joint_position 功能（直接发送关节位置命令）。

运行方式：python test/test_send_joint.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexiv_robot import FlexivRobot, load_robot_sn_from_yaml


def main():
    # 从配置文件加载机器人序列号
    try:
        robot_sn = load_robot_sn_from_yaml("robot.yaml")
    except FileNotFoundError:
        robot_sn = "Rizon4s-063239"
    
    print(f"连接机器人: {robot_sn}")
    robot = FlexivRobot(robot_sn)
    
    # 获取当前关节位置
    current_joints = robot.get_joint_positions()
    print(f"当前关节位置 (deg): {np.round(np.degrees(current_joints), 2).tolist()}")
    
    # 目标：J7 转 -30 度
    target_joints = np.array(current_joints)
    target_joints[6] += np.deg2rad(-30)
    print(f"目标关节位置 (deg): {np.round(np.degrees(target_joints), 2).tolist()}")
    
    print("\n开始发送关节位置命令 (按 Ctrl+C 停止)...")
    print("注意：机器人将持续向目标位置运动")
    
    try:
        while True:
            robot.send_joint_position(target_joints=target_joints)
            time.sleep(0.02)  # 50Hz
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        robot.stop()
        print("机器人已停止")


if __name__ == "__main__":
    main()