#!/usr/bin/env python
"""
gripper_test.py

直接使用 flexivrdk 底层 API 测试夹爪功能。

运行方式：python test/gripper_test.py
"""
import sys
import os
import time
import flexivrdk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexiv_robot import load_robot_config


def main():
    # 从配置文件加载机器人配置
    try:
        config = load_robot_config("robot.yaml")
        robot_sn = config["robot_sn"]
        gripper_name = config.get("gripper_name", "Flexiv-GN01")
    except FileNotFoundError:
        robot_sn = "Rizon4s-063239"
        gripper_name = "Flexiv-GN01"
    
    print(f"连接机器人: {robot_sn}")
    robot = flexivrdk.Robot(robot_sn)
    robot.Enable()
    
    print("等待机器人就绪...")
    while not robot.operational():
        time.sleep(0.5)
    print("机器人已就绪")
    
    # 创建夹爪接口
    gripper = flexivrdk.Gripper(robot)
    
    # 使能夹爪
    print(f"\n使能夹爪: {gripper_name}")
    gripper.Enable(gripper_name)
    
    # 获取参数
    params = gripper.params()
    print(f"\n夹爪参数:")
    print(f"  名称: {params.name}")
    print(f"  宽度范围: {params.min_width*1000:.1f} - {params.max_width*1000:.1f} mm")
    print(f"  速度范围: {params.min_vel*1000:.1f} - {params.max_vel*1000:.1f} mm/s")
    print(f"  力范围: {params.min_force:.1f} - {params.max_force:.1f} N")
    
    # 关闭夹爪
    print("\n关闭夹爪...")
    gripper.Move(0.01, 0.1, 20)
    time.sleep(2)
    
    # 获取状态
    states = gripper.states()
    print(f"  当前宽度: {states.width*1000:.1f} mm")
    print(f"  当前力: {states.force:.1f} N")
    
    # 打开夹爪
    print("\n打开夹爪...")
    gripper.Move(params.max_width, 0.1, 20)
    time.sleep(2)
    
    # 获取状态
    states = gripper.states()
    print(f"  当前宽度: {states.width*1000:.1f} mm")
    print(f"  当前力: {states.force:.1f} N")
    
    # 停止夹爪
    gripper.Stop()
    print("\n测试完成")


if __name__ == "__main__":
    main()