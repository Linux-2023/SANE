#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单夹爪延迟测试

发送夹爪命令，实时读取状态，测量延迟。
"""

import time
import sys
import numpy as np
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexiv_robot import FlexivRobot, load_robot_config


def main():
    # 读取配置
    config = load_robot_config('robot.yaml')
    robot_sn = config['robot_sn']
    gripper_name = config.get('gripper_name')
    
    print(f"连接机器人 {robot_sn}...")
    robot = FlexivRobot(robot_sn)
    
    print(f"使能夹爪 {gripper_name}...")
    robot.gripper_enable(gripper_name)
    
    params = robot.get_gripper_params()
    print(f"夹爪范围: {params.min_width*1000:.1f} - {params.max_width*1000:.1f} mm")
    
    # 设置测试参数
    TOLERANCE_MM = 1  # 到达判定容差
    SAMPLE_INTERVAL = 0.002  # 采样间隔 2ms (500Hz)
    
    try:
        while True:
            # 获取当前宽度
            current_width = robot.get_gripper_width()
            print(f"\n当前宽度: {current_width*1000:.1f} mm")
            
            # 输入目标宽度
            user_input = input("输入目标宽度 (mm), 或 'q' 退出: ").strip()
            if user_input.lower() == 'q':
                break
            
            try:
                target_mm = float(user_input)
            except ValueError:
                print("无效输入")
                continue
            
            target_m = target_mm / 1000.0
            if target_m < params.min_width or target_m > params.max_width:
                print(f"超出范围 ({params.min_width*1000:.1f} - {params.max_width*1000:.1f} mm)")
                continue
            
            # 记录初始状态
            start_width = robot.get_gripper_width()
            start_moving = robot.get_gripper_states().is_moving
            
            print(f"\n>>> 发送命令: {start_width*1000:.1f} -> {target_mm:.1f} mm")
            print("-" * 50)
            
            # 发送命令并记录时间
            t_send = time.perf_counter()
            robot.gripper_move(target_m, velocity=params.max_vel, wait=False)
            t_after_send = time.perf_counter()
            
            # 实时读取状态
            t_response = None  # 开始移动的时间
            t_reach = None     # 到达目标的时间
            
            timeout = 5.0
            while time.perf_counter() - t_send < timeout:
                t_now = time.perf_counter()
                states = robot.get_gripper_states()
                width = states.width
                is_moving = states.is_moving
                
                elapsed_ms = (t_now - t_send) * 1000
                
                # 检测开始移动
                if t_response is None:
                    if is_moving or abs(width - start_width) > 0.00001:
                        t_response = t_now
                        print(f"[{elapsed_ms:6.1f} ms] 开始移动, 宽度: {width*1000:.2f} mm")
                
                # 检测到达目标
                if t_reach is None:
                    if abs(width - target_m) <= TOLERANCE_MM / 1000.0:
                        t_reach = t_now
                        print(f"[{elapsed_ms:6.1f} ms] 到达目标, 宽度: {width*1000:.2f} mm")
                
                # 检测停止移动
                if t_response is not None and not is_moving:
                    print(f"[{elapsed_ms:6.1f} ms] 停止移动, 宽度: {width*1000:.2f} mm")
                    break
                
                time.sleep(SAMPLE_INTERVAL)
            
            # 打印结果
            print("-" * 50)
            send_latency = (t_after_send - t_send) * 1000
            print(f"发送延迟: {send_latency:.2f} ms")
            
            if t_response:
                response_latency = (t_response - t_send) * 1000
                print(f"响应延迟: {response_latency:.1f} ms")
            else:
                print("响应延迟: 未检测到移动")
            
            if t_reach:
                reach_latency = (t_reach - t_send) * 1000
                print(f"到达延迟: {reach_latency:.1f} ms")
            else:
                print(f"到达延迟: 未到达目标 (当前: {robot.get_gripper_width()*1000:.1f} mm)")
    
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        robot.stop()
        print("退出")


if __name__ == "__main__":
    main()

