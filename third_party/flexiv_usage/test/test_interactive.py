#!/usr/bin/env python
"""
test_interactive.py

交互式测试脚本，用于手动测试各项功能。
运行方式: python test/test_interactive.py

这个脚本会逐步测试每个功能，并等待用户确认后继续。
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexiv_robot import FlexivRobot, MotionParams, ForceControlFrame, load_robot_config, load_robot_sn_from_yaml


def wait_user_confirm(msg: str = "按 Enter 继续，输入 'q' 退出..."):
    """等待用户确认"""
    response = input(msg)
    if response.lower() == 'q':
        print("用户退出测试")
        sys.exit(0)


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_initialization(robot_sn: str) -> FlexivRobot:
    """测试初始化"""
    print_separator("测试 1: 机器人初始化")
    
    print(f"正在连接机器人 [{robot_sn}]...")
    robot = FlexivRobot(robot_sn, auto_init=True, verbose=True)
    
    print(f"\n初始化结果:")
    print(f"  - 连接状态: {'已连接' if robot.is_ready() else '未连接'}")
    print(f"  - 自由度: {robot.get_dof()}")
    
    return robot


def test_get_states(robot: FlexivRobot):
    """测试状态获取"""
    print_separator("测试 2: 获取机器人状态")
    robot.enable()
    
    # 关节位置
    joints = robot.get_joint_positions()
    print(f"\n关节位置 (rad):")
    for i, j in enumerate(joints):
        print(f"  J{i+1}: {j:.4f} rad ({math.degrees(j):.2f}°)")
    
    # TCP 位姿
    pose = robot.get_tcp_pose()
    print(f"\nTCP 位姿:")
    print(f"  位置: [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}] m")
    print(f"  姿态: [{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}, {pose[6]:.4f}] (qw,qx,qy,qz)")
    
    # 外部力
    wrench = robot.get_external_wrench("world")
    print(f"\n外部力/力矩 (世界坐标系):")
    print(f"  力: [{wrench[0]:.2f}, {wrench[1]:.2f}, {wrench[2]:.2f}] N")
    print(f"  力矩: [{wrench[3]:.2f}, {wrench[4]:.2f}, {wrench[5]:.2f}] Nm")
    
    robot.stop()
    wait_user_confirm()


def test_go_home(robot: FlexivRobot):
    """测试回 Home"""
    print_separator("测试 3: 回到 Home 位置")
    robot.enable()
    
    # 显示当前 Home 位置
    home = robot.get_home_joints()
    print(f"当前 Home 位置: {np.round(np.degrees(home), 1).tolist()}°")
    
    # 询问是否修改 Home
    print("\n[1] 使用当前 Home 位置")
    print("[2] 将当前关节位置设为 Home")
    choice = input("请选择: ")
    
    if choice == "2":
        robot.set_current_as_home()
        home = robot.get_home_joints()
        print(f"新 Home 位置: {np.round(np.degrees(home), 1).tolist()}°")
    
    # 先做一个小运动
    print("\n先做一个小运动（J6 +15°）以测试 go_home...")
    current = robot.get_joint_positions()
    target = current.copy()
    target[5] += math.radians(15)
    
    wait_user_confirm("确认机器人周围安全后按 Enter 继续...")
    
    robot.move_joint(target, timeout=10.0)
    print(f"当前 J6: {math.degrees(robot.get_joint_positions()[5]):.2f}°")
    
    # 回到 Home
    print("\n执行 go_home...")
    success = robot.go_home(wait=True, timeout=15.0)
    
    if success:
        print("已成功回到 Home 位置")
        final = robot.get_joint_positions()
        print(f"最终关节角: {np.round(np.degrees(final), 1).tolist()}°")
    else:
        print("回 Home 失败")
    
    robot.stop()
    wait_user_confirm()


def test_move_joint(robot: FlexivRobot):
    """测试关节运动"""
    print_separator("测试 4: 阻塞式关节运动")
    robot.enable()
    
    current = robot.get_joint_positions()
    print(f"当前 J6: {math.degrees(current[5]):.2f}°")
    
    # 目标：J6 转 20 度
    target = current.copy()
    target[5] += math.radians(20)
    print(f"目标 J6: {math.degrees(target[5]):.2f}° (转动 +20°)")
    
    wait_user_confirm("确认安全后按 Enter 开始运动...")
    
    print("开始关节运动...")
    success = robot.move_joint(target, timeout=15.0)
    
    final = robot.get_joint_positions()
    print(f"运动结果: {'成功' if success else '失败'}")
    print(f"最终 J6: {math.degrees(final[5]):.2f}°")
    
    # 返回
    print("\n返回原位...")
    robot.move_joint(current, timeout=15.0)
    
    robot.stop()
    wait_user_confirm()


def test_move_cartesian(robot: FlexivRobot):
    """测试笛卡尔运动"""
    print_separator("测试 5: 阻塞式笛卡尔运动")
    robot.enable()
    
    current = robot.get_tcp_pose()
    print(f"当前 TCP Z: {current[2]*1000:.1f} mm")
    
    # 目标：Z 方向上移 5cm
    target = current.copy()
    target[2] += 0.05
    print(f"目标 TCP Z: {target[2]*1000:.1f} mm (上移 50mm)")
    
    wait_user_confirm("确认安全后按 Enter 开始运动...")
    
    print("开始笛卡尔运动...")
    success = robot.move_cartesian(target, timeout=15.0)
    
    final = robot.get_tcp_pose()
    print(f"运动结果: {'成功' if success else '失败'}")
    print(f"最终 TCP Z: {final[2]*1000:.1f} mm")
    
    # 返回
    print("\n返回原位...")
    robot.move_cartesian(current, timeout=15.0)
    
    robot.stop()
    wait_user_confirm()


def test_move_async(robot: FlexivRobot):
    """测试非阻塞运动"""
    print_separator("测试 6: 非阻塞关节运动")
    robot.enable()
    
    current = robot.get_joint_positions()
    target = current.copy()
    target[5] += math.radians(15)
    
    print(f"将 J6 从 {math.degrees(current[5]):.2f}° 转到 {math.degrees(target[5]):.2f}°")
    
    wait_user_confirm("确认安全后按 Enter 开始运动...")
    
    completed = {"success": None}
    
    def callback(success):
        completed["success"] = success
        print(f"\n[回调] 运动完成: {'成功' if success else '失败'}")
    
    print("启动非阻塞运动...")
    robot.move_joint_async(target, callback=callback, timeout=15.0)
    
    # 在运动过程中打印状态
    print("运动中，实时显示 J6 角度:")
    while robot.is_moving():
        j6 = math.degrees(robot.get_joint_positions()[5])
        print(f"  J6: {j6:.2f}°", end="\r")
        time.sleep(0.1)
    print()
    
    # 返回
    print("\n返回原位...")
    robot.move_joint(current, timeout=15.0)
    
    robot.stop()
    wait_user_confirm()


def test_cancel_motion(robot: FlexivRobot):
    """测试取消运动"""
    print_separator("测试 7: 取消运动")
    robot.enable()
    
    current = robot.get_joint_positions()
    target = current.copy()
    target[5] += math.radians(30)
    
    print(f"将启动一个较长的运动（J6 +30°），然后在 1 秒后取消")
    
    wait_user_confirm("确认安全后按 Enter 开始...")
    
    robot.move_joint_async(target, timeout=30.0)
    
    print("运动已启动，1 秒后取消...")
    time.sleep(1.0)
    
    robot.cancel_motion()
    print("已发送取消指令")
    
    time.sleep(0.3)
    stopped_pos = robot.get_joint_positions()
    print(f"停止位置 J6: {math.degrees(stopped_pos[5]):.2f}°")
    
    # 返回原位
    print("\n返回原位...")
    robot.move_joint(current, timeout=15.0)
    
    robot.stop()
    wait_user_confirm()


def test_force_control(robot: FlexivRobot):
    """测试力控制"""
    print_separator("测试 8: 力位混合控制")
    robot.enable()
    
    print("先清零力/力矩传感器...")
    robot.zero_ft_sensor(timeout=30.0)
    
    current = robot.get_tcp_pose()
    
    print("\n将执行 Z 轴力控制（5N 下压），X/Y 轴位置保持")
    print("持续 5 秒后自动结束")
    
    wait_user_confirm("确认安全后按 Enter 开始...")
    
    robot.move_cartesian_force(
        target_pose=current,
        target_wrench=[0, 0, 5, 0, 0, 0],
        force_axes=[False, False, True, False, False, False],
        force_frame=ForceControlFrame.WORLD,
        timeout=5.0
    )
    
    print("力控制测试完成")
    
    robot.stop()
    wait_user_confirm()


def test_gripper_control(robot: FlexivRobot):
    """测试夹爪控制"""
    print_separator("测试 9: 夹爪控制")
    robot.enable()
    
    # 从配置文件读取默认夹爪名称
    default_gripper = "Flexiv-GN01"
    try:
        config = load_robot_config("robot.yaml")
        default_gripper = config.get("gripper_name", default_gripper)
    except FileNotFoundError:
        pass
    
    # 获取夹爪名称
    gripper_name = input(f"请输入夹爪名称 (默认: {default_gripper}): ").strip()
    if not gripper_name:
        gripper_name = default_gripper
    
    try:
        # 使能夹爪
        print(f"\n使能夹爪 [{gripper_name}]...")
        robot.gripper_enable(gripper_name)
        
        # 显示参数
        params = robot.get_gripper_params()
        print(f"\n夹爪参数:")
        print(f"  名称: {params.name}")
        print(f"  宽度范围: {params.min_width*1000:.1f} - {params.max_width*1000:.1f} mm")
        print(f"  速度范围: {params.min_vel*1000:.1f} - {params.max_vel*1000:.1f} mm/s")
        print(f"  力范围: {params.min_force:.1f} - {params.max_force:.1f} N")
        
        # 测试张开
        print("\n[1] 测试张开夹爪...")
        wait_user_confirm("按 Enter 张开夹爪...")
        robot.gripper_open(wait=True)
        states = robot.get_gripper_states()
        print(f"  当前宽度: {states.width*1000:.1f} mm")
        print(f"  当前力: {states.force:.1f} N")
        
        # 测试移动到中间位置
        mid_width = (params.min_width + params.max_width) / 2
        print(f"\n[2] 测试移动到中间位置 ({mid_width*1000:.1f} mm)...")
        wait_user_confirm("按 Enter 继续...")
        robot.gripper_move(mid_width, wait=True)
        states = robot.get_gripper_states()
        print(f"  当前宽度: {states.width*1000:.1f} mm")
        
        # 测试非阻塞闭合
        print("\n[3] 测试非阻塞闭合...")
        wait_user_confirm("按 Enter 继续...")
        robot.gripper_close(force_limit=20.0, wait=False)
        print("  夹爪开始闭合...")
        
        while robot.is_gripper_moving():
            states = robot.get_gripper_states()
            print(f"  宽度: {states.width*1000:.1f} mm, 力: {states.force:.1f} N, 运动中: {states.is_moving}")
            time.sleep(0.2)
        
        states = robot.get_gripper_states()
        print(f"  闭合完成 - 宽度: {states.width*1000:.1f} mm, 力: {states.force:.1f} N")
        
        # 测试张开
        print("\n[4] 张开夹爪...")
        robot.gripper_open(wait=True)
        print("  夹爪已张开")
        
        # 禁用夹爪
        print("\n禁用夹爪...")
        robot.gripper_disable()
        print("夹爪测试完成")
        
    except Exception as e:
        print(f"\n夹爪测试出错: {e}")
        if robot.is_gripper_enabled():
            robot.gripper_stop()
            robot.gripper_disable()
    
    robot.stop()
    wait_user_confirm()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  FlexivRobot 交互式功能测试")
    print("=" * 60)
    print("\n警告：此测试会使机器人实际运动，请确保：")
    print("  1. 机器人周围无障碍物")
    print("  2. 急停按钮在手边")
    print("  3. 随时准备按下急停")
    
    # 获取机器人序列号
    try:
        robot_sn = load_robot_sn_from_yaml("robot.yaml")
    except FileNotFoundError:
        robot_sn = input("\n请输入机器人序列号: ")
    
    print(f"\n机器人序列号: {robot_sn}")
    wait_user_confirm("\n按 Enter 开始测试，输入 'q' 退出...")
    
    try:
        # 1. 初始化
        robot = test_initialization(robot_sn)
        
        # 2. 状态获取
        test_get_states(robot)
        
        # 3. 回 Home
        test_go_home(robot)
        
        # 4. 关节运动
        test_move_joint(robot)
        
        # 5. 笛卡尔运动
        test_move_cartesian(robot)
        
        # 6. 非阻塞运动
        test_move_async(robot)
        
        # 7. 取消运动
        test_cancel_motion(robot)
        
        # 8. 力控制（可选）
        response = input("\n是否测试力控制? (y/n): ")
        if response.lower() == 'y':
            test_force_control(robot)
        
        # 9. 夹爪控制（可选）
        response = input("\n是否测试夹爪控制? (y/n): ")
        if response.lower() == 'y':
            test_gripper_control(robot)
        
        # 结束
        print_separator("测试完成")
        print("所有测试已完成！")
        
        robot.enable()
        robot.go_home(wait=True)
        robot.stop()
        
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        raise


if __name__ == "__main__":
    main()

