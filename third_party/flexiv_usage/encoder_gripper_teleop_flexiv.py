#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角度编码器夹爪遥操作程序

使用磁编码器读取手柄角度，实时映射控制 Flexiv 夹爪开合。

功能：
- 编码器角度 -> 夹爪开口宽度 的线性映射
- 空格键离合：按住暂停，松开恢复
- 支持校准零点、调节参数

使用方法:
    python encoder_gripper_teleop.py --port /dev/ttyUSB0
"""

import time
import threading
import argparse
import sys
import numpy as np
from typing import Optional

try:
    from pynput import keyboard
except ImportError:
    print("未找到 pynput，尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pynput"])
    from pynput import keyboard

from t265_action_reader import ActionReader
from flexiv_robot import FlexivRobot, load_robot_config


class EncoderGripperTeleop:
    """
    编码器夹爪遥操作控制器
    
    将编码器角度映射到夹爪开口宽度：
    - 0° (闭合) -> min_width
    - max_angle° (张开) -> max_width
    """
    
    def __init__(
        self,
        action_reader: ActionReader,
        flexiv_robot: FlexivRobot,
        control_frequency: float = 20.0,
        max_angle: float = 45.0,
        min_angle: float = 0.0,
        velocity: float = 0.1,
        force_limit: Optional[float] = None,
        verbose: bool = True
    ):
        """
        初始化编码器夹爪遥操作控制器
        
        Args:
            action_reader: 编码器读取器
            flexiv_robot: Flexiv 机器人实例
            control_frequency: 控制频率 [Hz]
            max_angle: 最大张开角度 [度]，对应夹爪最大宽度
            min_angle: 最小角度 [度]，对应夹爪最小宽度
            velocity: 夹爪移动速度 [m/s]
            force_limit: 夹爪力限 [N]，None 使用默认值
            verbose: 是否打印详细信息
        """
        self.reader = action_reader
        self.robot = flexiv_robot
        self.dt = 1.0 / control_frequency
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.velocity = velocity
        self.force_limit = force_limit
        self.verbose = verbose
        
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # 离合状态：默认跟随 (True)，按住空格断开 (False)
        self.is_following = True
        
        # 上次发送时间（用于限频）
        self.last_send_time = 0.0
        self.send_interval = 1.0 / control_frequency
        
        # 当前目标宽度（用于打印）
        self.current_width = 0.0
        self.current_angle = 0.0
        
    def start(self):
        """启动控制循环"""
        if self._running:
            return
        
        # 启动键盘监听
        self.kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.kb_listener.start()
        
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Teleop] 控制循环已启动")
    
    def stop(self):
        """停止控制循环"""
        self._running = False
        if hasattr(self, 'kb_listener') and self.kb_listener:
            self.kb_listener.stop()
        if self._thread:
            self._thread.join()
        print("[Teleop] 控制循环已停止")
    
    def _on_press(self, key):
        """空格按下 -> 断开控制"""
        if key == keyboard.Key.space:
            if self.is_following:
                self.is_following = False
                if self.verbose:
                    print("\n[Clutch] 断开控制 (Space Pressed)")
    
    def _on_release(self, key):
        """空格松开 -> 恢复跟随"""
        if key == keyboard.Key.space:
            self.is_following = True
            if self.verbose:
                print("\n[Clutch] 恢复跟随 (Space Released)")
    
    def _angle_to_width(self, angle: float) -> float:
        """
        将编码器角度映射到夹爪宽度
        
        Args:
            angle: 编码器角度 [度]
            
        Returns:
            夹爪宽度 [m]
        """
        # 获取夹爪参数
        params = self.robot.get_gripper_params()
        min_width = params.min_width
        max_width = params.max_width
        
        # 限制角度范围
        angle = max(self.min_angle, min(self.max_angle, angle))
        
        # 线性映射: min_angle -> min_width, max_angle -> max_width
        ratio = (angle - self.min_angle) / (self.max_angle - self.min_angle) if self.max_angle > self.min_angle else 0
        width = min_width + ratio * (max_width - min_width)
        
        return width
    
    def _loop(self):
        """主控制循环"""
        while self._running:
            start_time = time.time()
            
            if self.is_following:
                try:
                    # 读取编码器角度
                    action = self.reader.get_action()
                    angle = action.joint_angle  # 校准后的角度
                    
                    # 映射到夹爪宽度
                    target_width = self._angle_to_width(angle)
                    
                    # 更新状态（用于打印）
                    self.current_angle = angle
                    self.current_width = target_width
                    
                    # 发送夹爪指令（限频）
                    now = time.time()
                    if now - self.last_send_time >= self.send_interval:
                        self.robot.gripper_move(
                            width=target_width,
                            velocity=self.velocity,
                            force_limit=self.force_limit,
                            wait=False
                        )
                        self.last_send_time = now
                    
                    # 打印状态
                    if self.verbose:
                        params = self.robot.get_gripper_params()
                        print(f"\r[Follow] 角度: {angle:+7.2f}° -> 宽度: {target_width*1000:.1f}mm "
                              f"(范围: {params.min_width*1000:.1f}-{params.max_width*1000:.1f}mm)", end='')
                        
                except Exception as e:
                    print(f"\n[Error] {e}")
            else:
                # 断开状态，不发送指令
                if self.verbose:
                    print(f"\r[Hold] 角度: {self.current_angle:+7.2f}° | 宽度保持: {self.current_width*1000:.1f}mm  ", end='')
            
            # 频率控制
            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="角度编码器夹爪遥操作程序")
    parser.add_argument('--robot-config', default='robot.yaml', help='机器人配置文件')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='编码器串口')
    parser.add_argument('--max-angle', type=float, default=45.0, help='最大张开角度 [度]，默认 45°')
    parser.add_argument('--min-angle', type=float, default=0.0, help='最小闭合角度 [度]，默认 0°')
    parser.add_argument('--velocity', type=float, default=0.1, help='夹爪移动速度 [m/s]，默认 0.1')
    parser.add_argument('--freq', type=float, default=50.0, help='控制频率 [Hz]，默认 20')
    parser.add_argument('--reverse', action='store_true', help='反转编码器方向（张开=减少）')
    parser.add_argument('--angle-scale', type=float, default=2.0, help='编码器角度倍率，默认 2.0')
    args = parser.parse_args()
    
    # 1. 读取机器人配置
    print("=" * 50)
    print("  角度编码器夹爪遥操作")
    print("=" * 50)
    
    try:
        robot_config = load_robot_config(args.robot_config)
        robot_sn = robot_config['robot_sn']
        gripper_name = robot_config.get('gripper_name')
    except Exception as e:
        print(f"无法读取 {args.robot_config}：{e}")
        sys.exit(1)
    
    if not gripper_name:
        print("配置文件中未指定 gripper_name，请检查。")
        sys.exit(1)
    
    print(f"机器人序列号: {robot_sn}")
    print(f"夹爪名称: {gripper_name}")
    print(f"编码器端口: {args.port}")
    
    # 2. 初始化编码器读取器（只用编码器，不用 T265）
    print("\n正在连接编码器...")
    direction = -1 if args.reverse else 1
    reader = ActionReader(
        encoder_port=args.port,
        use_t265=False,  # 不使用 T265
        use_encoder=True,
        angle_direction=direction,
        angle_scale=args.angle_scale
    )
    reader.connect()
    
    if not reader.use_encoder:
        print("编码器连接失败，退出。")
        sys.exit(1)
    
    # 3. 校准编码器零点
    print("\n>>> 编码器零点校准")
    print("    请将手柄移动到【闭合/初始位置】")
    time.sleep(2.0)
    reader.calibrate(print_info=True)
    
    # 4. 初始化机器人
    print("\n正在连接 Flexiv Robot...")
    robot = FlexivRobot(robot_sn)
    
    # 5. 使能夹爪
    print(f"\n>>> 正在使能夹爪 [{gripper_name}]...")
    try:
        robot.gripper_enable(gripper_name, switch_tool=True)
    except Exception as e:
        print(f"夹爪使能失败: {e}")
        reader.disconnect()
        sys.exit(1)
    
    # 打印夹爪参数
    params = robot.get_gripper_params()
    print(f"    宽度范围: {params.min_width*1000:.1f} - {params.max_width*1000:.1f} mm")
    print(f"    速度范围: {params.min_vel*1000:.1f} - {params.max_vel*1000:.1f} mm/s")
    print(f"    力范围: {params.min_force:.1f} - {params.max_force:.1f} N")
    
    # 6. 启动遥操作控制器
    controller = EncoderGripperTeleop(
        action_reader=reader,
        flexiv_robot=robot,
        control_frequency=args.freq,
        max_angle=args.max_angle,
        min_angle=args.min_angle,
        velocity=args.velocity,
        verbose=True
    )
    
    controller.start()
    
    print("\n" + "=" * 50)
    print("  夹爪遥操作已就绪")
    print("  [Space] 松开 (默认): 跟随模式 (Follow)")
    print("  [Space] 按住:        暂停模式 (Hold)")
    print("  [Ctrl+C] 退出程序")
    print("=" * 50 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n正在停止...")
    finally:
        controller.stop()
        reader.disconnect()
        robot.stop()
        print("程序已退出")


if __name__ == "__main__":
    main()

