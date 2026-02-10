#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角度编码器夹爪遥操作程序 (Modbus 夹爪版)

使用磁编码器读取手柄角度，实时映射控制 Modbus RS485 夹爪开合。

功能：
- 编码器角度 -> 夹爪位置 的线性映射
- 空格键离合：按住暂停，松开恢复
- 支持校准零点、调节参数

使用方法:
    python encoder_gripper_teleop_modbus.py --encoder-port /dev/ttyUSB0 --gripper-port /dev/ttyUSB1
"""

import time
import threading
import argparse
import sys
from typing import Optional

try:
    from pynput import keyboard
except ImportError:
    print("未找到 pynput，尝试安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pynput"])
    from pynput import keyboard

from t265_action_reader import ActionReader
from gripper_controller import GripperController


class EncoderGripperTeleopModbus:
    """
    编码器夹爪遥操作控制器 (Modbus 版)
    
    将编码器角度映射到夹爪位置：
    - 0° (闭合) -> max_position (夹爪闭合)
    - max_angle° (张开) -> min_position (夹爪张开)
    """
    
    def __init__(
        self,
        action_reader: ActionReader,
        gripper: GripperController,
        control_frequency: float = 20.0,
        max_angle: float = 45.0,
        min_angle: float = 0.0,
        min_position: int = 0,
        max_position: int = 6000,
        speed: int = 100,
        force: int = 100,
        verbose: bool = True
    ):
        """
        初始化编码器夹爪遥操作控制器
        
        Args:
            action_reader: 编码器读取器
            gripper: Modbus 夹爪控制器实例
            control_frequency: 控制频率 [Hz]
            max_angle: 最大张开角度 [度]，对应夹爪最小位置（张开）
            min_angle: 最小角度 [度]，对应夹爪最大位置（闭合）
            min_position: 夹爪张开位置（最小值），默认 0
            max_position: 夹爪闭合位置（最大值），默认 6000
            speed: 夹爪移动速度 (0-100)
            force: 夹爪力 (0-100)
            verbose: 是否打印详细信息
        """
        self.reader = action_reader
        self.gripper = gripper
        self.dt = 1.0 / control_frequency
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.min_position = min_position
        self.max_position = max_position
        self.speed = speed
        self.force = force
        self.verbose = verbose
        
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # 离合状态：默认跟随 (True)，按住空格断开 (False)
        self.is_following = True
        
        # 上次发送时间（用于限频）
        self.last_send_time = 0.0
        self.send_interval = 1.0 / control_frequency
        
        # 当前目标位置（用于打印）
        self.current_position = 0
        self.current_angle = 0.0
        
        # 上次发送的位置（避免重复发送相同位置）
        self.last_sent_position = -1
        self.position_threshold = 10  # 位置变化阈值，避免频繁发送
        
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
    
    def _angle_to_position(self, angle: float) -> int:
        """
        将编码器角度映射到夹爪位置
        
        Args:
            angle: 编码器角度 [度]
            
        Returns:
            夹爪位置 (整数)
        
        映射关系：
        - min_angle (闭合) -> max_position (夹爪闭合)
        - max_angle (张开) -> min_position (夹爪张开)
        """
        # 限制角度范围
        angle = max(self.min_angle, min(self.max_angle, angle))
        
        # 线性映射: min_angle -> max_position, max_angle -> min_position
        # 反向映射：角度越大，位置越小（夹爪越张开）
        if self.max_angle > self.min_angle:
            ratio = (angle - self.min_angle) / (self.max_angle - self.min_angle)
        else:
            ratio = 0
        
        position = int(self.max_position - ratio * (self.max_position - self.min_position))
        
        return position
    
    def _loop(self):
        """主控制循环"""
        while self._running:
            start_time = time.time()
            
            if self.is_following:
                try:
                    # 读取编码器角度
                    action = self.reader.get_action()
                    angle = action.joint_angle  # 校准后的角度
                    
                    # 映射到夹爪位置
                    target_position = self._angle_to_position(angle)
                    
                    # 更新状态（用于打印）
                    self.current_angle = angle
                    self.current_position = target_position
                    
                    # 发送夹爪指令（限频 + 位置变化阈值）
                    now = time.time()
                    position_changed = abs(target_position - self.last_sent_position) > self.position_threshold
                    
                    if now - self.last_send_time >= self.send_interval and position_changed:
                        self.gripper.set_position_32bit(target_position)
                        self.gripper.trigger_motion()
                        self.last_send_time = now
                        self.last_sent_position = target_position
                    
                    # 打印状态
                    if self.verbose:
                        print(f"\r[Follow] 角度: {angle:+7.2f}° -> 位置: {target_position:5d} "
                              f"(范围: {self.min_position}-{self.max_position})", end='')
                        
                except Exception as e:
                    print(f"\n[Error] {e}")
            else:
                # 断开状态，不发送指令
                if self.verbose:
                    print(f"\r[Hold] 角度: {self.current_angle:+7.2f}° | 位置保持: {self.current_position:5d}  ", end='')
            
            # 频率控制
            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="角度编码器夹爪遥操作程序 (Modbus 版)")
    parser.add_argument('--encoder-port', default='/dev/ttyUSB0', help='编码器串口')
    parser.add_argument('--gripper-port', default='/dev/ttyUSB1', help='夹爪串口 (自动检测)')
    parser.add_argument('--max-angle', type=float, default=45.0, help='最大张开角度 [度]，默认 45°')
    parser.add_argument('--min-angle', type=float, default=0.0, help='最小闭合角度 [度]，默认 0°')
    parser.add_argument('--min-position', type=int, default=0, help='夹爪张开位置，默认 0')
    parser.add_argument('--max-position', type=int, default=6000, help='夹爪闭合位置，默认 6000')
    parser.add_argument('--speed', type=int, default=100, help='夹爪速度 (0-100)，默认 100')
    parser.add_argument('--force', type=int, default=20, help='夹爪力 (0-100)，默认 100')
    parser.add_argument('--freq', type=float, default=10.0, help='控制频率 [Hz]，默认 50')
    parser.add_argument('--reverse', action='store_true', help='反转编码器方向（张开=减少）')
    parser.add_argument('--angle-scale', type=float, default=2.0, help='编码器角度倍率，默认 2.0')
    args = parser.parse_args()
    
    # 1. 打印标题
    print("=" * 50)
    print("  角度编码器夹爪遥操作 (Modbus 版)")
    print("=" * 50)
    
    print(f"编码器端口: {args.encoder_port}")
    print(f"夹爪端口: {args.gripper_port or '自动检测'}")
    print(f"位置范围: {args.min_position} - {args.max_position}")
    print(f"角度范围: {args.min_angle}° - {args.max_angle}°")
    
    # 2. 初始化编码器读取器（只用编码器，不用 T265）
    print("\n正在连接编码器...")
    direction = -1 if args.reverse else 1
    reader = ActionReader(
        encoder_port=args.encoder_port,
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
    
    # 4. 初始化 Modbus 夹爪
    print("\n正在连接 Modbus 夹爪...")
    try:
        gripper = GripperController(
            port=args.gripper_port,
            baudrate=115200,
            slave_id=1,
            timeout=0.5
        )
        if not gripper.connect():
            print("夹爪连接失败，退出。")
            reader.disconnect()
            sys.exit(1)
    except Exception as e:
        print(f"夹爪连接失败: {e}")
        reader.disconnect()
        sys.exit(1)
    
    # 5. 使能夹爪
    print("\n>>> 正在使能夹爪...")
    gripper.enable_gripper(True)
    time.sleep(0.5)
    
    # 设置运动参数
    gripper.set_motion_parameters(
        speed=args.speed,
        force=args.force,
        accel=100,
        decel=100
    )
    
    # 读取当前位置
    current_pos = gripper.get_current_position()
    print(f"    当前位置: {current_pos}")
    print(f"    速度: {args.speed}%")
    print(f"    力: {args.force}%")
    
    # 6. 启动遥操作控制器
    controller = EncoderGripperTeleopModbus(
        action_reader=reader,
        gripper=gripper,
        control_frequency=args.freq,
        max_angle=args.max_angle,
        min_angle=args.min_angle,
        min_position=args.min_position,
        max_position=args.max_position,
        speed=args.speed,
        force=args.force,
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
        gripper.enable_gripper(False)
        gripper.disconnect()
        print("程序已退出")


if __name__ == "__main__":
    main()

