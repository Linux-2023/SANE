#!/usr/bin/env python3
"""
夹爪控制类
支持百分比控制和角度控制，自动从配置文件加载标定数据
"""

import sys
import json
import time
from typing import Optional

sys.path.append(".")
from scservo_sdk import *


class Gripper:
    """
    夹爪控制器
    
    支持百分比控制和角度控制。
    角度定义：0° = 完全关闭，max_angle = 完全打开
    注意：夹爪角度 = 电机角度 × 2
    
    使用示例:
        # 方式1: 使用 with 语句（推荐）
        with Gripper(side='left') as gripper:
            gripper.open()
            gripper.close()
            gripper.set_angle(30)
            gripper.set_percent(50)
        
        # 方式2: 手动管理连接
        gripper = Gripper(side='left')
        gripper.connect()
        gripper.open()
        gripper.disconnect()
    """
    
    CONFIG_FILE = 'gripper_config.json'
    
    def __init__(self, side: str = 'left', config_file: str = None):
        """
        初始化夹爪
        
        Args:
            side: 'left' 或 'right'，选择左夹爪或右夹爪
            config_file: 配置文件路径，默认使用 gripper_config.json
        """
        self.side = side
        self.config_file = config_file or self.CONFIG_FILE
        
        # 从配置文件加载参数
        self._load_config()
        
        # 初始化通信
        self._port_handler = PortHandler(self._port)
        self._packet_handler = sms_sts(self._port_handler)
        self._connected = False
    
    # ==================== 配置加载 ====================
    
    def _load_config(self):
        """从配置文件加载标定数据"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_file}\n"
                f"请先运行标定: python3 gripper_calibration.py --side {self.side}"
            )
        
        if self.side not in config.get('grippers', {}):
            raise ValueError(
                f"配置文件中没有 {self.side} 夹爪的数据\n"
                f"请先运行标定: python3 gripper_calibration.py --side {self.side}"
            )
        
        # 通信参数
        # 波特率从顶级配置读取（所有夹爪共享）
        self._baudrate = config.get('baudrate', 1000000)
        
        # 夹爪参数（每个夹爪有自己的端口）
        gripper_config = config['grippers'][self.side]
        self._port = gripper_config.get('port', '/dev/ttyACM0')  # 从夹爪配置读取端口
        self._servo_id = gripper_config['servo_id']
        self._min_position = gripper_config['min_position']  # 关闭位置
        self._max_position = gripper_config['max_position']  # 打开位置
        
        # 计算角度范围
        # 电机角度 = 位置差 / 4096 * 360
        # 夹爪角度 = 电机角度 × 2
        motor_angle = abs(self._max_position - self._min_position) / 4096 * 360
        self._max_angle = motor_angle * 2  # 夹爪最大角度
    
    # ==================== 连接管理 ====================
    
    def connect(self) -> bool:
        """
        连接夹爪
        
        Returns:
            bool: 是否连接成功
        """
        if self._connected:
            return True
        
        if not self._port_handler.openPort():
            print(f"❌ 无法打开串口: {self._port}")
            return False
        
        if not self._port_handler.setBaudRate(self._baudrate):
            print(f"❌ 无法设置波特率: {self._baudrate}")
            return False
        
        self._connected = True
        return True
    
    def disconnect(self):
        """断开连接"""
        if self._connected:
            self._port_handler.closePort()
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected
    
    # ==================== 属性 ====================
    
    @property
    def servo_id(self) -> int:
        """舵机ID"""
        return self._servo_id
    
    @property
    def max_angle(self) -> float:
        """夹爪最大开合角度（度）"""
        return self._max_angle
    
    # ==================== 位置转换 ====================
    
    def _percent_to_position(self, percent: float) -> int:
        """百分比转电机位置
        
        注意：如果 min_position > max_position（反向安装），会自动处理
        """
        percent = max(0, min(100, percent))
        
        if self._min_position < self._max_position:
            # 正常情况
            return int(self._min_position + (self._max_position - self._min_position) * percent / 100)
        else:
            # 反向情况：百分比越大，位置越小
            return int(self._min_position - (self._min_position - self._max_position) * percent / 100)
    
    def _position_to_percent(self, position: int) -> float:
        """电机位置转百分比
        
        注意：如果 min_position > max_position（反向安装），会自动处理
        """
        if self._max_position == self._min_position:
            return 0
        
        # 确定实际的最小和最大位置（处理反向情况）
        actual_min = min(self._min_position, self._max_position)
        actual_max = max(self._min_position, self._max_position)
        
        # 计算百分比（0% = 关闭位置，100% = 打开位置）
        if self._min_position < self._max_position:
            # 正常情况：min_position 对应 0%（关闭），max_position 对应 100%（打开）
            percent = (position - self._min_position) / (self._max_position - self._min_position) * 100
        else:
            # 反向情况：min_position 对应 0%（关闭），但 min_position > max_position
            # 所以 position 越小，百分比越大
            percent = (self._min_position - position) / (self._min_position - self._max_position) * 100
        
        return max(0, min(100, percent))
    
    def _angle_to_percent(self, angle: float) -> float:
        """夹爪角度转百分比"""
        if self._max_angle == 0:
            return 0
        return max(0, min(100, angle / self._max_angle * 100))
    
    def _percent_to_angle(self, percent: float) -> float:
        """百分比转夹爪角度"""
        return self._max_angle * percent / 100
    
    # ==================== 读取状态 ====================
    
    def _read_position(self) -> int:
        """读取电机当前位置"""
        if not self._connected:
            return -1
        pos, result, error = self._packet_handler.read2ByteTxRx(
            self._servo_id, SMS_STS_PRESENT_POSITION_L
        )
        return pos if result == COMM_SUCCESS else -1
    
    def get_percent(self) -> Optional[float]:
        """
        获取当前开度百分比
        
        Returns:
            float: 0-100，0=关闭，100=打开。失败返回 None
        """
        pos = self._read_position()
        if pos < 0:
            return None
        return self._position_to_percent(pos)
    
    def get_angle(self) -> Optional[float]:
        """
        获取当前夹爪角度（度）
        
        Returns:
            float: 0 ~ max_angle。失败返回 None
        """
        percent = self.get_percent()
        if percent is None:
            return None
        return self._percent_to_angle(percent)
    
    def is_moving(self) -> bool:
        """检查是否在移动中"""
        if not self._connected:
            return False
        moving, result, error = self._packet_handler.ReadMoving(self._servo_id)
        return bool(moving) if result == COMM_SUCCESS else False
    
    # ==================== 运动控制 ====================
    
    def _move_to_position(self, position: int, speed: int = 1500, acc: int = 50) -> bool:
        """移动到指定电机位置"""
        if not self._connected:
            return False
        
        speed = max(0, min(4095, speed))
        acc = max(0, min(255, acc))
        
        result, error = self._packet_handler.WritePosEx(
            self._servo_id, position, speed, acc
        )
        return result == COMM_SUCCESS
    
    def set_percent(self, percent: float, speed: int = 1500, acc: int = 50, 
                    wait: bool = False) -> bool:
        """
        设置夹爪开度（百分比）
        
        Args:
            percent: 目标开度 0-100（0=关闭，100=打开）
            speed: 速度 (0-4095)，默认 1500
            acc: 加速度 (0-255)，默认 50
            wait: 是否等待移动完成
            
        Returns:
            bool: 是否成功
        """
        position = self._percent_to_position(percent)
        success = self._move_to_position(position, speed, acc)
        
        if success and wait:
            self.wait_until_done()
        
        return success
    
    def set_normalized(self, normalized: float, speed: int = 1500, acc: int = 50,
                       wait: bool = False) -> bool:
        """
        设置夹爪开度（归一化值）
        
        Args:
            normalized: 目标开度 0-1（0=关闭，1=打开）
            speed: 速度 (0-4095)，默认 1500
            acc: 加速度 (0-255)，默认 50
            wait: 是否等待移动完成
            
        Returns:
            bool: 是否成功
        """
        # 将归一化值 (0-1) 转换为百分比 (0-100)
        percent = max(0.0, min(1.0, normalized)) * 100.0
        return self.set_percent(percent, speed, acc, wait)
    
    def set_angle(self, angle: float, speed: int = 1500, acc: int = 50,
                  wait: bool = False) -> bool:
        """
        设置夹爪角度
        
        Args:
            angle: 目标角度（0 ~ max_angle）
            speed: 速度 (0-4095)，默认 1500
            acc: 加速度 (0-255)，默认 50
            wait: 是否等待移动完成
            
        Returns:
            bool: 是否成功
        """
        angle = max(0, min(self._max_angle, angle))
        percent = self._angle_to_percent(angle)
        return self.set_percent(percent, speed, acc, wait)
    
    def open(self, speed: int = 1500, acc: int = 50, wait: bool = True) -> bool:
        """
        打开夹爪（100%开度）
        
        Args:
            speed: 速度
            acc: 加速度
            wait: 是否等待完成
        """
        return self.set_percent(100, speed, acc, wait)
    
    def close(self, speed: int = 1500, acc: int = 50, wait: bool = True) -> bool:
        """
        关闭夹爪（0%开度）
        
        Args:
            speed: 速度
            acc: 加速度
            wait: 是否等待完成
        """
        return self.set_percent(0, speed, acc, wait)
    
    def wait_until_done(self, timeout: float = 5.0) -> bool:
        """
        等待移动完成
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否在超时前完成
        """
        start = time.time()
        time.sleep(0.1)
        
        while time.time() - start < timeout:
            if not self.is_moving():
                return True
            time.sleep(0.05)
        
        return False
    
    # ==================== 状态信息 ====================
    
    def get_status(self) -> dict:
        """
        获取夹爪完整状态
        
        Returns:
            dict: 包含角度、百分比、是否移动等信息
        """
        percent = self.get_percent()
        angle = self._percent_to_angle(percent) if percent is not None else None
        
        return {
            'side': self.side,
            'servo_id': self._servo_id,
            'connected': self._connected,
            'percent': percent,
            'angle': angle,
            'max_angle': self._max_angle,
            'moving': self.is_moving() if self._connected else None
        }
    
    def __str__(self) -> str:
        side_name = "左" if self.side == "left" else "右"
        if not self._connected:
            return f"<Gripper {side_name} (未连接)>"
        
        percent = self.get_percent()
        if percent is not None:
            angle = self._percent_to_angle(percent)
            return f"<Gripper {side_name}: {angle:.1f}° ({percent:.0f}%)>"
        return f"<Gripper {side_name} (读取失败)>"
    
    # ==================== 上下文管理器 ====================
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ==================== 使用示例 ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='夹爪控制示例')
    parser.add_argument('--side', '-s', choices=['left', 'right'], default='left')
    parser.add_argument('--action', '-a', help='动作: open, close, 或角度值')
    args = parser.parse_args()
    
    with Gripper(side=args.side) as gripper:
        print(f"\n夹爪信息:")
        print(f"  侧边: {'左' if args.side == 'left' else '右'}")
        print(f"  舵机ID: {gripper.servo_id}")
        print(f"  最大角度: {gripper.max_angle:.1f}°")
        print(f"  当前状态: {gripper}")
        
        if args.action:
            if args.action == 'open':
                print("\n打开夹爪...")
                gripper.open()
            elif args.action == 'close':
                print("\n关闭夹爪...")
                gripper.close()
            else:
                try:
                    angle = float(args.action)
                    print(f"\n移动到 {angle}°...")
                    gripper.set_angle(angle, wait=True)
                except ValueError:
                    print(f"无效动作: {args.action}")
            
            print(f"  完成: {gripper}")


