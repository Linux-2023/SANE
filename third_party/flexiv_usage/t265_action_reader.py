#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时动作数据获取类

同时读取:
- T265 相机位姿 (位置 + 姿态)
- 磁编码器关节角度

使用示例:
    from action_reader import ActionReader
    
    # 初始化
    reader = ActionReader(encoder_port='/dev/ttyUSB0')
    reader.connect()
    
    # 获取数据
    action = reader.get_action()
    print(f"位置: {action['position']}")
    print(f"角度: {action['joint_angle']}")
    
    # 关闭
    reader.disconnect()
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import threading


@dataclass
class ActionData:
    """动作数据结构"""
    # 位姿数据 (来自T265)
    position: np.ndarray      # [x, y, z] 位置 (米)
    quaternion: np.ndarray    # [qw, qx, qy, qz] 四元数
    velocity: np.ndarray      # [vx, vy, vz] 线速度
    euler: np.ndarray         # [roll, pitch, yaw] 欧拉角 (度)
    
    # 关节角度 (来自编码器)
    joint_angle: float        # 校准后的关节角度 (度) - 相对于零点的张开角度
    joint_angle_raw: float    # 原始角度 (度) - 未校准
    joint_raw: int            # 原始编码器值
    joint_turns: int          # 圈数
    
    # 时间戳
    timestamp: float
    
    @property
    def pose(self) -> np.ndarray:
        """完整位姿 [x, y, z, qw, qx, qy, qz]"""
        return np.concatenate([self.position, self.quaternion])
    
    @property
    def pose_matrix(self) -> np.ndarray:
        """4x4 变换矩阵"""
        from scipy.spatial.transform import Rotation
        
        T = np.eye(4)
        # 四元数转旋转矩阵 (scipy用 [qx,qy,qz,qw] 格式)
        rot = Rotation.from_quat([
            self.quaternion[1], self.quaternion[2], 
            self.quaternion[3], self.quaternion[0]
        ])
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = self.position
        return T
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'pose': self.pose.tolist(),
            'position': self.position.tolist(),
            'quaternion': self.quaternion.tolist(),
            'euler': self.euler.tolist(),
            'velocity': self.velocity.tolist(),
            'joint_angle': self.joint_angle,
            'joint_angle_raw': self.joint_angle_raw,
            'joint_raw': self.joint_raw,
            'joint_turns': self.joint_turns,
            'timestamp': self.timestamp
        }
    
    def to_array(self) -> np.ndarray:
        """转换为数组 [x,y,z, qw,qx,qy,qz, angle]"""
        return np.array([
            *self.position,
            *self.quaternion,
            self.joint_angle  # 使用校准后的角度
        ])


class ActionReader:
    """
    实时动作数据读取器
    
    同时从T265相机和磁编码器读取数据
    """
    
    # 编码器默认配置
    ENCODER_RESOLUTION = 65536  # 16位分辨率
    REG_ANGLE_HIGH = 0x40
    REG_ANGLE_LOW = 0x41
    REG_TURNS = 0x44
    
    def __init__(self,
                 encoder_port: str = '/dev/ttyUSB0',
                 encoder_slave_id: int = 1,
                 encoder_baudrate: int = 115200,
                 encoder_resolution: int = 65536,
                 use_t265: bool = True,
                 use_encoder: bool = True,
                 angle_direction: int = 1,
                 angle_scale: float = 2.0):
        """
        初始化动作读取器
        
        Args:
            encoder_port: 编码器串口路径
            encoder_slave_id: 编码器从站地址
            encoder_baudrate: 编码器波特率
            encoder_resolution: 编码器分辨率
            use_t265: 是否使用T265
            use_encoder: 是否使用编码器
            angle_direction: 角度方向，1表示编码器增加=张开，-1表示编码器增加=闭合
            angle_scale: 角度倍率，默认2.0（编码器角度 * 倍率 = 实际张开角度）
        """
        self.encoder_port = encoder_port
        self.encoder_slave_id = encoder_slave_id
        self.encoder_baudrate = encoder_baudrate
        self.ENCODER_RESOLUTION = encoder_resolution
        
        self.use_t265 = use_t265
        self.use_encoder = use_encoder
        
        # 角度校准参数
        self._angle_zero_raw = 0      # 零点时的原始角度值
        self._angle_zero_turns = 0    # 零点时的圈数
        self._angle_direction = angle_direction  # 1或-1
        self._angle_scale = angle_scale  # 角度倍率
        self._calibrated = False
        
        # 累积角度追踪（处理0~360跳变）
        self._last_raw_angle = 0.0    # 上一次的原始角度
        self._accumulated_turns = 0   # 累积的圈数（自己追踪）
        self._use_internal_turns = True  # 使用内部圈数追踪
        
        # 设备对象
        self._t265_pipe = None
        self._t265_cfg = None
        self._encoder = None
        
        # 状态
        self._connected = False
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """
        连接所有设备
        
        Returns:
            是否成功连接
        """
        success = True
        
        # 连接 T265
        if self.use_t265:
            try:
                import pyrealsense2 as rs
                
                self._t265_pipe = rs.pipeline()
                self._t265_cfg = rs.config()
                self._t265_cfg.enable_stream(rs.stream.pose)
                self._t265_pipe.start(self._t265_cfg)
                
                # 等待稳定
                for _ in range(10):
                    self._t265_pipe.wait_for_frames()
                
                print("[T265] ✅ 连接成功")
            except Exception as e:
                print(f"[T265] ❌ 连接失败: {e}")
                self.use_t265 = False
                success = False
        
        # 连接编码器
        if self.use_encoder:
            try:
                import minimalmodbus
                
                self._encoder = minimalmodbus.Instrument(
                    self.encoder_port, 
                    self.encoder_slave_id
                )
                self._encoder.serial.baudrate = self.encoder_baudrate
                self._encoder.serial.bytesize = 8
                self._encoder.serial.parity = minimalmodbus.serial.PARITY_NONE
                self._encoder.serial.stopbits = 1
                self._encoder.serial.timeout = 0.5
                self._encoder.mode = minimalmodbus.MODE_RTU
                self._encoder.clear_buffers_before_each_transaction = True
                
                # 测试读取
                self._read_encoder_angle()
                print(f"[编码器] ✅ 连接成功 ({self.encoder_port})")
            except Exception as e:
                print(f"[编码器] ❌ 连接失败: {e}")
                self.use_encoder = False
                success = False
        
        self._connected = self.use_t265 or self.use_encoder
        return success
    
    def disconnect(self):
        """断开所有设备"""
        if self._t265_pipe:
            try:
                self._t265_pipe.stop()
            except:
                pass
            self._t265_pipe = None
        
        if self._encoder:
            try:
                self._encoder.serial.close()
            except:
                pass
            self._encoder = None
        
        self._connected = False
        print("[ActionReader] 已断开所有设备")
    
    def _read_t265_pose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """读取T265位姿"""
        import pyrealsense2 as rs
        import math
        
        frames = self._t265_pipe.wait_for_frames()
        pose_frame = frames.get_pose_frame()
        
        if not pose_frame:
            return (np.zeros(3), np.array([1, 0, 0, 0]), 
                    np.zeros(3), np.zeros(3))
        
        data = pose_frame.get_pose_data()
        
        # 位置
        position = np.array([
            data.translation.x,
            data.translation.y,
            data.translation.z
        ])
        
        # 四元数 [qw, qx, qy, qz]
        quaternion = np.array([
            data.rotation.w,
            data.rotation.x,
            data.rotation.y,
            data.rotation.z
        ])
        
        # 速度
        velocity = np.array([
            data.velocity.x,
            data.velocity.y,
            data.velocity.z
        ])
        
        # 欧拉角 (roll, pitch, yaw)
        w, x, y, z = data.rotation.w, data.rotation.x, data.rotation.y, data.rotation.z
        
# 1. 计算绕 X 轴 (T265 的 Pitch) 的角度
        # 数学公式对应标准 Euler 的 phi
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        angle_x = math.atan2(sinr_cosp, cosr_cosp)

        # 2. 计算绕 Y 轴 (T265 的 Yaw) 的角度
        # 数学公式对应标准 Euler 的 theta
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            # 处理万向节死锁 (Gimbal Lock) 附近的数值溢出
            angle_y = math.copysign(math.pi / 2, sinp)
        else:
            angle_y = math.asin(sinp)

        # 3. 计算绕 Z 轴 (T265 的 Roll) 的角度
        # 数学公式对应标准 Euler 的 psi
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        angle_z = math.atan2(siny_cosp, cosy_cosp)

        # --- 关键：T265 物理含义映射 ---
        # T265 坐标系定义: X=右, Y=上, Z=后
        
        pitch = math.degrees(angle_x)  # 绕 X 轴点头
        yaw   = math.degrees(angle_y)  # 绕 Y 轴转头
        roll  = math.degrees(angle_z)  # 绕 Z 轴歪头

        euler = np.array([roll, pitch, yaw])
        
        return position, quaternion, velocity, euler
    
    def _read_encoder_angle(self) -> Tuple[float, int, int]:
        """读取编码器原始角度"""
        # 读取角度
        angle_high = self._encoder.read_register(self.REG_ANGLE_HIGH, 0, 3, False)
        angle_low = self._encoder.read_register(self.REG_ANGLE_LOW, 0, 3, False)
        raw_value = (angle_high << 16) | angle_low
        angle_degrees = (raw_value / self.ENCODER_RESOLUTION) * 360.0
        
        # 读取圈数
        try:
            turns = self._encoder.read_register(self.REG_TURNS, 0, 3, False)
        except:
            turns = 0
        
        return angle_degrees, raw_value, turns
    
    def calibrate(self, print_info: bool = True) -> float:
        """
        校准角度零点
        
        将当前位置设为零点（闭合/初始位置）
        之后读取的角度将是相对于此零点的张开角度
        
        Returns:
            校准时的原始角度值
        """
        if not self.use_encoder or not self._encoder:
            print("[校准] ❌ 编码器未连接")
            return 0.0
        
        with self._lock:
            try:
                angle_raw, raw_value, turns = self._read_encoder_angle()
                self._angle_zero_raw = angle_raw
                self._angle_zero_turns = turns
                self._calibrated = True
                
                # 重置累积追踪
                self._last_raw_angle = angle_raw
                self._accumulated_turns = 0
                
                if print_info:
                    print(f"[校准] ✅ 零点已设置")
                    print(f"       原始角度: {angle_raw:.2f}°")
                    print(f"       原始值: {raw_value}")
                    print(f"       圈数: {turns}")
                    print(f"       方向: {'张开=增加' if self._angle_direction > 0 else '张开=减少'}")
                    print(f"       倍率: ×{self._angle_scale}")
                
                return angle_raw
            except Exception as e:
                print(f"[校准] ❌ 失败: {e}")
                return 0.0
    
    def set_direction(self, direction: int):
        """
        设置角度方向
        
        Args:
            direction: 1 表示编码器角度增加 = 张开
                      -1 表示编码器角度减少 = 张开
        """
        self._angle_direction = 1 if direction >= 0 else -1
        print(f"[方向] 已设置: {'张开=增加' if self._angle_direction > 0 else '张开=减少'}")
    
    def set_scale(self, scale: float):
        """
        设置角度倍率
        
        Args:
            scale: 倍率值（编码器角度 × 倍率 = 实际角度）
        """
        self._angle_scale = scale
        print(f"[倍率] 已设置: ×{scale}")
    
    def _get_calibrated_angle(self, raw_angle: float, turns: int) -> float:
        """
        计算校准后的角度
        
        处理0~360度的环绕问题：
        - 当角度从359°变到1°时，实际是+2°，不是-358°
        - 当角度从1°变到359°时，实际是-2°，不是+358°
        
        使用内部累积追踪来正确处理跨越0/360边界的情况
        
        Args:
            raw_angle: 原始角度 (度), 范围 0~360
            turns: 当前圈数 (可能不可靠，所以同时使用内部追踪)
            
        Returns:
            校准后的张开角度 (度)
        """
        if not self._calibrated:
            return raw_angle
        
        # 检测是否跨越了 0/360 边界
        delta = raw_angle - self._last_raw_angle
        
        # 如果变化超过180度，说明发生了环绕
        if delta > 180:
            # 从大角度跳到小角度，例如 350° -> 10°，实际是逆时针
            # delta = 10 - 350 = -340，但显示为正，所以实际减少了一圈
            self._accumulated_turns -= 1
        elif delta < -180:
            # 从小角度跳到大角度，例如 10° -> 350°，实际是顺时针
            # delta = 350 - 10 = 340，但显示为负，所以实际增加了一圈
            self._accumulated_turns += 1
        
        # 更新上次角度
        self._last_raw_angle = raw_angle
        
        # 计算相对于零点的角度变化
        angle_diff = raw_angle - self._angle_zero_raw
        
        # 加上累积的圈数
        total_diff = self._accumulated_turns * 360.0 + angle_diff
        
        # 应用方向和倍率
        calibrated_angle = total_diff * self._angle_direction * self._angle_scale
        
        return calibrated_angle
    
    @property
    def is_calibrated(self) -> bool:
        """是否已校准"""
        return self._calibrated
    
    def reset_calibration(self):
        """重置校准"""
        self._angle_zero_raw = 0
        self._angle_zero_turns = 0
        self._calibrated = False
        self._last_raw_angle = 0.0
        self._accumulated_turns = 0
        print("[校准] 已重置")
    
    def get_action(self) -> ActionData:
        """
        获取当前动作数据
        
        Returns:
            ActionData: 包含位姿和关节角度的数据
                - joint_angle: 校准后的张开角度（相对于零点）
                - joint_angle_raw: 原始绝对角度
        """
        with self._lock:
            timestamp = time.time()
            
            # 默认值
            position = np.zeros(3)
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
            velocity = np.zeros(3)
            euler = np.zeros(3)
            joint_angle_raw = 0.0
            joint_raw = 0
            joint_turns = 0
            
            # 读取 T265
            if self.use_t265 and self._t265_pipe:
                try:
                    position, quaternion, velocity, euler = self._read_t265_pose()
                except Exception as e:
                    print(f"[T265] 读取错误: {e}")
            
            # 读取编码器
            if self.use_encoder and self._encoder:
                try:
                    joint_angle_raw, joint_raw, joint_turns = self._read_encoder_angle()
                except Exception as e:
                    print(f"[编码器] 读取错误: {e}")
            
            # 计算校准后的角度
            joint_angle = self._get_calibrated_angle(joint_angle_raw, joint_turns)
            
            return ActionData(
                position=position,
                quaternion=quaternion,
                velocity=velocity,
                euler=euler,
                joint_angle=joint_angle,
                joint_angle_raw=joint_angle_raw,
                joint_raw=joint_raw,
                joint_turns=joint_turns,
                timestamp=timestamp
            )
    
    def get_pose(self) -> np.ndarray:
        """
        获取完整位姿
        
        Returns:
            pose: [x, y, z, qw, qx, qy, qz] 位置+四元数
        """
        action = self.get_action()
        return action.pose
    
    def get_pose_matrix(self) -> np.ndarray:
        """
        获取4x4变换矩阵
        
        Returns:
            4x4 齐次变换矩阵
        """
        action = self.get_action()
        return action.pose_matrix
    
    def get_pose_euler(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取位置和欧拉角
        
        Returns:
            (position, euler): position=[x,y,z], euler=[roll,pitch,yaw] (度)
        """
        action = self.get_action()
        return action.position, action.euler
    
    def get_angle(self) -> float:
        """
        只获取关节角度
        
        Returns:
            角度 (度)
        """
        action = self.get_action()
        return action.joint_angle
    
    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected
    
    def __enter__(self):
        """支持 with 语句"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.disconnect()


# ============ 便捷函数 ============

def create_reader(encoder_port: str = '/dev/ttyUSB0',
                  use_t265: bool = True,
                  use_encoder: bool = True) -> ActionReader:
    """
    创建并连接读取器
    
    Args:
        encoder_port: 编码器串口
        use_t265: 是否使用T265
        use_encoder: 是否使用编码器
    
    Returns:
        已连接的 ActionReader
    """
    reader = ActionReader(
        encoder_port=encoder_port,
        use_t265=use_t265,
        use_encoder=use_encoder
    )
    reader.connect()
    return reader


# ============ 测试代码 ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="动作数据读取器测试")
    parser.add_argument('--port', default='/dev/ttyUSB0', help='编码器串口')
    parser.add_argument('--no-t265', action='store_true', help='不使用T265')
    parser.add_argument('--no-encoder', action='store_true', help='不使用编码器')
    parser.add_argument('--freq', type=float, default=10, help='读取频率(Hz)')
    parser.add_argument('--reverse', action='store_true', help='反转角度方向（张开=减少）')
    args = parser.parse_args()
    
    print("=" * 60)
    print("动作数据读取器测试")
    print("=" * 60)
    
    # 创建读取器
    direction = -1 if args.reverse else 1
    reader = ActionReader(
        encoder_port=args.port,
        use_t265=not args.no_t265,
        use_encoder=not args.no_encoder,
        angle_direction=direction
    )
    
    # 连接
    if not reader.connect():
        print("连接失败!")
        exit(1)
    
    # 校准提示
    if reader.use_encoder:
        print("\n" + "=" * 60)
        print("角度校准")
        print("=" * 60)
        print("请将机构移动到初始位置（闭合状态）")
        input("按 Enter 键进行校准...")
        reader.calibrate()
    
    print("\n开始读取数据 (按 Ctrl+C 停止)...")
    print("-" * 60)
    
    interval = 1.0 / args.freq
    count = 0
    
    try:
        while True:
            action = reader.get_action()
            count += 1
            
            # 打印数据
            pos = action.position
            euler = action.euler
            print(f"\r[{count:5d}] "
                  f"位置:[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
                  f"姿态:[R{euler[0]:+6.1f}° P{euler[1]:+6.1f}° Y{euler[2]:+6.1f}°] "
                  f"张开:{action.joint_angle:+7.2f}°", end='')
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n停止读取")
    finally:
        reader.disconnect()

