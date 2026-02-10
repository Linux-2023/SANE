#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂遥操作系统--单人收数据版本

基于 DualArmEnvironment 实现，通过 T265 读取位姿并转换为关节角控制机器人。
支持记录模式，可以保存轨迹数据到 HDF5 文件。

控制键：
- 程序启动后自动开始记录数据
- Space: 按住暂停记录（断开跟随），松开恢复记录（恢复跟随）
- ESC: 保存数据并退出程序
"""

import time
import threading
import numpy as np
from typing import Optional, Dict, Any
import sys
import os
import h5py
from datetime import datetime
import cv2

# 添加 flexiv_usage 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../third_party/flexiv_usage'))

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("错误: 需要安装 scipy")
    sys.exit(1)

try:
    from pynput import keyboard
except ImportError:
    print("错误: 需要安装 pynput")
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    print("错误: 需要安装 pyrealsense2")
    sys.exit(1)

import flexivrdk
from flexiv_robot import FlexivRobot
from env_dual import DualArmEnvironment, create_dual_arm_env_from_config


class EncoderReader:
    """
    磁编码器读取器
    """
    # 编码器默认配置
    ENCODER_RESOLUTION = 65536
    REG_ANGLE_HIGH = 0x40
    REG_ANGLE_LOW = 0x41
    REG_TURNS = 0x44
    
    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        slave_id: int = 1,
        baudrate: int = 115200,
        direction: int = 1,
        scale: float = 2.0
    ):
        self.port = port
        self.slave_id = slave_id
        self.baudrate = baudrate
        self._direction = direction
        self._scale = scale
        
        self._encoder = None
        self._connected = False
        self._lock = threading.Lock()
        
        # 校准参数
        self._angle_zero_raw = 0.0
        self._calibrated = False
        self._last_raw_angle = 0.0
        self._accumulated_turns = 0
    
    def connect(self) -> bool:
        """连接编码器"""
        try:
            import minimalmodbus
            
            self._encoder = minimalmodbus.Instrument(self.port, self.slave_id)
            self._encoder.serial.baudrate = self.baudrate
            self._encoder.serial.bytesize = 8
            self._encoder.serial.parity = minimalmodbus.serial.PARITY_NONE
            self._encoder.serial.stopbits = 1
            self._encoder.serial.timeout = 0.5
            self._encoder.mode = minimalmodbus.MODE_RTU
            self._encoder.clear_buffers_before_each_transaction = True
            
            # 测试读取
            self._read_raw_angle()
            self._connected = True
            print(f"[编码器] ✅ 连接成功 ({self.port})")
            return True
        except Exception as e:
            print(f"[编码器] ❌ 连接失败 ({self.port}): {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._encoder:
            try:
                self._encoder.serial.close()
            except:
                pass
            self._encoder = None
        self._connected = False
    
    def _read_raw_angle(self) -> float:
        """读取原始角度"""
        angle_high = self._encoder.read_register(self.REG_ANGLE_HIGH, 0, 3, False)
        angle_low = self._encoder.read_register(self.REG_ANGLE_LOW, 0, 3, False)
        raw_value = (angle_high << 16) | angle_low
        return (raw_value / self.ENCODER_RESOLUTION) * 360.0
    
    def calibrate(self, print_info: bool = True) -> float:
        """校准零点"""
        if not self._connected:
            print("[校准] ❌ 编码器未连接")
            return 0.0
        
        with self._lock:
            try:
                angle_raw = self._read_raw_angle()
                self._angle_zero_raw = angle_raw
                self._calibrated = True
                self._last_raw_angle = angle_raw
                self._accumulated_turns = 0
                
                if print_info:
                    print(f"[校准] ✅ 零点已设置 ({self.port})")
                    print(f"       原始角度: {angle_raw:.2f}°")
                    print(f"       方向: {'张开=增加' if self._direction > 0 else '张开=减少'}")
                    print(f"       倍率: ×{self._scale}")
                
                return angle_raw
            except Exception as e:
                print(f"[校准] ❌ 失败: {e}")
                return 0.0
    
    def get_angle(self) -> float:
        """获取校准后的角度"""
        if not self._connected:
            return 0.0
        
        with self._lock:
            try:
                raw_angle = self._read_raw_angle()
                
                if not self._calibrated:
                    return raw_angle
                
                # 处理跨越 0/360 边界
                delta = raw_angle - self._last_raw_angle
                if delta > 180:
                    self._accumulated_turns -= 1
                elif delta < -180:
                    self._accumulated_turns += 1
                
                self._last_raw_angle = raw_angle
                
                # 计算相对于零点的角度
                angle_diff = raw_angle - self._angle_zero_raw
                total_diff = self._accumulated_turns * 360.0 + angle_diff
                
                return total_diff * self._direction * self._scale
            except Exception as e:
                print(f"[编码器] 读取错误: {e}")
                return 0.0
    
    @property
    def is_connected(self) -> bool:
        return self._connected


def display_observations(obs: dict, window_name: str = "Observations", recording: bool = False):
    """
    显示观测中的所有图像
    
    Args:
        obs: 观测字典，包含 'images' 键
        window_name: 窗口名称
        recording: 是否正在记录
    """
    images_dict = obs.get('images', {})
    if not images_dict:
        return 0

    display_images = []
    for cam_name, img in images_dict.items():
        # 转换格式: (C, H, W) -> (H, W, C)
        img_hwc = np.transpose(img, (1, 2, 0))
        
        # RGB -> BGR (OpenCV 使用 BGR 格式)
        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
        
        # 添加标签
        label_color = (0, 0, 255) if recording else (0, 255, 0)
        label_text = f"{cam_name} (RECORDING)" if recording else cam_name
        cv2.putText(img_bgr, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        
        display_images.append(img_bgr)
    
    # 水平拼接所有图像
    if len(display_images) > 1:
        combined = np.hstack(display_images)
    else:
        combined = display_images[0] if display_images else np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 在底部添加 prompt 信息
    prompt = obs.get('prompt', '')
    if prompt:
        # 创建一个底部横幅来显示 prompt
        banner_height = 60
        banner = np.zeros((banner_height, combined.shape[1], 3), dtype=np.uint8)
        # 使用较小的字体显示 prompt
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)
        cv2.putText(banner, f"Task: {prompt}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, text_color, thickness)
        # 将 banner 添加到 combined 图像的底部
        combined = np.vstack([combined, banner])
    
    cv2.imshow(window_name, combined)
    return cv2.waitKey(1) & 0xFF

def display_raw_observations(obs: dict, window_name: str = "Raw Observations"):
    """
    显示原始观测中的所有图像
    """
    raw_images_dict = obs.get('raw_images', {})
    if not raw_images_dict:
        return 0
    
    display_images = []
    for cam_name, img in raw_images_dict.items():
        display_images.append(img)
    
    # 按要求拼接显示：左右手眼 (1600x1200) 水平在第一行，全局(800x600)在第二行
    # 假设命名约定为 cam_left_wrist, cam_right_wrist, cam_high
    # 先分组
    raw_images_dict_keys = list(raw_images_dict.keys())
    left = raw_images_dict.get("cam_left_wrist", None)
    right = raw_images_dict.get("cam_right_wrist", None)
    high = raw_images_dict.get("cam_high", None)

    # 目标显示窗口尺寸
    wrist_target_size = (800, 600)  # 原图1600x1200缩小一半
    high_target_size = (800, 600)   # 原图就是800x600

    wrist_row = []
    if left is not None:
        # 保证能缩放，避免None
        left_disp = cv2.resize(left, wrist_target_size, interpolation=cv2.INTER_AREA)
        # 上下和左右翻转图像
        left_disp = cv2.flip(left_disp, -1)
        wrist_row.append(left_disp)
    if right is not None:
        right_disp = cv2.resize(right, wrist_target_size, interpolation=cv2.INTER_AREA)
        # 上下和左右翻转图像
        right_disp = cv2.flip(right_disp, -1)
        wrist_row.append(right_disp)

    # 第一行：左右手眼
    if len(wrist_row) > 0:
        row1 = np.hstack(wrist_row)
    else:
        row1 = np.zeros((wrist_target_size[1], wrist_target_size[0] * 2, 3), dtype=np.uint8)

    # 第二行：全局相机
    if high is not None:
        # 若已经是(800,600)，则不缩放
        if high.shape[1] != high_target_size[0] or high.shape[0] != high_target_size[1]:
            high_disp = cv2.resize(high, high_target_size, interpolation=cv2.INTER_AREA)
        else:
            high_disp = high
        row2 = high_disp
    else:
        # 空行填黑
        row2 = np.zeros((high_target_size[1], high_target_size[0], 3), dtype=np.uint8)

    # 使列宽对齐：row1可能宽度1600(两手), 800(单手), row2为800
    # 横向填黑让row2也达到row1的宽度
    row1_width = row1.shape[1]
    row2_width = row2.shape[1]
    if row2_width < row1_width:
        pad = np.zeros((row2.shape[0], row1_width - row2_width, 3), dtype=np.uint8)
        row2 = np.hstack([row2, pad])
    elif row1_width < row2_width:
        pad = np.zeros((row1.shape[0], row2_width - row1_width, 3), dtype=np.uint8)
        row1 = np.hstack([row1, pad])

    # 最终拼接
    combined = np.vstack([row1, row2])
    cv2.imshow(window_name, combined)
    return cv2.waitKey(1) & 0xFF
    
class T265Reader:
    """
    支持指定序列号的 T265 读取器
    """
    def __init__(self, serial_number: str = ""):
        """
        Args:
            serial_number: T265 序列号，留空则自动选择
        """
        self.serial_number = serial_number
        self._pipe = None
        self._cfg = None
        self._connected = False
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """连接 T265"""
        try:
            self._pipe = rs.pipeline()
            self._cfg = rs.config()
            
            # 如果指定了序列号，则使用该设备
            if self.serial_number:
                self._cfg.enable_device(self.serial_number)
                
            self._cfg.enable_stream(rs.stream.pose)
            self._pipe.start(self._cfg)
            
            # 等待稳定
            for _ in range(10):
                self._pipe.wait_for_frames()
            
            # 获取实际连接的设备序列号
            if not self.serial_number:
                device = self._pipe.get_active_profile().get_device()
                self.serial_number = device.get_info(rs.camera_info.serial_number)
            
            self._connected = True
            print(f"[T265] ✅ 连接成功 (SN: {self.serial_number})")
            return True
        except Exception as e:
            print(f"[T265] ❌ 连接失败: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._pipe:
            try:
                self._pipe.stop()
            except:
                pass
            self._pipe = None
        self._connected = False
    
    def get_pose(self):
        """
        获取当前位姿
        
        Returns:
            (position, quaternion): 位置 [x,y,z] 和四元数 [qw, qx, qy, qz]
        """
        if not self._connected:
            return np.zeros(3), np.array([1, 0, 0, 0])
        
        with self._lock:
            try:
                frames = self._pipe.wait_for_frames()
                pose_frame = frames.get_pose_frame()
                
                if not pose_frame:
                    return np.zeros(3), np.array([1, 0, 0, 0])
                
                data = pose_frame.get_pose_data()
                
                position = np.array([
                    data.translation.x,
                    data.translation.y,
                    data.translation.z
                ])
                
                quaternion = np.array([
                    data.rotation.w,
                    data.rotation.x,
                    data.rotation.y,
                    data.rotation.z
                ])
                
                return position, quaternion
            except Exception as e:
                print(f"[T265] 读取错误: {e}")
                return np.zeros(3), np.array([1, 0, 0, 0])
    
    @property
    def is_connected(self) -> bool:
        return self._connected


class DualArmTeleopSystem:
    """
    双臂遥操作系统
    
    主要功能：
    1. 从 T265 读取位姿
    2. 转换为关节角
    3. 控制 DualArmEnvironment
    4. 支持记录模式
    """
    
    def __init__(
        self,
        env: DualArmEnvironment,
        left_t265_serial: str = "",
        right_t265_serial: str = "",
        left_encoder_port: str = "",
        right_encoder_port: str = "",
        left_encoder_direction: int = 1,
        right_encoder_direction: int = 1,
        left_encoder_scale: float = 2.0,
        right_encoder_scale: float = 2.0,
        control_frequency: float = 20.0,
        position_scale: float = 1.5,
        filter_alpha: float = 0.8,
        rot_deadband_deg: float = 1.0,
        rot_fullspeed_deg: float = 8.0,
        rot_alpha_min: float = 0.05,
        pos_deadband_mm: float = 1.0,
        pos_fullspeed_mm: float = 10.0,
        pos_alpha_min: float = 0.05,
        pos_alpha_max: float = 0.6,
        t265_rot_center_offset_m: Optional[np.ndarray] = None,
        control_mode: str = "joint",
        verbose: bool = True,
        human_to_robot_direction: str = "opposite"
    ):
        """
        Args:
            env: DualArmEnvironment 实例
            left_t265_serial: 左臂 T265 序列号
            right_t265_serial: 右臂 T265 序列号
            left_encoder_port: 左臂编码器端口
            right_encoder_port: 右臂编码器端口
            left_encoder_direction: 左臂编码器方向 (1 或 -1)
            right_encoder_direction: 右臂编码器方向 (1 或 -1)
            left_encoder_scale: 左臂编码器倍率
            right_encoder_scale: 右臂编码器倍率
            control_frequency: 控制频率 (Hz)
            position_scale: 位置缩放因子
            filter_alpha: 滤波系数（最大值）
            rot_deadband_deg: 姿态死区角度 (度)
            rot_fullspeed_deg: 姿态全速角度 (度)
            rot_alpha_min: 姿态最小滤波系数
            pos_deadband_mm: 位置死区距离 (毫米)
            pos_fullspeed_mm: 位置全速距离 (毫米)
            pos_alpha_min: 位置最小滤波系数
            pos_alpha_max: 位置最大滤波系数
            t265_rot_center_offset_m: T265 旋转中心偏移 [x, y, z] (米)
            control_mode: 控制模式，"joint" 或 "cartesian"
            verbose: 是否打印详细信息
            human_to_robot_direction: 人与机械臂的相对方向，'same'为同向，'opposite'为面对面
        """
        self.env = env
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.position_scale = position_scale
        self.control_mode = control_mode.lower()  # 控制模式: "joint" 或 "cartesian"
        self.verbose = verbose
        
        # T265 旋转中心偏移
        if t265_rot_center_offset_m is None:
            self.t265_rot_center_offset_m = np.array([0.0, -0.04, 0.0], dtype=float)
        else:
            self.t265_rot_center_offset_m = np.array(t265_rot_center_offset_m, dtype=float).reshape(3)
        
        # ========== 自适应滤波参数 ==========
        # 姿态滤波参数
        self.rot_deadband_rad = float(np.deg2rad(rot_deadband_deg))
        self.rot_fullspeed_rad = float(np.deg2rad(rot_fullspeed_deg))
        self.rot_alpha_min = float(rot_alpha_min)
        self.rot_alpha_max = float(filter_alpha)
        
        # 位置滤波参数
        self.pos_deadband_m = float(pos_deadband_mm) / 1000.0
        self.pos_fullspeed_m = float(pos_fullspeed_mm) / 1000.0
        self.pos_alpha_min = float(pos_alpha_min)
        self.pos_alpha_max = float(pos_alpha_max)
        
        # ========== 滤波状态变量（左臂）==========
        self.left_last_filtered_pos: Optional[np.ndarray] = None
        self.left_last_filtered_quat: Optional[np.ndarray] = None
        
        # ========== 滤波状态变量（右臂）==========
        self.right_last_filtered_pos: Optional[np.ndarray] = None
        self.right_last_filtered_quat: Optional[np.ndarray] = None
        
        # T265 读取器
        self.left_t265 = T265Reader(serial_number=left_t265_serial)
        self.right_t265 = T265Reader(serial_number=right_t265_serial)
        
        # ========== 编码器读取器 ==========
        self.left_encoder: Optional[EncoderReader] = None
        self.right_encoder: Optional[EncoderReader] = None
        self._left_encoder_port = left_encoder_port
        self._right_encoder_port = right_encoder_port
        self._left_encoder_direction = left_encoder_direction
        self._right_encoder_direction = right_encoder_direction
        self._left_encoder_scale = left_encoder_scale
        self._right_encoder_scale = right_encoder_scale
        
        # 夹爪控制状态
        self.left_last_gripper_move_time = 0.0
        self.right_last_gripper_move_time = 0.0
        self.gripper_move_interval = 0.2  # 夹爪控制间隔
        self.gripper_max_angle = 10.0  # 编码器最大角度对应夹爪全开
        
        # IK Model（用于位姿到关节角的转换）
        self._left_model: Optional[flexivrdk.Model] = None
        self._right_model: Optional[flexivrdk.Model] = None
        
        # 参考位姿（用于增量控制）
        self.left_t265_ref_pos: Optional[np.ndarray] = None
        self.left_t265_ref_rot: Optional[R] = None
        self.left_robot_ref_pos: Optional[np.ndarray] = None
        self.left_robot_ref_rot: Optional[R] = None
        
        self.right_t265_ref_pos: Optional[np.ndarray] = None
        self.right_t265_ref_rot: Optional[R] = None
        self.right_robot_ref_pos: Optional[np.ndarray] = None
        self.right_robot_ref_rot: Optional[R] = None
        
        self.human_to_robot_direction = human_to_robot_direction

        # 旋转轴映射矩阵
        if self.human_to_robot_direction == "opposite": # 面对面
            self.M_rotvec = np.array([
                [0, 1,  0],
                [1, 0,  0],
                [0, 0, -1],
            ], dtype=float)
        else:
            # 同向
            self.M_rotvec = np.array([
                [0, -1,  0],
                [-1, 0,  0],
                [0, 0, -1],
            ], dtype=float)
        
        # ========== 离合器状态 ==========
        self.is_following = True  # True=跟随模式，False=断开模式
        
        # 运行状态
        self._running = False
        self._record_mode = True  # 初始化后自动开始记录
        
        # 记录数据
        self._episode_data = []
        self._record_directory = "recorded_data"
        
        # 键盘监听
        self._kb_listener: Optional[keyboard.Listener] = None
        self._kb_lock = threading.Lock()
        self._key_pressed = {'s': False, 'q': False, 'esc': False}
        self._space_pressed = False  # 空格键状态
        
        # 初始化
        self._init_t265()
        self._init_encoders()
        # 仅在关节控制模式下初始化 IK 模型
        if self.control_mode == "joint":
            self._init_ik_models()
        else:
            if self.verbose:
                print(f"✅ 使用笛卡尔控制模式，跳过 IK 模型初始化")
        self._reset_reference()
    
    def _init_t265(self):
        """初始化 T265 设备"""
        print("\n初始化 T265 设备...")
        
        if not self.left_t265.connect():
            print("警告: 左臂 T265 连接失败")
        
        if not self.right_t265.connect():
            print("警告: 右臂 T265 连接失败")
    
    def _init_encoders(self):
        """初始化编码器"""
        print("\n初始化编码器...")
        
        # 左臂编码器
        if self._left_encoder_port:
            self.left_encoder = EncoderReader(
                port=self._left_encoder_port,
                direction=self._left_encoder_direction,
                scale=self._left_encoder_scale
            )
            if self.left_encoder.connect():
                time.sleep(0.5)
                self.left_encoder.calibrate(print_info=True)
            else:
                print("警告: 左臂编码器连接失败")
                self.left_encoder = None
        
        # 右臂编码器
        if self._right_encoder_port:
            self.right_encoder = EncoderReader(
                port=self._right_encoder_port,
                direction=self._right_encoder_direction,
                scale=self._right_encoder_scale
            )
            if self.right_encoder.connect():
                time.sleep(0.5)
                self.right_encoder.calibrate(print_info=True)
            else:
                print("警告: 右臂编码器连接失败")
                self.right_encoder = None

        self.last_left_encoder_angle = 0.0
        self.last_right_encoder_angle = 0.0

    @staticmethod
    def _smoothstep01(x: float) -> float:
        """平滑过渡函数 (0-1 范围内)"""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)
    
    def _adaptive_alpha(self, err: float, deadband: float, fullspeed: float, 
                        alpha_min: float, alpha_max: float) -> float:
        """
        自适应滤波系数计算
        
        Args:
            err: 误差值
            deadband: 死区阈值（小于此值返回 0）
            fullspeed: 全速阈值（大于此值返回 alpha_max）
            alpha_min: 最小滤波系数
            alpha_max: 最大滤波系数
            
        Returns:
            滤波系数 alpha
        """
        if err <= deadband:
            return 0.0
        if fullspeed <= deadband:
            return alpha_max
        t = (err - deadband) / (fullspeed - deadband)
        k = self._smoothstep01(t)
        return alpha_min + k * (alpha_max - alpha_min)
    
    def _init_ik_models(self):
        """初始化 IK 模型"""
        try:
            # 获取 FlexivRobot 的 robot 对象
            left_robot_obj = self.env._left_robot._robot
            right_robot_obj = self.env._right_robot._robot
            
            self._left_model = flexivrdk.Model(left_robot_obj)
            self._right_model = flexivrdk.Model(right_robot_obj)
            
            if self.verbose:
                print("✅ IK 模型已初始化")
        except Exception as e:
            print(f"⚠️ IK 模型初始化失败: {e}")
            self._left_model = None
            self._right_model = None
    
    def _reset_reference(self):
        """重置参考位姿
        
        当松开空格键时调用，重新建立 T265 和机器人的映射关系。
        即使 T265 在断开期间移动了，也会以当前状态为新的参考点。
        """
        # 左臂参考位姿
        if self.left_t265.is_connected:
            pos, quat = self.left_t265.get_pose()
            q = quat  # [qw, qx, qy, qz]
            self.left_t265_ref_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            self.left_t265_ref_pos = self._t265_center_pos(pos, self.left_t265_ref_rot)
        
        # 获取机器人当前位姿作为参考
        if self.env._left_robot is not None and self.env._left_robot_enabled:
            try:
                robot_pose = self.env._left_robot.get_tcp_pose()
                self.left_robot_ref_pos = robot_pose[:3].copy()
                rq = robot_pose[3:]
                self.left_robot_ref_rot = R.from_quat([rq[1], rq[2], rq[3], rq[0]])
                
                # 重置滤波器状态为当前机器人位姿（避免松开空格后跳变）
                self.left_last_filtered_pos = self.left_robot_ref_pos.copy()
                self.left_last_filtered_quat = rq.copy()
            except Exception as e:
                if self.verbose:
                    print(f"警告: 无法获取左臂当前位姿: {e}")
        
        # 右臂参考位姿
        if self.right_t265.is_connected:
            pos, quat = self.right_t265.get_pose()
            q = quat
            self.right_t265_ref_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            self.right_t265_ref_pos = self._t265_center_pos(pos, self.right_t265_ref_rot)
        
        if self.env._right_robot is not None and self.env._right_robot_enabled:
            try:
                robot_pose = self.env._right_robot.get_tcp_pose()
                self.right_robot_ref_pos = robot_pose[:3].copy()
                rq = robot_pose[3:]
                self.right_robot_ref_rot = R.from_quat([rq[1], rq[2], rq[3], rq[0]])
                
                # 重置滤波器状态为当前机器人位姿（避免松开空格后跳变）
                self.right_last_filtered_pos = self.right_robot_ref_pos.copy()
                self.right_last_filtered_quat = rq.copy()
            except Exception as e:
                if self.verbose:
                    print(f"警告: 无法获取右臂当前位姿: {e}")
        
        if self.verbose:
            print("参考位姿已重置")
    
    def _t265_center_pos(self, pos: np.ndarray, rot: R) -> np.ndarray:
        """补偿旋转中心"""
        return np.array(pos, dtype=float).reshape(3) + rot.apply(self.t265_rot_center_offset_m)
    
    def _solve_ik(self, target_pose: np.ndarray, model: flexivrdk.Model, seed_joints: np.ndarray) -> tuple:
        """
        使用 Model.reachable 求解 IK
        
        Args:
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]
            model: IK 模型
            seed_joints: 种子关节角度
            
        Returns:
            (is_reachable, ik_solution): 是否可达和关节解
        """
        if model is None:
            return False, None
        
        try:
            is_reachable, ik_solution = model.reachable(
                list(target_pose),
                list(seed_joints),
                False  # 不允许自由姿态
            )
            
            if is_reachable:
                return True, np.array(ik_solution)
            else:
                return False, None
        except Exception as e:
            if self.verbose:
                print(f"IK 求解失败: {e}")
            return False, None
    
    def _t265_to_robot_pose(self, t265_pos: np.ndarray, t265_quat: np.ndarray, 
                            t265_ref_pos: np.ndarray, t265_ref_rot: R,
                            robot_ref_pos: np.ndarray, robot_ref_rot: R) -> np.ndarray:
        """
        将 T265 位姿转换为机器人位姿
        
        Args:
            t265_pos: T265 当前位置 [x, y, z]
            t265_quat: T265 当前四元数 [qw, qx, qy, qz]
            t265_ref_pos: T265 参考位置
            t265_ref_rot: T265 参考旋转
            robot_ref_pos: 机器人参考位置
            robot_ref_rot: 机器人参考旋转
            
        Returns:
            机器人目标位姿 [x, y, z, qw, qx, qy, qz]
        """
        # 计算 T265 增量
        curr_t265_rot = R.from_quat([t265_quat[1], t265_quat[2], t265_quat[3], t265_quat[0]])
        curr_t265_pos = self._t265_center_pos(t265_pos, curr_t265_rot)
        
        delta_p_t265 = curr_t265_pos - t265_ref_pos
        delta_r_t265 = t265_ref_rot.inv() * curr_t265_rot
        
        delta_p_robot = np.zeros(3)
        if self.human_to_robot_direction == "opposite": # 面对面
            delta_p_robot[0] = -delta_p_t265[0]
            delta_p_robot[1] = delta_p_t265[2]
            delta_p_robot[2] = delta_p_t265[1]
        else:
            # 同向
            delta_p_robot[0] = delta_p_t265[0]
            delta_p_robot[1] = -delta_p_t265[2]
            delta_p_robot[2] = delta_p_t265[1]

        
        # 应用缩放
        delta_p_robot *= self.position_scale
        
        # 姿态映射
        rotvec_t265 = delta_r_t265.as_rotvec()
        rotvec_tcp = self.M_rotvec @ rotvec_t265
        delta_r_robot = R.from_rotvec(rotvec_tcp)
        target_rot = robot_ref_rot * delta_r_robot
        
        # 计算目标位姿
        target_pos = robot_ref_pos + delta_p_robot
        target_quat = target_rot.as_quat()  # [qx, qy, qz, qw]
        target_quat_flexiv = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        
        return np.concatenate([target_pos, target_quat_flexiv])
    
    def _apply_filter(
        self,
        target_pos: np.ndarray,
        target_quat_flexiv: np.ndarray,
        last_filtered_pos: Optional[np.ndarray],
        last_filtered_quat: Optional[np.ndarray]
    ) -> tuple:
        """
        应用自适应滤波
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_quat_flexiv: 目标四元数 [qw, qx, qy, qz]
            last_filtered_pos: 上一次滤波后的位置
            last_filtered_quat: 上一次滤波后的四元数
            
        Returns:
            (filtered_pos, filtered_quat): 滤波后的位置和四元数
        """
        # 初始化滤波状态
        if last_filtered_pos is None:
            last_filtered_pos = target_pos.copy()
        if last_filtered_quat is None:
            last_filtered_quat = target_quat_flexiv.copy()
        
        # ========== 位置滤波 ==========
        pos_err = float(np.linalg.norm(target_pos - last_filtered_pos))
        pos_alpha = self._adaptive_alpha(
            err=pos_err,
            deadband=self.pos_deadband_m,
            fullspeed=self.pos_fullspeed_m,
            alpha_min=self.pos_alpha_min,
            alpha_max=self.pos_alpha_max,
        )
        
        if pos_alpha <= 0.0:
            filtered_pos = last_filtered_pos.copy()
        else:
            filtered_pos = last_filtered_pos + pos_alpha * (target_pos - last_filtered_pos)
        
        # ========== 姿态滤波 ==========
        # 确保四元数在同一半球（避免插值走远路）
        target_quat_for_filter = target_quat_flexiv.copy()
        if np.dot(last_filtered_quat, target_quat_for_filter) < 0:
            target_quat_for_filter = -target_quat_for_filter
        
        # 计算姿态误差
        last_r = R.from_quat([
            last_filtered_quat[1], last_filtered_quat[2],
            last_filtered_quat[3], last_filtered_quat[0]
        ])
        targ_r = R.from_quat([
            target_quat_for_filter[1], target_quat_for_filter[2],
            target_quat_for_filter[3], target_quat_for_filter[0]
        ])
        r_err = last_r.inv() * targ_r
        rot_err = float(np.linalg.norm(r_err.as_rotvec()))
        
        rot_alpha = self._adaptive_alpha(
            err=rot_err,
            deadband=self.rot_deadband_rad,
            fullspeed=self.rot_fullspeed_rad,
            alpha_min=self.rot_alpha_min,
            alpha_max=self.rot_alpha_max,
        )
        
        if rot_alpha <= 0.0:
            filtered_quat = last_filtered_quat.copy()
        else:
            # 线性插值四元数
            filtered_quat = (1.0 - rot_alpha) * last_filtered_quat + rot_alpha * target_quat_for_filter
            # 归一化
            n = float(np.linalg.norm(filtered_quat))
            if n > 1e-6:
                filtered_quat = filtered_quat / n
            else:
                filtered_quat = last_filtered_quat.copy()
        
        return filtered_pos, filtered_quat
    
    def _read_t265_and_convert_to_joints(self) -> Optional[np.ndarray]:
        """
        从 T265 读取位姿并转换为 16 维关节角向量
        
        Returns:
            16 维向量 [左臂关节x7, 左臂夹爪, 右臂关节x7, 右臂夹爪]，失败返回 None
        """
        action = np.zeros(16, dtype=np.float32)
        
        # ========== 左臂 ==========
        if self.left_t265.is_connected and self._left_model is not None:
            try:
                t265_pos, t265_quat = self.left_t265.get_pose()
                
                if self.left_t265_ref_pos is not None and self.left_robot_ref_pos is not None:
                    # 转换为机器人位姿（未滤波）
                    robot_pose = self._t265_to_robot_pose(
                        t265_pos, t265_quat,
                        self.left_t265_ref_pos, self.left_t265_ref_rot,
                        self.left_robot_ref_pos, self.left_robot_ref_rot
                    )
                    
                    # 应用滤波
                    target_pos = robot_pose[:3]
                    target_quat = robot_pose[3:]
                    
                    filtered_pos, filtered_quat = self._apply_filter(
                        target_pos, target_quat,
                        self.left_last_filtered_pos, self.left_last_filtered_quat
                    )
                    
                    # 更新滤波状态
                    self.left_last_filtered_pos = filtered_pos.copy()
                    self.left_last_filtered_quat = filtered_quat.copy()
                    
                    # 构建滤波后的位姿
                    filtered_pose = np.concatenate([filtered_pos, filtered_quat])
                    
                    # IK 求解
                    seed_joints = np.array(self.env._left_robot.get_joint_positions())
                    is_reachable, ik_solution = self._solve_ik(filtered_pose, self._left_model, seed_joints)
                    
                    if is_reachable:
                        action[0:7] = ik_solution
                    else:
                        # IK 失败，使用当前关节角
                        action[0:7] = seed_joints
                        if self.verbose:
                            print("⚠️ 左臂 IK 求解失败，使用当前关节角")
                else:
                    # 参考位姿未设置，使用当前关节角
                    action[0:7] = self.env._left_robot.get_joint_positions()
            except Exception as e:
                if self.verbose:
                    print(f"左臂位姿读取错误: {e}")
                action[0:7] = self.env._left_robot.get_joint_positions()
        else:
            # T265 未连接，使用当前关节角
            if self.env._left_robot is not None:
                action[0:7] = self.env._left_robot.get_joint_positions()
        
        # ========== 左臂夹爪（从编码器读取）==========
        if self.env._left_gripper_enabled:
            if self.left_encoder is not None and self.left_encoder.is_connected:
                # 从编码器读取角度并映射到夹爪开度
                angle = self.left_encoder.get_angle()
                angle = max(0.0, min(self.gripper_max_angle, angle))
                # action[7] = angle / self.gripper_max_angle  # 归一化到 [0, 1]
                if abs(angle - self.last_left_encoder_angle) > 0.1:
                    if angle > self.last_left_encoder_angle:
                        action[7] = 1.0 #夹爪打开
                    else:
                        action[7] = 0.0 #夹爪闭合
                else:
                    action[7] = self.env._left_robot.gripper_state
                self.last_left_encoder_angle = angle
            else:
                # 编码器未连接，保持当前夹爪状态
                # try:
                #     current_gripper = self.env._left_robot.get_gripper_width()
                #     action[7] = min(current_gripper / self.env._gripper_max_width, 1.0)
                # except:
                #     action[7] = 0.0
                action[7] = self.env._left_robot.gripper_state
        
        # ========== 右臂 ==========
        if self.right_t265.is_connected and self._right_model is not None:
            try:
                t265_pos, t265_quat = self.right_t265.get_pose()
                
                if self.right_t265_ref_pos is not None and self.right_robot_ref_pos is not None:
                    # 转换为机器人位姿（未滤波）
                    robot_pose = self._t265_to_robot_pose(
                        t265_pos, t265_quat,
                        self.right_t265_ref_pos, self.right_t265_ref_rot,
                        self.right_robot_ref_pos, self.right_robot_ref_rot
                    )
                    
                    # 应用滤波
                    target_pos = robot_pose[:3]
                    target_quat = robot_pose[3:]
                    
                    filtered_pos, filtered_quat = self._apply_filter(
                        target_pos, target_quat,
                        self.right_last_filtered_pos, self.right_last_filtered_quat
                    )
                    
                    # 更新滤波状态
                    self.right_last_filtered_pos = filtered_pos.copy()
                    self.right_last_filtered_quat = filtered_quat.copy()
                    
                    # 构建滤波后的位姿
                    filtered_pose = np.concatenate([filtered_pos, filtered_quat])
                    
                    # IK 求解
                    seed_joints = np.array(self.env._right_robot.get_joint_positions())
                    is_reachable, ik_solution = self._solve_ik(filtered_pose, self._right_model, seed_joints)
                    
                    if is_reachable:
                        action[8:15] = ik_solution
                    else:
                        # IK 失败，使用当前关节角
                        action[8:15] = seed_joints
                        if self.verbose:
                            print("⚠️ 右臂 IK 求解失败，使用当前关节角")
                else:
                    # 参考位姿未设置，使用当前关节角
                    action[8:15] = self.env._right_robot.get_joint_positions()
            except Exception as e:
                if self.verbose:
                    print(f"右臂位姿读取错误: {e}")
                action[8:15] = self.env._right_robot.get_joint_positions()
        else:
            # T265 未连接，使用当前关节角
            if self.env._right_robot is not None:
                action[8:15] = self.env._right_robot.get_joint_positions()
        
        # ========== 右臂夹爪（从编码器读取）==========
        if self.env._right_gripper_enabled:
            if self.right_encoder is not None and self.right_encoder.is_connected:
                # 从编码器读取角度并映射到夹爪开度
                angle = self.right_encoder.get_angle()
                angle = max(0.0, min(self.gripper_max_angle, angle))
                # action[15] = angle / self.gripper_max_angle  # 归一化到 [0, 1]
                if abs(angle - self.last_right_encoder_angle) > 0.1:
                    if angle > self.last_right_encoder_angle:
                        action[15] = 1.0 #夹爪打开
                    else:
                        action[15] = 0.0 #夹爪闭合
                else:
                    action[15] = self.env._right_robot.gripper_state
                self.last_right_encoder_angle = angle
            else:
                # 编码器未连接，保持当前夹爪状态
                # try:
                #     current_gripper = self.env._right_robot.get_gripper_width()
                #     action[15] = min(current_gripper / self.env._gripper_max_width, 1.0)
                # except:
                #     action[15] = 0.0
                action[15] = self.env._right_robot.gripper_state
        
        return action
    
    def _read_t265_to_cartesian_action(self) -> Optional[np.ndarray]:
        """
        从 T265 读取位姿并转换为 16 维笛卡尔动作向量（无需 IK）
        
        Returns:
            16 维向量 [左臂位姿x7, 左臂夹爪, 右臂位姿x7, 右臂夹爪]，失败返回 None
            位姿格式: [x, y, z, qw, qx, qy, qz]
        """
        action = np.zeros(16, dtype=np.float32)
        
        # ========== 左臂 ==========
        if self.left_t265.is_connected:
            try:
                t265_pos, t265_quat = self.left_t265.get_pose()
                
                if self.left_t265_ref_pos is not None and self.left_robot_ref_pos is not None:
                    # 转换为机器人位姿（未滤波）
                    robot_pose = self._t265_to_robot_pose(
                        t265_pos, t265_quat,
                        self.left_t265_ref_pos, self.left_t265_ref_rot,
                        self.left_robot_ref_pos, self.left_robot_ref_rot
                    )
                    
                    # 应用滤波
                    target_pos = robot_pose[:3]
                    target_quat = robot_pose[3:]
                    
                    filtered_pos, filtered_quat = self._apply_filter(
                        target_pos, target_quat,
                        self.left_last_filtered_pos, self.left_last_filtered_quat
                    )
                    
                    # 更新滤波状态
                    self.left_last_filtered_pos = filtered_pos.copy()
                    self.left_last_filtered_quat = filtered_quat.copy()
                    
                    # 直接使用滤波后的位姿
                    action[0:3] = filtered_pos
                    action[3:7] = filtered_quat
                else:
                    # 参考位姿未设置，使用当前机器人位姿
                    if self.env._left_robot is not None:
                        current_pose = self.env._left_robot.get_tcp_pose()
                        action[0:7] = current_pose
            except Exception as e:
                if self.verbose:
                    print(f"左臂位姿读取错误: {e}")
                # 使用当前机器人位姿
                if self.env._left_robot is not None:
                    current_pose = self.env._left_robot.get_tcp_pose()
                    action[0:7] = current_pose
        else:
            # T265 未连接，使用当前机器人位姿
            if self.env._left_robot is not None:
                current_pose = self.env._left_robot.get_tcp_pose()
                action[0:7] = current_pose
        
        # ========== 左臂夹爪（从编码器读取）==========
        if self.env._left_gripper_enabled:
            if self.left_encoder is not None and self.left_encoder.is_connected:
                angle = self.left_encoder.get_angle()
                angle = max(0.0, min(self.gripper_max_angle, angle))
                # action[7] = angle / self.gripper_max_angle
                if abs(angle - self.last_left_encoder_angle) > 0.1:
                    if angle > self.last_left_encoder_angle:
                        action[7] = 1.0 #夹爪打开
                    else:
                        action[7] = 0.0 #夹爪闭合
                else:
                    action[7] = self.env._left_robot.gripper_state
                self.last_left_encoder_angle = angle
            else:
                # try:
                #     current_gripper = self.env._left_robot.get_gripper_width()
                #     action[7] = min(current_gripper / self.env._gripper_max_width, 1.0)
                # except:
                #     action[7] = 0.0
                action[7] = self.env._left_robot.gripper_state
        
        # ========== 右臂 ==========
        if self.right_t265.is_connected:
            try:
                t265_pos, t265_quat = self.right_t265.get_pose()
                
                if self.right_t265_ref_pos is not None and self.right_robot_ref_pos is not None:
                    # 转换为机器人位姿（未滤波）
                    robot_pose = self._t265_to_robot_pose(
                        t265_pos, t265_quat,
                        self.right_t265_ref_pos, self.right_t265_ref_rot,
                        self.right_robot_ref_pos, self.right_robot_ref_rot
                    )
                    
                    # 应用滤波
                    target_pos = robot_pose[:3]
                    target_quat = robot_pose[3:]
                    
                    filtered_pos, filtered_quat = self._apply_filter(
                        target_pos, target_quat,
                        self.right_last_filtered_pos, self.right_last_filtered_quat
                    )
                    
                    # 更新滤波状态
                    self.right_last_filtered_pos = filtered_pos.copy()
                    self.right_last_filtered_quat = filtered_quat.copy()
                    
                    # 直接使用滤波后的位姿
                    action[8:11] = filtered_pos
                    action[11:15] = filtered_quat
                else:
                    # 参考位姿未设置，使用当前机器人位姿
                    if self.env._right_robot is not None:
                        current_pose = self.env._right_robot.get_tcp_pose()
                        action[8:15] = current_pose
            except Exception as e:
                if self.verbose:
                    print(f"右臂位姿读取错误: {e}")
                # 使用当前机器人位姿
                if self.env._right_robot is not None:
                    current_pose = self.env._right_robot.get_tcp_pose()
                    action[8:15] = current_pose
        else:
            # T265 未连接，使用当前机器人位姿
            if self.env._right_robot is not None:
                current_pose = self.env._right_robot.get_tcp_pose()
                action[8:15] = current_pose
        
        # ========== 右臂夹爪（从编码器读取）==========
        if self.env._right_gripper_enabled:
            if self.right_encoder is not None and self.right_encoder.is_connected:
                angle = self.right_encoder.get_angle()
                angle = max(0.0, min(self.gripper_max_angle, angle))
                # action[15] = angle / self.gripper_max_angle
                if abs(angle - self.last_right_encoder_angle) > 0.1:
                    if angle > self.last_right_encoder_angle:
                        action[15] = 1.0 #夹爪打开
                    else:
                        action[15] = 0.0 #夹爪闭合
                else:
                    action[15] = self.env._right_robot.gripper_state
                self.last_right_encoder_angle = angle
            else:
                # try:
                #     current_gripper = self.env._right_robot.get_gripper_width()
                #     action[15] = min(current_gripper / self.env._gripper_max_width, 1.0)
                # except:
                #     action[15] = 0.0
                action[15] = self.env._right_robot.gripper_state
        
        return action
    
    def _on_key_press(self, key):
        """键盘按下事件"""
        try:
            if key == keyboard.KeyCode.from_char('s'):
                with self._kb_lock:
                    self._key_pressed['s'] = True
            elif key == keyboard.KeyCode.from_char('q'):
                with self._kb_lock:
                    self._key_pressed['q'] = True
            elif key == keyboard.Key.esc:
                with self._kb_lock:
                    self._key_pressed['esc'] = True
            elif key == keyboard.Key.space:
                # 空格键按下：断开离合器
                with self._kb_lock:
                    if not self._space_pressed:  # 避免重复触发
                        self._space_pressed = True
                        self.is_following = False
                        if self.verbose:
                            print("\n[离合器] 断开控制 (Space Pressed)")
        except AttributeError:
            pass
    
    def _on_key_release(self, key):
        """键盘释放事件"""
        try:
            if key == keyboard.KeyCode.from_char('s'):
                with self._kb_lock:
                    self._key_pressed['s'] = False
            elif key == keyboard.KeyCode.from_char('q'):
                with self._kb_lock:
                    self._key_pressed['q'] = False
            elif key == keyboard.Key.space:
                # 空格键释放：恢复离合器，重置参考位姿
                with self._kb_lock:
                    self._space_pressed = False
                    self.is_following = True
                    if self.verbose:
                        print("\n[离合器] 恢复跟随 (Space Released)")
                # 重置参考位姿（在锁外执行，避免死锁）
                self._reset_reference()
        except AttributeError:
            pass
    
    def _handle_keyboard_input(self):
        """处理键盘输入"""
        with self._kb_lock:
            # s 和 q 键已禁用，保留代码但不处理
            if self._key_pressed['s']:
                self._key_pressed['s'] = False  # 重置标志，但不执行任何操作
            
            if self._key_pressed['q']:
                self._key_pressed['q'] = False  # 重置标志，但不执行任何操作
            
            if self._key_pressed['esc']:
                # ESC 键：保存数据并退出程序
                if self._record_mode and self._episode_data:
                    print("\n💾 保存记录数据...")
                    save_episode_to_hdf5(self._episode_data, self._record_directory, self.env._prompt)
                    print("✅ 记录已保存")
                    self._episode_data = []  # 清空数据
                else:
                    if self._episode_data:
                        print("\n💾 保存记录数据...")
                        save_episode_to_hdf5(self._episode_data, self._record_directory, self.env._prompt)
                        print("✅ 记录已保存")
                    else:
                        print("\n⚠️  没有可保存的记录数据")
                print("\n🛑 退出程序...")
                self._running = False
                self._key_pressed['esc'] = False
    
    def run(self):
        """运行遥操作系统主循环"""
        print("\n" + "=" * 50)
        print("  双臂遥操作系统已就绪")
        print(f"  控制模式: {'关节控制 (Joint)' if self.control_mode == 'joint' else '笛卡尔控制 (Cartesian)'}")
        print("  ✅ 数据记录已自动开启")
        print("  [Space] 松开 (默认): 跟随模式 (Follow)")
        print("  [Space] 按住:        断开模式 (Hold) - 暂停记录")
        print("  [ESC] 保存数据并退出程序")
        if self.env._prompt:
            print(f"  任务: {self.env._prompt}")
        print(f"  记录目录: {self._record_directory}")
        print("=" * 50 + "\n")
        
        # 创建显示窗口
        window_name = "Dual Arm Observations"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # 启动键盘监听
        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self._kb_listener.start()
        
        self._running = True
        try:
            while self._running:
                start_time = time.time()
                
                # 处理键盘输入
                self._handle_keyboard_input()
                
                try:
                    obs = self.env.get_observation()
                    # 显示图像（不处理键盘输入，键盘输入由 pynput 处理）
                    display_observations(obs, window_name=window_name, recording=self._record_mode)
                    # display_raw_observations(obs, window_name=window_name)
                    # 在记录模式下，记录观测和动作
                    if self._record_mode:
                        # 记录观测和对应的动作
                        step_data = {
                            'observation': obs.copy(),
                        }
                        self._episode_data.append(step_data)
                        if self.verbose and len(self._episode_data) % 50 == 0:
                            print(f"记录中... 步数: {len(self._episode_data)}", end='\r')
                except Exception as e:
                    if self.verbose:
                        print(f"获取观测失败: {e}")
                    # 如果获取观测失败，跳过本次动作
                    continue
                
                # 根据离合器状态决定是否执行动作
                if self.is_following:
                    if self.control_mode == "joint":
                        # 关节控制模式：从 T265 读取位姿并通过 IK 转换为关节角
                        action_data = self._read_t265_and_convert_to_joints()
                        if action_data is not None:
                            action = {"actions": action_data}
                            self.env.apply_action(action)
                    else:
                        # 笛卡尔控制模式：从 T265 读取位姿直接发送
                        action_data = self._read_t265_to_cartesian_action()
                        if action_data is not None:
                            action = {"actions": action_data}
                            self.env.apply_cartesian_action(action)
                else:
                    # 断开模式：保持当前位置不动（发送当前状态作为动作）
                    # 这样可以让机器人保持在当前位置
                    pass  # 不发送动作，机器人会保持当前位置
                
                # 频率控制
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n收到中断信号，正在退出...")
            # 如果正在记录，保存部分数据
            if self._episode_data:
                print("保存部分记录的数据...")
                save_episode_to_hdf5(self._episode_data, self._record_directory, self.env._prompt)
        finally:
            # # 在清理前，如果还有未保存的数据，尝试保存
            # if self._episode_data:
            #     try:
            #         print("\n保存剩余记录数据...")
            #         save_episode_to_hdf5(self._episode_data, self._record_directory, self.env._prompt)
            #     except Exception as e:
            #         print(f"保存数据时出错: {e}")
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        self._running = False
        
        if self._kb_listener:
            self._kb_listener.stop()
        
        # 断开 T265
        if self.left_t265.is_connected:
            self.left_t265.disconnect()
        
        if self.right_t265.is_connected:
            self.right_t265.disconnect()
        
        # 断开编码器
        if self.left_encoder is not None and self.left_encoder.is_connected:
            self.left_encoder.disconnect()
        
        if self.right_encoder is not None and self.right_encoder.is_connected:
            self.right_encoder.disconnect()
        
        # 关闭 OpenCV 窗口
        cv2.destroyAllWindows()
        
        if self.verbose:
            print("遥操作系统已关闭")


def save_episode_to_hdf5(episode_data: list, directory: str = "recorded_data", prompt: str = ""):
    """
    保存 episode 数据到 HDF5 文件
    
    Args:
        episode_data: 包含观测和动作的列表，每个元素为 {'observation': obs, 'action': action}
        directory: 保存目录
        prompt: 任务提示文本
    """
    if not episode_data:
        print("No data to save.")
        return

    # 确保保存目录存在
    os.makedirs(directory, exist_ok=True)
    
    # 生成唯一文件名
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 将prompt转换为文件名安全的格式
    if prompt:
        safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_prompt = safe_prompt.replace(",", "").replace(".", "")
        max_prompt_len = 80
        if len(safe_prompt) > max_prompt_len:
            safe_prompt = safe_prompt[:max_prompt_len]
        filename = os.path.join(directory, f"episode_{timestamp_str}_{safe_prompt}.hdf5")
    else:
        filename = os.path.join(directory, f"episode_{timestamp_str}.hdf5")
    
    print(f"Saving episode to {filename}...")

    with h5py.File(filename, "w") as f:
        first_obs = episode_data[0]['observation']
        #first_action = episode_data[0]['action']
        num_steps = len(episode_data)

        # 创建数据集
        obs_group = f.create_group("observations")
        img_group = obs_group.create_group("images")

        state_shape = first_obs['state'].shape
        obs_group.create_dataset("qpos", (num_steps,) + state_shape, dtype=first_obs['state'].dtype)

        if 'pose_state' in first_obs:
            pose_state_shape = first_obs['pose_state'].shape
            obs_group.create_dataset("pose", (num_steps,) + pose_state_shape, dtype=first_obs['pose_state'].dtype)

        # 如果有 prev_state，则创建数据集
        if 'prev_state' in first_obs:
            prev_state_shape = first_obs['prev_state'].shape
            obs_group.create_dataset("prev_qpos", (num_steps,) + prev_state_shape, dtype=first_obs['prev_state'].dtype)

        if 'prev_pose_state' in first_obs:
            prev_pose_state_shape = first_obs['prev_pose_state'].shape
            obs_group.create_dataset("prev_pose", (num_steps,) + prev_pose_state_shape, dtype=first_obs['prev_pose_state'].dtype)

        action_shape = state_shape #first_action.shape
        f.create_dataset("action", (num_steps,) + action_shape, dtype=first_obs['state'].dtype)
        f.create_dataset("action_pose", (num_steps,) + action_shape, dtype=first_obs['state'].dtype)

        for cam_name, img in first_obs['images'].items():
            img_shape = img.shape
            img_group.create_dataset(cam_name, (num_steps,) + img_shape, dtype=img.dtype)

        if 'prompt' in first_obs:
            prompt_len = len(first_obs['prompt'])
            fix_prompt_type = np.dtype(f'S{prompt_len}')
            f.create_dataset("task", (num_steps,), dtype=fix_prompt_type)

        if 'timestamps' in first_obs and first_obs['timestamps']:
            timestamps_group = obs_group.create_group("timestamps")
            for ts_key in first_obs['timestamps'].keys():
                timestamps_group.create_dataset(ts_key, (num_steps,), dtype=np.float64)

        # 填充数据
        for i in range(num_steps):
            obs_group['qpos'][i] = episode_data[i]['observation']['state']
            obs_group['pose'][i] = episode_data[i]['observation']['pose_state']
            #f['action'][i] = episode_data[i]['action']
            action_index = min(i+1, num_steps - 1)
            f['action'][i] = episode_data[action_index]['observation']['state'] 
            f['action_pose'][i] = episode_data[action_index]['observation']['pose_state']

            # 如果有 prev_state，则保存
            if 'prev_state' in first_obs and 'prev_state' in episode_data[i]['observation']:
                obs_group['prev_qpos'][i] = episode_data[i]['observation']['prev_state']
            if 'prev_pose_state' in first_obs and 'prev_pose_state' in episode_data[i]['observation']:
                obs_group['prev_pose'][i] = episode_data[i]['observation']['prev_pose_state']
            
            for cam_name, img in episode_data[i]['observation']['images'].items():
                img_group[cam_name][i] = img
                
            if 'prompt' in first_obs:
                f['task'][i] = episode_data[i]['observation']['prompt']

            if 'timestamps' in first_obs and 'timestamps' in episode_data[i]['observation']:
                for ts_key, ts_value in episode_data[i]['observation']['timestamps'].items():
                    if ts_key in timestamps_group:
                        timestamps_group[ts_key][i] = ts_value

    print(f"Successfully saved {num_steps} steps.")


def list_t265_devices():
    """列出所有可用的 T265 设备序列号"""
    ctx = rs.context()
    devices = ctx.query_devices()
    t265_serials = []
    for dev in devices:
        if dev.get_info(rs.camera_info.name) == "Intel RealSense T265":
            serial = dev.get_info(rs.camera_info.serial_number)
            t265_serials.append(serial)
    return t265_serials

def load_dual_arm_teleop_config(config_path: str) -> Dict[str, Any]:
    """
    加载双臂遥操作配置文件
    """
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="双臂遥操作系统")
    parser.add_argument('--env-config', default='examples/flexiv/dual_arm_env_config.yaml',
                        help='双臂环境配置文件路径')
    parser.add_argument('--teleop-config', default='examples/flexiv/dual_arm_teleop_config.yaml',
                        help='双臂遥操作配置文件路径')
    parser.add_argument('--record-dir', default='recorded_data', help='记录数据保存目录')
    parser.add_argument('--prompt', default='', help='任务提示文本')
    args = parser.parse_args()
    
    teleop_config = load_dual_arm_teleop_config(args.teleop_config)

    # 列出可用的 T265 设备
    print("\n检测 T265 设备...")
    available_t265 = list_t265_devices()
    print(f"发现 {len(available_t265)} 个 T265: {available_t265}")
    
    if len(available_t265) < 2:
        print("警告: 检测到的 T265 数量少于 2，双臂控制可能无法正常工作!")
        return 0
    
    # 从配置文件中获取参数
    left_arm_config = teleop_config.get('left_arm', {})
    right_arm_config = teleop_config.get('right_arm', {})
    control_config = teleop_config.get('control', {})
    
    # T265 序列号
    left_t265_serial = left_arm_config.get('t265_serial', '')
    right_t265_serial = right_arm_config.get('t265_serial', '')
    
    # 编码器配置
    left_encoder_port = left_arm_config.get('encoder_port', '')
    right_encoder_port = right_arm_config.get('encoder_port', '')
    left_encoder_direction = left_arm_config.get('encoder_direction', 1)
    right_encoder_direction = right_arm_config.get('encoder_direction', 1)
    left_encoder_scale = left_arm_config.get('encoder_scale', 2.0)
    right_encoder_scale = right_arm_config.get('encoder_scale', 2.0)
    
    # 控制参数
    control_mode = control_config.get('control_mode', 'joint')
    control_frequency = control_config.get('frequency', 20.0)
    position_scale = control_config.get('position_scale', 1.5)
    human_to_robot_direction = control_config.get('human_to_robot_direction', 'opposite')
    
    # 滤波参数
    filter_alpha = control_config.get('alpha', 0.8)
    rot_deadband_deg = control_config.get('rot_deadband_deg', 1.0)
    rot_fullspeed_deg = control_config.get('rot_fullspeed_deg', 8.0)
    rot_alpha_min = control_config.get('rot_alpha_min', 0.05)
    pos_deadband_mm = control_config.get('pos_deadband_mm', 1.0)
    pos_fullspeed_mm = control_config.get('pos_fullspeed_mm', 10.0)
    pos_alpha_min = control_config.get('pos_alpha_min', 0.05)
    pos_alpha_max = control_config.get('pos_alpha_max', 0.6)

    # 创建环境
    print(f"\n加载配置文件: {args.env_config}")
    env = create_dual_arm_env_from_config(args.env_config)
    print('env._prompt:', env._prompt)
    
    try:
        env.reset()
        
        # 创建遥操作系统
        teleop = DualArmTeleopSystem(
            env=env,
            left_t265_serial=left_t265_serial,
            right_t265_serial=right_t265_serial,
            left_encoder_port=left_encoder_port,
            right_encoder_port=right_encoder_port,
            left_encoder_direction=left_encoder_direction,
            right_encoder_direction=right_encoder_direction,
            left_encoder_scale=left_encoder_scale,
            right_encoder_scale=right_encoder_scale,
            control_frequency=control_frequency,
            position_scale=position_scale,
            filter_alpha=filter_alpha,
            rot_deadband_deg=rot_deadband_deg,
            rot_fullspeed_deg=rot_fullspeed_deg,
            rot_alpha_min=rot_alpha_min,
            pos_deadband_mm=pos_deadband_mm,
            pos_fullspeed_mm=pos_fullspeed_mm,
            pos_alpha_min=pos_alpha_min,
            pos_alpha_max=pos_alpha_max,
            control_mode=control_mode,
            verbose=True,
            human_to_robot_direction=human_to_robot_direction
        )
        
        # 设置记录目录
        teleop._record_directory = args.record_dir
        
        # 运行遥操作系统
        teleop.run()
    
    finally:
        env.close()
        print("程序已退出")


if __name__ == "__main__":
    main()

