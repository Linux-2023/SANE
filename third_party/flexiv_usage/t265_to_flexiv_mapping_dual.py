#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂 T265 -> Flexiv 映射控制器
基于 t265_to_flexiv_mapping.py 扩展为双臂版本。
所有配置从 YAML 文件读取。
"""

import time
import threading
import numpy as np
import argparse
import yaml
from typing import Optional, Dict, Any, List
import sys
import subprocess

# 自动检查并安装依赖
def install_package(package):
    print(f"正在安装 {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"无法安装 {package}: {e}")
        print(f"请手动运行: pip install {package}")
        sys.exit(1)

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("未找到 scipy，尝试安装...")
    install_package("scipy")
    from scipy.spatial.transform import Rotation as R

try:
    from pynput import keyboard
except ImportError:
    print("未找到 pynput，尝试安装...")
    install_package("pynput")
    from pynput import keyboard

try:
    import pyrealsense2 as rs
except ImportError:
    print("未找到 pyrealsense2，尝试安装...")
    install_package("pyrealsense2")
    import pyrealsense2 as rs

import flexivrdk
from flexiv_robot import FlexivRobot
from collision_check.dual_arm_collision_checker import create_checker, DualArmCollisionChecker


def list_t265_devices() -> List[str]:
    """列出所有可用的 T265 设备序列号"""
    ctx = rs.context()
    devices = ctx.query_devices()
    t265_serials = []
    for dev in devices:
        if dev.get_info(rs.camera_info.name) == "Intel RealSense T265":
            serial = dev.get_info(rs.camera_info.serial_number)
            t265_serials.append(serial)
    return t265_serials


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


class SingleArmController:
    """
    单臂控制器（用于双臂系统中的每个臂）
    支持通过 IK 转换位姿到关节，并进行碰撞检测
    """
    def __init__(
        self,
        arm_name: str,
        t265_reader: T265Reader,
        encoder_reader: Optional[EncoderReader],
        flexiv_robot: FlexivRobot,
        control_frequency: float = 20.0,
        position_scale: float = 1.5,
        filter_alpha: float = 0.8,
        rot_deadband_deg: float = 1.0,
        rot_fullspeed_deg: float = 8.0,
        rot_alpha_min: float = 0.05,
        pos_deadband_m: float = 0.001,
        pos_fullspeed_m: float = 0.01,
        pos_alpha_min: float = 0.05,
        pos_alpha_max: float = 0.6,
        t265_rot_center_offset_m: Optional[np.ndarray] = None,
        gripper_enabled: bool = False,
        verbose: bool = True,
        is_left_arm: bool = True,
        collision_enabled: bool = True,
        collision_safety_threshold: float = 0.03
    ):
        self.arm_name = arm_name
        self.t265 = t265_reader
        self.encoder = encoder_reader
        self.robot = flexiv_robot
        self.dt = 1.0 / control_frequency
        self.pos_scale = position_scale
        
        # 姿态自适应滤波参数
        self.rot_deadband_rad = float(np.deg2rad(rot_deadband_deg))
        self.rot_fullspeed_rad = float(np.deg2rad(rot_fullspeed_deg))
        self.rot_alpha_min = float(rot_alpha_min)
        self.rot_alpha_max = float(filter_alpha)
        
        # 位置自适应滤波参数
        self.pos_deadband_m = float(pos_deadband_m)
        self.pos_fullspeed_m = float(pos_fullspeed_m)
        self.pos_alpha_min = float(pos_alpha_min)
        self.pos_alpha_max = float(pos_alpha_max)

        self.gripper_enabled = gripper_enabled
        self.verbose = verbose

        # T265 旋转中心偏移
        if t265_rot_center_offset_m is None:
            self.t265_rot_center_offset_m = np.array([0.0, -0.04, 0.0], dtype=float)
        else:
            self.t265_rot_center_offset_m = np.array(t265_rot_center_offset_m, dtype=float).reshape(3)

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # 离合状态
        self.is_following = True
        
        # 引用位姿
        self.t265_ref_pos: Optional[np.ndarray] = None
        self.t265_ref_rot: Optional[R] = None
        self.robot_ref_pos: Optional[np.ndarray] = None
        self.robot_ref_rot: Optional[R] = None
        
        # 滤波状态
        self.last_filtered_pos: Optional[np.ndarray] = None
        self.last_filtered_quat: Optional[np.ndarray] = None
        
        # 夹爪控制状态
        self.last_gripper_move_time = 0.0
        self.gripper_move_interval = 0.2
        
        # 旋转轴映射矩阵：人对着机械臂
        # self.M_rotvec = np.array([
        #     [0, 1,  0],
        #     [1, 0,  0],
        #     [0, 0, -1],
        # ], dtype=float)

        # 人与机器人同向时的旋转矩阵
        self.M_rotvec = np.array([
            [0, -1,  0],
            [-1, 0,  0],
            [0, 0, -1],
        ], dtype=float)
        
        # 碰撞检测相关
        self.is_left_arm = is_left_arm
        self.collision_enabled = collision_enabled
        self.collision_safety_threshold = collision_safety_threshold
        
        # 这些由 DualArmController 设置
        self._collision_checker: Optional[DualArmCollisionChecker] = None
        self._model: Optional[flexivrdk.Model] = None
        self._get_other_arm_joints: Optional[callable] = None
        self._update_own_joints: Optional[callable] = None
        
        # 当前关节角度（用于共享）
        self._current_joints: Optional[np.ndarray] = None
        self._joints_lock = threading.Lock()

    def setup_collision_detection(
        self,
        collision_checker: DualArmCollisionChecker,
        model: flexivrdk.Model,
        get_other_arm_joints: callable,
        update_own_joints: callable
    ):
        """设置碰撞检测相关组件（由 DualArmController 调用）"""
        self._collision_checker = collision_checker
        self._model = model
        self._get_other_arm_joints = get_other_arm_joints
        self._update_own_joints = update_own_joints
        
        # 初始化当前关节角度
        self._current_joints = np.array(self.robot.get_joint_positions())
        if self.verbose:
            print(f"[{self.arm_name}] 碰撞检测已配置")
    
    def get_current_joints(self) -> np.ndarray:
        """获取当前关节角度（线程安全）"""
        with self._joints_lock:
            if self._current_joints is None:
                return np.array(self.robot.get_joint_positions())
            return self._current_joints.copy()
    
    def _update_current_joints(self, joints: np.ndarray):
        """更新当前关节角度（线程安全）"""
        with self._joints_lock:
            self._current_joints = joints.copy()
    
    def _solve_ik(self, target_pose: np.ndarray) -> tuple:
        """
        使用 Model.reachable 求解 IK
        
        Args:
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]
            
        Returns:
            (is_reachable, ik_solution): 是否可达和关节解
        """
        if self._model is None:
            return False, None
        
        try:
            # 获取当前关节作为 IK 种子
            seed_joints = list(self.get_current_joints())
            
            # 调用 reachable 接口
            is_reachable, ik_solution = self._model.reachable(
                list(target_pose), 
                seed_joints, 
                False  # 不允许自由姿态
            )
            
            if is_reachable:
                return True, np.array(ik_solution)
            else:
                return False, None
        except Exception as e:
            if self.verbose:
                print(f"[{self.arm_name}] IK 求解失败: {e}")
            return False, None
    
    def _check_collision(self, target_joints: np.ndarray) -> tuple:
        """
        检查碰撞
        
        Args:
            target_joints: 本臂的目标关节角度
            
        Returns:
            (is_safe, min_distance): 是否安全和最小距离
        """
        if self._collision_checker is None or self._get_other_arm_joints is None:
            return True, float('inf')
        
        try:
            other_joints = self._get_other_arm_joints()
            if other_joints is None:
                return True, float('inf')
            
            if self.is_left_arm:
                q_left = target_joints
                q_right = other_joints
            else:
                q_left = other_joints
                q_right = target_joints
            
            # 根据阈值选择检测方式
            if self.collision_safety_threshold <= 0:
                # 只检测是否真正碰撞（快速）
                is_collision = self._collision_checker.check_collision(q_left, q_right)
                is_safe = not is_collision
                min_dist = -1.0 if is_collision else 1.0  # 简化的距离指示
            else:
                # 基于距离阈值检测（需要计算距离）
                min_dist = self._collision_checker.compute_min_distance(q_left, q_right)
                is_safe = min_dist > self.collision_safety_threshold
            
            return is_safe, min_dist
        except Exception as e:
            if self.verbose:
                print(f"[{self.arm_name}] 碰撞检测失败: {e}")
            return True, float('inf')

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[{self.arm_name}] 控制循环已启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        print(f"[{self.arm_name}] 控制循环已停止")

    def set_following(self, following: bool):
        """设置跟随状态"""
        was_following = self.is_following
        self.is_following = following
        if following and not was_following:
            # 从断开恢复到跟随，重置参考点
            self.reset_reference()

    def reset_reference(self):
        """重置参考位姿"""
        with self._lock:
            # 获取 T265 当前位姿
            pos, quat = self.t265.get_pose()
            q = quat  # [qw, qx, qy, qz]
            self.t265_ref_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            self.t265_ref_pos = self._t265_center_pos(pos, self.t265_ref_rot)

            # 获取 Robot 当前位姿
            robot_pose = self.robot.get_tcp_pose()
            self.robot_ref_pos = robot_pose[:3].copy()
            rq = robot_pose[3:]
            self.robot_ref_rot = R.from_quat([rq[1], rq[2], rq[3], rq[0]])
            
            # 重置滤波器状态
            self.last_filtered_pos = self.robot_ref_pos.copy()
            self.last_filtered_quat = rq.copy()
            
            if self.verbose:
                print(f"[{self.arm_name}] 参考位姿已重置")

    def _t265_center_pos(self, pos: np.ndarray, rot: R) -> np.ndarray:
        """补偿旋转中心"""
        return np.array(pos, dtype=float).reshape(3) + rot.apply(self.t265_rot_center_offset_m)

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _adaptive_alpha(self, err: float, deadband: float, fullspeed: float, alpha_min: float, alpha_max: float) -> float:
        if err <= deadband:
            return 0.0
        if fullspeed <= deadband:
            return alpha_max
        t = (err - deadband) / (fullspeed - deadband)
        k = self._smoothstep01(t)
        return alpha_min + k * (alpha_max - alpha_min)

    def _loop(self):
        self.reset_reference()

        while self._running:
            start_time = time.time()

            if self.is_following:
                try:
                    # 获取 T265 当前位姿
                    pos, quat = self.t265.get_pose()
                    q = quat
                    curr_t265_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                    curr_t265_pos = self._t265_center_pos(pos, curr_t265_rot)

                    with self._lock:
                        if self.t265_ref_pos is None or self.robot_ref_pos is None:
                            continue
                        
                        # 计算增量
                        delta_p_t265 = curr_t265_pos - self.t265_ref_pos
                        delta_r_t265 = self.t265_ref_rot.inv() * curr_t265_rot

                        delta_p_robot = np.zeros(3)

                        # # 位置映射：人对着机械臂
                        # delta_p_robot[0] = -delta_p_t265[0]
                        # delta_p_robot[1] = delta_p_t265[2]
                        # delta_p_robot[2] = delta_p_t265[1]

                        # 位置映射：人与机械臂同向
                        delta_p_robot[0] = delta_p_t265[0]
                        delta_p_robot[1] = -delta_p_t265[2]
                        delta_p_robot[2] = delta_p_t265[1]

                        # 平移放缩
                        delta_p_robot *= self.pos_scale

                        # 姿态映射
                        rotvec_t265 = delta_r_t265.as_rotvec()
                        rotvec_tcp = self.M_rotvec @ rotvec_t265
                        delta_r_robot = R.from_rotvec(rotvec_tcp)
                        target_rot = self.robot_ref_rot * delta_r_robot
                        
                        # 计算目标位姿
                        target_pos = self.robot_ref_pos + delta_p_robot
                        target_quat = target_rot.as_quat()  # [qx, qy, qz, qw]
                        target_quat_flexiv = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])

                        # 滤波
                        if self.last_filtered_pos is None:
                            self.last_filtered_pos = target_pos.copy()
                        if self.last_filtered_quat is None:
                            self.last_filtered_quat = target_quat_flexiv.copy()

                        # 位置滤波
                        pos_err = float(np.linalg.norm(target_pos - self.last_filtered_pos))
                        pos_alpha = self._adaptive_alpha(
                            err=pos_err,
                            deadband=self.pos_deadband_m,
                            fullspeed=self.pos_fullspeed_m,
                            alpha_min=self.pos_alpha_min,
                            alpha_max=self.pos_alpha_max,
                        )
                        if pos_alpha <= 0.0:
                            filtered_pos = self.last_filtered_pos
                        else:
                            filtered_pos = self.last_filtered_pos + pos_alpha * (target_pos - self.last_filtered_pos)

                        # 姿态滤波
                        if np.dot(self.last_filtered_quat, target_quat_flexiv) < 0:
                            target_quat_flexiv = -target_quat_flexiv

                        last_r = R.from_quat([
                            self.last_filtered_quat[1], self.last_filtered_quat[2], 
                            self.last_filtered_quat[3], self.last_filtered_quat[0]
                        ])
                        targ_r = R.from_quat([
                            target_quat_flexiv[1], target_quat_flexiv[2], 
                            target_quat_flexiv[3], target_quat_flexiv[0]
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
                            filtered_quat = self.last_filtered_quat
                        else:
                            filtered_quat = (1.0 - rot_alpha) * self.last_filtered_quat + rot_alpha * target_quat_flexiv
                            n = float(np.linalg.norm(filtered_quat))
                            if n > 1e-6:
                                filtered_quat = filtered_quat / n
                            else:
                                filtered_quat = self.last_filtered_quat

                        self.last_filtered_pos = filtered_pos
                        self.last_filtered_quat = filtered_quat

                        cmd_pose = np.concatenate([filtered_pos, filtered_quat])
                        
                        # 碰撞检测逻辑
                        should_send = True
                        target_joints = None
                        
                        if self.collision_enabled and self._model is not None:
                            # 使用 IK 将位姿转换为关节角度
                            is_reachable, ik_solution = self._solve_ik(cmd_pose)
                            
                            if not is_reachable:
                                # 位姿不可达，不发送命令
                                should_send = False
                                if self.verbose:
                                    print(f"[{self.arm_name}] ⚠️ 目标位姿不可达，跳过")
                            else:
                                target_joints = ik_solution
                                
                                # 检查碰撞
                                is_safe, min_dist = self._check_collision(target_joints)
                                
                                if not is_safe:
                                    should_send = False
                                    if self.verbose:
                                        if self.collision_safety_threshold <= 0:
                                            print(f"[{self.arm_name}] ⚠️ 碰撞! 跳过")
                                        else:
                                            print(f"[{self.arm_name}] ⚠️ 碰撞风险! 最小距离: {min_dist*1000:.1f}mm，跳过")
                        
                        if should_send:
                            if target_joints is not None and self.collision_enabled:
                                # 使用关节位置控制（更精确）
                                self.robot.send_joint_position(target_joints)
                                self._update_current_joints(target_joints)
                                if self._update_own_joints:
                                    self._update_own_joints(target_joints)
                            else:
                                # 回退到笛卡尔控制
                                self.robot.send_cartesian_pose(cmd_pose)
                                # 更新当前关节
                                curr_joints = np.array(self.robot.get_joint_positions())
                                self._update_current_joints(curr_joints)
                                if self._update_own_joints:
                                    self._update_own_joints(curr_joints)

                        # 夹爪控制
                        if self.gripper_enabled and self.encoder and self.robot.is_gripper_enabled():
                            MAX_ANGLE = 45.0
                            angle = self.encoder.get_angle()
                            angle = max(0.0, min(MAX_ANGLE, angle))
                            
                            params = self.robot.get_gripper_params()
                            min_w = params.min_width
                            max_w = params.max_width
                            
                            ratio = angle / MAX_ANGLE
                            target_width = min_w + (max_w - min_w) * ratio
                            
                            if time.time() - self.last_gripper_move_time > self.gripper_move_interval:
                                self.robot.gripper_move(target_width, velocity=0.1, wait=False)
                                self.last_gripper_move_time = time.time()

                except Exception as e:
                    print(f"[{self.arm_name} Error] {e}")
            else:
                # 断开控制期间维持 Hold
                try:
                    curr_pose = self.robot.get_tcp_pose()
                    self.robot.send_cartesian_pose(curr_pose)
                    
                    # 更新关节状态（用于碰撞检测）
                    if self.collision_enabled:
                        curr_joints = np.array(self.robot.get_joint_positions())
                        self._update_current_joints(curr_joints)
                        if self._update_own_joints:
                            self._update_own_joints(curr_joints)
                except:
                    pass

            # 频率控制
            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)


class DualArmController:
    """
    双臂控制器
    管理两个单臂控制器、统一的键盘监听和碰撞检测
    """
    def __init__(
        self, 
        left_arm: SingleArmController, 
        right_arm: SingleArmController, 
        verbose: bool = True,
        collision_enabled: bool = True,
        collision_safety_threshold: float = 0.03
    ):
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.verbose = verbose
        self.kb_listener = None
        
        # 碰撞检测相关
        self.collision_enabled = collision_enabled
        self.collision_safety_threshold = collision_safety_threshold
        self._collision_checker: Optional[DualArmCollisionChecker] = None
        self._left_model: Optional[flexivrdk.Model] = None
        self._right_model: Optional[flexivrdk.Model] = None
        
        # 共享关节状态（线程安全）
        self._left_joints = np.zeros(7)
        self._right_joints = np.zeros(7)
        self._joints_lock = threading.Lock()
        
        # 初始化碰撞检测
        if collision_enabled:
            self._setup_collision_detection()
    
    def _setup_collision_detection(self):
        """初始化碰撞检测系统"""
        try:
            # 创建碰撞检测器
            print("[DualArm] 初始化碰撞检测器...")
            self._collision_checker = create_checker(mode="capsule")
            print(f"[DualArm] 碰撞检测器已创建，检测对数: {self._collision_checker.n_collision_pairs}")
            
            # 创建 Model 实例用于 IK
            print("[DualArm] 创建 IK 模型...")
            self._left_model = flexivrdk.Model(self.left_arm.robot.robot)
            self._right_model = flexivrdk.Model(self.right_arm.robot.robot)
            
            # 初始化共享关节状态
            self._left_joints = np.array(self.left_arm.robot.get_joint_positions())
            self._right_joints = np.array(self.right_arm.robot.get_joint_positions())
            
            # 设置各臂的碰撞检测配置
            self.left_arm.setup_collision_detection(
                collision_checker=self._collision_checker,
                model=self._left_model,
                get_other_arm_joints=self._get_right_joints,
                update_own_joints=self._update_left_joints
            )
            
            self.right_arm.setup_collision_detection(
                collision_checker=self._collision_checker,
                model=self._right_model,
                get_other_arm_joints=self._get_left_joints,
                update_own_joints=self._update_right_joints
            )
            
            if self.collision_safety_threshold <= 0:
                print(f"[DualArm] ✅ 碰撞检测系统就绪，模式: 仅检测碰撞")
            else:
                print(f"[DualArm] ✅ 碰撞检测系统就绪，安全距离阈值: {self.collision_safety_threshold*1000:.1f}mm")
            
        except Exception as e:
            print(f"[DualArm] ⚠️ 碰撞检测初始化失败: {e}")
            print("[DualArm] 将禁用碰撞检测，使用笛卡尔控制")
            self.collision_enabled = False
            self.left_arm.collision_enabled = False
            self.right_arm.collision_enabled = False
    
    def _get_left_joints(self) -> np.ndarray:
        """获取左臂当前关节（线程安全）"""
        with self._joints_lock:
            return self._left_joints.copy()
    
    def _get_right_joints(self) -> np.ndarray:
        """获取右臂当前关节（线程安全）"""
        with self._joints_lock:
            return self._right_joints.copy()
    
    def _update_left_joints(self, joints: np.ndarray):
        """更新左臂关节（线程安全）"""
        with self._joints_lock:
            self._left_joints = joints.copy()
    
    def _update_right_joints(self, joints: np.ndarray):
        """更新右臂关节（线程安全）"""
        with self._joints_lock:
            self._right_joints = joints.copy()
        
    def start(self):
        """启动双臂控制"""
        # 启动键盘监听
        self.kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.kb_listener.start()
        
        # 启动各臂控制循环
        self.left_arm.start()
        self.right_arm.start()
        
        print("[DualArm] 双臂控制器已启动")
    
    def stop(self):
        """停止双臂控制"""
        if self.kb_listener:
            self.kb_listener.stop()
        self.left_arm.stop()
        self.right_arm.stop()
        print("[DualArm] 双臂控制器已停止")
    
    def _on_press(self, key):
        if key == keyboard.Key.space:
            # 按下空格，断开两臂控制
            self.left_arm.set_following(False)
            self.right_arm.set_following(False)
            if self.verbose:
                print("\n[Clutch] 双臂断开控制 (Space Pressed)")

    def _on_release(self, key):
        if key == keyboard.Key.space:
            # 松开空格，恢复两臂跟随
            self.left_arm.set_following(True)
            self.right_arm.set_following(True)
            if self.verbose:
                print("\n[Clutch] 双臂恢复跟随 (Space Released)")


def load_dual_arm_config(config_path: str) -> Dict[str, Any]:
    """加载双臂配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_joint_list_deg(data) -> Optional[np.ndarray]:
    """解析关节角度列表（度）"""
    if data is None:
        return None
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return None
        s = s.replace('[', '').replace(']', '')
        parts = [p for p in s.replace(',', ' ').split() if p]
        try:
            vals = [float(p) for p in parts]
            return np.array(vals, dtype=float)
        except:
            return None
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return None


def setup_single_arm(
    arm_name: str,
    arm_config: Dict[str, Any],
    control_config: Dict[str, Any],
    available_t265_serials: List[str],
    used_t265_serials: List[str],
    is_left_arm: bool = True,
    collision_enabled: bool = True,
    collision_safety_threshold: float = 0.03
) -> tuple:
    """
    设置单臂
    
    Returns:
        (SingleArmController, T265Reader, EncoderReader, FlexivRobot)
    """
    robot_sn = arm_config['robot_sn']
    gripper_name = arm_config.get('gripper_name', '')
    t265_serial = arm_config.get('t265_serial', '')
    encoder_port = arm_config.get('encoder_port', '')
    encoder_direction = arm_config.get('encoder_direction', 1)
    encoder_scale = arm_config.get('encoder_scale', 2.0)
    init_joints_deg = parse_joint_list_deg(arm_config.get('init_joints_deg'))
    
    print(f"\n{'='*50}")
    print(f"正在初始化 [{arm_name}]")
    print(f"{'='*50}")
    
    # 1. 连接 T265
    print(f"[{arm_name}] 连接 T265...")
    
    # 如果未指定序列号，自动分配未使用的
    if not t265_serial and available_t265_serials:
        for serial in available_t265_serials:
            if serial not in used_t265_serials:
                t265_serial = serial
                break
    
    t265_reader = T265Reader(serial_number=t265_serial)
    if not t265_reader.connect():
        print(f"[{arm_name}] T265 连接失败!")
        sys.exit(1)
    
    used_t265_serials.append(t265_reader.serial_number)
    
    # 2. 连接编码器（如果有夹爪且指定了端口）
    encoder_reader = None
    want_gripper = bool(gripper_name) and bool(encoder_port)
    
    if want_gripper:
        print(f"[{arm_name}] 连接编码器...")
        encoder_reader = EncoderReader(
            port=encoder_port,
            direction=encoder_direction,
            scale=encoder_scale
        )
        if not encoder_reader.connect():
            print(f"[{arm_name}] 编码器连接失败，禁用夹爪控制")
            encoder_reader = None
            want_gripper = False
    
    # 3. 连接机器人
    print(f"[{arm_name}] 连接 Flexiv Robot ({robot_sn})...")
    robot = FlexivRobot(robot_sn)
    
    # 4. 初始化关节运动
    if init_joints_deg is not None:
        try:
            dof = robot.get_dof()
        except:
            dof = len(init_joints_deg)
        
        if len(init_joints_deg) != dof:
            print(f"[{arm_name}] 警告: init_joints_deg 长度与 DoF 不匹配，跳过")
        else:
            target_joints_rad = np.deg2rad(init_joints_deg)
            print(f"[{arm_name}] 初始化关节运动: {np.round(init_joints_deg, 2).tolist()}°")
            timeout = control_config.get('init_joints_timeout', 60.0)
            ok = robot.move_joint(target_joints_rad, timeout=float(timeout))
            if not ok:
                print(f"[{arm_name}] 警告: 初始化关节运动超时")
            else:
                print(f"[{arm_name}] 初始化关节运动完成")
    
    # 5. 初始化夹爪
    gripper_enabled = False
    if want_gripper and gripper_name:
        print(f"[{arm_name}] 启用夹爪 [{gripper_name}]...")
        try:
            robot.gripper_enable(gripper_name, switch_tool=False)
            gripper_enabled = True
        except Exception as e:
            print(f"[{arm_name}] 启用夹爪失败: {e}")
    
    # 6. 切换工具为 Flange
    print(f"[{arm_name}] 切换工具为 Flange...")
    try:
        tool = flexivrdk.Tool(robot.robot)
        tool.Switch("Flange")
        time.sleep(1)
        print(f"[{arm_name}] 工具已切换为 Flange")
    except Exception as e:
        print(f"[{arm_name}] 切换工具失败: {e}")
    
    # 7. 编码器校准
    if encoder_reader and encoder_reader.is_connected:
        print(f"[{arm_name}] 编码器零点校准...")
        time.sleep(1)
        encoder_reader.calibrate(print_info=True)
    
    # 8. 创建控制器
    controller = SingleArmController(
        arm_name=arm_name,
        t265_reader=t265_reader,
        encoder_reader=encoder_reader,
        flexiv_robot=robot,
        control_frequency=control_config.get('frequency', 20.0),
        position_scale=control_config.get('position_scale', 1.5),
        filter_alpha=control_config.get('alpha', 0.8),
        rot_deadband_deg=control_config.get('rot_deadband_deg', 1.0),
        rot_fullspeed_deg=control_config.get('rot_fullspeed_deg', 8.0),
        rot_alpha_min=control_config.get('rot_alpha_min', 0.05),
        pos_deadband_m=control_config.get('pos_deadband_mm', 1.0) / 1000.0,
        pos_fullspeed_m=control_config.get('pos_fullspeed_mm', 10.0) / 1000.0,
        pos_alpha_min=control_config.get('pos_alpha_min', 0.05),
        pos_alpha_max=control_config.get('pos_alpha_max', 0.6),
        gripper_enabled=gripper_enabled,
        is_left_arm=is_left_arm,
        collision_enabled=collision_enabled,
        collision_safety_threshold=collision_safety_threshold
    )
    
    return controller, t265_reader, encoder_reader, robot


def main():
    parser = argparse.ArgumentParser(description="双臂 T265 -> Flexiv 位姿映射控制")
    parser.add_argument('--config', default='dual_arm_config.yaml', help='双臂配置文件路径')
    parser.add_argument('--no-gripper', action='store_true', help='禁用所有夹爪控制')
    parser.add_argument('--no-collision', action='store_true', help='禁用碰撞检测')
    parser.add_argument('--collision-threshold', type=float, default=None, 
                        help='碰撞安全距离阈值 (米)，默认从配置文件读取或使用 0.03')
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    try:
        config = load_dual_arm_config(args.config)
    except Exception as e:
        print(f"无法加载配置文件: {e}")
        sys.exit(1)
    
    left_arm_config = config.get('left_arm', {})
    right_arm_config = config.get('right_arm', {})
    control_config = config.get('control', {})
    collision_config = config.get('collision', {})
    
    if args.no_gripper:
        left_arm_config['gripper_name'] = ''
        right_arm_config['gripper_name'] = ''
    
    # 碰撞检测配置
    collision_enabled = not args.no_collision and collision_config.get('enabled', True)
    collision_safety_threshold = args.collision_threshold or collision_config.get('safety_threshold', 0.03)
    
    if collision_enabled:
        if collision_safety_threshold <= 0:
            print(f"\n碰撞检测: 已启用，模式: 仅检测碰撞（无安全距离）")
        else:
            print(f"\n碰撞检测: 已启用，安全距离阈值: {collision_safety_threshold*1000:.1f}mm")
    else:
        print("\n碰撞检测: 已禁用")
    
    # 列出可用的 T265 设备
    print("\n检测 T265 设备...")
    available_t265 = list_t265_devices()
    print(f"发现 {len(available_t265)} 个 T265: {available_t265}")
    
    if len(available_t265) < 2:
        print("警告: 检测到的 T265 数量少于 2，双臂控制可能无法正常工作!")
    
    used_t265_serials = []
    
    # 设置左臂
    left_controller, left_t265, left_encoder, left_robot = setup_single_arm(
        arm_name="左臂",
        arm_config=left_arm_config,
        control_config=control_config,
        available_t265_serials=available_t265,
        used_t265_serials=used_t265_serials,
        is_left_arm=True,
        collision_enabled=collision_enabled,
        collision_safety_threshold=collision_safety_threshold
    )
    
    # 设置右臂
    right_controller, right_t265, right_encoder, right_robot = setup_single_arm(
        arm_name="右臂",
        arm_config=right_arm_config,
        control_config=control_config,
        available_t265_serials=available_t265,
        used_t265_serials=used_t265_serials,
        is_left_arm=False,
        collision_enabled=collision_enabled,
        collision_safety_threshold=collision_safety_threshold
    )
    
    # 创建双臂控制器
    dual_controller = DualArmController(
        left_controller, 
        right_controller,
        collision_enabled=collision_enabled,
        collision_safety_threshold=collision_safety_threshold
    )
    dual_controller.start()
    
    print("\n" + "=" * 50)
    print("  双臂映射控制已就绪")
    print("  [Space] 松开 (默认): 跟随模式 (Follow)")
    print("  [Space] 按住:        断开模式 (Hold)")
    if collision_enabled:
        if collision_safety_threshold <= 0:
            print("  碰撞检测: 已启用 (仅检测碰撞)")
        else:
            print(f"  碰撞检测: 已启用 (安全距离 {collision_safety_threshold*1000:.0f}mm)")
    else:
        print("  碰撞检测: 已禁用")
    print("  [Ctrl+C] 退出程序")
    print("=" * 50 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        dual_controller.stop()
        
        # 断开连接
        left_t265.disconnect()
        right_t265.disconnect()
        if left_encoder:
            left_encoder.disconnect()
        if right_encoder:
            right_encoder.disconnect()
        left_robot.stop()
        right_robot.stop()


if __name__ == "__main__":
    main()

