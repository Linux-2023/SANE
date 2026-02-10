#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flexiv_teleop.py

Flexiv Robot Teleoperation (No ROS dependency)

Two control modes:
1. UI Manual Control: Control robot with button interface
2. T265 Pose Teleop: Control robot with T265 camera pose

Usage:
    python flexiv_teleop.py
    
    Then select mode from the GUI

Dependencies:
    pip install flexivrdk numpy scipy

Author: LeRobot-Anything
"""

import time
import threading
import numpy as np
from typing import Optional, List
from scipy.spatial.transform import Rotation

from flexiv_robot import FlexivRobot, MotionParams


# ==================== 配置参数 ====================

# Default configuration
CONFIG = {
    'robot_sn': 'Rizon4s-063239',
    'gripper_name': 'Flexiv-GN01',
    'encoder_port': '/dev/ttyUSB0',
    'control_rate': 50.0,
    'pos_scale': 0.1,   # T265 1m -> Robot 1cm (1:100 ratio)
    'rot_scale': 0.1,    # T265 rotation -> Robot rotation (1:1 ratio)
}


# ==================== T265 位姿遥操作控制器 ====================

class T265PoseTeleop:
    """
    T265 位姿遥操作控制器
    
    将 T265 相机的位姿变化实时映射到机器人末端位姿控制。
    """
    
    def __init__(
        self,
        robot: FlexivRobot,
        encoder_port: str = '/dev/ttyUSB0',
        control_frequency: float = 50.0,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_name: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化 T265 位姿遥操作控制器
        
        Args:
            robot: FlexivRobot 实例
            encoder_port: 编码器串口
            control_frequency: 控制频率 [Hz]
            position_scale: 位置缩放因子
            rotation_scale: 旋转缩放因子
            gripper_name: 夹爪名称（None 表示不使用夹爪）
            verbose: 是否打印详细信息
        """
        self.robot = robot
        self.control_frequency = control_frequency
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_name = gripper_name
        self.verbose = verbose
        
        # 导入 ActionReader
        from t265_action_reader import ActionReader
        
        # 初始化 ActionReader
        print(f"[T265Teleop] Initializing ActionReader...")
        self.action_reader = ActionReader(
            encoder_port=encoder_port,
            use_t265=True,
            use_encoder=True
        )
        
        if not self.action_reader.connect():
            raise RuntimeError("ActionReader connection failed!")
        
        # 使能夹爪（如果有）
        if gripper_name:
            print(f"[T265Teleop] Enabling gripper [{gripper_name}]...")
            self.robot.gripper_enable(gripper_name)
        
        # 线程控制
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        
        # 参考位姿
        self._t265_reference_pose: Optional[np.ndarray] = None
        self._robot_reference_pose: Optional[np.ndarray] = None
        self._t265_reference_angle: Optional[float] = None
        
        # 坐标系映射矩阵（T265 坐标系 -> 机器人坐标系）
        # 默认假设：T265 的 X -> 机器人的 X, Y -> Y, Z -> Z
        # 可以根据实际安装调整
        self.axis_mapping = np.array([
            [1, 0, 0],   # T265 X -> Robot X
            [0, 1, 0],   # T265 Y -> Robot Y
            [0, 0, 1]    # T265 Z -> Robot Z
        ], dtype=float)
        
        # 夹爪参数
        # 角度 0° = 闭合 (gripper_min), 角度 90° = 张开 (gripper_max)
        self.gripper_min = 0.0      # 闭合时宽度 (m)
        self.gripper_max = 0.08     # 最大张开宽度 (m)
        self.gripper_angle_max = 90.0  # 最大角度 (度)
        
        # 夹爪启用状态
        self.gripper_enabled = False
        if gripper_name:
            try:
                self.gripper_enabled = True
            except:
                self.gripper_enabled = False
        
        # 夹爪控制状态
        self._last_gripper_width = -1.0  # 上次发送的夹爪宽度
        self._gripper_threshold = 0.0005  # 宽度变化阈值 (0.5mm)
        self._filtered_gripper_width = 0.0  # 滤波后的夹爪宽度
        
        # ====== 滤波参数 ======
        # 位姿滤波（指数移动平均）
        # alpha 越小越平滑，但响应越慢；越大响应越快但越抖
        self.position_filter_alpha = 0.3   # 位置滤波系数 (0.0-1.0)
        self.rotation_filter_alpha = 0.3   # 旋转滤波系数 (0.0-1.0)
        self.gripper_filter_alpha = 0.2    # 夹爪滤波系数 (0.0-1.0)
        
        # 滤波状态
        self._filtered_position = None     # 滤波后的位置增量
        self._filtered_rotation = None     # 滤波后的旋转增量
        
        print("[T265Teleop] Initialized")
        print(f"  Position filter alpha: {self.position_filter_alpha}")
        print(f"  Rotation filter alpha: {self.rotation_filter_alpha}")
        print(f"  Gripper filter alpha: {self.gripper_filter_alpha}")
    
    def set_axis_mapping(self, mapping: np.ndarray):
        """
        设置坐标系映射矩阵
        
        Args:
            mapping: 3x3 矩阵，将 T265 坐标变换到机器人坐标
        """
        self.axis_mapping = np.array(mapping, dtype=float)
        print(f"[T265Teleop] Axis mapping updated:\n{self.axis_mapping}")
    
    def calibrate(self):
        """校准编码器零点 - 将当前位置设为 0 度（闭合状态）"""
        if self.action_reader.use_encoder:
            print("[T265Teleop] === GRIPPER CALIBRATION ===")
            print("  Please move the gripper to CLOSED position (0 degrees)")
            print("  This position will be the reference for gripper control")
            print("  Gripper mapping: 0° = Closed, 90° = Fully Open")
            input("Press Enter when gripper is in closed position...")
            self.action_reader.calibrate()
            print("[T265Teleop] Calibration complete - current position is now 0°")
    
    def reset_reference(self):
        """
        重置参考位姿
        
        将当前 T265 位姿和机器人位姿设为参考。
        同时将夹爪设置为闭合状态。
        """
        with self._lock:
            action = self.action_reader.get_action()
            self._t265_reference_pose = action.pose.copy()
            self._t265_reference_angle = action.joint_angle  # 记录初始角度作为 0 度参考
            self._robot_reference_pose = self.robot.get_tcp_pose().copy()
            
            # 重置滤波状态
            self._filtered_position = np.zeros(3)
            self._filtered_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数 [qw, qx, qy, qz]
            self._filtered_gripper_width = self.gripper_min
            self._last_gripper_width = -1.0
            
            # 初始化夹爪为闭合状态 (宽度 = 0)
            if self.gripper_enabled and self.robot.is_gripper_enabled():
                print("[T265Teleop] Initializing gripper to closed position...")
                self.robot.gripper_move(self.gripper_min, wait=True)
            
            print("[T265Teleop] Reference pose reset")
            print(f"  T265 position: {np.round(self._t265_reference_pose[:3], 4)}")
            print(f"  Robot position: {np.round(self._robot_reference_pose[:3], 4)}")
            print(f"  Gripper reference angle: {self._t265_reference_angle:.1f}° (this is now 0°)")
    
    def _compute_pose_delta(self, reference_pose: np.ndarray, current_pose: np.ndarray) -> dict:
        """计算位姿变化量"""
        # 位置差（应用坐标系映射）
        delta_pos_t265 = current_pose[:3] - reference_pose[:3]
        delta_position = self.axis_mapping @ delta_pos_t265 * self.position_scale
        
        # 旋转差（四元数）
        ref_rot = Rotation.from_quat([
            reference_pose[4], reference_pose[5],
            reference_pose[6], reference_pose[3]
        ])
        curr_rot = Rotation.from_quat([
            current_pose[4], current_pose[5],
            current_pose[6], current_pose[3]
        ])
        
        relative_rot = curr_rot * ref_rot.inv()
        
        # 应用旋转缩放
        if self.rotation_scale != 1.0:
            axis_angle = relative_rot.as_rotvec()
            scaled_axis_angle = axis_angle * self.rotation_scale
            relative_rot = Rotation.from_rotvec(scaled_axis_angle)
        
        delta_quat = relative_rot.as_quat()
        delta_quaternion = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])
        
        return {'position': delta_position, 'quaternion': delta_quaternion}
    
    def _apply_pose_delta(self, reference_pose: np.ndarray, delta: dict) -> np.ndarray:
        """将位姿变化量应用到参考位姿"""
        target_position = reference_pose[:3] + delta['position']
        
        ref_rot = Rotation.from_quat([
            reference_pose[4], reference_pose[5],
            reference_pose[6], reference_pose[3]
        ])
        delta_rot = Rotation.from_quat([
            delta['quaternion'][1], delta['quaternion'][2],
            delta['quaternion'][3], delta['quaternion'][0]
        ])
        
        target_rot = ref_rot * delta_rot
        target_quat = target_rot.as_quat()
        target_quaternion = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        
        return np.concatenate([target_position, target_quaternion])
    
    def _apply_filter(self, current: np.ndarray, filtered: np.ndarray, alpha: float) -> np.ndarray:
        """
        应用指数移动平均滤波
        
        filtered = alpha * current + (1 - alpha) * filtered
        """
        return alpha * current + (1 - alpha) * filtered
    
    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
        """
        四元数球面线性插值 (SLERP) 用于平滑旋转
        q1, q2: [qw, qx, qy, qz]
        """
        # 转换为 scipy 格式 [qx, qy, qz, qw]
        r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
        r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
        
        # 使用 SLERP 插值
        from scipy.spatial.transform import Slerp
        key_times = [0, 1]
        key_rots = Rotation.concatenate([r1, r2])
        slerp = Slerp(key_times, key_rots)
        
        result = slerp(alpha)
        q_out = result.as_quat()  # [qx, qy, qz, qw]
        return np.array([q_out[3], q_out[0], q_out[1], q_out[2]])  # [qw, qx, qy, qz]
    
    def _control_loop(self):
        """控制循环（带滤波）"""
        period = 1.0 / self.control_frequency
        
        print("[T265Teleop] Control loop started (with filtering)...")
        print("  Move T265 camera to control robot end-effector, press Ctrl+C to exit")
        
        while not self._stop_flag.is_set():
            try:
                t0 = time.perf_counter()
                
                if self._t265_reference_pose is None or self._robot_reference_pose is None:
                    time.sleep(period)
                    continue
                
                # 读取 T265 位姿
                action = self.action_reader.get_action()
                t265_current_pose = action.pose
                
                # 计算原始变化量
                delta = self._compute_pose_delta(self._t265_reference_pose, t265_current_pose)
                
                # ====== 应用位置滤波 ======
                if self._filtered_position is None:
                    self._filtered_position = delta['position'].copy()
                else:
                    self._filtered_position = self._apply_filter(
                        delta['position'], 
                        self._filtered_position, 
                        self.position_filter_alpha
                    )
                
                # ====== 应用旋转滤波 ======
                if self._filtered_rotation is None:
                    self._filtered_rotation = delta['quaternion'].copy()
                else:
                    self._filtered_rotation = self._slerp_quaternion(
                        self._filtered_rotation,
                        delta['quaternion'],
                        self.rotation_filter_alpha
                    )
                
                # 使用滤波后的增量
                filtered_delta = {
                    'position': self._filtered_position,
                    'quaternion': self._filtered_rotation
                }
                
                # 应用到机器人
                target_pose = self._apply_pose_delta(self._robot_reference_pose, filtered_delta)
                
                # 发送位姿命令
                self.robot.send_cartesian_pose(target_pose)
                
                # ====== 控制夹爪（带滤波）======
                if self.gripper_enabled and self.robot.is_gripper_enabled():
                    # 计算相对于初始位置的角度变化（取绝对值）
                    angle_delta = abs(action.joint_angle - self._t265_reference_angle)
                    
                    # 限制角度在 0-90 度范围内
                    clamped_angle = np.clip(angle_delta, 0.0, self.gripper_angle_max)
                    
                    # 线性映射: 0° -> gripper_min, 90° -> gripper_max
                    ratio = clamped_angle / self.gripper_angle_max
                    raw_gripper_width = self.gripper_min + (self.gripper_max - self.gripper_min) * ratio
                    
                    # 应用夹爪滤波（平滑过渡）
                    self._filtered_gripper_width = (
                        self.gripper_filter_alpha * raw_gripper_width + 
                        (1 - self.gripper_filter_alpha) * self._filtered_gripper_width
                    )
                    
                    # 只有当滤波后的宽度变化超过阈值时才发送命令
                    if abs(self._filtered_gripper_width - self._last_gripper_width) > self._gripper_threshold:
                        self.robot.gripper_move(self._filtered_gripper_width, wait=False)
                        self._last_gripper_width = self._filtered_gripper_width
                
                # 控制频率
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
                    
            except Exception as e:
                print(f"[T265Teleop ERROR] {e}")
                import traceback
                traceback.print_exc()
                break
    
    def start(self):
        """启动遥操作"""
        if self._running:
            print("[WARN] Teleop already running")
            return
        
        self.reset_reference()
        
        self._stop_flag.clear()
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """停止遥操作"""
        self._stop_flag.set()
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        self.robot.stop()
        self.action_reader.disconnect()
        print("[T265Teleop] 已停止")
    
    def run_blocking(self):
        """阻塞式运行"""
        self._running = True
        self.reset_reference()
        
        try:
            self._control_loop()
        except KeyboardInterrupt:
            print("\n[T265Teleop] 收到退出信号...")
        finally:
            self.stop()


# ==================== UI 手动控制模式 ====================

class UIManualControl:
    """
    UI 手动控制模式
    
    通过 tkinter 界面按钮控制机械臂的 6 个自由度。
    """
    
    def __init__(
        self,
        robot: FlexivRobot,
        control_frequency: float = 50.0,
        position_step: float = 0.005,  # 0.5cm
        angle_step: float = 1.0,       # 1 deg
        gripper_step: float = 0.005,   # 0.5cm
        gripper_name: str = "Flexiv-GN01"
    ):
        """
        Initialize UI Manual Control
        
        Args:
            robot: FlexivRobot instance
            control_frequency: Control frequency [Hz]
            position_step: Position step [m], default 0.5cm
            angle_step: Angle step [deg], default 1 deg
            gripper_step: Gripper step [m], default 0.5cm
            gripper_name: Gripper name (always enabled)
        """
        self.robot = robot
        self.control_frequency = control_frequency
        self.position_step = position_step
        self.angle_step = angle_step
        self.gripper_step = gripper_step
        self.gripper_name = gripper_name
        
        # Always enable gripper
        print(f"[UIControl] Enabling gripper [{gripper_name}]...")
        try:
            self.robot.gripper_enable(gripper_name)
            self.gripper_enabled = True
            print(f"[UIControl] Gripper enabled successfully")
        except Exception as e:
            print(f"[UIControl] Warning: Failed to enable gripper: {e}")
            self.gripper_enabled = False
        
        # Current target pose [x, y, z, qw, qx, qy, qz]
        self.target_pose = self.robot.get_tcp_pose().copy()
        
        # Current euler angles (deg) - for display and modification
        self._update_euler_from_pose()
        
        # Gripper width
        self.gripper_width = 0.04  # Default 4cm
        self.gripper_min = 0.0
        self.gripper_max = 0.08   # 8cm max
        
        # Thread control
        self._running = False
        self._control_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        print("[UIControl] Initialized")
        print(f"  Position step: {position_step * 100:.1f} cm")
        print(f"  Angle step: {angle_step:.1f} deg")
        print(f"  Gripper step: {gripper_step * 100:.1f} cm")
    
    def _update_euler_from_pose(self):
        """从当前位姿更新欧拉角"""
        quat = self.target_pose[3:7]  # [qw, qx, qy, qz]
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [qx, qy, qz, qw]
        euler = rot.as_euler('xyz', degrees=True)
        self.current_euler = euler  # [roll, pitch, yaw]
    
    def _update_pose_from_euler(self):
        """从欧拉角更新位姿的四元数部分"""
        rot = Rotation.from_euler('xyz', self.current_euler, degrees=True)
        quat = rot.as_quat()  # [qx, qy, qz, qw]
        self.target_pose[3:7] = [quat[3], quat[0], quat[1], quat[2]]  # [qw, qx, qy, qz]
    
    def modify_pose(self, axis: str, direction: int):
        """
        修改位姿
        
        Args:
            axis: 'x', 'y', 'z', 'roll', 'pitch', 'yaw'
            direction: +1 或 -1
        """
        with self._lock:
            if axis == 'x':
                self.target_pose[0] += direction * self.position_step
            elif axis == 'y':
                self.target_pose[1] += direction * self.position_step
            elif axis == 'z':
                self.target_pose[2] += direction * self.position_step
            elif axis == 'roll':
                self.current_euler[0] += direction * self.angle_step
                self._update_pose_from_euler()
            elif axis == 'pitch':
                self.current_euler[1] += direction * self.angle_step
                self._update_pose_from_euler()
            elif axis == 'yaw':
                self.current_euler[2] += direction * self.angle_step
                self._update_pose_from_euler()
    
    def modify_gripper(self, direction: int):
        """Modify gripper width"""
        with self._lock:
            self.gripper_width += direction * self.gripper_step
            self.gripper_width = np.clip(self.gripper_width, self.gripper_min, self.gripper_max)
    
    def _control_loop(self):
        """Control loop"""
        period = 1.0 / self.control_frequency
        
        while self._running:
            try:
                t0 = time.perf_counter()
                
                with self._lock:
                    # Send pose command
                    self.robot.send_cartesian_pose(self.target_pose.copy())
                    
                    # Control gripper (always)
                    if self.gripper_enabled and self.robot.is_gripper_enabled():
                        self.robot.gripper_move(self.gripper_width, wait=False)
                
                # Control frequency
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
                    
            except Exception as e:
                print(f"[UIControl ERROR] {e}")
                break
    
    def start_control_thread(self):
        """启动控制线程"""
        if self._running:
            return
        
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
    
    def stop_control_thread(self):
        """停止控制线程"""
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
    
    def get_current_values(self) -> dict:
        """获取当前值（用于 UI 显示）"""
        with self._lock:
            return {
                'x': self.target_pose[0],
                'y': self.target_pose[1],
                'z': self.target_pose[2],
                'roll': self.current_euler[0],
                'pitch': self.current_euler[1],
                'yaw': self.current_euler[2],
                'gripper': self.gripper_width
            }
    
    def run_ui(self):
        """Run tkinter UI with long-press support"""
        import tkinter as tk
        from tkinter import font as tkfont
        
        # Colors - Modern dark theme
        BG_DARK = '#1a1a2e'
        BG_CARD = '#16213e'
        BG_HEADER = '#0f3460'
        TEXT_PRIMARY = '#eaeaea'
        TEXT_SECONDARY = '#a0a0a0'
        ACCENT_BLUE = '#4fc3f7'
        ACCENT_GREEN = '#66bb6a'
        ACCENT_RED = '#ef5350'
        ACCENT_ORANGE = '#ffa726'
        VALUE_COLOR = '#00e676'
        
        # Create main window
        root = tk.Tk()
        root.title("Flexiv Robot Manual Control")
        root.geometry("720x700")
        root.configure(bg=BG_DARK)
        root.resizable(True, True)
        
        # Fonts
        font_title = tkfont.Font(family='Segoe UI', size=18, weight='bold')
        font_header = tkfont.Font(family='Segoe UI', size=11, weight='bold')
        font_label = tkfont.Font(family='Segoe UI', size=12)
        font_value = tkfont.Font(family='Consolas', size=16, weight='bold')
        font_btn = tkfont.Font(family='Segoe UI', size=12, weight='bold')
        font_info = tkfont.Font(family='Segoe UI', size=10)
        
        # Store value labels and buttons for long-press
        value_labels = {}
        active_buttons = {}  # Track which buttons are being held
        repeat_jobs = {}     # Store after() job IDs for cancellation
        
        # Long-press repeat settings
        REPEAT_DELAY = 300   # Initial delay before repeat starts (ms)
        REPEAT_INTERVAL = 50 # Interval between repeats (ms)
        
        # Title
        title_frame = tk.Frame(root, bg=BG_HEADER, height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="FLEXIV ROBOT MANUAL CONTROL",
            font=font_title, 
            fg=TEXT_PRIMARY, 
            bg=BG_HEADER
        )
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(root, bg=BG_DARK, padx=25, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        def on_button_action(axis: str, direction: int):
            """Execute single action"""
            if axis == 'gripper':
                self.modify_gripper(direction)
            else:
                self.modify_pose(axis, direction)
            update_display()
        
        def on_button_press(axis: str, direction: int, btn_id: str):
            """Handle button press - start action and schedule repeat"""
            active_buttons[btn_id] = True
            on_button_action(axis, direction)
            
            def repeat_action():
                if active_buttons.get(btn_id, False):
                    on_button_action(axis, direction)
                    repeat_jobs[btn_id] = root.after(REPEAT_INTERVAL, repeat_action)
            
            # Schedule first repeat after delay
            repeat_jobs[btn_id] = root.after(REPEAT_DELAY, repeat_action)
        
        def on_button_release(btn_id: str):
            """Handle button release - stop repeating"""
            active_buttons[btn_id] = False
            if btn_id in repeat_jobs:
                root.after_cancel(repeat_jobs[btn_id])
                del repeat_jobs[btn_id]
        
        def create_section(parent, title: str, icon: str, color: str) -> tk.Frame:
            """Create a styled section frame"""
            section = tk.Frame(parent, bg=BG_CARD, padx=15, pady=12)
            section.pack(fill=tk.X, pady=8)
            
            # Header
            header = tk.Frame(section, bg=BG_CARD)
            header.pack(fill=tk.X, pady=(0, 10))
            
            header_label = tk.Label(
                header,
                text=f"{icon}  {title}",
                font=font_header,
                fg=color,
                bg=BG_CARD
            )
            header_label.pack(side=tk.LEFT)
            
            return section
        
        def create_control_row(parent, label: str, axis: str, unit: str, color: str):
            """Create a control row with long-press enabled buttons"""
            row = tk.Frame(parent, bg=BG_CARD)
            row.pack(fill=tk.X, pady=6)
            
            # Axis label
            lbl = tk.Label(
                row, text=label, font=font_label, fg=TEXT_PRIMARY, bg=BG_CARD, width=7, anchor='e'
            )
            lbl.pack(side=tk.LEFT, padx=(0, 15))
            
            # Minus button
            btn_id_minus = f"{axis}_minus"
            btn_minus = tk.Button(
                row, text="  −  ", font=font_btn,
                fg='white', bg=ACCENT_RED, activebackground='#c62828',
                relief=tk.FLAT, cursor='hand2', width=5
            )
            btn_minus.pack(side=tk.LEFT, padx=5)
            btn_minus.bind('<ButtonPress-1>', lambda e: on_button_press(axis, -1, btn_id_minus))
            btn_minus.bind('<ButtonRelease-1>', lambda e: on_button_release(btn_id_minus))
            btn_minus.bind('<Leave>', lambda e: on_button_release(btn_id_minus))
            
            # Value display
            value_frame = tk.Frame(row, bg='#0d1b2a', padx=10, pady=5)
            value_frame.pack(side=tk.LEFT, padx=15)
            
            value_lbl = tk.Label(
                value_frame, text="+0.00", font=font_value,
                fg=VALUE_COLOR, bg='#0d1b2a', width=8
            )
            value_lbl.pack()
            value_labels[axis] = value_lbl
            
            # Unit label
            unit_lbl = tk.Label(
                row, text=unit, font=font_label, fg=TEXT_SECONDARY, bg=BG_CARD, width=4
            )
            unit_lbl.pack(side=tk.LEFT)
            
            # Plus button
            btn_id_plus = f"{axis}_plus"
            btn_plus = tk.Button(
                row, text="  +  ", font=font_btn,
                fg='white', bg=ACCENT_GREEN, activebackground='#2e7d32',
                relief=tk.FLAT, cursor='hand2', width=5
            )
            btn_plus.pack(side=tk.LEFT, padx=5)
            btn_plus.bind('<ButtonPress-1>', lambda e: on_button_press(axis, +1, btn_id_plus))
            btn_plus.bind('<ButtonRelease-1>', lambda e: on_button_release(btn_id_plus))
            btn_plus.bind('<Leave>', lambda e: on_button_release(btn_id_plus))
        
        def update_display():
            """Update all value displays"""
            values = self.get_current_values()
            value_labels['x'].config(text=f"{values['x']*100:+.2f}")
            value_labels['y'].config(text=f"{values['y']*100:+.2f}")
            value_labels['z'].config(text=f"{values['z']*100:+.2f}")
            value_labels['roll'].config(text=f"{values['roll']:+.1f}")
            value_labels['pitch'].config(text=f"{values['pitch']:+.1f}")
            value_labels['yaw'].config(text=f"{values['yaw']:+.1f}")
            value_labels['gripper'].config(text=f"{values['gripper']*100:.1f}")
        
        # Position Control Section
        pos_section = create_section(main_frame, "POSITION (Step: 0.5 cm)", "◈", ACCENT_BLUE)
        create_control_row(pos_section, "X", "x", "cm", ACCENT_BLUE)
        create_control_row(pos_section, "Y", "y", "cm", ACCENT_BLUE)
        create_control_row(pos_section, "Z", "z", "cm", ACCENT_BLUE)
        
        # Orientation Control Section
        rot_section = create_section(main_frame, "ORIENTATION (Step: 1°)", "◎", ACCENT_ORANGE)
        create_control_row(rot_section, "Roll", "roll", "deg", ACCENT_ORANGE)
        create_control_row(rot_section, "Pitch", "pitch", "deg", ACCENT_ORANGE)
        create_control_row(rot_section, "Yaw", "yaw", "deg", ACCENT_ORANGE)
        
        # Gripper Control Section (always shown)
        gripper_section = create_section(main_frame, "GRIPPER (Step: 0.5 cm)", "✋", ACCENT_GREEN)
        create_control_row(gripper_section, "Width", "gripper", "cm", ACCENT_GREEN)
        
        # Info bar
        info_frame = tk.Frame(root, bg=BG_DARK, pady=10)
        info_frame.pack(fill=tk.X)
        
        info_label = tk.Label(
            info_frame,
            text="Click or hold buttons to adjust robot pose  |  Close window to exit",
            font=font_info,
            fg=TEXT_SECONDARY,
            bg=BG_DARK
        )
        info_label.pack()
        
        # Start control thread
        self.start_control_thread()
        
        # Initialize display
        update_display()
        
        # Periodic display update
        def periodic_update():
            if self._running:
                update_display()
                root.after(100, periodic_update)
        
        root.after(100, periodic_update)
        
        # Window close handler
        def on_closing():
            print("[UIControl] Shutting down...")
            # Cancel all pending repeat jobs
            for job_id in repeat_jobs.values():
                root.after_cancel(job_id)
            self.stop_control_thread()
            self.robot.stop()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        print("[UIControl] UI started")
        root.mainloop()
        print("[UIControl] UI closed")


# ==================== 舵机遥操作（原有功能）====================

class ServoReader:
    """中林舵机主臂读取器"""
    
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        import serial
        import re
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.re = re
        print(f"[ServoReader] 串口已打开: {port}")
        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7
        self._init_servos()

    def send_command(self, cmd):
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        return self.ser.read_all().decode('ascii', errors='ignore')

    def pwm_to_angle(self, response_str, pwm_min=500, pwm_max=2500, angle_range=270):
        match = self.re.search(r'P(\d{4})', response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        angle = (pwm_val - pwm_min) / (pwm_max - pwm_min) * angle_range
        return angle

    def _init_servos(self):
        self.send_command('#000PVER!')
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i] = angle if angle is not None else 0.0
        print("[ServoReader] 舵机零位校准完成")

    def get_action_offset(self) -> List[float]:
        angle_offset = [0.0] * 7
        for i in range(7):
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip())
            if angle is not None:
                angle_offset[i] = angle - self.zero_angles[i]
        return angle_offset

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


class FeetechServoReader:
    """飞特舵机主臂读取器"""
    
    def __init__(self, port='/dev/ttyACM0', baudrate=1000000):
        import sys
        feetech_path = '/home/cjh/work/ELM/data/LeRobot-Anything-U-Arm/src/uarm/scripts/Uarm_teleop/Feetech_servo'
        sys.path.insert(0, feetech_path)
        
        from scservo_sdk import (
            PortHandler, sms_sts, GroupSyncRead,
            SMS_STS_PRESENT_POSITION_L, SMS_STS_OFS_L, COMM_SUCCESS
        )
        
        self.portHandler = PortHandler(port)
        self.packetHandler = sms_sts(self.portHandler)
        
        if not self.portHandler.openPort():
            raise RuntimeError(f"无法打开串口 {port}")
        if not self.portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"无法设置波特率 {baudrate}")
        
        print(f"[FeetechServoReader] 串口已打开: {port}")
        
        self.SMS_STS_PRESENT_POSITION_L = SMS_STS_PRESENT_POSITION_L
        self.SMS_STS_OFS_L = SMS_STS_OFS_L
        self.COMM_SUCCESS = COMM_SUCCESS
        self.GroupSyncRead = GroupSyncRead
        
        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 8
        self.num_servos = 8
        self._init_servos()

    def _init_servos(self):
        print("[FeetechServoReader] 正在校准舵机零位...")
        
        for i in range(self.num_servos):
            scs_id = i + 1
            self.packetHandler.unLockEprom(scs_id)
            time.sleep(0.1)
            
            comm, error = self.packetHandler.write2ByteTxRx(scs_id, self.SMS_STS_OFS_L, 0)
            time.sleep(0.1)
            
            raw_pos, result, error = self.packetHandler.read2ByteTxRx(
                scs_id, self.SMS_STS_PRESENT_POSITION_L
            )
            
            Homing_Offset = raw_pos - 2047
            if Homing_Offset < 0:
                encoded_offset = (1 << 11) | abs(Homing_Offset)
            else:
                encoded_offset = Homing_Offset
            
            comm, error = self.packetHandler.write2ByteTxRx(scs_id, self.SMS_STS_OFS_L, encoded_offset)
            time.sleep(0.1)
            self.packetHandler.LockEprom(scs_id)
            time.sleep(0.1)
        
        groupSyncRead = self.GroupSyncRead(self.packetHandler, self.SMS_STS_PRESENT_POSITION_L, 4)
        for i in range(self.num_servos):
            groupSyncRead.addParam(i + 1)
        
        groupSyncRead.txRxPacket()
        
        for i in range(self.num_servos):
            scs_id = i + 1
            available, error = groupSyncRead.isAvailable(scs_id, self.SMS_STS_PRESENT_POSITION_L, 4)
            if available:
                self.zero_angles[i] = groupSyncRead.getData(scs_id, self.SMS_STS_PRESENT_POSITION_L, 2)
            else:
                self.zero_angles[i] = 2047
        
        groupSyncRead.clearParam()
        print("[FeetechServoReader] 舵机零位校准完成")

    def get_action_offset(self) -> List[float]:
        angle_offset = [0.0] * self.num_servos
        
        groupSyncRead = self.GroupSyncRead(self.packetHandler, self.SMS_STS_PRESENT_POSITION_L, 4)
        for i in range(self.num_servos):
            groupSyncRead.addParam(i + 1)
        
        result = groupSyncRead.txRxPacket()
        if result != self.COMM_SUCCESS:
            groupSyncRead.clearParam()
            return angle_offset
        
        for i in range(self.num_servos):
            scs_id = i + 1
            available, error = groupSyncRead.isAvailable(scs_id, self.SMS_STS_PRESENT_POSITION_L, 4)
            if available:
                current_pos = groupSyncRead.getData(scs_id, self.SMS_STS_PRESENT_POSITION_L, 2)
                if current_pos is not None:
                    angle_offset[i] = (current_pos - self.zero_angles[i]) / 4096.0 * 360.0
        
        groupSyncRead.clearParam()
        return angle_offset

    def close(self):
        if self.portHandler:
            self.portHandler.closePort()


class ServoJointTeleop:
    """舵机关节遥操作控制器（原有功能）"""
    
    def __init__(
        self,
        robot_sn: str,
        servo_port: str = '/dev/ttyUSB0',
        servo_type: str = 'zhonglin',
        control_rate: float = 50.0,
        gripper_name: Optional[str] = None
    ):
        self.control_rate = control_rate
        self.gripper_name = gripper_name
        
        motion_params = MotionParams(
            joint_max_vel=1.0,
            joint_max_acc=1.5,
            cart_max_linear_vel=0.3,
            cart_max_angular_vel=0.8,
            control_frequency=int(control_rate)
        )
        
        # 初始化舵机
        print(f"[ServoTeleop] 正在连接舵机 [{servo_port}]...")
        if servo_type == 'feetech':
            self.servo_reader = FeetechServoReader(port=servo_port)
        else:
            self.servo_reader = ServoReader(port=servo_port)
        
        # 初始化机器人
        print(f"[ServoTeleop] 正在连接机器人 [{robot_sn}]...")
        self.robot = FlexivRobot(robot_sn=robot_sn, auto_init=True, motion_params=motion_params, verbose=True)
        
        if gripper_name:
            self.robot.gripper_enable(gripper_name)
        
        self.init_qpos = self.robot.get_joint_positions()
        self.target_joint_pos = self.init_qpos.copy()
        
        self.num_joints = 7
        self.num_dof = 8
        
        self.joint_direction = np.array([-1, 1, 1, -1, 1, -1, -1])
        self.joint_scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.gripper_servo_idx = 7
        self.gripper_min = 0.0
        self.gripper_max = 0.08
        self.servo_gripper_range = 60.0
        
        self.filter_alpha = 0.3
        self.filtered_servo_deg = np.zeros(self.num_dof)
        
        self._running = False
        self._thread = None
        
        print("[ServoTeleop] 初始化完成")

    def map_servo_to_flexiv(self, servo_deg: List[float]) -> np.ndarray:
        servo_deg = np.array(servo_deg)
        num_servos = min(len(servo_deg), self.num_dof)
        self.filtered_servo_deg[:num_servos] = (
            self.filter_alpha * servo_deg[:num_servos] + 
            (1 - self.filter_alpha) * self.filtered_servo_deg[:num_servos]
        )
        
        servo_rad = np.radians(self.filtered_servo_deg[:self.num_joints])
        servo_rad = servo_rad * self.joint_direction * self.joint_scale
        joint_angles = servo_rad + self.init_qpos
        
        return joint_angles

    def angle_to_gripper(self, servo_angle_deg: float) -> float:
        ratio = np.clip(1 - (servo_angle_deg / self.servo_gripper_range), 0.0, 1.0)
        
        if self.gripper_name and self.robot.is_gripper_enabled():
            try:
                params = self.robot.get_gripper_params()
                return float(params.min_width + (params.max_width - params.min_width) * ratio)
            except:
                pass
        
        return float(self.gripper_min + (self.gripper_max - self.gripper_min) * ratio)

    def _control_loop(self):
        period = 1.0 / self.control_rate
        
        print("[ServoTeleop] 遥操作开始运行...")
        
        while self._running:
            try:
                t0 = time.perf_counter()
                
                servo_offset_deg = self.servo_reader.get_action_offset()
                self.target_joint_pos = self.map_servo_to_flexiv(servo_offset_deg)
                self.robot.send_joint_position(self.target_joint_pos)
                
                if self.gripper_name and self.robot.is_gripper_enabled():
                    if len(servo_offset_deg) > self.gripper_servo_idx:
                        gripper_width = self.angle_to_gripper(servo_offset_deg[self.gripper_servo_idx])
                        self.robot.gripper_move(gripper_width, wait=False)
                
                elapsed = time.perf_counter() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
                    
            except Exception as e:
                print(f"[ServoTeleop ERROR] {e}")
                break

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.robot.stop()
        self.servo_reader.close()
        print("[ServoTeleop] 已停止")

    def run_blocking(self):
        self._running = True
        try:
            self._control_loop()
        except KeyboardInterrupt:
            print("\n[ServoTeleop] Received exit signal...")
        finally:
            self.stop()




# ==================== 模式选择界面 ====================

def show_mode_selector() -> Optional[str]:
    """
    Display mode selection UI
    
    Returns:
        Selected mode ('ui' or 't265') or None if cancelled
    """
    import tkinter as tk
    from tkinter import font as tkfont
    
    selected_mode = [None]  # Use list to allow modification in nested function
    
    # Colors
    BG_DARK = '#1a1a2e'
    BG_CARD = '#16213e'
    BG_HEADER = '#0f3460'
    TEXT_PRIMARY = '#eaeaea'
    TEXT_SECONDARY = '#a0a0a0'
    ACCENT_BLUE = '#4fc3f7'
    ACCENT_GREEN = '#66bb6a'
    ACCENT_ORANGE = '#ffa726'
    
    # Create window
    root = tk.Tk()
    root.title("Flexiv Robot Teleoperation")
    root.geometry("500x400")
    root.configure(bg=BG_DARK)
    root.resizable(False, False)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 500) // 2
    y = (root.winfo_screenheight() - 400) // 2
    root.geometry(f"500x400+{x}+{y}")
    
    # Fonts
    font_title = tkfont.Font(family='Segoe UI', size=20, weight='bold')
    font_subtitle = tkfont.Font(family='Segoe UI', size=11)
    font_btn = tkfont.Font(family='Segoe UI', size=14, weight='bold')
    font_desc = tkfont.Font(family='Segoe UI', size=10)
    
    # Header
    header = tk.Frame(root, bg=BG_HEADER, height=80)
    header.pack(fill=tk.X)
    header.pack_propagate(False)
    
    title = tk.Label(header, text="FLEXIV ROBOT TELEOP", font=font_title, fg=TEXT_PRIMARY, bg=BG_HEADER)
    title.pack(pady=(15, 5))
    
    subtitle = tk.Label(header, text="Select Control Mode", font=font_subtitle, fg=TEXT_SECONDARY, bg=BG_HEADER)
    subtitle.pack()
    
    # Main content
    content = tk.Frame(root, bg=BG_DARK, padx=40, pady=30)
    content.pack(fill=tk.BOTH, expand=True)
    
    def select_mode(mode: str):
        selected_mode[0] = mode
        root.destroy()
    
    def create_mode_button(parent, text: str, desc: str, mode: str, color: str, row: int):
        frame = tk.Frame(parent, bg=BG_CARD, padx=20, pady=15)
        frame.pack(fill=tk.X, pady=10)
        
        btn = tk.Button(
            frame,
            text=text,
            font=font_btn,
            fg='white',
            bg=color,
            activebackground=color,
            relief=tk.FLAT,
            cursor='hand2',
            width=20,
            height=2,
            command=lambda: select_mode(mode)
        )
        btn.pack(pady=(0, 8))
        
        desc_label = tk.Label(frame, text=desc, font=font_desc, fg=TEXT_SECONDARY, bg=BG_CARD)
        desc_label.pack()
    
    # UI Mode Button
    create_mode_button(
        content,
        "🎮  UI Manual Control",
        "Control robot with button interface\nAdjust position, orientation, and gripper",
        "ui",
        ACCENT_BLUE,
        0
    )
    
    # T265 Mode Button
    create_mode_button(
        content,
        "📷  T265 Pose Teleop",
        "Control robot with T265 camera pose\nReal-time position and orientation tracking",
        "t265",
        ACCENT_ORANGE,
        1
    )
    
    # Info
    info = tk.Label(
        root,
        text=f"Robot: {CONFIG['robot_sn']}  |  Gripper: {CONFIG['gripper_name']}",
        font=font_desc,
        fg=TEXT_SECONDARY,
        bg=BG_DARK
    )
    info.pack(pady=10)
    
    # Handle window close
    def on_closing():
        selected_mode[0] = None
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()
    
    return selected_mode[0]


# ==================== 主程序 ====================

def main():
    print("=" * 60)
    print("        FLEXIV ROBOT TELEOPERATION")
    print("=" * 60)
    print(f"  Robot SN: {CONFIG['robot_sn']}")
    print(f"  Gripper: {CONFIG['gripper_name']}")
    print(f"  Control Rate: {CONFIG['control_rate']} Hz")
    print("=" * 60)
    
    # Show mode selector
    mode = show_mode_selector()
    
    if mode is None:
        print("\n[Main] No mode selected, exiting...")
        return
    
    print(f"\n[Main] Selected mode: {mode}")
    
    if mode == 'ui':
        # UI Manual Control Mode
        print("[Main] Starting UI manual control mode...")
        
        motion_params = MotionParams(
            cart_max_linear_vel=0.2,
            cart_max_angular_vel=0.5,
            control_frequency=int(CONFIG['control_rate'])
        )
        
        robot = FlexivRobot(
            robot_sn=CONFIG['robot_sn'],
            auto_init=True,
            motion_params=motion_params,
            verbose=True
        )
        
        controller = UIManualControl(
            robot=robot,
            control_frequency=CONFIG['control_rate'],
            position_step=0.005,  # 0.5cm
            angle_step=1.0,       # 1 deg
            gripper_name=CONFIG['gripper_name']
        )
        
        controller.run_ui()
        
    elif mode == 't265':
        # T265 Teleop Mode
        print("[Main] Starting T265 teleop mode...")
        
        motion_params = MotionParams(
            cart_max_linear_vel=0.3,
            cart_max_angular_vel=0.8,
            control_frequency=int(CONFIG['control_rate'])
        )
        
        robot = FlexivRobot(
            robot_sn=CONFIG['robot_sn'],
            auto_init=True,
            motion_params=motion_params,
            verbose=True
        )
        
        controller = T265PoseTeleop(
            robot=robot,
            encoder_port=CONFIG['encoder_port'],
            control_frequency=CONFIG['control_rate'],
            position_scale=CONFIG['pos_scale'],
            rotation_scale=CONFIG['rot_scale'],
            gripper_name=CONFIG['gripper_name']
        )
        
        # Calibrate encoder
        controller.calibrate()
        
        print("\nReady to start teleop...")
        print("Please move the T265 camera to the initial alignment position")
        input("Press Enter to start control...")
        
        controller.run_blocking()
    
    print("\nProgram ended")


if __name__ == "__main__":
    main()
