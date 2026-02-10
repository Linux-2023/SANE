#!/usr/bin/env python
"""
flexiv_robot.py

Flexiv 机器人控制封装类，提供常用的机器人控制方法。

功能包括：
- 机器人初始化（连接、清除故障、使能、等待就绪）
- 阻塞式/非阻塞式关节运动
- 阻塞式/非阻塞式笛卡尔位置运动
- 阻塞式/非阻塞式力位混合运动
- 获取当前状态（关节角、末端位姿等）
"""

import time
import math
import threading
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import flexivrdk


class ForceControlFrame(Enum):
    """力控制参考坐标系"""
    WORLD = "world"
    TCP = "tcp"


@dataclass
class MotionParams:
    """运动参数配置"""
    # 关节运动参数
    joint_max_vel: float = 1.0       # 最大关节速度 [rad/s]
    joint_max_acc: float = 1.5       # 最大关节加速度 [rad/s²]
    
    # 笛卡尔运动参数
    cart_max_linear_vel: float = 0.5   # 最大线速度 [m/s]
    cart_max_angular_vel: float = 1.0  # 最大角速度 [rad/s]
    cart_max_linear_acc: float = 2.0   # 最大线加速度 [m/s²]
    cart_max_angular_acc: float = 5.0  # 最大角加速度 [rad/s²]
    
    # 控制频率
    control_frequency: int = 50      # 控制频率 [Hz]
    
    # 到达判定阈值
    joint_tolerance: float = 0.01    # 关节角度误差阈值 [rad] (~0.57°)
    position_tolerance: float = 0.001  # 位置误差阈值 [m] (1mm)
    orientation_tolerance: float = 0.01  # 姿态误差阈值 (四元数点积)


class FlexivRobot:
    """Flexiv 机器人控制封装类"""
    
    # 默认 Home 关节角度 [rad]（7 个关节全为 0）
    DEFAULT_HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def __init__(
        self,
        robot_sn: str,
        auto_init: bool = True,
        motion_params: Optional[MotionParams] = None,
        home_joints: Optional[Union[List[float], np.ndarray]] = None,
        verbose: bool = True
    ):
        """
        初始化 Flexiv 机器人。
        
        Args:
            robot_sn: 机器人序列号，如 "Rizon4s-063239"
            auto_init: 是否自动完成初始化流程（连接、清除故障、使能、等待就绪）
            motion_params: 运动参数配置，None 则使用默认值
            home_joints: Home 位置的关节角度 [rad]，None 则使用初始化时的当前位置
            verbose: 是否打印详细信息
        """
        self.robot_sn = robot_sn
        self.verbose = verbose
        self.params = motion_params or MotionParams()
        
        # Home 位置（初始化后设置）
        self._home_joints: Optional[np.ndarray] = None
        if home_joints is not None:
            self._home_joints = np.array(home_joints, dtype=float)
        
        # 内部状态
        self._robot: Optional[flexivrdk.Robot] = None
        self._is_initialized = False
        self._motion_thread: Optional[threading.Thread] = None
        self._stop_motion_flag = threading.Event()
        self._motion_lock = threading.Lock()
        
        # 夹爪相关
        self._gripper: Optional[flexivrdk.Gripper] = None
        self._tool: Optional[flexivrdk.Tool] = None
        self._gripper_name: Optional[str] = None
        
        if auto_init:
            self.initialize()
    
    # ==================== 初始化相关 ====================
    
    def initialize(self, timeout_sec: float = 60.0) -> bool:
        """
        完整的机器人初始化流程：连接 -> 清除故障 -> 使能 -> 等待就绪
        
        Args:
            timeout_sec: 等待就绪的超时时间 [s]
            
        Returns:
            是否初始化成功
        """
        try:
            # 1. 连接机器人
            self._log(f"正在连接机器人 [{self.robot_sn}] ...")
            self._robot = flexivrdk.Robot(self.robot_sn, verbose=self.verbose)
            self._log("机器人连接成功")
            
            # 2. 清除故障（如有）
            if self._robot.fault():
                self._log("检测到故障，尝试清除...")
                if not self._robot.ClearFault(timeout_sec=30):
                    self._log("故障无法清除，请检查机器人状态", level="error")
                    return False
                self._log("故障已清除")
            
            # 3. 使能机器人
            self._log("使能机器人...")
            self.enable()
            
            # 4. 等待就绪
            self._log("等待机器人进入 operational 状态...")
            start_time = time.time()
            while not self._robot.operational():
                if time.time() - start_time > timeout_sec:
                    self._log("等待机器人就绪超时", level="error")
                    return False
                time.sleep(0.5)
            
            self._is_initialized = True
            
            # 5. 设置 Home 位置
            if self._home_joints is None:
                # 如果没有指定 home_joints，使用当前位置作为 Home
                self._home_joints = np.array(self._robot.states().q)
                self._log(f"Home 位置设置为当前关节角: {np.round(np.degrees(self._home_joints), 1).tolist()}°")
            else:
                self._log(f"Home 位置已指定: {np.round(np.degrees(self._home_joints), 1).tolist()}°")
            
            self._log("机器人初始化完成，已就绪")
            return True
            
        except Exception as e:
            self._log(f"初始化失败: {e}", level="error")
            return False
    
    def enable(self):
        """使能机器人"""
        if self._robot is not None:
            self._robot.Enable()
        self._log("机器人已使能")

    def is_ready(self) -> bool:
        """检查机器人是否已就绪可操作"""
        return self._is_initialized and self._robot is not None and self._robot.operational()
    
    def stop(self):
        """停止机器人运动并切换到 IDLE 模式"""
        self._stop_motion_flag.set()
        if self._robot is not None:
            self._robot.Stop()
    
    # ==================== 状态获取 ====================
    
    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节角度
        
        Returns:
            关节角度数组 [rad]，shape=(n_joints,)
        """
        self._check_ready()
        return np.array(self._robot.states().q)
    
    def get_tcp_pose(self) -> np.ndarray:
        """
        获取当前 TCP 位姿
        
        Returns:
            TCP 位姿数组 [x, y, z, qw, qx, qy, qz]，单位 [m] 和 []
        """
        self._check_ready()
        return np.array(self._robot.states().tcp_pose)
    
    def get_tcp_position(self) -> np.ndarray:
        """
        获取当前 TCP 位置
        
        Returns:
            TCP 位置数组 [x, y, z]，单位 [m]
        """
        return self.get_tcp_pose()[:3]
    
    def get_tcp_quaternion(self) -> np.ndarray:
        """
        获取当前 TCP 姿态（四元数）
        
        Returns:
            四元数数组 [qw, qx, qy, qz]
        """
        return self.get_tcp_pose()[3:7]
    
    def get_flange_pose(self) -> np.ndarray:
        """
        获取当前法兰位姿
        
        Returns:
            法兰位姿数组 [x, y, z, qw, qx, qy, qz]
        """
        self._check_ready()
        return np.array(self._robot.states().flange_pose)
    
    def get_joint_torques(self) -> np.ndarray:
        """
        获取当前关节力矩
        
        Returns:
            关节力矩数组 [Nm]
        """
        self._check_ready()
        return np.array(self._robot.states().tau)
    
    def get_external_wrench(self, frame: str = "world") -> np.ndarray:
        """
        获取外部力/力矩
        
        Args:
            frame: 参考坐标系，"world" 或 "tcp"
            
        Returns:
            力/力矩数组 [fx, fy, fz, mx, my, mz]，单位 [N, Nm]
        """
        self._check_ready()
        states = self._robot.states()
        if frame.lower() == "world":
            return np.array(states.ext_wrench_in_world)
        else:
            return np.array(states.ext_wrench_in_tcp)
    
    def get_robot_info(self) -> flexivrdk.RobotInfo:
        """获取机器人信息"""
        self._check_ready()
        return self._robot.info()
    
    def get_dof(self) -> int:
        """获取机器人自由度"""
        return self.get_robot_info().DoF
    
    # ==================== 阻塞式运动 ====================
    
    def move_joint(
        self,
        target_joints: Union[List[float], np.ndarray],
        timeout: float = 30.0,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        阻塞式关节运动：运动到目标关节角度，等待到达后返回
        
        Args:
            target_joints: 目标关节角度 [rad]
            timeout: 超时时间 [s]
            tolerance: 到达判定阈值 [rad]，None 则使用默认值
            
        Returns:
            是否成功到达目标
        """
        self._check_ready()
        target = np.array(target_joints, dtype=float)
        tol = tolerance if tolerance is not None else self.params.joint_tolerance
        
        # 切换到关节位置控制模式
        self._robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        
        dof = len(target)
        target_vel = [0.0] * dof
        target_acc = [0.0] * dof
        max_vel = [self.params.joint_max_vel] * dof
        max_acc = [self.params.joint_max_acc] * dof
        
        period = 1.0 / self.params.control_frequency
        start_time = time.time()
        self._stop_motion_flag.clear()
        
        while not self._stop_motion_flag.is_set():
            time.sleep(period)
            
            # 检查故障
            if self._robot.fault():
                self._log("运动过程中发生故障", level="error")
                return False
            
            # 发送命令
            self._robot.SendJointPosition(
                list(target), target_vel, target_acc, max_vel, max_acc
            )
            
            # 检查是否到达
            current = self.get_joint_positions()
            error = np.max(np.abs(current - target))
            if error < tol:
                self._log(f"关节运动完成，最大误差: {error:.4f} rad")
                return True
            
            # 超时检查
            if time.time() - start_time > timeout:
                self._log(f"关节运动超时，当前误差: {error:.4f} rad", level="warning")
                return False
        
        return False
    
    def move_cartesian(
        self,
        target_pose: Union[List[float], np.ndarray],
        timeout: float = 30.0,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None
    ) -> bool:
        """
        阻塞式笛卡尔位置运动：运动到目标 TCP 位姿，等待到达后返回
        
        Args:
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]，单位 [m]
            timeout: 超时时间 [s]
            position_tolerance: 位置误差阈值 [m]
            orientation_tolerance: 姿态误差阈值（四元数点积）
            
        Returns:
            是否成功到达目标
        """
        self._check_ready()
        self.zero_ft_sensor()
        target = np.array(target_pose, dtype=float)
        pos_tol = position_tolerance if position_tolerance is not None else self.params.position_tolerance
        ori_tol = orientation_tolerance if orientation_tolerance is not None else self.params.orientation_tolerance
        
        # 切换到笛卡尔运动力控制模式
        self._robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        
        # 设置为纯运动控制（所有轴关闭力控）
        self._robot.SetForceControlAxis([False] * 6)
        
        period = 1.0 / self.params.control_frequency
        start_time = time.time()
        self._stop_motion_flag.clear()
        
        while not self._stop_motion_flag.is_set():
            time.sleep(period)
            
            # 检查故障
            if self._robot.fault():
                self._log("运动过程中发生故障", level="error")
                return False
            
            # 发送命令
            self._robot.SendCartesianMotionForce(
                pose=list(target),
                wrench=[0.0] * 6,
                max_linear_vel=self.params.cart_max_linear_vel,
                max_angular_vel=self.params.cart_max_angular_vel,
                max_linear_acc=self.params.cart_max_linear_acc,
                max_angular_acc=self.params.cart_max_angular_acc
            )
            
            # 检查是否到达
            current = self.get_tcp_pose()
            pos_error = np.linalg.norm(current[:3] - target[:3])
            ori_error = 1.0 - abs(np.dot(current[3:7], target[3:7]))  # 四元数距离
            
            if pos_error < pos_tol and ori_error < ori_tol:
                self._log(f"笛卡尔运动完成，位置误差: {pos_error*1000:.2f} mm, 姿态误差: {ori_error:.4f}")
                return True
            
            # 超时检查
            if time.time() - start_time > timeout:
                self._log(f"笛卡尔运动超时，位置误差: {pos_error*1000:.2f} mm", level="warning")
                return False
        
        return False
    
    def move_cartesian_force(
        self,
        target_pose: Union[List[float], np.ndarray],
        target_wrench: Union[List[float], np.ndarray],
        force_axes: List[bool],
        force_frame: ForceControlFrame = ForceControlFrame.WORLD,
        timeout: float = 30.0,
        position_tolerance: Optional[float] = None
    ) -> bool:
        """
        阻塞式力位混合运动：同时进行位置控制和力控制
        
        Args:
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]
            target_wrench: 目标力/力矩 [fx, fy, fz, mx, my, mz]，单位 [N, Nm]
            force_axes: 哪些轴启用力控制 [X, Y, Z, Rx, Ry, Rz]
            force_frame: 力控制参考坐标系
            timeout: 超时时间 [s]
            position_tolerance: 位置误差阈值 [m]（仅用于运动控制轴的判定）
            
        Returns:
            是否成功完成运动
        """
        self._check_ready()
        self.zero_ft_sensor()
        target = np.array(target_pose, dtype=float)
        wrench = np.array(target_wrench, dtype=float)
        pos_tol = position_tolerance if position_tolerance is not None else self.params.position_tolerance
        
        # 切换到笛卡尔运动力控制模式
        self._robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
        
        # 设置力控制参考坐标系
        if force_frame == ForceControlFrame.WORLD:
            self._robot.SetForceControlFrame(flexivrdk.CoordType.WORLD)
        else:
            self._robot.SetForceControlFrame(flexivrdk.CoordType.TCP)
        
        # 设置力控制轴
        self._robot.SetForceControlAxis(force_axes)
        
        period = 1.0 / self.params.control_frequency
        start_time = time.time()
        self._stop_motion_flag.clear()
        
        # 确定哪些轴是运动控制的（用于到达判定）
        motion_axes = [not f for f in force_axes[:3]]  # 只检查位置轴
        
        while not self._stop_motion_flag.is_set():
            time.sleep(period)
            
            # 检查故障
            if self._robot.fault():
                self._log("运动过程中发生故障", level="error")
                return False
            
            # 发送命令
            self._robot.SendCartesianMotionForce(
                pose=list(target),
                wrench=list(wrench),
                max_linear_vel=self.params.cart_max_linear_vel,
                max_angular_vel=self.params.cart_max_angular_vel,
                max_linear_acc=self.params.cart_max_linear_acc,
                max_angular_acc=self.params.cart_max_angular_acc
            )
            
            # 检查运动控制轴是否到达
            current = self.get_tcp_pose()
            pos_errors = np.abs(current[:3] - target[:3])
            motion_error = max([pos_errors[i] for i in range(3) if motion_axes[i]], default=0)
            
            if motion_error < pos_tol:
                self._log(f"力位混合运动完成，运动轴误差: {motion_error*1000:.2f} mm")
                return True
            
            # 超时检查
            if time.time() - start_time > timeout:
                self._log(f"力位混合运动超时，误差: {motion_error*1000:.2f} mm", level="warning")
                return False
        
        return False
    
    # ==================== 非阻塞式运动 ====================
    
    def move_joint_async(
        self,
        target_joints: Union[List[float], np.ndarray],
        callback: Optional[callable] = None,
        timeout: float = 30.0,
        tolerance: Optional[float] = None
    ) -> threading.Thread:
        """
        非阻塞式关节运动：在后台线程执行运动
        
        Args:
            target_joints: 目标关节角度 [rad]
            callback: 运动完成后的回调函数，接收参数 (success: bool)
            timeout: 超时时间 [s]
            tolerance: 到达判定阈值 [rad]
            
        Returns:
            运动线程对象
        """
        def motion_task():
            success = self.move_joint(target_joints, timeout, tolerance)
            if callback is not None:
                callback(success)
        
        self._stop_motion_flag.clear()
        thread = threading.Thread(target=motion_task, daemon=True)
        thread.start()
        self._motion_thread = thread
        return thread
    
    def move_cartesian_async(
        self,
        target_pose: Union[List[float], np.ndarray],
        callback: Optional[callable] = None,
        timeout: float = 30.0,
        position_tolerance: Optional[float] = None,
        orientation_tolerance: Optional[float] = None
    ) -> threading.Thread:
        """
        非阻塞式笛卡尔位置运动：在后台线程执行运动
        
        Args:
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]
            callback: 运动完成后的回调函数，接收参数 (success: bool)
            timeout: 超时时间 [s]
            position_tolerance: 位置误差阈值 [m]
            orientation_tolerance: 姿态误差阈值
            
        Returns:
            运动线程对象
        """
        def motion_task():
            success = self.move_cartesian(target_pose, timeout, position_tolerance, orientation_tolerance)
            if callback is not None:
                callback(success)
        
        self._stop_motion_flag.clear()
        thread = threading.Thread(target=motion_task, daemon=True)
        thread.start()
        self._motion_thread = thread
        return thread
    
    def move_cartesian_force_async(
        self,
        target_pose: Union[List[float], np.ndarray],
        target_wrench: Union[List[float], np.ndarray],
        force_axes: List[bool],
        force_frame: ForceControlFrame = ForceControlFrame.WORLD,
        callback: Optional[callable] = None,
        timeout: float = 30.0,
        position_tolerance: Optional[float] = None
    ) -> threading.Thread:
        """
        非阻塞式力位混合运动：在后台线程执行运动
        
        Args:
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]
            target_wrench: 目标力/力矩 [fx, fy, fz, mx, my, mz]
            force_axes: 哪些轴启用力控制 [X, Y, Z, Rx, Ry, Rz]
            force_frame: 力控制参考坐标系
            callback: 运动完成后的回调函数
            timeout: 超时时间 [s]
            position_tolerance: 位置误差阈值 [m]
            
        Returns:
            运动线程对象
        """
        def motion_task():
            success = self.move_cartesian_force(
                target_pose, target_wrench, force_axes, force_frame, timeout, position_tolerance
            )
            if callback is not None:
                callback(success)
        
        self._stop_motion_flag.clear()
        thread = threading.Thread(target=motion_task, daemon=True)
        thread.start()
        self._motion_thread = thread
        return thread
    
    def wait_motion_complete(self, timeout: Optional[float] = None) -> bool:
        """
        等待当前异步运动完成
        
        Args:
            timeout: 等待超时时间 [s]，None 则无限等待
            
        Returns:
            运动线程是否已结束
        """
        if self._motion_thread is not None:
            self._motion_thread.join(timeout=timeout)
            return not self._motion_thread.is_alive()
        return True
    
    def cancel_motion(self):
        """取消当前异步运动"""
        self._stop_motion_flag.set()
        self.stop()
    
    def is_moving(self) -> bool:
        """检查是否有运动正在执行"""
        return self._motion_thread is not None and self._motion_thread.is_alive()
    
    # ==================== 直接发送命令（单次，非阻塞） ====================
    
    def send_joint_position(
        self,
        target_joints: Union[List[float], np.ndarray],
        target_vel: Optional[Union[List[float], np.ndarray]] = None,
        target_acc: Optional[Union[List[float], np.ndarray]] = None
    ):
        """
        直接发送一次关节位置命令（非阻塞，需要用户自己循环调用）
        
        Args:
            target_joints: 目标关节角度 [rad]
            target_vel: 目标关节速度 [rad/s]，None 则为 0
            target_acc: 目标关节加速度 [rad/s²]，None 则为 0
        """
        self._check_ready()
        
        # 确保在正确的模式
        if self._robot.mode() != flexivrdk.Mode.NRT_JOINT_POSITION:
            self._robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
                    
        target = list(np.array(target_joints, dtype=float))
        dof = len(target)
        vel = list(target_vel) if target_vel is not None else [0.0] * dof
        acc = list(target_acc) if target_acc is not None else [0.0] * dof
        max_vel = [self.params.joint_max_vel] * dof
        max_acc = [self.params.joint_max_acc] * dof
        
        self._robot.SendJointPosition(target, vel, acc, max_vel, max_acc)
        self._log(f"发送关节位置命令: {target}")
    
    def send_cartesian_pose(
        self,
        target_pose: Union[List[float], np.ndarray],
        target_wrench: Optional[Union[List[float], np.ndarray]] = None
    ):
        """
        直接发送一次笛卡尔位姿命令（非阻塞，需要用户自己循环调用）
        
        Args:
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]
            target_wrench: 目标力/力矩，None 则为 0
        """
        self._check_ready()
        # 确保在正确的模式
        if self._robot.mode() != flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE:
            self.zero_ft_sensor()
            self._robot.SwitchMode(flexivrdk.Mode.NRT_CARTESIAN_MOTION_FORCE)
            self._robot.SetForceControlAxis([False] * 6)
        
        pose = list(np.array(target_pose, dtype=float))
        wrench = list(target_wrench) if target_wrench is not None else [0.0] * 6
        
        self._robot.SendCartesianMotionForce(
            pose=pose,
            wrench=wrench,
            max_linear_vel=self.params.cart_max_linear_vel,
            max_angular_vel=self.params.cart_max_angular_vel,
            max_linear_acc=self.params.cart_max_linear_acc,
            max_angular_acc=self.params.cart_max_angular_acc
        )
    
    # ==================== 夹爪控制 ====================
    
    def gripper_enable(self, gripper_name: str, switch_tool: bool = True):
        """
        使能夹爪
        
        Args:
            gripper_name: 夹爪名称，如 "Flexiv-GN01"
            switch_tool: 是否同时切换机器人工具到夹爪（用于重力补偿和 TCP 位置）
        """
        self._check_ready()
        
        # 创建夹爪和工具接口
        if self._gripper is None:
            self._gripper = flexivrdk.Gripper(self._robot)
        if self._tool is None:
            self._tool = flexivrdk.Tool(self._robot)
        
        # 使能夹爪
        self._log(f"使能夹爪 [{gripper_name}]...")
        self._gripper.Enable(gripper_name)
        self._gripper_name = gripper_name
        
        # 切换工具
        if switch_tool:
            self._log(f"切换工具到 [{gripper_name}]...")
            self._tool.Switch(gripper_name)
        
        # 打印参数
        params = self._gripper.params()
        self._log(f"夹爪参数: 宽度范围={params.min_width*1000:.1f}-{params.max_width*1000:.1f}mm, "
                  f"力范围={params.min_force:.1f}-{params.max_force:.1f}N")
    
    def gripper_disable(self):
        """禁用当前夹爪"""
        if self._gripper is not None and self._gripper_name is not None:
            self._log(f"禁用夹爪 [{self._gripper_name}]...")
            self._gripper.Disable()
            self._gripper_name = None
    
    def gripper_init(self, wait: bool = True, timeout: float = 10.0) -> bool:
        """
        手动初始化夹爪（某些夹爪需要）
        
        Args:
            wait: 是否等待初始化完成
            timeout: 等待超时时间 [s]
            
        Returns:
            是否初始化成功（wait=False 时始终返回 True）
        """
        self._check_gripper_enabled()
        self._log("触发夹爪初始化...")
        self._gripper.Init()
        
        if wait:
            return self._wait_gripper_idle(timeout)
        return True
    
    def gripper_move(
        self, 
        width: float, 
        velocity: Optional[float] = None, 
        force_limit: Optional[float] = None,
        wait: bool = True,
        timeout: float = 10.0
    ) -> bool:
        """
        移动夹爪到指定开口宽度
        
        Args:
            width: 目标开口宽度 [m]
            velocity: 移动速度 [m/s]，None 则使用最大速度的 50%
            force_limit: 最大接触力 [N]，None 则使用最大力的 50%
            wait: 是否等待运动完成
            timeout: 等待超时时间 [s]
            
        Returns:
            是否成功到达目标（wait=False 时始终返回 True）
        """
        self._check_gripper_enabled()
        
        params = self._gripper.params()
        vel = velocity if velocity is not None else (params.max_vel)
        force = force_limit if force_limit is not None else (params.max_force)
        
        # 限制参数范围
        width = max(params.min_width, min(params.max_width, width))
        vel = max(params.min_vel, min(params.max_vel, vel))
        force = max(params.min_force, min(params.max_force, force))
        
        self._log(f"夹爪移动: 宽度={width*1000:.1f}mm, 速度={vel*1000:.1f}mm/s, 力限={force:.1f}N")
        self._gripper.Move(width, vel, force)
        
        if wait:
            return self._wait_gripper_idle(timeout)
        return True
    
    def gripper_grasp(
        self, 
        force: float,
        wait: bool = True,
        timeout: float = 10.0
    ) -> bool:
        """
        力控夹取（需要夹爪支持力控）
        
        Args:
            force: 目标夹持力 [N]，正值为闭合力，负值为张开力
            wait: 是否等待夹取稳定
            timeout: 等待超时时间 [s]
            
        Returns:
            是否成功（wait=False 时始终返回 True）
        """
        self._check_gripper_enabled()
        
        params = self._gripper.params()
        force = max(-params.max_force, min(params.max_force, force))
        
        self._log(f"夹爪力控夹取: 力={force:.1f}N")
        self._gripper.Grasp(force)
        
        if wait:
            return self._wait_gripper_idle(timeout)
        return True
    
    def gripper_open(
        self, 
        velocity: Optional[float] = None,
        wait: bool = True,
        timeout: float = 10.0
    ) -> bool:
        """
        完全张开夹爪
        
        Args:
            velocity: 移动速度 [m/s]，None 则使用最大速度的 50%
            wait: 是否等待运动完成
            timeout: 等待超时时间 [s]
            
        Returns:
            是否成功（wait=False 时始终返回 True）
        """
        self._check_gripper_enabled()
        params = self._gripper.params()
        return self.gripper_move(params.max_width, velocity, wait=wait, timeout=timeout)
    
    def gripper_close(
        self,
        force_limit: Optional[float] = None,
        velocity: Optional[float] = None,
        wait: bool = True,
        timeout: float = 10.0
    ) -> bool:
        """
        完全闭合夹爪
        
        Args:
            force_limit: 最大接触力 [N]，None 则使用最大力的 50%
            velocity: 移动速度 [m/s]，None 则使用最大速度的 50%
            wait: 是否等待运动完成
            timeout: 等待超时时间 [s]
            
        Returns:
            是否成功（wait=False 时始终返回 True）
        """
        self._check_gripper_enabled()
        params = self._gripper.params()
        return self.gripper_move(params.min_width, velocity, force_limit, wait=wait, timeout=timeout)
    
    def gripper_stop(self):
        """停止夹爪运动并保持当前位置"""
        if self._gripper is not None:
            self._gripper.Stop()
            self._log("夹爪已停止")
    
    def is_gripper_moving(self) -> bool:
        """
        检查夹爪是否正在运动
        
        Returns:
            True 正在运动，False 已停止
        """
        if self._gripper is None:
            return False
        return self._gripper.states().is_moving
    
    def wait_gripper(self, timeout: float = 10.0) -> bool:
        """
        等待夹爪运动完成
        
        Args:
            timeout: 超时时间 [s]
            
        Returns:
            是否在超时前完成
        """
        return self._wait_gripper_idle(timeout)
    
    def get_gripper_width(self) -> float:
        """
        获取当前夹爪开口宽度
        
        Returns:
            开口宽度 [m]
        """
        self._check_gripper_enabled()
        return self._gripper.states().width
    
    def get_gripper_force(self) -> float:
        """
        获取当前夹爪夹持力
        
        Returns:
            夹持力 [N]，正值为张开力，负值为闭合力
        """
        self._check_gripper_enabled()
        return self._gripper.states().force
    
    def get_gripper_states(self) -> flexivrdk.GripperStates:
        """
        获取夹爪完整状态
        
        Returns:
            GripperStates 对象，包含 width, force, is_moving
        """
        self._check_gripper_enabled()
        return self._gripper.states()
    
    def get_gripper_params(self) -> flexivrdk.GripperParams:
        """
        获取夹爪参数
        
        Returns:
            GripperParams 对象，包含 name, min_width, max_width, min_vel, max_vel, min_force, max_force
        """
        self._check_gripper_enabled()
        return self._gripper.params()
    
    def is_gripper_enabled(self) -> bool:
        """检查夹爪是否已使能"""
        return self._gripper is not None and self._gripper_name is not None
    
    def _check_gripper_enabled(self):
        """检查夹爪是否已使能"""
        if not self.is_gripper_enabled():
            raise RuntimeError("夹爪未使能，请先调用 gripper_enable()")
    
    def _wait_gripper_idle(self, timeout: float) -> bool:
        """等待夹爪停止运动"""
        start = time.time()
        # 先等待一小段时间让运动开始
        time.sleep(0.1)
        while time.time() - start < timeout:
            if not self._gripper.states().is_moving:
                return True
            time.sleep(0.02)
        self._log("等待夹爪超时", level="warning")
        return False
    
    # ==================== 工具方法 ====================
    
    def set_motion_params(self, params: MotionParams):
        """设置运动参数"""
        self.params = params
    
    def switch_mode(self, mode: flexivrdk.Mode):
        """切换控制模式"""
        self._check_ready()
        self._robot.SwitchMode(mode)
    
    def execute_primitive(self, name: str, params: dict):
        """
        执行原语
        
        Args:
            name: 原语名称，如 "Home", "MoveL", "ZeroFTSensor" 等
            params: 原语参数
        """
        self._check_ready()
        if self._robot.mode() != flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION:
            self._robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self._robot.ExecutePrimitive(name, params)
    
    def execute_plan(self, name: str, wait: bool = True):
        """
        执行计划
        
        Args:
            name: 计划名称
            wait: 是否等待计划执行完成
        """
        self._check_ready()
        if self._robot.mode() != flexivrdk.Mode.NRT_PLAN_EXECUTION:
            self._robot.SwitchMode(flexivrdk.Mode.NRT_PLAN_EXECUTION)
        self._robot.ExecutePlan(name)
        
        if wait:
            while self._robot.busy():
                time.sleep(0.1)
    
    def go_home(self, wait: bool = True, timeout: float = 30.0) -> bool:
        """
        回到 Home 位置（初始化时设定的关节角度）
        
        Args:
            wait: 是否等待运动完成
            timeout: 超时时间 [s]（仅在 wait=True 时有效）
            
        Returns:
            是否成功到达 Home 位置（wait=False 时始终返回 True）
        """
        self._check_ready()
        
        if self._home_joints is None:
            self._log("Home 位置未设置", level="error")
            return False
        
        self._log(f"回到 Home 位置: {np.round(np.degrees(self._home_joints), 1).tolist()}°")
        
        if wait:
            success = self.move_joint(self._home_joints, timeout=timeout)
            if success:
                self._log("已到达 Home 位置")
            return success
        else:
            self.move_joint_async(self._home_joints, timeout=timeout)
            return True
    
    def set_home_joints(self, home_joints: Union[List[float], np.ndarray]):
        """
        设置 Home 位置的关节角度
        
        Args:
            home_joints: Home 位置的关节角度 [rad]
        """
        self._home_joints = np.array(home_joints, dtype=float)
        self._log(f"Home 位置已更新: {np.round(np.degrees(self._home_joints), 1).tolist()}°")
    
    def set_current_as_home(self):
        """将当前关节角度设置为 Home 位置"""
        self._check_ready()
        self._home_joints = self.get_joint_positions()
        self._log(f"当前位置已设为 Home: {np.round(np.degrees(self._home_joints), 1).tolist()}°")
    
    def get_home_joints(self) -> Optional[np.ndarray]:
        """
        获取 Home 位置的关节角度
        
        Returns:
            Home 关节角度 [rad]，未设置时返回 None
        """
        return self._home_joints.copy() if self._home_joints is not None else None
    
    def zero_ft_sensor(self, timeout: float = 30.0):
        """
        清零力/力矩传感器
        
        Args:
            timeout: 超时时间 [s]
        """
        self._check_ready()
        self._robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self._robot.ExecutePrimitive("ZeroFTSensor", {})
        
        start = time.time()
        while time.time() - start < timeout:
            states = self._robot.primitive_states()
            if states.get("terminated", 0) == 1 or states.get("reachedTarget", 0) == 1:
                self._log("力/力矩传感器清零完成")
                return
            time.sleep(0.1)
        
        self._log("力/力矩传感器清零超时", level="warning")
    
    @property
    def robot(self) -> flexivrdk.Robot:
        """获取底层 Robot 对象（用于高级操作）"""
        return self._robot
    
    # ==================== 私有方法 ====================
    
    def _check_ready(self):
        """检查机器人是否就绪"""
        if not self.is_ready():
            raise RuntimeError("机器人未就绪，请先调用 initialize() 或检查机器人状态")
    
    def _log(self, msg: str, level: str = "info"):
        """打印日志"""
        if self.verbose:
            prefix = {"info": "[INFO]", "warning": "[WARNING]", "error": "[ERROR]"}.get(level, "[INFO]")
            print(f"{prefix} {msg}")


# ==================== 工具函数 ====================

def load_robot_config(path: str) -> dict:
    """
    从 YAML 文件加载机器人配置
    
    示例文件格式:
        name: Rizon4s-063239 # arm serial number
        gripper_name: Flexiv-GN01
    
    Args:
        path: YAML 配置文件路径
        
    Returns:
        配置字典，包含 'robot_sn' 和可选的 'gripper_name'
    """
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()  # 去除注释
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            if key == "name":
                config["robot_sn"] = value
            elif key == "gripper_name":
                config["gripper_name"] = value
            else:
                config[key] = value
    
    return config


def load_robot_sn_from_yaml(path: str) -> str:
    """
    从 YAML 文件加载机器人序列号（向后兼容）
    
    示例文件格式:
        name: Rizon4s-063239 # serial number
        
    Args:
        path: YAML 配置文件路径
        
    Returns:
        机器人序列号
    """
    config = load_robot_config(path)
    if "robot_sn" not in config:
        raise ValueError(f"配置文件 {path} 中未找到 'name' 字段")
    return config["robot_sn"]


def quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """四元数转旋转矩阵"""
    R = np.zeros((3, 3))
    R[0, 0] = 1 - 2 * (qy**2 + qz**2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx**2 + qz**2)
    R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


def pose_to_matrix(pose: Union[List[float], np.ndarray]) -> np.ndarray:
    """位姿转 4x4 变换矩阵"""
    pose = np.array(pose)
    T = np.eye(4)
    T[:3, :3] = quat_to_rotation_matrix(pose[3], pose[4], pose[5], pose[6])
    T[:3, 3] = pose[:3]
    return T


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import math
    
    # 从配置文件加载序列号
    robot_sn = load_robot_sn_from_yaml("robot.yaml")
    
    # 创建机器人实例（自动初始化）
    robot = FlexivRobot(robot_sn)
    
    # 获取当前状态
    print(f"当前关节角: {robot.get_joint_positions()}")
    print(f"当前 TCP 位姿: {robot.get_tcp_pose()}")
    
    # 示例 1: 阻塞式关节运动
    current_joints = robot.get_joint_positions()
    target_joints = current_joints.copy()
    # target_joints[5] += math.radians(10)  # J6 转 10 度
    # robot.move_joint(target_joints)
    
    # 示例 2: 阻塞式笛卡尔运动
    # current_pose = robot.get_tcp_pose()
    # target_pose = current_pose.copy()
    # target_pose[2] += 0.05  # Z 方向上移 5cm
    # robot.move_cartesian(target_pose)
    
    # 示例 3: 非阻塞式运动
    def on_complete(success):
        print(f"运动完成: {success}")
    robot.move_joint_async(target_joints, callback=on_complete)
    # 可以做其他事情...
    robot.wait_motion_complete()
    
    # 停止
    robot.stop()
    print("程序结束")

