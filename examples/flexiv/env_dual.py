"""
双臂机器人环境

基于 env.py 扩展，支持双臂控制，兼容 OpenPI 环境接口。

支持两种控制模式：
1. 关节角控制模式 (use_pose_control=False, 默认)
   - 状态维度：16维 [左臂关节角(7) + 左臂夹爪(1) + 右臂关节角(7) + 右臂夹爪(1)]
   - 动作维度：16维 [左臂关节角(7) + 左臂夹爪(1) + 右臂关节角(7) + 右臂夹爪(1)]
   
2. 位姿控制模式 (use_pose_control=True)
   - 状态维度：16维 [左臂位姿(7: x,y,z,qw,qx,qy,qz) + 左臂夹爪(1) + 右臂位姿(7) + 右臂夹爪(1)]
   - 动作维度：16维 [左臂位姿(7: x,y,z,qw,qx,qy,qz) + 左臂夹爪(1) + 右臂位姿(7) + 右臂夹爪(1)]

相机：
- cam_left_wrist: 左臂腕部相机
- cam_right_wrist: 右臂腕部相机
- cam_high: 全局相机（可选）
"""

import numpy as np
from openpi_client.runtime import environment as _environment
from typing_extensions import override
from typing import Optional, List, Dict, Any
import time
import threading
import cv2
import h5py
from datetime import datetime
import os
import sys

# 添加 flexiv_usage 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../third_party/flexiv_usage'))

from flexiv_robot import FlexivRobot, MotionParams
from cameras import RealSenseCamera, USBCamera, GoproCamera

# 导入四元数跳变修复函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
try:
    from quat2rotvec import fix_quaternion_discontinuity
    QUAT_FIX_AVAILABLE = True
except ImportError:
    print("Warning: quat2rotvec module not available, quaternion discontinuity fix will be disabled")
    QUAT_FIX_AVAILABLE = False
    fix_quaternion_discontinuity = None

# 尝试导入碰撞检测模块
try:
    from collision_check.dual_arm_collision_checker import create_checker, DualArmCollisionChecker
    COLLISION_CHECK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Collision check module not available: {e}")
    COLLISION_CHECK_AVAILABLE = False
    DualArmCollisionChecker = None


class DualArmEnvironment(_environment.Environment):
    """双臂机器人真机环境，兼容 OpenPI 环境接口。"""

    def __init__(
        self,
        # 左臂配置
        left_robot_sn: str = "",
        left_gripper_name: str = "",
        left_wrist_camera_id: int = 0,
        left_wrist_camera_type: str = "usb",  # "usb" or "realsense"
        # 右臂配置
        right_robot_sn: str = "",
        right_gripper_name: str = "",
        right_wrist_camera_id: int = 2,
        right_wrist_camera_type: str = "usb",
        # 全局相机配置
        high_camera_id: Optional[int] = None,
        high_camera_type: str = "usb",
        # 相机分辨率
        wrist_camera_width: int = 320,
        wrist_camera_height: int = 240,
        high_camera_width: int = 1280,
        high_camera_height: int = 960,
        camera_fps: int = 30,
        # 环境参数
        max_episode_steps: int = 500,
        initial_left_joint_pos: Optional[List[float]] = None,
        initial_right_joint_pos: Optional[List[float]] = None,
        seed: int = 0,
        tele_mode: bool = False,
        prompt: str = "",
        watchdog_timeout: float = 5.0,
        show_camera_preview: bool = False,
        gripper_max_width: float = 0.08,  # 夹爪最大开口宽度 (米)
        action_history_size: int = 30,
        action_change_threshold: float = 0.01,
        record_mode: bool = False,
        record_directory: str = "recorded_data",
        verbose: bool = True,
        # 碰撞检测配置
        collision_check_enabled: bool = True,
        collision_safety_threshold: float = 0.03,  # 安全距离阈值 (米)，<=0 表示仅检测实际碰撞
        # 控制模式配置
        use_pose_control: bool = False,  # True: 位姿控制, False: 关节角控制
        # 夹爪闭合阈值
        close_threshold: float = 0.05,
        open_threshold: float = 0.95,
    ) -> None:
        """初始化双臂机器人环境。

        Args:
            left_robot_sn: 左臂机器人序列号
            left_gripper_name: 左臂夹爪名称
            left_wrist_camera_id: 左臂腕部相机 ID
            left_wrist_camera_type: 左臂腕部相机类型 ("usb", "realsense", "gopro")
            right_robot_sn: 右臂机器人序列号
            right_gripper_name: 右臂夹爪名称
            right_wrist_camera_id: 右臂腕部相机 ID
            right_wrist_camera_type: 右臂腕部相机类型 ("usb", "realsense", "gopro")
            high_camera_id: 全局相机 ID (None 表示不使用)
            high_camera_type: 全局相机类型
            wrist_camera_width: 腕部相机宽度
            wrist_camera_height: 腕部相机高度
            high_camera_width: 全局相机宽度
            high_camera_height: 全局相机高度
            camera_fps: 相机帧率
            max_episode_steps: 每个 episode 的最大步数
            initial_left_joint_pos: 左臂初始关节位置 [j1-j7, gripper]，共8维
            initial_right_joint_pos: 右臂初始关节位置 [j1-j7, gripper]，共8维
            seed: 随机种子
            tele_mode: 是否为遥操作模式（禁用机器人运动命令）
            prompt: 任务提示文本
            watchdog_timeout: 看门狗超时时间
            show_camera_preview: 是否显示相机预览
            gripper_max_width: 夹爪最大开口宽度
            action_history_size: 动作历史队列大小
            action_change_threshold: 动作变化阈值
            record_mode: 是否记录轨迹数据
            record_directory: 记录数据保存目录
            verbose: 是否打印详细信息
            collision_check_enabled: 是否启用碰撞检测
            collision_safety_threshold: 碰撞安全距离阈值 (米)，<=0 表示仅检测实际碰撞
            use_pose_control: 控制模式，True 为位姿控制，False 为关节角控制
        """
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)
        self._verbose = verbose
        self._use_pose_control = use_pose_control
        # 夹爪阈值
        self._close_threshold = close_threshold
        self._open_threshold = open_threshold
        # 线程同步
        self._obs_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._update_rate = camera_fps

        # 看门狗机制
        self._watchdog_timeout = watchdog_timeout
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10
        self._device_healthy = True

        # 机器人状态
        self._left_robot_enabled = False
        self._right_robot_enabled = False
        
        # 夹爪配置
        self._gripper_max_width = gripper_max_width
        self._left_gripper_enabled = False
        self._right_gripper_enabled = False

        # 初始化硬件
        self._init_robots(
            left_robot_sn, left_gripper_name,
            right_robot_sn, right_gripper_name
        )
        self._init_cameras(
            left_wrist_camera_id, left_wrist_camera_type,
            right_wrist_camera_id, right_wrist_camera_type,
            high_camera_id, high_camera_type,
            wrist_camera_width, wrist_camera_height,
            high_camera_width, high_camera_height,
            camera_fps
        )

        # Episode 跟踪
        self._done = True
        self._episode_reward = 0.0
        self._step_count = 0
        self._max_episode_steps = max_episode_steps

        # 默认初始位置（8维：7关节 + 夹爪）
        if initial_left_joint_pos is None:
            self._initial_left_joint_pos = [0.0] * 7 + [0.0]  # 夹爪关闭
        else:
            self._initial_left_joint_pos = list(initial_left_joint_pos)
            if len(self._initial_left_joint_pos) == 7:
                self._initial_left_joint_pos.append(0.0)  # 添加夹爪

        if initial_right_joint_pos is None:
            self._initial_right_joint_pos = [0.0] * 7 + [0.0]
        else:
            self._initial_right_joint_pos = list(initial_right_joint_pos)
            if len(self._initial_right_joint_pos) == 7:
                self._initial_right_joint_pos.append(0.0)

        # 启动硬件
        self._start_hardware()
        
        if self._verbose:
            print(f"DualArmEnvironment initialized")
            print(f"  - Control mode: {'Pose Control' if self._use_pose_control else 'Joint Control'}")
            print(f"  - Left robot: {left_robot_sn}")
            print(f"  - Right robot: {right_robot_sn}")
            print(f"  - Cameras: left_wrist, right_wrist" + (", cam_high" if high_camera_id is not None else ""))

        self._tele_mode = tele_mode
        self._prompt = prompt

        # 相机预览设置
        self._show_camera_preview = show_camera_preview
        self._preview_window_name = "Dual Arm Camera Preview"
        self._preview_window_initialized = False

        # 动作历史队列
        self._action_history_size = action_history_size
        self._action_history = []
        self._action_change_threshold = action_change_threshold

        # 轨迹记录
        self._record_mode = record_mode
        self._record_directory = record_directory
        self._episode_data = []
        self._last_obs = None
        self._prev_state = None  # 保存上一次的 state
        self._prev_joint_state = None  # 保存上一次的关节状态
        self._prev_pose_state = None  # 保存上一次的位姿状态
        self._last_pose_state = None  # 用于四元数跳变修复的上一次pose状态

        # 碰撞检测
        self._collision_check_enabled = collision_check_enabled and COLLISION_CHECK_AVAILABLE
        self._collision_safety_threshold = collision_safety_threshold
        self._collision_checker = None
        self._collision_skip_count = 0  # 统计因碰撞跳过的动作数
        
        if self._collision_check_enabled:
            self._setup_collision_detection()


    def _init_robots(
        self,
        left_robot_sn: str,
        left_gripper_name: str,
        right_robot_sn: str,
        right_gripper_name: str
    ):
        """初始化双臂机器人。"""
        # 运动参数配置
        motion_params = MotionParams(
            joint_max_vel=1.0,
            joint_max_acc=1.5,
            control_frequency=50
        )

        # 左臂
        if left_robot_sn:
            if self._verbose:
                print(f"Connecting to left robot ({left_robot_sn})...")
            self._left_robot = FlexivRobot(
                robot_sn=left_robot_sn,
                auto_init=True,
                motion_params=motion_params,
                verbose=self._verbose
            )
            self._left_robot_enabled = self._left_robot.is_ready()
            
            # 初始化左臂夹爪
            if left_gripper_name and self._left_robot_enabled:
                try:
                    self._left_robot.gripper_enable(left_gripper_name, switch_tool=False)
                    self._left_gripper_enabled = True
                    self._left_robot.gripper_state = 0 #初始化夹爪状态为关闭
                    if self._verbose:
                        print(f"Left gripper [{left_gripper_name}] enabled")
                except Exception as e:
                    print(f"Warning: Failed to enable left gripper: {e}")
        else:
            self._left_robot = None
            print("Warning: No left robot SN provided")

        # 右臂
        if right_robot_sn:
            if self._verbose:
                print(f"Connecting to right robot ({right_robot_sn})...")
            self._right_robot = FlexivRobot(
                robot_sn=right_robot_sn,
                auto_init=True,
                motion_params=motion_params,
                verbose=self._verbose
            )
            self._right_robot_enabled = self._right_robot.is_ready()
            
            # 初始化右臂夹爪
            if right_gripper_name and self._right_robot_enabled:
                try:
                    self._right_robot.gripper_enable(right_gripper_name, switch_tool=False)
                    self._right_gripper_enabled = True
                    self._right_robot.gripper_state = 0 #初始化夹爪状态为关闭

                    if self._verbose:
                        print(f"Right gripper [{right_gripper_name}] enabled")
                except Exception as e:
                    print(f"Warning: Failed to enable right gripper: {e}")
        else:
            self._right_robot = None
            print("Warning: No right robot SN provided")

    def _setup_collision_detection(self):
        """初始化碰撞检测系统。"""
        if not COLLISION_CHECK_AVAILABLE:
            if self._verbose:
                print("Collision check module not available, skipping...")
            self._collision_check_enabled = False
            return
        
        try:
            if self._verbose:
                print("Initializing collision checker...")
            
            # 创建碰撞检测器
            self._collision_checker = create_checker(mode="mesh")
            
            if self._verbose:
                print(f"✅ Collision checker initialized, pairs: {self._collision_checker.n_collision_pairs}")
                if self._collision_safety_threshold <= 0:
                    print("   Mode: Collision detection only (no safety distance)")
                else:
                    print(f"   Safety distance threshold: {self._collision_safety_threshold * 1000:.1f}mm")
        
        except Exception as e:
            print(f"⚠️ Failed to initialize collision checker: {e}")
            print("   Collision check will be disabled")
            self._collision_check_enabled = False
            self._collision_checker = None

    def _check_collision(self, left_joints: List[float], right_joints: List[float]) -> tuple:
        """检查双臂碰撞。
        
        Args:
            left_joints: 左臂关节角度 (7维)
            right_joints: 右臂关节角度 (7维)
            
        Returns:
            (is_safe, min_distance): 是否安全和最小距离
        """
        if not self._collision_check_enabled or self._collision_checker is None:
            return True, float('inf')
        
        try:
            q_left = np.array(left_joints, dtype=float)
            q_right = np.array(right_joints, dtype=float)
            
            if self._collision_safety_threshold <= 0:
                # 仅检测是否真正碰撞（快速模式）
                is_collision = self._collision_checker.check_collision(q_right,q_left)
                is_safe = not is_collision
                min_dist = -1.0 if is_collision else 1.0  # 简化的距离指示
            else:
                # 基于距离阈值检测
                min_dist = self._collision_checker.compute_min_distance(q_right, q_left)
                is_safe = min_dist > self._collision_safety_threshold
            
            return is_safe, min_dist
        
        except Exception as e:
            if self._verbose:
                print(f"⚠️ Collision check failed: {e}")
            return True, float('inf')  # 检测失败时默认安全

    def _create_camera(
        self,
        camera_type: str,
        camera_id: int,
        width: int,
        height: int,
        fps: int,
        name: str = "camera",
        warmup_time: float = 1.0,
        center_crop: bool = False
    ):
        """根据相机类型创建相机实例。
        
        Args:
            camera_type: 相机类型 ("usb", "realsense", "gopro")
            camera_id: 相机设备 ID
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            name: 相机名称（仅用于 GoPro）
            warmup_time: 预热时间（秒，仅用于 GoPro）
            center_crop: 是否启用中心裁剪
        Returns:
            相机实例
        """
        if camera_type == "realsense":
            return RealSenseCamera(width=width, height=height, fps=fps)
        elif camera_type == "gopro":
            return GoproCamera(
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps,
                name=name,
                warmup_time=warmup_time,
                center_crop=center_crop
            )
        else:  # 默认使用 USB 相机
            return USBCamera(camera_id=camera_id, width=width, height=height, fps=fps)

    def _init_cameras(
        self,
        left_wrist_camera_id: int,
        left_wrist_camera_type: str,
        right_wrist_camera_id: int,
        right_wrist_camera_type: str,
        high_camera_id: Optional[int],
        high_camera_type: str,
        wrist_camera_width: int,
        wrist_camera_height: int,
        high_camera_width: int,
        high_camera_height: int,
        camera_fps: int
    ):
        """初始化相机。
        
        支持的相机类型:
        - "usb": 普通 USB 摄像头
        - "realsense": Intel RealSense 摄像头
        - "gopro": GoPro（通过 Elgato 采集卡）
        """
        # 左臂腕部相机
        self._left_wrist_camera = self._create_camera(
            camera_type=left_wrist_camera_type,
            camera_id=left_wrist_camera_id,
            width=wrist_camera_width,
            height=wrist_camera_height, 
            fps=camera_fps,
            name="cam_left_wrist"
        )
        self._left_wrist_camera_width = wrist_camera_width
        self._left_wrist_camera_height = wrist_camera_height

        # 右臂腕部相机
        self._right_wrist_camera = self._create_camera(
            camera_type=right_wrist_camera_type,
            camera_id=right_wrist_camera_id,
            width=wrist_camera_width,
            height=wrist_camera_height,
            fps=camera_fps,
            name="cam_right_wrist"
        )
        self._right_wrist_camera_width = wrist_camera_width
        self._right_wrist_camera_height = wrist_camera_height

        # 全局相机（可选）
        self._use_high_camera = high_camera_id is not None
        if self._use_high_camera:
            self._high_camera = self._create_camera(
                camera_type=high_camera_type,
                camera_id=high_camera_id,
                width=high_camera_width,
                height=high_camera_height,
                fps=camera_fps,
                name="cam_high",
                center_crop=False #全局相机可能启用中心裁剪，确保只有桌面
            )
            self._high_camera_width = high_camera_width
            self._high_camera_height = high_camera_height
        else:
            self._high_camera = None
            self._high_camera_width = 0
            self._high_camera_height = 0

    def _start_hardware(self):
        """启动相机硬件。"""
        # 启动相机
        if self._verbose:
            print("Starting cameras...")
        
        self._left_wrist_camera.start()
        self._right_wrist_camera.start()
        if self._use_high_camera:
            self._high_camera.start()

        # 等待相机初始化
        time.sleep(1.0)
        
        if self._verbose:
            print("Cameras started successfully")

    def _get_robot_state(self) -> np.ndarray:
        """获取双臂状态（16维）。
        
        根据控制模式返回不同的状态：
        - 关节角模式: [左臂关节角x7, 左臂夹爪x1, 右臂关节角x7, 右臂夹爪x1]
        - 位姿模式: [左臂位姿x7, 左臂夹爪x1, 右臂位姿x7, 右臂夹爪x1]
        
        Returns:
            16维数组
        """
        if self._use_pose_control:
            return self._get_robot_state_pose()
        else:
            return self._get_robot_state_joint()
    
    def _get_robot_state_joint(self) -> np.ndarray:
        """获取双臂关节角状态（16维）。
        
        Returns:
            16维数组: [左臂关节角x7, 左臂夹爪x1, 右臂关节角x7, 右臂夹爪x1]
        """
        state = np.zeros(16, dtype=np.float32)
        
        # 左臂关节角度 (0-6)
        if self._left_robot is not None and self._left_robot_enabled:
            try:
                left_joints = self._left_robot.get_joint_positions()
                state[0:7] = left_joints[:7]
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get left robot state: {e}")
        
        # 左臂夹爪 (7)
        if self._left_gripper_enabled:
            try:
                state[7] = self._left_robot.gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get left gripper state: {e}")
        
        # 右臂关节角度 (8-14)
        if self._right_robot is not None and self._right_robot_enabled:
            try:
                right_joints = self._right_robot.get_joint_positions()
                state[8:15] = right_joints[:7]
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get right robot state: {e}")
        
        # 右臂夹爪 (15)
        if self._right_gripper_enabled:
            try:
                state[15] = self._right_robot.gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get right gripper state: {e}")
        
        return state
    
    def _get_robot_state_pose(self) -> np.ndarray:
        """获取双臂位姿状态（16维）。
        
        如果启用了四元数跳变修复，会自动修复相邻帧之间的四元数跳变。
        
        Returns:
            16维数组: [左臂位姿x7(x,y,z,qw,qx,qy,qz), 左臂夹爪x1, 右臂位姿x7, 右臂夹爪x1]
        """
        state = np.zeros(16, dtype=np.float32)
        
        # 左臂位姿 (0-6): [x, y, z, qw, qx, qy, qz]
        if self._left_robot is not None and self._left_robot_enabled:
            try:
                left_pose = self._left_robot.get_tcp_pose()
                state[0:7] = left_pose[:7]
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get left robot pose: {e}")
        
        # 左臂夹爪 (7)
        if self._left_gripper_enabled:
            try:
                state[7] = self._left_robot.gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get left gripper state: {e}")
        
        # 右臂位姿 (8-14): [x, y, z, qw, qx, qy, qz]
        if self._right_robot is not None and self._right_robot_enabled:
            try:
                right_pose = self._right_robot.get_tcp_pose()
                state[8:15] = right_pose[:7]
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get right robot pose: {e}")
        
        # 右臂夹爪 (15)
        if self._right_gripper_enabled:
            try:
                state[15] = self._right_robot.gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get right gripper state: {e}")
        
        # 修复四元数跳变（如果可用且不是第一次获取状态）
        if QUAT_FIX_AVAILABLE and fix_quaternion_discontinuity is not None:
            if self._last_pose_state is not None:
                state = fix_quaternion_discontinuity(self._last_pose_state, state)
            # 更新上一次状态
            self._last_pose_state = state.copy()
        
        return state

    def _read_camera_frame(self, camera, camera_name: str):
        """读取相机帧，支持多线程模式（GoproCamera）和同步模式。
        
        Returns:
            (frame, timestamp) 或 (None, timestamp)
        """
        try:
            # 检查是否是 GoproCamera 且启用了多线程
            if hasattr(camera, 'get_latest_frame') and hasattr(camera, 'use_threading') and camera.use_threading:
                # 使用多线程模式，获取最新帧（非阻塞）
                frame_data = camera.get_latest_frame()
                if frame_data is not None:
                    return frame_data.frame, frame_data.timestamp
                else:
                    # 队列为空，尝试等待一帧（带超时）
                    frame_data = camera.get_frame(timeout=0.05)
                    if frame_data is not None:
                        return frame_data.frame, frame_data.timestamp
                    else:
                        if self._verbose:
                            print(f"Warning: {camera_name} camera queue is empty")
                        return None, time.time()
            else:
                # 使用同步模式
                frame = camera.read(timeout=0.5)
                return frame, time.time()
        except Exception as e:
            if self._verbose:
                print(f"Warning: {camera_name} camera read error: {e}")
            return None, time.time()

    def _update_observation(self) -> Dict[str, Any]:
        """收集所有设备数据并构建观测。"""
        start_ts = time.time()

        # 读取左臂腕部相机（支持多线程模式）
        left_wrist_frame, left_wrist_ts = self._read_camera_frame(
            self._left_wrist_camera, "Left wrist"
        )

        # 读取右臂腕部相机（支持多线程模式）
        right_wrist_frame, right_wrist_ts = self._read_camera_frame(
            self._right_wrist_camera, "Right wrist"
        )

        # 读取全局相机（如果启用，支持多线程模式）
        high_frame = None
        high_ts = time.time()
        if self._use_high_camera:
            high_frame, high_ts = self._read_camera_frame(
                self._high_camera, "High"
            )

        # 读取机器人状态
        robot_ts = time.time()
        state = self._get_robot_state()
        joint_state = self._get_robot_state_joint()
        pose_state = self._get_robot_state_pose()

        # 检查是否所有相机都失败
        if left_wrist_frame is None and right_wrist_frame is None:
            raise RuntimeError("All wrist cameras failed to read data")

        raw_left_wrist_img = left_wrist_frame
        raw_right_wrist_img = right_wrist_frame
        # 处理图像
        left_wrist_img = self._process_camera_frame(
            left_wrist_frame, self._left_wrist_camera_width, self._left_wrist_camera_height
        )
        right_wrist_img = self._process_camera_frame(
            right_wrist_frame, self._right_wrist_camera_width, self._right_wrist_camera_height
        )


        # 构建观测字典
        images = {
            "cam_left_wrist": left_wrist_img,
            "cam_right_wrist": right_wrist_img,
        }
        timestamps = {
            "start": start_ts,
            "robot": robot_ts,
            "cam_left_wrist": left_wrist_ts,
            "cam_right_wrist": right_wrist_ts,
        }
        raw_images = {
            "cam_left_wrist": raw_left_wrist_img,
            "cam_right_wrist": raw_right_wrist_img,
        }

        # 添加全局相机（如果启用）
        if self._use_high_camera:
            raw_images["cam_high"] = high_frame
            high_img = self._process_camera_frame(
                high_frame, self._high_camera_width, self._high_camera_height
            )
            images["cam_high"] = high_img
            timestamps["cam_high"] = high_ts

        # 准备上一次的 state（如果不存在则使用零向量）
        prev_state = self._prev_state.copy() if self._prev_state is not None else state #np.zeros_like(state)
        prev_joint_state = self._prev_joint_state.copy() if self._prev_joint_state is not None else joint_state
        prev_pose_state = self._prev_pose_state.copy() if self._prev_pose_state is not None else pose_state

        obs = {
            "state": state,
            "joint_state": joint_state,
            "pose_state": pose_state,
            "prev_state": prev_state,
            "prev_joint_state": prev_joint_state,
            "prev_pose_state": prev_pose_state,
            "images": images,
            # "raw_images": raw_images,
            "timestamps": timestamps,
            "prompt": self._prompt,
        }

        # 保存当前 state 作为下一次的 prev_state
        self._prev_state = state.copy()
        self._prev_joint_state = joint_state.copy()
        self._prev_pose_state = pose_state.copy()
        # 可选：显示相机预览
        if self._show_camera_preview:
            self._display_camera_preview(left_wrist_frame, right_wrist_frame, high_frame)

        return obs

    def _display_camera_preview(
        self,
        left_frame: Optional[np.ndarray],
        right_frame: Optional[np.ndarray],
        high_frame: Optional[np.ndarray]
    ) -> None:
        """显示相机预览窗口。"""
        try:
            # 调整大小用于显示
            display_size = (320, 240)
            
            frames = []
            if left_frame is not None:
                left_resized = cv2.resize(left_frame, display_size)
                cv2.putText(left_resized, "Left Wrist", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frames.append(left_resized)
            
            if right_frame is not None:
                right_resized = cv2.resize(right_frame, display_size)
                cv2.putText(right_resized, "Right Wrist", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frames.append(right_resized)
            
            if high_frame is not None:
                high_resized = cv2.resize(high_frame, display_size)
                cv2.putText(high_resized, "High", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                frames.append(high_resized)
            
            if frames:
                if not self._preview_window_initialized:
                    cv2.namedWindow(self._preview_window_name, cv2.WINDOW_NORMAL)
                    self._preview_window_initialized = True
                
                # 水平拼接所有相机画面
                combined = np.hstack(frames)
                cv2.imshow(self._preview_window_name, combined)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self._show_camera_preview = False
                    cv2.destroyWindow(self._preview_window_name)
                    self._preview_window_initialized = False
        except Exception as e:
            if self._verbose:
                print(f"Camera preview error: {e}")
            self._show_camera_preview = False

    def _process_camera_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """处理相机帧：调整大小、转换颜色、转置。"""
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        target_size = 224
        cur_height, cur_width = frame.shape[:2]

        # 计算缩放比例以保持宽高比
        ratio = max(cur_width / target_size, cur_height / target_size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        #print(f"mother fuckerrrr resized_width: {resized_width}, resized_height: {resized_height}")

        # 使用 OpenCV 调整大小
        if ratio > 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
        # resized = frame

        # BGR 转 RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 填充到目标大小
        pad_h = (target_size - resized_height) // 2
        pad_w = (target_size - resized_width) // 2
        pad_h_remainder = target_size - resized_height - pad_h
        pad_w_remainder = target_size - resized_width - pad_w
        # print(f"mother fuckerrrr pad_h: {pad_h}, pad_w: {pad_w}, pad_h_remainder: {pad_h_remainder}, pad_w_remainder: {pad_w_remainder}")

        padded = np.pad(
            resized_rgb,
            ((pad_h, pad_h_remainder), (pad_w, pad_w_remainder), (0, 0)),
            mode='constant',
            constant_values=0
        )

        # 转换轴顺序 [H, W, C] --> [C, H, W]
        return np.transpose(padded, (2, 0, 1))
        # return np.transpose(resized_rgb, (2, 0, 1))

    @override
    def reset(self) -> None:
        """重置环境：移动到初始位置，重置计数器。"""
        if not self._tele_mode:
            # 移动到初始位置
            if self._verbose:
                print("Resetting to initial joint positions...")

            # 并行移动双臂到初始位置
            left_thread = None
            right_thread = None

            if self._left_robot is not None and self._left_robot_enabled:
                def move_left():
                    self._left_robot.move_joint(
                        self._initial_left_joint_pos[:7],
                        timeout=10.0
                    )
                    if self._left_gripper_enabled:
                        target_width = self._initial_left_joint_pos[7] * self._gripper_max_width
                        self._left_robot.gripper_move(target_width, wait=True)
                
                left_thread = threading.Thread(target=move_left, daemon=True)
                left_thread.start()

            if self._right_robot is not None and self._right_robot_enabled:
                def move_right():
                    self._right_robot.move_joint(
                        self._initial_right_joint_pos[:7],
                        timeout=10.0
                    )
                    if self._right_gripper_enabled:
                        target_width = self._initial_right_joint_pos[7] * self._gripper_max_width
                        self._right_robot.gripper_move(target_width, wait=True)
                
                right_thread = threading.Thread(target=move_right, daemon=True)
                right_thread.start()

            # 等待双臂运动完成
            if left_thread:
                left_thread.join()
            if right_thread:
                right_thread.join()

            time.sleep(1.0)

        self._done = False
        self._episode_reward = 0.0
        self._step_count = 0
        self._action_history = []

        #if self._record_mode:
        self._episode_data = []
        self._last_obs = None
        self._prev_state = None  # 重置上一次的 state
        self._last_pose_state = None  # 重置上一次的 pose 状态（用于四元数跳变修复）

        if self._verbose:
            print("Environment reset complete.")

    @override
    def is_episode_complete(self) -> bool:
        """检查 episode 是否完成。"""
        if self._step_count >= self._max_episode_steps:
            self._done = True
        # elif len(self._action_history) >= self._action_history_size:
        #     action_array = np.array(self._action_history)
        #     std_per_dim = np.std(action_array, axis=0)
        #     if np.all(std_per_dim < self._action_change_threshold):
        #         self._done = True

        return self._done

    @override
    def get_observation(self) -> Dict[str, Any]:
        """获取当前观测。"""
        if not self._device_healthy:
            raise RuntimeError("Device is unhealthy. Please restart.")

        start_time = time.time()
        while time.time() - start_time < self._watchdog_timeout:
            try:
                obs = self._update_observation()
                if obs is not None:
                    #if self._record_mode:
                    self._last_obs = obs.copy()
                    return obs
            except Exception as e:
                if self._verbose:
                    print(f"Observation update error: {e}")
            time.sleep(0.001)
        raise RuntimeError(f"Watchdog timeout: {self._watchdog_timeout}s")

    @override
    def apply_action(self, action: Dict[str, Any]) -> None:
        """应用动作到双臂。

        根据控制模式调用相应的控制函数：
        - 关节角模式: 调用 _apply_joint_action
        - 位姿模式: 调用 _apply_pose_action

        Expected action format:
        {
            "actions": np.ndarray of shape (16,) containing:
                关节角模式: [left_j1, left_j2, ..., left_j7, left_gripper,
                            right_j1, right_j2, ..., right_j7, right_gripper]
                位姿模式: [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
                          right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
        }
        """
        if self._use_pose_control:
            self._apply_pose_action(action)
        else:
            self._apply_joint_action(action)
    
    def _apply_joint_action(self, action: Dict[str, Any]) -> None:
        """应用关节角动作到双臂。

        Expected action format:
        {
            "actions": np.ndarray of shape (16,) containing:
                [left_j1, left_j2, ..., left_j7, left_gripper,
                 right_j1, right_j2, ..., right_j7, right_gripper]
        }

        All joints in radians, gripper in normalized value (0-1)
        """
        joint_action = action.get("actions", None)
        if joint_action is None:
            raise ValueError("Action dict must contain 'actions' key with 16-dim joint positions.")

        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.tolist()

        if len(joint_action) != 16:
            raise ValueError(f"Expected 16-dim action, got {len(joint_action)}")

        # 记录模式
        if self._record_mode and self._last_obs is not None:
            step_data = {
                'observation': self._last_obs.copy(),
                'action': np.array(joint_action, dtype=np.float32)
            }
            self._episode_data.append(step_data)

        # 分解动作
        left_joints = joint_action[0:7]      # 左臂关节
        left_gripper = joint_action[7]       # 左臂夹爪 (归一化值)
        right_joints = joint_action[8:15]    # 右臂关节
        right_gripper = joint_action[15]     # 右臂夹爪 (归一化值)

        # 碰撞检测
        if self._collision_check_enabled:
            is_safe, min_dist = self._check_collision(left_joints, right_joints)
            if not is_safe:
                self._collision_skip_count += 1
                if self._verbose:
                    if self._collision_safety_threshold <= 0:
                        print(f"⚠️ Collision detected! Skipping action (total skipped: {self._collision_skip_count})")
                    else:
                        print(f"⚠️ Collision risk! min_dist={min_dist*1000:.1f}mm < threshold={self._collision_safety_threshold*1000:.1f}mm, skipping (total: {self._collision_skip_count})")
                # 碰撞时不执行动作，但仍然更新步数等
                self._step_count += 1
                return

        # 发送控制命令
        if not self._tele_mode:
            try:
                # 左臂关节控制
                if self._left_robot is not None and self._left_robot_enabled:
                    self._left_robot.send_joint_position(left_joints)
                    
                    # 左臂夹爪控制
                    if self._left_gripper_enabled:
                        # 传入夹爪是开还是闭合
                        if left_gripper < self._close_threshold: #闭合
                            self._left_robot.gripper_grasp(force=20.0,wait=False)
                            self._left_robot.gripper_state = 0 #夹爪状态为闭合
                        elif left_gripper > self._open_threshold: #打开
                            self._left_robot.gripper_grasp(force=-20.0,wait=False)
                            self._left_robot.gripper_state = 1 #夹爪状态为打开
                        # 0.2 <= left_gripper <= 0.8 时不控制，保持当前状态
                # 右臂关节控制
                if self._right_robot is not None and self._right_robot_enabled:
                    self._right_robot.send_joint_position(right_joints)
                    
                    # 右臂夹爪控制
                    if self._right_gripper_enabled:
                        # 传入夹爪是开还是闭合
                        if right_gripper < self._close_threshold: #闭合
                            self._right_robot.gripper_grasp(force=20.0,wait=False)
                            self._right_robot.gripper_state = 0 #夹爪状态为闭合
                        elif right_gripper > self._open_threshold: #打开
                            self._right_robot.gripper_grasp(force=-20.0,wait=False)
                            self._right_robot.gripper_state = 1 #夹爪状态为打开
                        # 0.2 <= right_gripper <= 0.8 时不控制，保持当前状态
            except Exception as e:
                print(f"Error applying joint action: {e}")
                self._done = True

        # 更新动作历史
        action_np = np.array(joint_action, dtype=np.float32)
        self._action_history.append(action_np)
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)

        self._step_count += 1
        reward = 0.0
        self._episode_reward += reward
    
    def _apply_pose_action(self, action: Dict[str, Any]) -> None:
        """应用位姿动作到双臂。

        Expected action format:
        {
            "actions": np.ndarray of shape (16,) containing:
                [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
                 right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
        }

        位置单位: 米 (m)
        姿态: 四元数 [qw, qx, qy, qz]
        夹爪: 归一化值 (0-1)
        """
        pose_action = action.get("actions", None)
        if pose_action is None:
            raise ValueError("Action dict must contain 'actions' key with 16-dim pose positions.")

        if isinstance(pose_action, np.ndarray):
            pose_action = pose_action.tolist()

        if len(pose_action) != 16:
            raise ValueError(f"Expected 16-dim action, got {len(pose_action)}")

        # 记录模式
        if self._record_mode and self._last_obs is not None:
            step_data = {
                'observation': self._last_obs.copy(),
                'action': np.array(pose_action, dtype=np.float32)
            }
            self._episode_data.append(step_data)

        # 分解动作
        left_pose = pose_action[0:7]       # 左臂末端 [x, y, z, qw, qx, qy, qz]
        left_gripper = pose_action[7]       # 左臂夹爪 (归一化值)
        right_pose = pose_action[8:15]      # 右臂末端 [x, y, z, qw, qx, qy, qz]
        right_gripper = pose_action[15]     # 右臂夹爪 (归一化值)

        # 碰撞检测（需要先通过逆运动学获取关节角）
        if self._collision_check_enabled:
            # 位姿模式下，碰撞检测需要关节角，这里暂时跳过碰撞检测
            # 如果需要，可以通过逆运动学计算关节角后再检测
            pass

        # 发送控制命令
        if not self._tele_mode:
            try:
                # 左臂位姿控制
                if self._left_robot is not None and self._left_robot_enabled:
                    self._left_robot.send_cartesian_pose(left_pose)
                    
                    # 左臂夹爪控制
                    if self._left_gripper_enabled:
                        if left_gripper < 0.5: #闭合
                            self._left_robot.gripper_grasp(force=20.0,wait=False)
                            self._left_robot.gripper_state = 0 #夹爪状态为闭合
                        else: #打开
                            self._left_robot.gripper_grasp(force=-20.0,wait=False)
                            self._left_robot.gripper_state = 1 #夹爪状态为打开

                # 右臂位姿控制
                if self._right_robot is not None and self._right_robot_enabled:
                    self._right_robot.send_cartesian_pose(right_pose)
                    
                    # 右臂夹爪控制
                    if self._right_gripper_enabled:
                        if right_gripper < 0.5: #闭合
                            self._right_robot.gripper_grasp(force=20.0,wait=False)
                            self._right_robot.gripper_state = 0 #夹爪状态为闭合
                        else: #打开
                            self._right_robot.gripper_grasp(force=-20.0,wait=False)
                            self._right_robot.gripper_state = 1 #夹爪状态为打开
            except Exception as e:
                print(f"Error applying pose action: {e}")
                self._done = True

        # 更新动作历史
        action_np = np.array(pose_action, dtype=np.float32)
        self._action_history.append(action_np)
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)

        self._step_count += 1
        reward = 0.0
        self._episode_reward += reward

    def apply_cartesian_action(self, action: Dict[str, Any]) -> None:
        """应用笛卡尔空间动作到双臂（末端位姿控制）。

        Expected action format:
        {
            "actions": np.ndarray of shape (16,) containing:
                [left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
                 right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
        }

        位置单位: 米 (m)
        姿态: 四元数 [qw, qx, qy, qz]
        夹爪: 归一化值 (0-1)
        
        该方法用于 T265 遥操作，直接发送笛卡尔位姿给机器人。
        """
        cartesian_action = action.get("actions", None)
        if cartesian_action is None:
            raise ValueError("Action dict must contain 'actions' key with 16-dim cartesian positions.")

        if isinstance(cartesian_action, np.ndarray):
            cartesian_action = cartesian_action.tolist()

        if len(cartesian_action) != 16:
            raise ValueError(f"Expected 16-dim action, got {len(cartesian_action)}")

        # 记录模式
        if self._record_mode and self._last_obs is not None:
            step_data = {
                'observation': self._last_obs.copy(),
                'action': np.array(cartesian_action, dtype=np.float32)
            }
            self._episode_data.append(step_data)

        # 分解动作
        left_pose = cartesian_action[0:7]       # 左臂末端 [x, y, z, qw, qx, qy, qz]
        left_gripper = cartesian_action[7]      # 左臂夹爪 (归一化值)
        right_pose = cartesian_action[8:15]     # 右臂末端 [x, y, z, qw, qx, qy, qz]
        right_gripper = cartesian_action[15]    # 右臂夹爪 (归一化值)

        # 发送控制命令
        if not self._tele_mode:
            try:
                # 左臂笛卡尔位姿控制
                if self._left_robot is not None and self._left_robot_enabled:
                    self._left_robot.send_cartesian_pose(left_pose)
                    
                    # 左臂夹爪控制
                    if self._left_gripper_enabled:
                        if left_gripper < 0.5: #闭合
                            self._left_robot.gripper_grasp(force=20.0,wait=False)
                            self._left_robot.gripper_state = 0 #夹爪状态为闭合
                        else: #打开
                            self._left_robot.gripper_grasp(force=-20.0,wait=False)
                            self._left_robot.gripper_state = 1 #夹爪状态为打开
                        # self._left_robot.gripper_move(target_width, wait=False)

                # 右臂笛卡尔位姿控制
                if self._right_robot is not None and self._right_robot_enabled:
                    self._right_robot.send_cartesian_pose(right_pose)
                    
                    # 右臂夹爪控制
                    if self._right_gripper_enabled:
                        if right_gripper < 0.5: #闭合
                            self._right_robot.gripper_grasp(force=20.0,wait=False)
                            self._right_robot.gripper_state = 0 #夹爪状态为闭合
                        else: #打开
                            self._right_robot.gripper_grasp(force=-20.0,wait=False)
                            self._right_robot.gripper_state = 1 #夹爪状态为打开
                        # self._right_robot.gripper_move(target_width, wait=False)

            except Exception as e:
                print(f"Error applying cartesian action: {e}")
                self._done = True

        # 更新动作历史
        action_np = np.array(cartesian_action, dtype=np.float32)
        self._action_history.append(action_np)
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)

        self._step_count += 1
        reward = 0.0
        self._episode_reward += reward

    def close(self) -> None:
        """清理资源。"""
        if self._verbose:
            print("Closing DualArmEnvironment...")

        self._stop_event.set()

        # 停止机器人
        if not self._tele_mode:
            if self._left_robot is not None:
                try:
                    if self._left_gripper_enabled:
                        self._left_robot.gripper_disable()
                    self._left_robot.stop()
                    if self._verbose:
                        print("Left robot stopped")
                except Exception as e:
                    print(f"Error stopping left robot: {e}")

            if self._right_robot is not None:
                try:
                    if self._right_gripper_enabled:
                        self._right_robot.gripper_disable()
                    self._right_robot.stop()
                    if self._verbose:
                        print("Right robot stopped")
                except Exception as e:
                    print(f"Error stopping right robot: {e}")

        # 停止相机
        try:
            self._left_wrist_camera.stop()
            if self._verbose:
                print("Left wrist camera stopped")
        except Exception as e:
            print(f"Error stopping left wrist camera: {e}")

        try:
            self._right_wrist_camera.stop()
            if self._verbose:
                print("Right wrist camera stopped")
        except Exception as e:
            print(f"Error stopping right wrist camera: {e}")

        if self._use_high_camera:
            try:
                self._high_camera.stop()
                if self._verbose:
                    print("High camera stopped")
            except Exception as e:
                print(f"Error stopping high camera: {e}")

        if self._preview_window_initialized:
            try:
                cv2.destroyWindow(self._preview_window_name)
            except:
                pass

        if self._verbose:
            print("DualArmEnvironment closed successfully")

    def _save_episode_to_hdf5(self):
        """保存当前 episode 数据到 HDF5 文件。"""
        if not self._episode_data:
            print("No episode data to save.")
            return

        os.makedirs(self._record_directory, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt = self._prompt if self._prompt else ""
        
        if prompt:
            safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
            safe_prompt = safe_prompt.replace(",", "").replace(".", "")
            max_prompt_len = 80
            if len(safe_prompt) > max_prompt_len:
                safe_prompt = safe_prompt[:max_prompt_len]
            filename = os.path.join(self._record_directory, f"episode_{timestamp_str}_{safe_prompt}.hdf5")
        else:
            filename = os.path.join(self._record_directory, f"episode_{timestamp_str}.hdf5")

        print(f"Saving episode to {filename}...")

        episode_obs_data = [step['observation'] for step in self._episode_data]
        if not episode_obs_data:
            print("No observation data to save.")
            return

        with h5py.File(filename, "w") as f:
            first_obs = episode_obs_data[0]
            first_action = self._episode_data[0]['action']
            num_steps = len(episode_obs_data)

            obs_group = f.create_group("observations")
            img_group = obs_group.create_group("images")

            state_shape = first_obs['state'].shape
            obs_group.create_dataset("qpos", (num_steps,) + state_shape, dtype=first_obs['state'].dtype)

            # 如果有 prev_state，则创建数据集
            if 'prev_state' in first_obs:
                prev_state_shape = first_obs['prev_state'].shape
                obs_group.create_dataset("prev_state", (num_steps,) + prev_state_shape, dtype=first_obs['prev_state'].dtype)

            action_shape = first_action.shape
            f.create_dataset("action", (num_steps,) + action_shape, dtype=first_action.dtype)

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

            for i in range(num_steps):
                obs_group['qpos'][i] = episode_obs_data[i]['state']
                f['action'][i] = self._episode_data[i]['action']
                
                # 如果有 prev_state，则保存
                if 'prev_state' in first_obs and 'prev_state' in episode_obs_data[i]:
                    obs_group['prev_state'][i] = episode_obs_data[i]['prev_state']
                
                for cam_name, img in episode_obs_data[i]['images'].items():
                    img_group[cam_name][i] = img
                    
                if 'prompt' in first_obs:
                    f['task'][i] = episode_obs_data[i]['prompt']

                if 'timestamps' in first_obs and 'timestamps' in episode_obs_data[i]:
                    for ts_key, ts_value in episode_obs_data[i]['timestamps'].items():
                        if ts_key in timestamps_group:
                            timestamps_group[ts_key][i] = ts_value

        print(f"Successfully saved {num_steps} steps.")
        self._episode_data = []

    def __del__(self):
        """析构函数，确保资源被释放。"""
        try:
            self.close()
        except Exception:
            pass


# 配置加载工具函数
def load_dual_arm_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件加载双臂配置。
    
    支持的相机类型:
    - "usb": 普通 USB 摄像头
    - "realsense": Intel RealSense 摄像头
    - "gopro": GoPro（通过 Elgato 采集卡）
    
    示例配置文件格式:
    ```yaml
    left_arm:
      robot_sn: "Rizon4s-063239"
      gripper_name: "Flexiv-GN01"
      wrist_camera_id: 0
      wrist_camera_type: "gopro"  # 支持 "usb", "realsense", "gopro"
      init_joints_deg: [0, 0, 0, 0, 0, 0, 0]
    
    right_arm:
      robot_sn: "Rizon4s-063240"
      gripper_name: "Flexiv-GN01"
      wrist_camera_id: 2
      wrist_camera_type: "gopro"
      init_joints_deg: [0, 0, 0, 0, 0, 0, 0]
    
    high_camera:
      camera_id: 4
      camera_type: "usb"
    
    camera:
      wrist_width: 480      # 腕部相机宽度
      wrist_height: 480     # 腕部相机高度（如果 width == height，GoPro 会自动启用 1:1 裁剪模式）
      high_width: 1280      # 全局相机宽度
      high_height: 960      # 全局相机高度
      fps: 30               # 相机帧率
    
    environment:
      max_episode_steps: 500
      gripper_max_width: 0.08
      use_pose_control: false  # true: 位姿控制, false: 关节角控制
    ```
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_dual_arm_env_from_config(config_path: str, **kwargs) -> DualArmEnvironment:
    """从配置文件创建双臂环境。"""
    config = load_dual_arm_config(config_path)
    
    left_arm = config.get('left_arm', {})
    right_arm = config.get('right_arm', {})
    high_camera = config.get('high_camera', {})
    camera_config = config.get('camera', {})
    env_config = config.get('environment', {})
    collision_config = config.get('collision', {})
    
    # 转换初始关节角度从度到弧度
    left_init_joints_deg = left_arm.get('init_joints_deg')
    right_init_joints_deg = right_arm.get('init_joints_deg')
    
    left_init_joints_rad = None
    if left_init_joints_deg is not None:
        left_init_joints_rad = [np.deg2rad(j) for j in left_init_joints_deg[:7]]
        if len(left_init_joints_deg) > 7:
            left_init_joints_rad.append(left_init_joints_deg[7])  # 夹爪值不转换
    
    right_init_joints_rad = None
    if right_init_joints_deg is not None:
        right_init_joints_rad = [np.deg2rad(j) for j in right_init_joints_deg[:7]]
        if len(right_init_joints_deg) > 7:
            right_init_joints_rad.append(right_init_joints_deg[7])
    
    # 合并参数
    params = {
        'left_robot_sn': left_arm.get('robot_sn', ''),
        'left_gripper_name': left_arm.get('gripper_name', ''),
        'left_wrist_camera_id': left_arm.get('wrist_camera_id', 0),
        'left_wrist_camera_type': left_arm.get('wrist_camera_type', 'usb'),
        'right_robot_sn': right_arm.get('robot_sn', ''),
        'right_gripper_name': right_arm.get('gripper_name', ''),
        'right_wrist_camera_id': right_arm.get('wrist_camera_id', 2),
        'right_wrist_camera_type': right_arm.get('wrist_camera_type', 'usb'),
        'high_camera_id': high_camera.get('camera_id'),
        'high_camera_type': high_camera.get('camera_type', 'usb'),
        'initial_left_joint_pos': left_init_joints_rad,
        'initial_right_joint_pos': right_init_joints_rad,
        # 相机分辨率配置（从 camera 部分读取，如果没有则使用默认值）
        'wrist_camera_width': camera_config.get('wrist_width', 320),
        'wrist_camera_height': camera_config.get('wrist_height', 240),
        'high_camera_width': camera_config.get('high_width', 1280),
        'high_camera_height': camera_config.get('high_height', 960),
        'camera_fps': camera_config.get('fps', 30),
        # 碰撞检测配置
        'collision_check_enabled': collision_config.get('enabled', True),
        'collision_safety_threshold': collision_config.get('safety_threshold', 0.03),
        # 控制模式配置
        'use_pose_control': env_config.get('use_pose_control', False),
        **env_config,
        **kwargs
    }
    
    return DualArmEnvironment(**params)


if __name__ == "__main__":
    # 实时可视化测试
    print("Testing DualArmEnvironment with real-time visualization...")
    print("按 'q' 键或 Ctrl+C 退出")
    
    # 使用示例配置
    env = create_dual_arm_env_from_config("examples/flexiv/dual_arm_env_config.yaml")
    
    try:
        env.reset()
        
        # 帧率统计
        frame_count = 0
        last_print_time = time.time()
        start_time = time.time()
        frames_in_last_second = 0
        
        window_name = "Dual Arm Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        print("开始实时可视化...")
        print("-" * 80)
        
        while True:
            # 获取当前时间（用于帧率统计）
            current_time = time.time()
            
            # 获取观测
            obs = env.get_observation()
            
            # 提取图像
            if 'cam_left_wrist' in obs['images'] and 'cam_right_wrist' in obs['images']:
                left_img = obs['images']['cam_left_wrist']
                right_img = obs['images']['cam_right_wrist']
                
                # 转换格式: (C, H, W) -> (H, W, C)
                left_img_hwc = np.transpose(left_img, (1, 2, 0))
                right_img_hwc = np.transpose(right_img, (1, 2, 0))
                
                # RGB -> BGR (OpenCV 使用 BGR 格式)
                left_img_bgr = cv2.cvtColor(left_img_hwc, cv2.COLOR_RGB2BGR)
                right_img_bgr = cv2.cvtColor(right_img_hwc, cv2.COLOR_RGB2BGR)
                
                # 水平拼接两张图像
                combined = np.hstack([left_img_bgr, right_img_bgr])
                
                # 统计帧率
                frame_count += 1
                frames_in_last_second += 1
                total_elapsed = current_time - start_time
                avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                
                # 在图像上添加帧率信息
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                color = (0, 255, 0)
                
                # 在左图（左臂相机）上添加标签
                text_left = "Left Wrist"
                cv2.putText(combined, text_left, (10, 30), font, font_scale, color, thickness)
                
                # 在右图（右臂相机）上添加标签
                h, w = left_img_bgr.shape[:2]
                text_right = "Right Wrist"
                cv2.putText(combined, text_right, (w + 10, 30), font, font_scale, color, thickness)
                
                # 在底部添加帧率信息
                fps_text = f"FPS: {avg_fps:.1f} | Frames: {frame_count}"
                cv2.putText(combined, fps_text, (10, h - 10), font, font_scale, color, thickness)
                
                # 显示图像
                cv2.imshow(window_name, combined)
            
            # 每秒打印一次统计信息
            elapsed = current_time - last_print_time
            if elapsed >= 1.0:
                fps_this_second = frames_in_last_second / elapsed
                print(f"[{total_elapsed:6.1f}s] 当前帧率: {fps_this_second:5.1f} FPS | "
                      f"平均帧率: {avg_fps:5.1f} FPS | 总帧数: {frame_count}")
                
                # 打印时间戳信息
                if 'timestamps' in obs:
                    left_ts = obs['timestamps'].get('cam_left_wrist', 0)
                    right_ts = obs['timestamps'].get('cam_right_wrist', 0)
                    print(f"  左相机时间戳: {left_ts:.6f}, 右相机时间戳: {right_ts:.6f}, "
                          f"时间差: {abs(left_ts - right_ts)*1000:.2f}ms")
                
                # 重置计数器
                last_print_time = current_time
                frames_in_last_second = 0
            
            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 检查是否episode完成
            if env.is_episode_complete():
                print("Episode complete.")
                break
            
            end_time = time.time()
            step_time = end_time - current_time
            time.sleep(max(0, 0.0083 - step_time))
                
    except KeyboardInterrupt:
        total_elapsed = time.time() - start_time
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
        print(f"\n\n测试中断")
        print(f"总运行时间: {total_elapsed:.2f} 秒")
        print(f"总帧数: {frame_count}")
        print(f"平均帧率: {avg_fps:.2f} FPS")
    finally:
        cv2.destroyWindow(window_name)
        env.close()
    
    print("Test complete.")

