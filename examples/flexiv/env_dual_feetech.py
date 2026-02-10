"""
Dual-arm Flexiv environment using external Feetech grippers.

在原有 `env_dual.DualArmEnvironment` 的基础上，重写所有与夹爪相关的逻辑，
改为使用 `utils/Feetech_servo/gripper.py` 里的 `Gripper`（左右各一个），其余接口保持兼容：

- 观测:
  - 关节控制模式: state 仍为 16 维 [左7关节, 左夹爪, 右7关节, 右夹爪]
    其中夹爪维度为 [0,1] 的归一化开合（由 Feetech 百分比/角度换算）
- 动作:
  - 关节控制模式: 动作仍为 16 维，同上，夹爪维度直接使用归一化值 (0-1) 转换为百分比 (0-100) 控制，不使用阈值

注意：
- 机械臂仍然使用 Flexiv 原装机械臂（`FlexivRobot`），只是末端夹爪改为独立的 Feetech 夹爪。
- 两侧末端都是 Feetech 夹爪，因此本环境会同时初始化左右两个 `Gripper`。
- **重要**：`left_gripper_name` 和 `right_gripper_name` 参数用于设置 TCP 工具坐标系。
  即使不使用 Flexiv 原装夹爪，仍需要通过 `gripper_enable(gripper_name, switch_tool=True)` 
  来设置正确的工具坐标系，以确保获取的 TCP 位姿与数据采集时的坐标系一致。
- **左右臂调换**：为了匹配模型训练时的左右臂定义，本环境会自动调换左右臂：
  - 观测（obs）中：物理左臂数据会出现在索引 8-15（模型右臂），物理右臂数据会出现在索引 0-7（模型左臂）
  - 动作（action）中：模型输出的"左臂"动作会应用到物理右臂，模型输出的"右臂"动作会应用到物理左臂
  - 图像观测中：物理左相机图像会作为 `cam_right_wrist`，物理右相机图像会作为 `cam_left_wrist`
- **动作缩放（action_scale）**：模型输出的是绝对位姿，通过 `action_scale` 参数可以将其转换为相对增量：
  - 计算方式：`target = prev_state + action_scale * (target_action - prev_state)`
  - 只对位姿/关节部分（索引 0-6, 8-14）应用 scale，夹爪部分（索引 7, 15）不受影响
  - `action_scale=1.0` 表示直接使用模型输出的绝对位姿，`action_scale<1.0` 会减小动作幅度
"""

# """
# 1. 数据采集到的是描述在某一个坐标系下的位姿轨迹 P_W
# 2. 通过设定gripper　的名称来去确定末端tcp的旋转状态
# 3. 当前，需要验证一下，如果我设定了gripper的名称，这个名称是经过我重新配置的，会不会得到与数采轨迹一样的结果
# 4. 如果是一样的，那么就可以直接使用这个名称来去确定末端tcp的旋转状态

# ===============
# 模型是怎么推理的
# 模型得到obs，然后，这个模型输出了一个相对于当前状态的累加了一定相对变换的位姿，来控制夹爪进行移动
# 这个相对变换的位姿，是根据当前状态和模型输出的位姿来计算的，所以，你需要保证，现在获取到的状态和数据里面的相对位姿态的描述是在他同一个坐标系下
# 那么，我当前需要确认的就是，我更改了夹爪的名字，获取到的tcp　位姿会有调整吗

# """


import os
import sys
import time
from typing import Optional, List, Dict, Any
from typing_extensions import override
from datetime import datetime

import numpy as np
import cv2
import h5py

# 用于四元数处理的库
try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, quaternion scaling will use simplified method")

# 添加 Feetech_servo 路径
CURRENT_DIR = os.path.dirname(__file__)
FEETECH_PATH = os.path.join(CURRENT_DIR, "utils", "Feetech_servo")
if FEETECH_PATH not in sys.path:
    sys.path.insert(0, FEETECH_PATH)

try:
    from gripper import Gripper
except ImportError as e:
    print(f"Warning: Could not import Feetech Gripper: {e}")
    Gripper = None  # type: ignore

from env_dual import DualArmEnvironment as _BaseDualArmEnv


class DualArmEnvironmentFeetech(_BaseDualArmEnv):
    """
    双臂 Flexiv 环境（Feetech 夹爪版本）。

    与原 `DualArmEnvironment` 的主要区别：
    - 不再使用 Flexiv 内置夹爪 API（gripper_enable / gripper_move / gripper_grasp 等）
    - 使用左右两个 Feetech `Gripper` 实例来控制夹爪开合，并读取当前开度
    - 观测和动作接口保持不变，便于与现有收集/推理脚本对接
    """

    def __init__(
        self,
        # 左臂配置
        left_robot_sn: str = "",
        left_gripper_name: str = "",  # 用于设置 TCP 工具坐标系，不实际控制 Flexiv 原装夹爪
        left_gripper_side: str = "left",
        left_gripper_config_file: Optional[str] = None,
        left_wrist_camera_id: int = 0,
        left_wrist_camera_type: str = "usb",
        # 右臂配置
        right_robot_sn: str = "",
        right_gripper_name: str = "",  # 用于设置 TCP 工具坐标系，不实际控制 Flexiv 原装夹爪
        right_gripper_side: str = "right",
        right_gripper_config_file: Optional[str] = None,
        right_wrist_camera_id: int = 2,
        right_wrist_camera_type: str = "usb",
        # 全局相机配置（已禁用，不使用 high camera）
        high_camera_id: Optional[int] = None,  # 强制设为 None，不使用全局相机
        high_camera_type: str = "usb",  # 保留参数以保持兼容性，但不使用
        # 相机分辨率
        wrist_camera_width: int = 320,
        wrist_camera_height: int = 240,
        high_camera_width: int = 1280,  # 保留参数以保持兼容性，但不使用
        high_camera_height: int = 960,  # 保留参数以保持兼容性，但不使用
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
        action_history_size: int = 30,
        action_change_threshold: float = 0.01,
        record_mode: bool = False,
        record_directory: str = "recorded_data",
        verbose: bool = True,
        # 碰撞检测配置
        collision_check_enabled: bool = True,
        collision_safety_threshold: float = 0.03,
        # 控制模式配置
        use_pose_control: bool = False,
        # 夹爪阈值
        close_threshold: float = 0.05,
        open_threshold: float = 0.95,
        # 动作缩放参数
        action_scale: float = 1.0,  # 动作缩放因子，只对位姿/关节部分生效，不对夹爪生效
    ) -> None:
        """
        额外参数：
            left_gripper_name / right_gripper_name: Flexiv 夹爪名称，用于设置 TCP 工具坐标系（不实际控制原装夹爪）
            left_gripper_side / right_gripper_side: 传给 Feetech `Gripper` 的 side（默认 "left"/"right"）
            left_gripper_config_file / right_gripper_config_file: 夹爪配置文件路径，None 时使用默认
            action_scale: 动作缩放因子，用于将模型输出的绝对位姿转换为相对位姿增量。
                         计算方式：target = prev_state + action_scale * (target_action - prev_state)
                         只对位姿/关节部分（前7维）生效，夹爪部分（第8维）不受影响，默认 1.0
        """
        # 保存动作缩放参数
        self._action_scale = action_scale
        
        # 先保存 Feetech 相关配置，供 _init_robots 使用
        self._left_gripper_name = left_gripper_name
        self._right_gripper_name = right_gripper_name
        self._left_gripper_side = left_gripper_side
        self._right_gripper_side = right_gripper_side
        self._left_gripper_config_file = left_gripper_config_file
        self._right_gripper_config_file = right_gripper_config_file

        # Feetech 夹爪实例及状态
        self._left_gripper: Optional[Gripper] = None
        self._right_gripper: Optional[Gripper] = None
        self._left_gripper_enabled: bool = False
        self._right_gripper_enabled: bool = False
        # 归一化夹爪状态 [0,1]，用于在读取失败时提供上一次已知值
        self._left_gripper_state: float = 0.0
        self._right_gripper_state: float = 0.0

        # 调用父类初始化（其中会调用我们重写的 _init_robots）
        # 注意：传入空字符串给父类，避免父类尝试控制 Flexiv 原装夹爪
        # 我们会在重写的 _init_robots 中自己处理 gripper_name 来设置工具坐标系
        # 强制禁用 high camera
        super().__init__(
            left_robot_sn=left_robot_sn,
            left_gripper_name="",  # 父类不控制夹爪，我们在 _init_robots 中自己处理
            left_wrist_camera_id=left_wrist_camera_id,
            left_wrist_camera_type=left_wrist_camera_type,
            right_robot_sn=right_robot_sn,
            right_gripper_name="",  # 父类不控制夹爪，我们在 _init_robots 中自己处理
            right_wrist_camera_id=right_wrist_camera_id,
            right_wrist_camera_type=right_wrist_camera_type,
            high_camera_id=None,  # 强制禁用全局相机
            high_camera_type=high_camera_type,
            wrist_camera_width=wrist_camera_width,
            wrist_camera_height=wrist_camera_height,
            high_camera_width=high_camera_width,
            high_camera_height=high_camera_height,
            camera_fps=camera_fps,
            max_episode_steps=max_episode_steps,
            initial_left_joint_pos=initial_left_joint_pos,
            initial_right_joint_pos=initial_right_joint_pos,
            seed=seed,
            tele_mode=tele_mode,
            prompt=prompt,
            watchdog_timeout=watchdog_timeout,
            show_camera_preview=show_camera_preview,
            gripper_max_width=0.08,  # 对 Feetech 来说不再使用，仅保持签名兼容
            action_history_size=action_history_size,
            action_change_threshold=action_change_threshold,
            record_mode=record_mode,
            record_directory=record_directory,
            verbose=verbose,
            collision_check_enabled=collision_check_enabled,
            collision_safety_threshold=collision_safety_threshold,
            use_pose_control=use_pose_control,
            close_threshold=close_threshold,
            open_threshold=open_threshold,
        )

        if self._verbose:
            print("DualArmEnvironmentFeetech initialized (using Feetech external grippers)")
            if self._action_scale != 1.0:
                print(f"  Action scale: {self._action_scale} (applied to pose/joint only, not gripper)")
    
    # ==================== 辅助函数：四元数缩放 ====================
    
    def _scale_pose_with_quaternion(
        self,
        prev_pose: np.ndarray,
        target_pose: np.ndarray,
        scale: float
    ) -> np.ndarray:
        """
        对位姿进行缩放，正确处理四元数部分。
        
        Args:
            prev_pose: 上一个时间步的位姿 [x, y, z, qw, qx, qy, qz]
            target_pose: 目标位姿 [x, y, z, qw, qx, qy, qz]
            scale: 缩放因子
            
        Returns:
            缩放后的位姿 [x, y, z, qw, qx, qy, qz]
        """
        result = np.zeros(7, dtype=np.float32)
        
        # 位置部分 (x, y, z)：线性缩放
        pos_prev = prev_pose[0:3]
        pos_target = target_pose[0:3]
        pos_diff = pos_target - pos_prev
        result[0:3] = pos_prev + scale * pos_diff
        
        # 四元数部分 (qw, qx, qy, qz)：使用相对旋转
        quat_prev = prev_pose[3:7]  # [qw, qx, qy, qz]
        quat_target = target_pose[3:7]  # [qw, qx, qy, qz]
        
        if SCIPY_AVAILABLE:
            # 使用 scipy 处理四元数
            # scipy 使用 [x, y, z, w] 格式，需要转换
            q_prev_scipy = np.array([quat_prev[1], quat_prev[2], quat_prev[3], quat_prev[0]])
            q_target_scipy = np.array([quat_target[1], quat_target[2], quat_target[3], quat_target[0]])
            
            r_prev = R.from_quat(q_prev_scipy)
            r_target = R.from_quat(q_target_scipy)
            
            # 计算相对旋转：r_diff = r_target * r_prev^(-1)
            r_diff = r_prev.inv() * r_target
            
            # 将相对旋转转换为旋转向量（轴角表示）
            rotvec_diff = r_diff.as_rotvec()
            
            # 对旋转向量应用缩放
            rotvec_scaled = scale * rotvec_diff
            
            # 将缩放后的旋转向量转换回旋转
            r_scaled_diff = R.from_rotvec(rotvec_scaled)
            
            # 计算最终旋转：r_final = r_prev * r_scaled_diff
            r_final = r_prev * r_scaled_diff
            
            # 转换回 [qw, qx, qy, qz] 格式
            q_final_scipy = r_final.as_quat()  # [x, y, z, w]
            result[3:7] = np.array([q_final_scipy[3], q_final_scipy[0], q_final_scipy[1], q_final_scipy[2]])
        else:
            # 简化方法：线性插值后归一化（不够精确，但可用）
            quat_diff = quat_target - quat_prev
            quat_scaled = quat_prev + scale * quat_diff
            # 归一化
            quat_norm = np.linalg.norm(quat_scaled)
            if quat_norm > 1e-6:
                result[3:7] = quat_scaled / quat_norm
            else:
                result[3:7] = quat_prev
        
        return result

    # ==================== 机器人 & 夹爪初始化 ====================

    def _init_robots(
        self,
        left_robot_sn: str,
        left_gripper_name: str,
        right_robot_sn: str,
        right_gripper_name: str,
    ):
        """
        重写机器人初始化：
        - 机器人部分沿用原逻辑（不启用 Flexiv 原装夹爪）
        - 额外初始化左右两个 Feetech 夹爪
        """
        from flexiv_robot import FlexivRobot, MotionParams

        # 运动参数配置与原实现保持一致
        motion_params = MotionParams(
            joint_max_vel=1.0,
            joint_max_acc=1.5,
            control_frequency=50,
        )

        # 左臂
        if left_robot_sn:
            if self._verbose:
                print(f"Connecting to left robot ({left_robot_sn})...")
            self._left_robot = FlexivRobot(
                robot_sn=left_robot_sn,
                auto_init=True,
                motion_params=motion_params,
                verbose=self._verbose,
            )
            self._left_robot_enabled = self._left_robot.is_ready()
            
            # 如果提供了 gripper_name，用于设置 TCP 工具坐标系（不实际控制原装夹爪）
            if self._left_gripper_name and self._left_robot_enabled:
                try:
                    # 使用 switch_tool=True 来切换工具坐标系
                    self._left_robot.gripper_enable(self._left_gripper_name, switch_tool=False)
                    if self._verbose:
                        print(f"Left robot TCP tool coordinate set to [{self._left_gripper_name}]")
                except Exception as e:
                    print(f"Warning: Failed to set left robot TCP tool coordinate: {e}")
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
                verbose=self._verbose,
            )
            self._right_robot_enabled = self._right_robot.is_ready()
            
            # 如果提供了 gripper_name，用于设置 TCP 工具坐标系（不实际控制原装夹爪）
            if self._right_gripper_name and self._right_robot_enabled:
                try:
                    # 使用 switch_tool=True 来切换工具坐标系
                    self._right_robot.gripper_enable(self._right_gripper_name, switch_tool=False)
                    if self._verbose:
                        print(f"Right robot TCP tool coordinate set to [{self._right_gripper_name}]")
                except Exception as e:
                    print(f"Warning: Failed to set right robot TCP tool coordinate: {e}")
        else:
            self._right_robot = None
            print("Warning: No right robot SN provided")

        # 初始化 Feetech 夹爪
        if Gripper is None:
            print("Warning: Feetech Gripper class not available, grippers will be disabled")
            self._left_gripper_enabled = False
            self._right_gripper_enabled = False
            return

        # 左侧 Feetech 夹爪
        try:
            cfg = self._left_gripper_config_file
            # 如果提供了配置文件路径，需要解析为绝对路径
            if cfg is not None:
                if not os.path.isabs(cfg):
                    # 相对路径：相对于项目根目录或当前文件所在目录
                    # 先尝试相对于项目根目录
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
                    abs_cfg = os.path.join(project_root, cfg)
                    if not os.path.exists(abs_cfg):
                        # 如果不存在，尝试相对于当前文件所在目录
                        abs_cfg = os.path.join(os.path.dirname(__file__), cfg)
                    cfg = abs_cfg if os.path.exists(abs_cfg) else cfg
            else:
                # None 时使用默认路径（Gripper 类会在 Feetech_servo 目录下查找）
                cfg = None
            
            self._left_gripper = Gripper(
                side=self._left_gripper_side,
                config_file=cfg,
            )
            if self._left_gripper.connect():
                self._left_gripper_enabled = True
                if self._verbose:
                    print(f"Left Feetech gripper [{self._left_gripper_side}] connected")
            else:
                print("Warning: Failed to connect left Feetech gripper")
                self._left_gripper_enabled = False
        except Exception as e:
            print(f"Warning: Left Feetech gripper init failed: {e}")
            self._left_gripper = None
            self._left_gripper_enabled = False

        # 右侧 Feetech 夹爪
        try:
            cfg = self._right_gripper_config_file
            # 如果提供了配置文件路径，需要解析为绝对路径
            if cfg is not None:
                if not os.path.isabs(cfg):
                    # 相对路径：相对于项目根目录或当前文件所在目录
                    # 先尝试相对于项目根目录
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
                    abs_cfg = os.path.join(project_root, cfg)
                    if not os.path.exists(abs_cfg):
                        # 如果不存在，尝试相对于当前文件所在目录
                        abs_cfg = os.path.join(os.path.dirname(__file__), cfg)
                    cfg = abs_cfg if os.path.exists(abs_cfg) else cfg
            else:
                # None 时使用默认路径（Gripper 类会在 Feetech_servo 目录下查找）
                cfg = None
            
            self._right_gripper = Gripper(
                side=self._right_gripper_side,
                config_file=cfg,
            )
            if self._right_gripper.connect():
                self._right_gripper_enabled = True
                if self._verbose:
                    print(f"Right Feetech gripper [{self._right_gripper_side}] connected")
            else:
                print("Warning: Failed to connect right Feetech gripper")
                self._right_gripper_enabled = False
        except Exception as e:
            print(f"Warning: Right Feetech gripper init failed: {e}")
            self._right_gripper = None
            self._right_gripper_enabled = False

        # 初始化时尝试将夹爪打开
        try:
            if self._left_gripper_enabled and self._left_gripper is not None:
                self._left_gripper.open(wait=False)
                self._left_gripper_state = 1.0
            if self._right_gripper_enabled and self._right_gripper is not None:
                self._right_gripper.open(wait=False)
                self._right_gripper_state = 1.0
        except Exception as e:
            print(f"Warning: Failed to open grippers during init: {e}")

    # ==================== 状态读取 ====================

    def _get_robot_state_joint(self) -> np.ndarray:
        """
        重写关节角状态读取，夹爪状态来自 Feetech 夹爪百分比（0~100）归一化。
        
        返回格式与原环境保持一致：
        [左臂关节角x7, 左臂夹爪x1, 右臂关节角x7, 右臂夹爪x1]
        """
        state = np.zeros(16, dtype=np.float32)
        
        # 左臂关节角度 (0-6)
        if getattr(self, "_left_robot", None) is not None and getattr(
            self, "_left_robot_enabled", False
        ):
            try:
                left_joints = self._left_robot.get_joint_positions()
                state[0:7] = left_joints[:7]
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get left robot state: {e}")

        # 左臂夹爪 (7)
        if self._left_gripper_enabled and self._left_gripper is not None:
            try:
                percent = self._left_gripper.get_percent()
                if percent is not None:
                    self._left_gripper_state = float(percent) / 100.0
                state[7] = self._left_gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get left Feetech gripper state: {e}")
                state[7] = self._left_gripper_state

        # 右臂关节角度 (8-14)
        if getattr(self, "_right_robot", None) is not None and getattr(
            self, "_right_robot_enabled", False
        ):
            try:
                right_joints = self._right_robot.get_joint_positions()
                state[8:15] = right_joints[:7]
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get right robot state: {e}")

        # 右臂夹爪 (15)
        if self._right_gripper_enabled and self._right_gripper is not None:
            try:
                percent = self._right_gripper.get_percent()
                if percent is not None:
                    self._right_gripper_state = float(percent) / 100.0
                state[15] = self._right_gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get right Feetech gripper state: {e}")
                state[15] = self._right_gripper_state

        return state

    def _get_robot_state_pose(self) -> np.ndarray:
        """
        位姿模式下，沿用原来的末端位姿获取逻辑，只修改夹爪状态来源为 Feetech。
        
        返回格式与原环境保持一致：
        [左臂位姿x7(x,y,z,qw,qx,qy,qz), 左臂夹爪x1, 右臂位姿x7, 右臂夹爪x1]
        """
        # 临时禁用父类的夹爪标志，避免父类尝试访问不存在的 gripper_state
        left_gripper_enabled_backup = self._left_gripper_enabled
        right_gripper_enabled_backup = self._right_gripper_enabled
        try:
            self._left_gripper_enabled = False
            self._right_gripper_enabled = False
            state = super()._get_robot_state_pose()
        finally:
            # 恢复标志
            self._left_gripper_enabled = left_gripper_enabled_backup
            self._right_gripper_enabled = right_gripper_enabled_backup

        # 左夹爪
        if self._left_gripper_enabled and self._left_gripper is not None:
            try:
                percent = self._left_gripper.get_percent()
                if percent is not None:
                    # 将百分比 (0-100) 转换为归一化值 (0-1)
                    self._left_gripper_state = float(percent) / 100.0
                state[7] = self._left_gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get left Feetech gripper state (pose): {e}")
                state[7] = self._left_gripper_state

        # 右夹爪
        if self._right_gripper_enabled and self._right_gripper is not None:
            try:
                percent = self._right_gripper.get_percent()
                if percent is not None:
                    # 将百分比 (0-100) 转换为归一化值 (0-1)
                    self._right_gripper_state = float(percent) / 100.0
                state[15] = self._right_gripper_state
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to get right Feetech gripper state (pose): {e}")
                state[15] = self._right_gripper_state

        return state

    # ==================== 观测更新 ====================
    
    def _update_observation(self) -> Dict[str, Any]:
        """
        观测更新逻辑直接沿用父类，实现与原环境一致的左右臂/相机语义。
        """
        return super()._update_observation()

    # ==================== 环境重置 & 关闭 ====================

    def reset(self) -> None:
        """
        重置时仍然移动两侧机械臂到初始位姿，
        同时尝试将两个 Feetech 夹爪打开作为初始状态。
        
        注意：需要临时禁用父类的夹爪标志，避免父类尝试调用 Flexiv 的 gripper_move()
        """
        # 临时禁用父类的夹爪标志，避免父类尝试调用 Flexiv 的 gripper_move()
        left_gripper_enabled_backup = self._left_gripper_enabled
        right_gripper_enabled_backup = self._right_gripper_enabled
        try:
            # 禁用父类的夹爪标志，这样父类不会尝试调用 gripper_move()
            self._left_gripper_enabled = False
            self._right_gripper_enabled = False
            super().reset()
        finally:
            # 恢复 Feetech 夹爪标志
            self._left_gripper_enabled = left_gripper_enabled_backup
            self._right_gripper_enabled = right_gripper_enabled_backup

        # 在父类完成关节运动后，再做 Feetech 夹爪的复位
        if not getattr(self, "_tele_mode", False):
            try:
                if self._left_gripper_enabled and self._left_gripper is not None:
                    self._left_gripper.open(wait=True)
                    self._left_gripper_state = 1.0
                if self._right_gripper_enabled and self._right_gripper is not None:
                    self._right_gripper.open(wait=True)
                    self._right_gripper_state = 1.0
            except Exception as e:
                if self._verbose:
                    print(f"Warning: Failed to open grippers in reset: {e}")
        
        # 重置上一个状态（用于动作缩放）
        self._prev_joint_state = None
        self._prev_pose_state = None

    def close(self) -> None:
        """
        关闭环境时，除了原有的相机和机器人清理，还需要断开 Feetech 夹爪。
        """
        # 先断开 Feetech 夹爪连接
        try:
            if self._left_gripper is not None:
                self._left_gripper.disconnect()
        except Exception as e:
            if self._verbose:
                print(f"Error disconnecting left Feetech gripper: {e}")

        try:
            if self._right_gripper is not None:
                self._right_gripper.disconnect()
        except Exception as e:
            if self._verbose:
                print(f"Error disconnecting right Feetech gripper: {e}")

        # 然后调用父类 close，但为了避免再次尝试控制 Flexiv 原装夹爪，
        # 这里临时关闭相关标志位
        left_enabled_backup = getattr(self, "_left_gripper_enabled", False)
        right_enabled_backup = getattr(self, "_right_gripper_enabled", False)
        try:
            # 父类里这些标志只是用于打印/逻辑判断，不再真正控制 Feetech 夹爪
            self._left_gripper_enabled = False
            self._right_gripper_enabled = False
            super().close()
        finally:
            self._left_gripper_enabled = left_enabled_backup
            self._right_gripper_enabled = right_enabled_backup

    # ==================== 动作应用（关节 / 位姿 / 笛卡尔） ====================


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
        """
        关节模式下的动作应用：
        - 关节 14 维仍由 Flexiv 机器人控制
        - 夹爪维度由 Feetech 夹爪控制，按照归一化值控制
        - 支持动作缩放和左右臂调换
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
                "observation": self._last_obs.copy(),
                "action": np.array(joint_action, dtype=np.float32),
            }
            self._episode_data.append(step_data)

        # 获取上一个时间步的状态（模型视角）
        prev_joint_state = getattr(self, "_prev_joint_state", None)
        if prev_joint_state is None:
            # 如果没有上一个状态，直接使用当前动作（相当于 scale=1.0）
            prev_joint_state = np.array(joint_action, dtype=np.float32)
        
        # 将动作转换为 numpy 数组
        target_action = np.array(joint_action, dtype=np.float32)
        
        # 计算 diff = target_action - prev_state
        diff = target_action - prev_joint_state
        
        # 应用缩放：只对关节部分（索引 0-6, 8-14）应用 scale，夹爪部分（索引 7, 15）不应用
        scaled_diff = diff.copy()
        scaled_diff[0:7] *= self._action_scale  # 模型左臂关节
        scaled_diff[8:15] *= self._action_scale  # 模型右臂关节
        # 夹爪部分（索引 7, 15）不缩放，直接使用 diff
        
        # 计算最终目标：target = prev_state + scaled_diff
        final_action = prev_joint_state + scaled_diff
        
        # 分解最终动作（模型视角 = 物理视角）
        left_joints = final_action[0:7].tolist()
        left_gripper_val = float(final_action[7])
        right_joints = final_action[8:15].tolist()
        right_gripper_val = float(final_action[15])

        # 碰撞检测沿用父类实现
        if self._collision_check_enabled:
            is_safe, min_dist = self._check_collision(left_joints, right_joints)
            if not is_safe:
                self._collision_skip_count += 1
                if self._verbose:
                    if self._collision_safety_threshold <= 0:
                        print(
                            f"⚠️ Collision detected! Skipping action (total skipped: {self._collision_skip_count})"
                        )
                    else:
                        print(
                            f"⚠️ Collision risk! min_dist={min_dist*1000:.1f}mm < "
                            f"threshold={self._collision_safety_threshold*1000:.1f}mm, "
                            f"skipping (total: {self._collision_skip_count})"
                        )
                self._step_count += 1
                return

        # 发送控制命令
        if not self._tele_mode:
            try:
                # 左臂关节控制
                if getattr(self, "_left_robot", None) is not None and getattr(
                    self, "_left_robot_enabled", False
                ):
                    self._left_robot.send_joint_position(left_joints)

                # 右臂关节控制
                if getattr(self, "_right_robot", None) is not None and getattr(
                    self, "_right_robot_enabled", False
                ):
                    self._right_robot.send_joint_position(right_joints)

                # 左 Feetech 夹爪控制
                if self._left_gripper_enabled and self._left_gripper is not None:
                    self._left_gripper.set_normalized(left_gripper_val, wait=False)
                    self._left_gripper_state = left_gripper_val  # 保存归一化值

                # 右 Feetech 夹爪控制
                if self._right_gripper_enabled and self._right_gripper is not None:
                    self._right_gripper.set_normalized(right_gripper_val, wait=False)
                    self._right_gripper_state = right_gripper_val  # 保存归一化值

            except Exception as e:
                print(f"Error applying joint action (Feetech grippers): {e}")
                self._done = True

        # 更新上一个状态（用于下一次动作缩放计算）
        # 注意：使用 final_action（已应用缩放后的动作），而不是原始 target_action
        self._prev_joint_state = final_action.copy()

        # 更新动作历史
        action_np = np.array(joint_action, dtype=np.float32)
        self._action_history.append(action_np)
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)

        self._step_count += 1
        reward = 0.0
        self._episode_reward += reward
    
    """
    

    def _apply_pose_action(self, action: Dict[str, Any]) -> None:

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
                "observation": self._last_obs.copy(),
                "action": np.array(pose_action, dtype=np.float32),
            }
            self._episode_data.append(step_data)

        # 获取上一个时间步的位姿状态（模型视角）
        prev_pose_state = getattr(self, "_prev_pose_state", None)
        if prev_pose_state is None:
            # 如果没有上一个状态，直接使用当前动作（相当于 scale=1.0）
            prev_pose_state = np.array(pose_action, dtype=np.float32)
        
        # 将动作转换为 numpy 数组
        target_action = np.array(pose_action, dtype=np.float32)
        
        # 计算最终动作：分别处理左右臂的位姿和夹爪
        final_action = np.zeros(16, dtype=np.float32)
        
        # 模型左臂位姿（索引 0-6）：正确处理四元数
        prev_left_pose = prev_pose_state[0:7]
        target_left_pose = target_action[0:7]
        final_action[0:7] = self._scale_pose_with_quaternion(
            prev_left_pose, target_left_pose, self._action_scale
        )
        
        # 模型左臂夹爪（索引 7）：不缩放，直接使用目标值
        final_action[7] = target_action[7]
        
        # 模型右臂位姿（索引 8-14）：正确处理四元数
        prev_right_pose = prev_pose_state[8:15]
        target_right_pose = target_action[8:15]
        final_action[8:15] = self._scale_pose_with_quaternion(
            prev_right_pose, target_right_pose, self._action_scale
        )
        
        # 模型右臂夹爪（索引 15）：不缩放，直接使用目标值
        final_action[15] = target_action[15]
        
        # 分解最终动作（模型视角 = 物理视角）
        left_pose = final_action[0:7].tolist()
        left_gripper_val = float(final_action[7])
        right_pose = final_action[8:15].tolist()
        right_gripper_val = float(final_action[15])

        if not self._tele_mode:
            try:
                # 左臂位姿控制
                if getattr(self, "_left_robot", None) is not None and getattr(
                    self, "_left_robot_enabled", False
                ):
                    self._left_robot.send_cartesian_pose(left_pose)

                # 右臂位姿控制
                if getattr(self, "_right_robot", None) is not None and getattr(
                    self, "_right_robot_enabled", False
                ):
                    self._right_robot.send_cartesian_pose(right_pose)

                # 左 Feetech 夹爪控制
                if self._left_gripper_enabled and self._left_gripper is not None:
                    self._left_gripper.set_normalized(left_gripper_val, wait=False)
                    self._left_gripper_state = left_gripper_val  # 保存归一化值

                # 右 Feetech 夹爪控制
                if self._right_gripper_enabled and self._right_gripper is not None:
                    self._right_gripper.set_normalized(right_gripper_val, wait=False)
                    self._right_gripper_state = right_gripper_val  # 保存归一化值

            except Exception as e:
                print(f"Error applying pose action (Feetech grippers): {e}")
                self._done = True

        # 更新上一个状态（用于下一次动作缩放计算）
        # 注意：使用 final_action（已应用缩放后的动作），而不是原始 target_action
        self._prev_pose_state = final_action.copy()

        # 更新动作历史
        action_np = np.array(pose_action, dtype=np.float32)
        self._action_history.append(action_np)
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)

        self._step_count += 1
        reward = 0.0
        self._episode_reward += reward


    
    """

    def _apply_pose_action(self, action: Dict[str, Any]) -> None:
        """
        位姿模式动作应用：
        - 直接使用 action，不进行缩放
        - 夹爪部分改为 Feetech 夹爪控制
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
                "observation": self._last_obs.copy(),
                "action": np.array(pose_action, dtype=np.float32),
            }
            self._episode_data.append(step_data)

        # 直接使用 action，不进行缩放
        final_action = np.array(pose_action, dtype=np.float32)
        
        # 分解最终动作（模型视角 = 物理视角）
        left_pose = final_action[0:7].tolist()
        left_gripper_val = float(final_action[7])
        right_pose = final_action[8:15].tolist()
        right_gripper_val = float(final_action[15])

        if not self._tele_mode:
            try:
                # 左臂位姿控制
                if getattr(self, "_left_robot", None) is not None and getattr(
                    self, "_left_robot_enabled", False
                ):
                    self._left_robot.send_cartesian_pose(left_pose)

                # 右臂位姿控制
                if getattr(self, "_right_robot", None) is not None and getattr(
                    self, "_right_robot_enabled", False
                ):
                    self._right_robot.send_cartesian_pose(right_pose)

                # 左 Feetech 夹爪控制
                if self._left_gripper_enabled and self._left_gripper is not None:
                    self._left_gripper.set_normalized(left_gripper_val, wait=False)
                    self._left_gripper_state = left_gripper_val  # 保存归一化值

                # 右 Feetech 夹爪控制
                if self._right_gripper_enabled and self._right_gripper is not None:
                    self._right_gripper.set_normalized(right_gripper_val, wait=False)
                    self._right_gripper_state = right_gripper_val  # 保存归一化值

            except Exception as e:
                print(f"Error applying pose action (Feetech grippers): {e}")
                self._done = True

        # 更新上一个状态（虽然不再用于缩放计算，但保留用于记录）
        self._prev_pose_state = final_action.copy()

        # 更新动作历史
        action_np = np.array(pose_action, dtype=np.float32)
        self._action_history.append(action_np)
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)

        self._step_count += 1
        reward = 0.0
        self._episode_reward += reward

    def apply_cartesian_action(self, action: Dict[str, Any]) -> None:
        """
        笛卡尔空间动作（T265 遥操作）：
        - 末端位姿控制保持不变
        - 夹爪改为 Feetech 控制，直接使用归一化值 (0-1) 转换为百分比 (0-100) 控制
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
                "observation": self._last_obs.copy(),
                "action": np.array(cartesian_action, dtype=np.float32),
            }
            self._episode_data.append(step_data)

        # 获取上一个时间步的位姿状态（模型视角）
        prev_pose_state = getattr(self, "_prev_pose_state", None)
        if prev_pose_state is None:
            # 如果没有上一个状态，直接使用当前动作（相当于 scale=1.0）
            prev_pose_state = np.array(cartesian_action, dtype=np.float32)
        
        # 将动作转换为 numpy 数组
        target_action = np.array(cartesian_action, dtype=np.float32)
        
        # 计算最终动作：分别处理左右臂的位姿和夹爪
        final_action = np.zeros(16, dtype=np.float32)
        
        # 模型左臂位姿（索引 0-6）：正确处理四元数
        prev_left_pose = prev_pose_state[0:7]
        target_left_pose = target_action[0:7]
        final_action[0:7] = self._scale_pose_with_quaternion(
            prev_left_pose, target_left_pose, self._action_scale
        )
        
        # 模型左臂夹爪（索引 7）：不缩放，直接使用目标值
        final_action[7] = target_action[7]
        
        # 模型右臂位姿（索引 8-14）：正确处理四元数
        prev_right_pose = prev_pose_state[8:15]
        target_right_pose = target_action[8:15]
        final_action[8:15] = self._scale_pose_with_quaternion(
            prev_right_pose, target_right_pose, self._action_scale
        )
        
        # 模型右臂夹爪（索引 15）：不缩放，直接使用目标值
        final_action[15] = target_action[15]
        
        # 分解最终动作（模型视角 = 物理视角）
        left_pose = final_action[0:7].tolist()
        left_gripper_val = float(final_action[7])
        right_pose = final_action[8:15].tolist()
        right_gripper_val = float(final_action[15])

        if not self._tele_mode:
            try:
                # 左臂
                if getattr(self, "_left_robot", None) is not None and getattr(
                    self, "_left_robot_enabled", False
                ):
                    self._left_robot.send_cartesian_pose(left_pose)

                # 右臂
                if getattr(self, "_right_robot", None) is not None and getattr(
                    self, "_right_robot_enabled", False
                ):
                    self._right_robot.send_cartesian_pose(right_pose)

                # 左 Feetech 夹爪
                if self._left_gripper_enabled and self._left_gripper is not None:
                    self._left_gripper.set_normalized(left_gripper_val, wait=False)
                    self._left_gripper_state = left_gripper_val  # 保存归一化值

                # 右 Feetech 夹爪
                if self._right_gripper_enabled and self._right_gripper is not None:
                    self._right_gripper.set_normalized(right_gripper_val, wait=False)
                    self._right_gripper_state = right_gripper_val  # 保存归一化值

            except Exception as e:
                print(f"Error applying cartesian action (Feetech grippers): {e}")
                self._done = True

        # 更新上一个状态（用于下一次动作缩放计算）
        # 注意：使用 final_action（已应用缩放后的动作），而不是原始 target_action
        self._prev_pose_state = final_action.copy()

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


def create_dual_arm_env_from_config(config_path: str, **kwargs) -> "DualArmEnvironment":
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
        'high_camera_id': None,  # 强制禁用全局相机
        'high_camera_type': high_camera.get('camera_type', 'usb'),  # 保留参数以保持兼容性
        'initial_left_joint_pos': left_init_joints_rad,
        'initial_right_joint_pos': right_init_joints_rad,
        # 相机分辨率配置
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
        # 环境参数
        **{k: v for k, v in env_config.items() if k != 'use_pose_control'},
        **kwargs
    }
    
    return DualArmEnvironment(**params)


# 为了方便直接替换导入，可以导出一个同名别名
DualArmEnvironment = DualArmEnvironmentFeetech


# 配置加载工具函数
def load_dual_arm_feetech_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件加载双臂配置（Feetech 夹爪版本）。
    
    支持的相机类型:
    - "usb": 普通 USB 摄像头
    - "realsense": Intel RealSense 摄像头
    - "gopro": GoPro（通过 Elgato 采集卡）
    
    示例配置文件格式:
    ```yaml
    left_arm:
      robot_sn: "Rizon4s-063239"
      gripper_name: "Flexiv-GN01"  # 用于设置 TCP 工具坐标系
      gripper_side: "left"
      gripper_config_file: null  # null 表示使用默认路径
      wrist_camera_id: 0
      wrist_camera_type: "usb"
      init_joints_deg: [0, 0, 0, 0, 0, 0, 0]
    
    right_arm:
      robot_sn: "Rizon4s-063240"
      gripper_name: "Flexiv-GN01"  # 用于设置 TCP 工具坐标系
      gripper_side: "right"
      gripper_config_file: null
      wrist_camera_id: 2
      wrist_camera_type: "usb"
      init_joints_deg: [0, 0, 0, 0, 0, 0, 0]
    
    gripper:
      close_threshold: 0.05
      open_threshold: 0.95
    ```
    """
    import yaml
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_dual_arm_feetech_env_from_config(config_path: str, **kwargs) -> DualArmEnvironmentFeetech:
    """从配置文件创建双臂环境（Feetech 夹爪版本）。"""
    config = load_dual_arm_feetech_config(config_path)
    
    left_arm = config.get('left_arm', {})
    right_arm = config.get('right_arm', {})
    high_camera = config.get('high_camera', {})
    camera_config = config.get('camera', {})
    env_config = config.get('environment', {})
    collision_config = config.get('collision', {})
    gripper_config = config.get('gripper', {})
    
    # 转换初始关节角度从度到弧度
    left_init_joints_deg = left_arm.get('init_joints_deg')
    right_init_joints_deg = right_arm.get('init_joints_deg')
    
    left_init_joints_rad = None
    if left_init_joints_deg is not None:
        left_init_joints_rad = [np.deg2rad(j) for j in left_init_joints_deg[:7]]
    
    right_init_joints_rad = None
    if right_init_joints_deg is not None:
        right_init_joints_rad = [np.deg2rad(j) for j in right_init_joints_deg[:7]]
    
    # 合并参数
    params = {
        'left_robot_sn': left_arm.get('robot_sn', ''),
        'left_gripper_name': left_arm.get('gripper_name', ''),  # 用于设置 TCP 工具坐标系
        'left_gripper_side': left_arm.get('gripper_side', 'left'),
        'left_gripper_config_file': left_arm.get('gripper_config_file'),
        'left_wrist_camera_id': left_arm.get('wrist_camera_id', 0),
        'left_wrist_camera_type': left_arm.get('wrist_camera_type', 'usb'),
        'right_robot_sn': right_arm.get('robot_sn', ''),
        'right_gripper_name': right_arm.get('gripper_name', ''),  # 用于设置 TCP 工具坐标系
        'right_gripper_side': right_arm.get('gripper_side', 'right'),
        'right_gripper_config_file': right_arm.get('gripper_config_file'),
        'right_wrist_camera_id': right_arm.get('wrist_camera_id', 2),
        'right_wrist_camera_type': right_arm.get('wrist_camera_type', 'usb'),
        'high_camera_id': None,  # 强制禁用全局相机
        'high_camera_type': high_camera.get('camera_type', 'usb'),  # 保留参数以保持兼容性
        'initial_left_joint_pos': left_init_joints_rad,
        'initial_right_joint_pos': right_init_joints_rad,
        # 相机分辨率配置
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
        # 夹爪阈值
        'close_threshold': gripper_config.get('close_threshold', 0.05),
        'open_threshold': gripper_config.get('open_threshold', 0.95),
        # 动作缩放参数
        'action_scale': env_config.get('action_scale', 0.1),
        # 环境参数
        **{k: v for k, v in env_config.items() if k not in ['use_pose_control', 'action_scale']},
        **kwargs
    }
    
    return DualArmEnvironmentFeetech(**params)


if __name__ == "__main__":
    # 简单自测：使用配置文件加载环境
    import os
    
    print("Testing DualArmEnvironmentFeetech (Feetech grippers)...")
    
    # 配置文件路径
    config_path = os.path.join(
        os.path.dirname(__file__),
        "dual_arm_env_feetech_config.yaml"
    )
    
    try:
        if os.path.exists(config_path):
            print(f"📄 从配置文件加载: {config_path}")
            env = create_dual_arm_feetech_env_from_config(
                config_path,
                tele_mode=True,  # 默认不开启真实机器人运动，方便在无机情况下测试图像/夹爪加载
                show_camera_preview=False,
            )
        else:
            print(f"⚠️  配置文件不存在: {config_path}")
            print("使用默认参数...")
            env = DualArmEnvironmentFeetech(
                left_robot_sn="",
                right_robot_sn="",
                tele_mode=True,
                show_camera_preview=False,
            )
        
        print("✅ 环境初始化成功")
        
        env.reset()
        print("✅ Reset 完成")
        
        print("⏱  开始连续获取 5 秒观测并实时显示相机画面（按 q 关闭窗口）...")
        start_time = time.time()
        while time.time() - start_time < 20.0:
            obs = env.get_observation()
            
            # 打印一次状态结构（可根据需要精简或注释掉）
            state = obs.get("state", None)
            if state is not None:
                print(f"\r当前 state 形状: {state.shape}", end="", flush=True)
            print("state: ", state)
            # 实时可视化图像（注意：obs['images'] 中是 [C,H,W] 的 RGB，需要转回 [H,W,C] BGR）
            images = obs.get("images", {})
            left_img = images.get("cam_left_wrist", None)
            right_img = images.get("cam_right_wrist", None)
            high_img = images.get("cam_high", None)

            # 兼容 Python3.8/3.9 的类型注解写法
            def chw_to_bgr(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if img is None:
                    return None
                if img.ndim == 3 and img.shape[0] in (1, 3):  # [C,H,W]
                    img_hwc = np.transpose(img, (1, 2, 0))  # -> [H,W,C], RGB
                    if img_hwc.shape[2] == 3:
                        return cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
                    return img_hwc
                return img

            left_vis = chw_to_bgr(left_img)
            right_vis = chw_to_bgr(right_img)
            high_vis = chw_to_bgr(high_img)

            env._display_camera_preview(left_vis, right_vis, high_vis)
        
        env.close()
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

