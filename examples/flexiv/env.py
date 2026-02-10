"""
Flexiv Robot Environment for OpenPi.

这个环境类用于 Flexiv 机器人的推理和数据采集，参考 examples/piper/env.py 实现。
- 从 Flexiv 机器人获取关节角度和夹爪宽度
- 从 GoPro（通过 Elgato 采集卡 v4l 设备）获取图像
- 执行关节位置控制和夹爪控制

使用示例:
    from env import FlexivEnvironment
    
    env = FlexivEnvironment(
        robot_sn="Rizon4s-063239",
        gripper_name="Flexiv-GN01",
        camera_ids=[0, 2],  # cam_high, cam_wrist
    )
    env.reset()
    obs = env.get_observation()
    env.apply_action({"actions": np.zeros(8)})  # 7 joints + 1 gripper
    env.close()
"""

import sys
import os
import time
import threading
from typing import List, Optional, Union

import numpy as np
import cv2

# 添加 third_party 路径以导入 flexiv_robot
FLEXIV_USAGE_PATH = os.path.join(os.path.dirname(__file__), '../../third_party/flexiv_usage')
sys.path.insert(0, FLEXIV_USAGE_PATH)

from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    from flexiv_robot import FlexivRobot, MotionParams
except ImportError as e:
    print(f"Warning: Could not import FlexivRobot: {e}")
    print(f"Please ensure flexiv_robot.py is in {FLEXIV_USAGE_PATH}")
    raise


class GoproCamera:
    """
    GoPro 相机封装类，通过 Elgato Cam Link 采集卡读取视频流。
    使用 OpenCV 的 VideoCapture 访问 v4l 设备。
    """
    
    def __init__(
        self,
        camera_id: Union[int, str] = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        name: str = "gopro",
        warmup_frames: int = 10
    ):
        """
        初始化 GoPro 相机。
        
        Args:
            camera_id: 相机设备 ID 或路径（如 0, "/dev/video0"）
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            name: 相机名称（用于日志）
            warmup_frames: 预热帧数，启动时读取并丢弃的帧数
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.warmup_frames = warmup_frames
        
        self._capture: Optional[cv2.VideoCapture] = None
        self._started = False
        self._lock = threading.Lock()
    
    def start(self, timeout: float = 5.0) -> bool:
        """启动相机"""
        if self._started:
            return True
        
        try:
            self._capture = cv2.VideoCapture(self.camera_id)
            
            if not self._capture.isOpened():
                print(f"[{self.name}] ❌ 无法打开相机 {self.camera_id}")
                return False
            
            # 设置相机属性
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 等待第一帧成功读取
            start_time = time.time()
            first_frame_read = False
            while time.time() - start_time < timeout:
                ret, frame = self._capture.read()
                if ret and frame is not None:
                    first_frame_read = True
                    break
                time.sleep(0.01)
            
            if not first_frame_read:
                print(f"[{self.name}] ⚠️ 无法读取第一帧，预热可能不完整")
            
            # 预热：读取并丢弃几帧以确保摄像头稳定
            if self.warmup_frames > 0:
                print(f"[{self.name}] 正在预热摄像头（读取 {self.warmup_frames} 帧）...")
                warmup_count = 0
                for i in range(self.warmup_frames):
                    ret, frame = self._capture.read()
                    if ret and frame is not None:
                        warmup_count += 1
                    time.sleep(0.033)  # 约 30fps 的间隔
                print(f"[{self.name}] ✅ 预热完成（成功读取 {warmup_count}/{self.warmup_frames} 帧）")
            
            # 获取实际分辨率
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._capture.get(cv2.CAP_PROP_FPS))
            
            self.width = actual_width
            self.height = actual_height
            
            self._started = True
            print(f"[{self.name}] ✅ 相机已启动: {actual_width}x{actual_height}@{actual_fps}fps")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ❌ 启动失败: {e}")
            return False
    
    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """读取一帧图像（BGR 格式）"""
        if not self._started or self._capture is None:
            return None
        
        with self._lock:
            try:
                ret, frame = self._capture.read()
                if ret and frame is not None:
                    return frame
                return None
            except Exception as e:
                print(f"[{self.name}] ❌ 读取错误: {e}")
                return None
    
    def stop(self):
        """停止相机"""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
        self._started = False
        print(f"[{self.name}] 相机已停止")
    
    @property
    def is_started(self) -> bool:
        return self._started


class FlexivEnvironment(_environment.Environment):
    """
    Flexiv 机器人环境类，用于实机推理。
    
    观测空间 (observation):
        - state: 机器人状态 [joint_1, ..., joint_7, gripper_width]，shape=(8,)
        - images:
            - cam_high: 全局相机图像，shape=(3, 224, 224)
            - cam_wrist (可选): 腕部相机图像，shape=(3, 224, 224)
        - prompt: 任务提示文本
    
    动作空间 (action):
        - actions: [joint_1, ..., joint_7, gripper_width]，shape=(8,)
                   关节角度单位为 rad，夹爪宽度单位为 m
    """
    
    def __init__(
        self,
        robot_sn: str,
        gripper_name: str = "",
        camera_ids: Optional[List[Union[int, str]]] = None,
        camera_names: Optional[List[str]] = None,
        camera_width: int = 1920,
        camera_height: int = 1080,
        camera_fps: int = 30,
        camera_warmup_frames: int = 10,
        max_episode_steps: int = 500,
        initial_joint_pos_deg: Optional[List[float]] = None,
        control_frequency: int = 20,
        tele_mode: bool = False,
        prompt: str = "",
        watchdog_timeout: float = 5.0,
        show_camera: bool = False,
        verbose: bool = True,
    ):
        """
        初始化 Flexiv 机器人环境。
        
        Args:
            robot_sn: 机器人序列号，如 "Rizon4s-063239"
            gripper_name: 夹爪名称，如 "Flexiv-GN01"，空字符串表示不使用夹爪
            camera_ids: 相机设备 ID 列表，如 [0, 2] 或 ["/dev/video0"]
            camera_names: 相机名称列表，如 ["cam_high", "cam_wrist"]
            camera_width: 相机采集宽度
            camera_height: 相机采集高度
            camera_fps: 相机帧率
            camera_warmup_frames: 相机预热帧数，启动时读取并丢弃的帧数
            max_episode_steps: 每个 episode 最大步数
            initial_joint_pos_deg: 初始关节角度（度），None 则使用当前位置
            control_frequency: 控制频率 (Hz)
            tele_mode: 是否为遥控/录制模式（禁用机器人运动）
            prompt: 任务提示文本
            watchdog_timeout: 看门狗超时时间 (秒)
            show_camera: 是否显示相机预览窗口
            verbose: 是否打印详细日志
        """
        self.robot_sn = robot_sn
        self.gripper_name = gripper_name
        self.verbose = verbose
        self._tele_mode = tele_mode
        self._prompt = prompt
        self._watchdog_timeout = watchdog_timeout
        self._show_camera = show_camera
        
        # Episode 追踪
        self._done = True
        self._episode_reward = 0.0
        self._step_count = 0
        self._max_episode_steps = max_episode_steps
        self._control_frequency = control_frequency
        
        # 初始关节位置
        if initial_joint_pos_deg is not None:
            self._initial_joint_pos_rad = np.deg2rad(initial_joint_pos_deg)
        else:
            self._initial_joint_pos_rad = None
        
        # 设备健康状态
        self._device_healthy = True
        self._robot_enabled = False
        self._gripper_enabled = False
        
        # 相机配置
        if camera_ids is None:
            camera_ids = [0]  # 默认只使用一个相机
        if camera_names is None:
            camera_names = ["cam_wrist"] if len(camera_ids) == 1 else ["cam_high", "cam_wrist"]
        
        assert len(camera_ids) == len(camera_names), "camera_ids 和 camera_names 长度必须相同"
        
        self._camera_names = camera_names
        self._cameras: List[GoproCamera] = []
        
        # 初始化相机
        for cam_id, cam_name in zip(camera_ids, camera_names):
            camera = GoproCamera(
                camera_id=cam_id,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                name=cam_name,
                warmup_frames=camera_warmup_frames
            )
            self._cameras.append(camera)
        
        # 初始化机器人
        self._robot: Optional[FlexivRobot] = None
        
        # 显示设置
        self._display_window_name = "Flexiv Camera Preview"
        self._display_window_initialized = False
        
        # 启动硬件
        self._start_hardware()
        
        self._log(f"FlexivEnvironment 初始化完成")
        self._log(f"  - 机器人: {robot_sn}")
        self._log(f"  - 夹爪: {gripper_name if gripper_name else '未配置'}")
        self._log(f"  - 相机: {list(zip(camera_names, camera_ids))}")
    
    def _log(self, msg: str, level: str = "info"):
        """打印日志"""
        if self.verbose:
            prefix = {"info": "[INFO]", "warning": "[WARNING]", "error": "[ERROR]"}.get(level, "[INFO]")
            print(f"{prefix} {msg}")
    
    def _start_hardware(self):
        """启动所有硬件设备"""
        # 1. 启动相机
        for camera in self._cameras:
            if not camera.start():
                raise RuntimeError(f"相机 {camera.name} 启动失败")
        
        time.sleep(0.5)  # 等待相机稳定
        
        # 2. 初始化机器人（非 tele_mode 下才连接真实机器人）
        if not self._tele_mode:
            self._log("正在连接 Flexiv 机器人...")
            try:
                # 配置运动参数
                motion_params = MotionParams()
                motion_params.joint_max_vel = 1.0  # rad/s
                motion_params.joint_max_acc = 1.5  # rad/s²
                motion_params.control_frequency = self._control_frequency
                
                self._robot = FlexivRobot(
                    robot_sn=self.robot_sn,
                    auto_init=True,
                    motion_params=motion_params,
                    verbose=self.verbose
                )
                self._robot_enabled = True
                self._log("✅ 机器人连接成功")
                
                # 3. 初始化夹爪
                if self.gripper_name:
                    self._log(f"正在初始化夹爪 [{self.gripper_name}]...")
                    try:
                        self._robot.gripper_enable(self.gripper_name, switch_tool=True)
                        self._gripper_enabled = True
                        self._log("✅ 夹爪初始化成功")
                    except Exception as e:
                        self._log(f"⚠️ 夹爪初始化失败: {e}", level="warning")
                        self._gripper_enabled = False
                
            except Exception as e:
                self._log(f"❌ 机器人初始化失败: {e}", level="error")
                raise RuntimeError(f"机器人初始化失败: {e}")
        else:
            self._log("录制模式：跳过机器人初始化")
    
    def _process_camera_frame(self, frame: Optional[np.ndarray], width: int, height: int) -> np.ndarray:
        """
        处理相机帧：调整大小、颜色转换、转置。
        
        Args:
            frame: 原始相机帧 (BGR)
            width: 原始宽度
            height: 原始高度
            
        Returns:
            处理后的图像，shape=(3, 224, 224)，RGB 格式
        """
        target_size = 224
        
        if frame is None:
            # 如果没有帧，返回黑色图像
            return np.zeros((3, target_size, target_size), dtype=np.uint8)
        
        cur_height, cur_width = frame.shape[:2]
        
        # 计算保持宽高比的缩放比例
        ratio = max(cur_width / target_size, cur_height / target_size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        
        # 使用 OpenCV 缩放（比 PIL 更快）
        if ratio > 1.0:
            interpolation = cv2.INTER_AREA  # 缩小用 INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR  # 放大用 INTER_LINEAR
        
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
        
        # BGR 转 RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 计算 padding
        pad_h = (target_size - resized_height) // 2
        pad_w = (target_size - resized_width) // 2
        pad_h_remainder = target_size - resized_height - pad_h
        pad_w_remainder = target_size - resized_width - pad_w
        
        # 黑色填充
        padded = np.pad(
            resized_rgb,
            ((pad_h, pad_h_remainder), (pad_w, pad_w_remainder), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # 转换轴顺序 [H, W, C] -> [C, H, W]
        return np.transpose(padded, (2, 0, 1))
    
    def _update_observation(self) -> dict:
        """
        收集所有设备数据并组装观测字典。
        
        Returns:
            观测字典
        """
        start_ts = time.time()
        
        # 读取所有相机图像
        images = {}
        camera_frames = []
        for camera in self._cameras:
            try:
                frame = camera.read(timeout=0.5)
                camera_frames.append(frame)
                if frame is not None:
                    processed_img = self._process_camera_frame(frame, camera.width, camera.height)
                    images[camera.name] = processed_img
                else:
                    self._log(f"⚠️ 相机 {camera.name} 读取超时", level="warning")
                    images[camera.name] = np.zeros((3, 224, 224), dtype=np.uint8)
            except Exception as e:
                self._log(f"❌ 相机 {camera.name} 读取错误: {e}", level="error")
                images[camera.name] = np.zeros((3, 224, 224), dtype=np.uint8)
        
        camera_ts = time.time()
        
        # 读取机器人状态
        if self._robot is not None and self._robot_enabled:
            try:
                joint_positions = self._robot.get_joint_positions()  # 7 个关节角度 [rad]
                
                # 获取夹爪宽度
                if self._gripper_enabled:
                    gripper_width = self._robot.get_gripper_width()  # [m]
                else:
                    gripper_width = 0.0
                
                # 组合状态: 7 关节 + 1 夹爪
                state = np.concatenate([joint_positions, [gripper_width]]).astype(np.float32)
                
            except Exception as e:
                self._log(f"❌ 机器人状态读取失败: {e}", level="error")
                state = np.zeros(8, dtype=np.float32)
        else:
            state = np.zeros(8, dtype=np.float32)
        
        robot_ts = time.time()
        
        # 组装观测字典
        obs = {
            "state": state,
            "images": images,
            "timestamps": {
                "start": start_ts,
                "camera": camera_ts,
                "robot": robot_ts,
            },
            "prompt": self._prompt,
        }
        
        # 可选：显示相机预览
        if self._show_camera and camera_frames:
            self._display_camera_preview(camera_frames[0])
        
        return obs
    
    def _display_camera_preview(self, frame: Optional[np.ndarray]):
        """显示相机预览窗口"""
        if frame is None:
            return
        
        try:
            if not self._display_window_initialized:
                cv2.namedWindow(self._display_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._display_window_name, 640, 480)
                self._display_window_initialized = True
            
            # 缩放显示
            display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow(self._display_window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._show_camera = False
                cv2.destroyWindow(self._display_window_name)
                self._display_window_initialized = False
                
        except Exception as e:
            self._log(f"❌ 相机预览错误: {e}", level="error")
            self._show_camera = False
    
    @override
    def reset(self) -> None:
        """重置环境：使能机器人，移动到初始位置，重置计数器"""
        if not self._tele_mode:
            # 移动到初始位置
            if self._initial_joint_pos_rad is not None and self._robot is not None:
                self._log("正在移动到初始关节位置...")
                try:
                    success = self._robot.move_joint(
                        self._initial_joint_pos_rad,
                        timeout=30.0
                    )
                    if success:
                        self._log("✅ 已到达初始位置")
                    else:
                        self._log("⚠️ 移动到初始位置超时", level="warning")
                except Exception as e:
                    self._log(f"❌ 移动到初始位置失败: {e}", level="error")
            
            # 打开夹爪
            if self._gripper_enabled and self._robot is not None:
                try:
                    self._robot.gripper_open(wait=True)
                except Exception as e:
                    self._log(f"⚠️ 夹爪打开失败: {e}", level="warning")
            
            time.sleep(1.0)  # 等待稳定
        
        self._done = False
        self._episode_reward = 0.0
        self._step_count = 0
        self._log("环境重置完成")
    
    @override
    def is_episode_complete(self) -> bool:
        """检查 episode 是否完成"""
        if self._step_count >= self._max_episode_steps:
            self._done = True
        return self._done
    
    @override
    def get_observation(self) -> dict:
        """获取当前观测"""
        if not self._device_healthy:
            raise RuntimeError("设备不健康，请检查硬件连接")
        
        start_time = time.time()
        while time.time() - start_time < self._watchdog_timeout:
            try:
                obs = self._update_observation()
                if obs is not None:
                    return obs
            except Exception as e:
                self._log(f"❌ 观测更新失败: {e}", level="error")
            time.sleep(0.001)
        
        raise RuntimeError(f"获取观测超时 ({self._watchdog_timeout}s)")
    
    @override
    def apply_action(self, action: dict) -> None:
        """
        执行动作。
        
        Args:
            action: 动作字典，包含:
                - actions: np.ndarray, shape=(8,)
                           [joint_1, ..., joint_7, gripper_width]
                           关节角度单位 rad，夹爪宽度单位 m
        """
        joint_action = action.get("actions", None)
        if joint_action is None:
            raise ValueError("动作字典必须包含 'actions' 键")
        
        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.flatten().tolist()
        
        if len(joint_action) != 8:
            raise ValueError(f"动作维度必须为 8，当前为 {len(joint_action)}")
        
        if not self._tele_mode and self._robot is not None:
            try:
                # 提取关节角度和夹爪宽度
                joint_positions = joint_action[:7]  # 7 个关节角度 [rad]
                gripper_width = joint_action[7]      # 夹爪宽度 [m]
                
                # 发送关节位置命令
                self._robot.send_joint_position(joint_positions)
                
                # 发送夹爪命令
                if self._gripper_enabled:
                    # 限制夹爪宽度范围
                    params = self._robot.get_gripper_params()
                    gripper_width = np.clip(gripper_width, params.min_width, params.max_width)
                    self._robot.gripper_move(gripper_width, wait=False)
                
            except Exception as e:
                self._log(f"❌ 执行动作失败: {e}", level="error")
                self._done = True
        
        self._step_count += 1
    
    def close(self) -> None:
        """关闭环境，清理资源"""
        self._log("正在关闭 FlexivEnvironment...")
        
        # 停止相机
        for camera in self._cameras:
            try:
                camera.stop()
            except Exception as e:
                self._log(f"❌ 关闭相机 {camera.name} 失败: {e}", level="error")
        
        # 停止机器人（仅在控制真实机器人模式下）
        if self._robot is not None and not self._tele_mode:
            try:
                # 禁用夹爪
                if self._gripper_enabled:
                    self._robot.gripper_disable()
                    self._gripper_enabled = False
                
                # 停止机器人
                self._robot.stop()
                self._robot_enabled = False
                self._log("✅ 机器人已停止")
            except Exception as e:
                self._log(f"❌ 停止机器人失败: {e}", level="error")
        
        # 关闭显示窗口
        if self._display_window_initialized:
            try:
                cv2.destroyWindow(self._display_window_name)
            except Exception:
                pass
        
        self._log("✅ FlexivEnvironment 已关闭")
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except Exception:
            pass
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()
        return False
    
    # ==================== 额外便捷方法 ====================
    
    def get_robot_state(self) -> dict:
        """
        获取机器人完整状态（用于调试）
        
        Returns:
            包含关节角度、TCP 位姿、夹爪宽度等的字典
        """
        if self._robot is None:
            return {}
        
        try:
            state = {
                "joint_positions": self._robot.get_joint_positions().tolist(),
                "tcp_pose": self._robot.get_tcp_pose().tolist(),
                "gripper_width": self._robot.get_gripper_width() if self._gripper_enabled else 0.0,
            }
            return state
        except Exception as e:
            self._log(f"❌ 获取机器人状态失败: {e}", level="error")
            return {}
    
    def move_to_joint_position(self, joint_positions_deg: List[float], timeout: float = 30.0) -> bool:
        """
        阻塞式移动到指定关节位置（用于调试或初始化）
        
        Args:
            joint_positions_deg: 目标关节角度（度）
            timeout: 超时时间（秒）
            
        Returns:
            是否成功到达目标位置
        """
        if self._robot is None or self._tele_mode:
            return False
        
        try:
            joint_positions_rad = np.deg2rad(joint_positions_deg)
            return self._robot.move_joint(joint_positions_rad, timeout=timeout)
        except Exception as e:
            self._log(f"❌ 关节运动失败: {e}", level="error")
            return False
    
    def set_prompt(self, prompt: str):
        """设置任务提示文本"""
        self._prompt = prompt


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flexiv Environment 测试")
    parser.add_argument("--robot_sn", type=str, default="Rizon4s-063239", help="机器人序列号")
    parser.add_argument("--gripper", type=str, default="Flexiv-GN01", help="夹爪名称")
    parser.add_argument("--camera_ids", type=int, nargs="+", default=[0], help="相机 ID 列表")
    parser.add_argument("--tele_mode", action="store_true", help="tele_mode / 录制模式（不控制机器人）")
    parser.add_argument("--show_camera", action="store_true", help="显示相机预览")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Flexiv Environment 测试")
    print("=" * 50)
    
    try:
        env = FlexivEnvironment(
            robot_sn=args.robot_sn,
            gripper_name=args.gripper,
            camera_ids=args.camera_ids,
            tele_mode=args.tele_mode,
            show_camera=args.show_camera,
            prompt="测试任务",
        )
        
        print("\n正在重置环境...")
        env.reset()
        
        print("\n获取观测...")
        obs = env.get_observation()
        print(f"  - state shape: {obs['state'].shape}")
        print(f"  - state: {obs['state']}")
        print(f"  - images keys: {list(obs['images'].keys())}")
        for key, img in obs['images'].items():
            print(f"    - {key} shape: {img.shape}")
            # 保存图片到本地，需要将 (C, H, W) 转换为 (H, W, C)
        
        print("\n获取机器人状态...")
        robot_state = env.get_robot_state()
        print(f"  - joint_positions (deg): {[round(np.rad2deg(j), 2) for j in robot_state.get('joint_positions', [])]}")
        print(f"  - tcp_pose: {robot_state.get('tcp_pose', [])}")
        print(f"  - gripper_width (mm): {robot_state.get('gripper_width', 0) * 1000:.2f}")
        
        if not args.tele_mode:
            print("\n测试执行动作（保持当前位置）...")
            current_state = obs['state']
            # current_state[6] = current_state[6] - 0.2
            action = {"actions": current_state}  # 保持当前位置
            env.apply_action(action)
            time.sleep(1.0)
            print("  - 动作已执行")
        
        print("\n关闭环境...")
        env.close()
        
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

