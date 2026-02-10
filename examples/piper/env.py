import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import time
import threading
import cv2
import h5py
from datetime import datetime
import os

from piper_controller import PiperController
from cameras import RealSenseCamera, USBCamera


class PiperEnvironment(_environment.Environment):
    """An environment for a Piper robot arm with RealSense camera (real hardware)."""

    def __init__(
        self,
        can_port: str = "can0",
        # camera_width: int = 224,
        # camera_height: int = 224,
        camera_fps: int = 30,
        high_camera_id: int = 6,
        left_wrist_camera_id: int = 4,
        max_episode_steps: int = 500,
        initial_joint_pos: list = None,
        seed: int = 0,
        tele_mode: bool = False,
        prompt: str = "",
        watchdog_timeout: float = 5.0,
        show_usb_camera: bool = False,
        gripper_norm: bool = True,
        action_history_size: int = 30,
        action_change_threshold: float = 0.01,
        record_mode: bool = False,
        record_directory: str = "recorded_data",
    ) -> None:
        """Initialize the Piper robot environment.
        
        Args:
            can_port: CAN bus port name for the Piper robot (default "can0").
            camera_width: RealSense camera width in pixels.
            camera_height: RealSense camera height in pixels.
            camera_fps: RealSense camera frames per second.
            usb_camera_id: USB camera device ID for cam_high (default 0).
            max_episode_steps: Maximum steps per episode before auto-termination.
            initial_joint_pos: Initial joint configuration [j1, j2, j3, j4, j5, j6, gripper].
                              If None, uses [0, 0, 0, 0, 0, 0, 0.001] as default.
            seed: Random seed (for compatibility; not used in real hardware).
            tele_mode: If True, disables robot motion commands for safe testing.
            prompt: Text prompt to be included in observations.
            show_usb_camera: If True, opens an OpenCV window to preview the USB camera in real time.
            record_mode: If True, records trajectory data during episodes.
            record_directory: Directory to save recorded episodes (default "recorded_data").
        """
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        # Threading for parallel hardware polling
        self._obs_lock = threading.Lock()
        self._stop_event = threading.Event()
        #self._last_obs = None  # This will be updated by the background thread
        self._update_thread = None
        self._update_rate = camera_fps  # Hz
        
        # 看门狗机制：检测后台线程是否卡死
        self._watchdog_timeout = watchdog_timeout
        #self._last_update_time = time.time()
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10  # 连续失败10次后认为设备故障
        self._device_healthy = True
        self._robot_enabled = False

        # Initialize hardware
        self._robot = PiperController(can_port=can_port, gripper_norm=gripper_norm)
        # cannot customize the resolution of the cameras, use the minimum resolution of them.
        self._wrist_camera = USBCamera(camera_id=left_wrist_camera_id, width=320, height=240, fps=camera_fps) #RealSenseCamera(width=camera_width, height=camera_height, fps=camera_fps)
        self._high_camera = USBCamera(camera_id=high_camera_id, width=1280, height=960, fps=camera_fps)
        
        # Episode tracking
        self._done = True
        self._episode_reward = 0.0
        self._step_count = 0
        self._max_episode_steps = max_episode_steps
        
        # Default initial position
        if initial_joint_pos is None:
            self._initial_joint_pos = [0.08250172,0.28386036,-0.81074035,0.00631809,0.7380648,0.03782129,0.0] # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001] # 
        else:
            self._initial_joint_pos = initial_joint_pos

        # Start hardware and background threads
        self._start_hardware()
        print(f"PiperEnvironment initialized with CAN port {can_port}")
        print(f"  - Wrist camera (USB ID {left_wrist_camera_id}): {320}x{240}@{camera_fps}fps")
        print(f"  - High camera (USB ID {high_camera_id}): {1280}x{960}@{camera_fps}fps")

        # Optional: record mode settings can be implemented here
        self._tele_mode = tele_mode
        
        # Store prompt for inclusion in observations
        self._prompt = prompt

        # USB 相机预览设置
        self._show_usb_camera = show_usb_camera
        self._usb_window_name = "USB Camera Preview"
        self._usb_window_initialized = False
        # 缓存显示尺寸（高分辨率相机的一半）
        self._display_width = self._high_camera.width // 2
        self._display_height = self._high_camera.height // 2
        # 显示帧计数器（用于跳帧显示）
        self._display_frame_counter = 0
        
        # Action 历史队列：记录最近接收的动作
        self._action_history_size = action_history_size
        self._action_history = []
        self._action_change_threshold = action_change_threshold
        
        # 轨迹记录相关
        self._record_mode = record_mode
        self._record_directory = record_directory
        self._episode_data = []  # 存储当前episode的数据
        self._last_obs = None  # 存储最新的observation

    def _start_hardware(self):
        """Starts camera hardware and the background observation polling thread."""
        # Start cameras
        self._wrist_camera.start()
        self._high_camera.start()
        try:
            self._robot.enable()
            self._robot_enabled = True
        except Exception as e:
            print(f"Warning during enable: {e}")
            raise RuntimeError(f"Failed to enable robot: {e}")
        # Wait a moment for cameras to initialize
        time.sleep(1.0)

        # Create and start the single polling thread
        # self._update_thread = threading.Thread(target=self._update_observation_loop, daemon=True)
        # self._update_thread.start()

    # def _update_observation_loop(self):
    #     """Continuously polls all hardware and updates the latest observation."""
    #     while not self._stop_event.is_set():
    #         start_time = time.time()
            
    #         try:
    #             # Atomically update the observation
    #             self._update_observation()
    #             # 更新看门狗时间戳
    #             self._last_update_time = time.time()
    #             # 重置连续失败计数
    #             if self._consecutive_failures > 0:
    #                 print(f"设备恢复正常，重置失败计数")
    #             self._consecutive_failures = 0
    #         except Exception as e:
    #             print(f"❌ 观测更新循环错误: {e}")
    #             import traceback
    #             traceback.print_exc()
    #             self._consecutive_failures += 1
                
    #             if self._consecutive_failures >= self._max_consecutive_failures:
    #                 print(f"⚠️ 连续失败 {self._consecutive_failures} 次，标记设备为不健康状态")
    #                 self._device_healthy = False
    #                 break

    #         # Sleep to maintain the desired update rate
    #         elapsed_time = time.time() - start_time
    #         sleep_time = (1.0 / self._update_rate) - elapsed_time
    #         if sleep_time > 0:
    #             time.sleep(sleep_time)
        
    #     print("⚠️ 观测更新线程已退出")

    def _update_observation(self):
        """Gathers data from all devices and updates self._last_obs."""
        # Poll all devices and record timestamps
        start_ts = time.time()
        
        # 读取全局相机 (带超时)
        try:
            high_frame = self._high_camera.read(timeout=0.5)
            high_ts = time.time()
            if high_frame is None:
                print(f"⚠️ 警告: 全局相机读取超时或失败")
        except Exception as e:
            print(f"❌ 警告: 全局相机读取异常: {e}")
            high_frame = None
            high_ts = time.time()
        
        # 读取腕部相机 (带超时)
        try:
            wrist_frame = self._wrist_camera.read(timeout=0.5)
            wrist_ts = time.time()
            if wrist_frame is None:
                print(f"⚠️ 警告: 腕部相机读取超时或失败")
        except Exception as e:
            print(f"❌ 警告: 腕部相机读取异常: {e}")
            wrist_frame = None
            wrist_ts = time.time()
        
        # 读取机械臂状态
        try:
            state = self._robot.get_full_joint_status()
            robot_ts = time.time()
        except Exception as e:
            print(f"❌ 警告: 机械臂状态读取失败: {e}")
            state = [0.0] * 7
            robot_ts = time.time()
        
        # 检查是否所有设备都失败
        if high_frame is None and wrist_frame is None:
            raise RuntimeError("所有相机都无法读取数据，可能设备已断开连接")
        
        #process_start_ts = time.time()
        # Process images
        wrist_img = self._process_camera_frame(
            wrist_frame, self._wrist_camera.width, self._wrist_camera.height
        )
        high_img = self._process_camera_frame(
            high_frame, self._high_camera.width, self._high_camera.height
        )

        # Assemble the observation dictionary
        obs = {
            "state": np.array(state, dtype=np.float32),
            "images": {
                "cam_left_wrist": wrist_img,
                "cam_high": high_img,
            },
            "timestamps": {
                "start": start_ts,
                "robot": robot_ts,
                "cam_left_wrist": wrist_ts,
                "cam_high": high_ts,
            },
            "prompt": self._prompt,
        }
        #process_end_ts = time.time()

        # Update the shared observation under a lock
        # with self._obs_lock:
        #     self._last_obs = obs
        #print(f"Delta time for process: {process_end_ts - process_start_ts} ")

        #display_start_ts = time.time()
        # 可选：实时显示 USB 相机画面
        if self._show_usb_camera:
            self._display_usb_camera_frame(high_frame)
        #display_end_ts = time.time()
        #print(f"Delta time for display: {display_end_ts - display_start_ts} ")
        
        return obs

    def _display_usb_camera_frame(self, frame: np.ndarray) -> None:
        """在单独窗口中实时显示 USB 相机画面。
        
        优化版本：
        1. 使用缓存的显示尺寸，避免重复计算
        2. 使用 INTER_AREA 插值（对缩小图像更快且质量更好）
        3. 可选：降低显示频率以减少开销（当前每帧都显示）
        """
        try:
            if frame is None:
                return

            # 可选：降低显示频率（每2帧显示一次，减少50%的显示开销）
            # 取消注释下面两行来启用跳帧显示
            # self._display_frame_counter += 1
            # if self._display_frame_counter % 2 != 0:
            #     return

            if not self._usb_window_initialized:
                cv2.namedWindow(self._usb_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._usb_window_name, self._display_width, self._display_height)
                self._usb_window_initialized = True

            # 检查是否需要resize（如果已经是目标大小，跳过）
            original_height, original_width = frame.shape[:2]
            if original_width == self._display_width and original_height == self._display_height:
                # 已经是目标大小，直接显示
                cv2.imshow(self._usb_window_name, frame)
            else:
                # 使用 INTER_AREA 插值（对缩小图像更快且质量更好）
                # 如果追求极致速度，可以使用 INTER_NEAREST（最快但质量较差）
                resized_frame = cv2.resize(
                    frame, 
                    (self._display_width, self._display_height), 
                    interpolation=cv2.INTER_AREA
                )
                cv2.imshow(self._usb_window_name, resized_frame)
            
            # waitKey 返回 -1 表示无按键，避免阻塞
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("⚠️ 检测到按键 'q'，已关闭 USB 相机实时预览。")
                self._show_usb_camera = False
                cv2.destroyWindow(self._usb_window_name)
                self._usb_window_initialized = False
        except Exception as e:
            print(f"❌ USB 相机预览异常: {e}")
            self._show_usb_camera = False
            try:
                cv2.destroyWindow(self._usb_window_name)
            except Exception:
                pass
            self._usb_window_initialized = False

    def _process_camera_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Processes a raw camera frame (resize, color convert, transpose).
        
        Optimized version using OpenCV instead of PIL for better performance.
        Performance improvements:
        1. Uses cv2.resize instead of PIL (much faster)
        2. Reduces memory copies by combining operations
        3. Uses optimized interpolation methods
        """
        if frame is None:
            # If no frame available, create a black image
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        target_size = 224
        cur_height, cur_width = frame.shape[:2]
        
        # Calculate resize ratio to maintain aspect ratio
        ratio = max(cur_width / target_size, cur_height / target_size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        
        # Resize using OpenCV (much faster than PIL)
        # Use INTER_AREA for downscaling (better quality) or INTER_LINEAR for upscaling
        if ratio > 1.0:
            interpolation = cv2.INTER_AREA  # Better for downscaling
        else:
            interpolation = cv2.INTER_LINEAR  # Better for upscaling
        
        # Resize using OpenCV (BGR format)
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
        
        # Convert BGR to RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Calculate padding
        pad_h = (target_size - resized_height) // 2
        pad_w = (target_size - resized_width) // 2
        pad_h_remainder = target_size - resized_height - pad_h
        pad_w_remainder = target_size - resized_width - pad_w
        
        # Pad with zeros (black padding)
        padded = np.pad(
            resized_rgb,
            ((pad_h, pad_h_remainder), (pad_w, pad_w_remainder), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # Convert axis order from [H, W, C] --> [C, H, W]
        return np.transpose(padded, (2, 0, 1))

    @override
    def reset(self) -> None:
        """Reset the environment: enable robot, move to initial position, reset counters."""
        # If in record mode, skip enabling and moving the robot
        if not self._tele_mode:
            # Enable robot if not already enabled
            try:
                if not self._robot_enabled:
                    self._robot.enable()
                    self._robot_enabled = True
                else:
                    print("Robot already enabled")
            except Exception as e:
                print(f"Warning during enable: {e}")
            
            # Move to initial position
            print("Resetting to initial joint position...")
            self._robot.control_full_joint(self._initial_joint_pos, velocity=100)
            time.sleep(2.0)  # wait for motion to complete and observation to be ready
        
        # Ensure the first observation is ready
        # while self._last_obs is None:
        #     time.sleep(0.1)

        self._done = False
        self._episode_reward = 0.0
        self._step_count = 0
        # 清空 action 历史队列
        self._action_history = []
        # 如果处于记录模式，初始化episode数据列表
        if self._record_mode:
            self._episode_data = []
            self._last_obs = None
        print("Environment reset complete.")

    @override
    def is_episode_complete(self) -> bool:
        """Check if the episode is done (manually set or max steps reached)."""
        if self._step_count >= self._max_episode_steps:
            self._done = True
        # 检查 action 历史队列中的动作变化幅度
        elif len(self._action_history) >= self._action_history_size:
            # 将历史队列转换为 numpy 数组
            action_array = np.array(self._action_history)
            # 计算每个维度上的标准差（动作变化幅度）
            std_per_dim = np.std(action_array, axis=0)            
            # 检查每个维度的变化是否都小于阈值，只有当所有维度都满足条件时才完成
            if np.all(std_per_dim < self._action_change_threshold):
                self._done = True
        
        return self._done

    @override
    def get_observation(self) -> dict:
        """Return the last observation from the background thread."""
        # 检查设备健康状态
        if not self._device_healthy:
            raise RuntimeError("⚠️ 设备处于不健康状态，可能已断开连接。请重启程序。")
        
        # 检查看门狗超时
        # time_since_update = time.time() - self._last_update_time
        # if time_since_update > self._watchdog_timeout:
        #     self._device_healthy = False
        #     raise RuntimeError(
        #         f"⚠️ 看门狗超时：后台线程已 {time_since_update:.2f} 秒无响应（超时阈值: {self._watchdog_timeout}s）。\n"
        #         f"这通常意味着设备（相机或机械臂）已断开连接或卡死。请检查硬件连接并重启程序。"
        #     )
        start_time = time.time()
        while time.time() - start_time < self._watchdog_timeout:
            obs = self._update_observation()
            if obs is not None:
                # 如果处于记录模式，保存最新的observation
                if self._record_mode:
                    self._last_obs = obs.copy()
                #print(f"Delta time from observation to return: {time.time() - obs['timestamps']['robot']} ")
                return obs
            time.sleep(0.0001)
        
        raise RuntimeError("⚠️ 看门狗超时：后台线程已无响应（超时阈值: {self._watchdog_timeout}s）。\n"
                           "这通常意味着设备（相机或机械臂）已断开连接或卡死。请检查硬件连接并重启程序。")

        # with self._obs_lock:
        #     if self._last_obs is None:
        #         raise RuntimeError("观测数据尚未准备好。请调用 reset() 并等待。")
        #     print(f"Obtained obs times used: {self._last_obs['timestamps']['robot']-self._last_obs['timestamps']['start']}")
        #     print(f"Delta with current time: {time.time() - self._last_obs['timestamps']['robot']} ")
        #     return self._last_obs.copy()


    @override
    def apply_action(self, action: dict) -> None:
        """Apply an action to the robot and update observation.
        
        Expected action format:
        {
            "actions": np.ndarray of shape (7,) containing [j1, j2, j3, j4, j5, j6, gripper]
                       where joints are in radians and gripper is opening width in meters.
        }
        """
        # Extract action (7-dim: 6 joints + gripper)
        joint_action = action.get("actions", None)
        if joint_action is None:
            raise ValueError("Action dict must contain 'actions' key with joint positions.")
        
        # Convert to list if numpy array
        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.tolist()
        
        # 如果处于记录模式，将最新的obs和当前的action组合成一个step存入数据库
        if self._record_mode and self._last_obs is not None:
            step_data = {
                'observation': self._last_obs.copy(),
                'action': np.array(joint_action, dtype=np.float32)
            }
            self._episode_data.append(step_data)
        
        # Send command to robot
        try:
            if not self._tele_mode:
                self._robot.control_full_joint(joint_action, velocity=100)
        except Exception as e:
            print(f"Error applying action: {e}")
            self._done = True
        
        # 将动作添加到历史队列
        action_np = np.array(joint_action, dtype=np.float32)
        self._action_history.append(action_np)
        # 保持队列大小不超过设定值
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)
        
        # Small delay to allow motion to start
        #time.sleep(1.0 / self._update_rate)
        
        # Observation is updated in the background, no need to call a getter here
        self._step_count += 1
        
        # Optional: compute reward (placeholder for now; extend as needed)
        # For real tasks, you might check task completion criteria here
        reward = 0.0
        self._episode_reward += reward

        #print(f"Step {self._step_count}: Applied action {joint_action}")

    def close(self) -> None:
        """Clean up: stop threads, disable robot, and stop cameras."""
        print("🔄 正在关闭 PiperEnvironment...")
        
        # Signal thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._update_thread:
            print("⏳ 等待后台线程结束...")
            self._update_thread.join(timeout=3)
            if self._update_thread.is_alive():
                print("⚠️ 警告: 后台线程未在超时时间内结束（可能卡死）")

        if not self._tele_mode:
            try:
                print("🔄 正在失能机械臂...")
                if self._robot_enabled:
                    self._robot.disable()
                    self._robot_enabled = False
                    print("✅ 机械臂已失能")
                else:
                    print("Robot already disabled")
            except Exception as e:
                print(f"❌ 机械臂失能时出错: {e}")

        try:
            print("🔄 正在关闭腕部相机...")
            self._wrist_camera.stop()
            print("✅ 腕部相机已关闭")
        except Exception as e:
            print(f"❌ 腕部相机关闭时出错: {e}")
    
        try:
            print("🔄 正在关闭全局相机...")
            self._high_camera.stop()
            print("✅ 全局相机已关闭")
        except Exception as e:
            print(f"❌ 全局相机关闭时出错: {e}")

        if self._usb_window_initialized:
            cv2.destroyWindow(self._usb_window_name)
        
        print("✅ PiperEnvironment 已完全关闭")

    def _save_episode_to_hdf5(self):
        """Save the current episode data to an HDF5 file."""
        if not self._episode_data:
            print("No episode data to save.")
            return
        
        # Ensure the save directory exists
        os.makedirs(self._record_directory, exist_ok=True)
        
        # Generate a unique filename with prompt
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 将prompt转换为文件名安全的格式
        prompt = self._prompt if self._prompt else ""
        if prompt:
            # 将空格替换为下划线，移除或替换特殊字符
            safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
            safe_prompt = safe_prompt.replace(",", "").replace(".", "")
            # 限制长度（避免文件名过长）
            max_prompt_len = 80
            if len(safe_prompt) > max_prompt_len:
                safe_prompt = safe_prompt[:max_prompt_len]
            filename = os.path.join(self._record_directory, f"episode_{timestamp_str}_{safe_prompt}.hdf5")
        else:
            filename = os.path.join(self._record_directory, f"episode_{timestamp_str}.hdf5")
        
        print(f"Saving episode to {filename}...")
        
        # 将episode_data转换为collect_data.py中的格式（只有observation）
        episode_obs_data = []
        for step_data in self._episode_data:
            episode_obs_data.append(step_data['observation'])
        
        # 如果没有observation数据，直接返回
        if not episode_obs_data:
            print("No observation data to save.")
            return
        
        with h5py.File(filename, "w") as f:
            # Pre-allocate numpy arrays by inspecting the first observation
            first_obs = episode_obs_data[0]
            first_action = self._episode_data[0]['action']
            num_steps = len(episode_obs_data)
            
            # Create datasets based on the expected structure
            obs_group = f.create_group("observations")
            img_group = obs_group.create_group("images")
            
            state_shape = first_obs['state'].shape
            obs_group.create_dataset("qpos", (num_steps,) + state_shape, dtype=first_obs['state'].dtype)
            
            # Action from the actual action stored in _episode_data
            action_shape = first_action.shape
            f.create_dataset("action", (num_steps,) + action_shape, dtype=first_action.dtype)
            
            for cam_name, img in first_obs['images'].items():
                img_shape = img.shape
                img_group.create_dataset(cam_name, (num_steps,) + img_shape, dtype=img.dtype)
                
            # Save the instruction prompt
            if 'prompt' in first_obs:
                prompt_len = len(first_obs['prompt'])
                fix_prompt_type = np.dtype(f'S{prompt_len}')
                f.create_dataset("task", (num_steps,), dtype=fix_prompt_type)
            
            # Save timestamps if available
            timestamps_group = None
            if 'timestamps' in first_obs and first_obs['timestamps']:
                timestamps_group = obs_group.create_group("timestamps")
                # Create datasets for each timestamp field
                for ts_key in first_obs['timestamps'].keys():
                    timestamps_group.create_dataset(ts_key, (num_steps,), dtype=np.float64)
            
            # Populate the datasets
            for i in range(num_steps):
                # Populate observations
                obs_group['qpos'][i] = episode_obs_data[i]['state']
                
                # Action from the actual action stored in _episode_data
                f['action'][i] = self._episode_data[i]['action']
                
                for cam_name, img in episode_obs_data[i]['images'].items():
                    img_group[cam_name][i] = img
                if 'prompt' in first_obs:
                    f['task'][i] = episode_obs_data[i]['prompt']
                
                # Save timestamps
                if timestamps_group is not None and 'timestamps' in episode_obs_data[i]:
                    for ts_key, ts_value in episode_obs_data[i]['timestamps'].items():
                        if ts_key in timestamps_group:
                            timestamps_group[ts_key][i] = ts_value
                    
        print(f"Successfully saved {num_steps} steps.")
        # 清空episode数据
        self._episode_data = []

    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.close()
        except Exception:
            pass
