"""Simple RealSense camera wrapper.

Provides a RealSenseCamera class that encapsulates pipeline setup, frame
acquisition, preview and cleanup. Designed to be drop-in replacement for the
inline script you provided.

Usage example:

from openpi.realsense_camera import RealSenseCamera

with RealSenseCamera(width=640, height=480, fps=30) as cam:
    cam.show_preview()

Or programmatically:

cam = RealSenseCamera()
cam.start()
frame = cam.read()
cam.stop()

"""
from typing import Optional, Callable
import time
from queue import Queue
from dataclasses import dataclass
import threading

try:
    import pyrealsense2 as rs
except Exception as e:  # pragma: no cover - hard to unit test without hardware
    raise ImportError("pyrealsense2 is required for RealSenseCamera: " + str(e))

import numpy as np
import cv2


class RealSenseCamera:
    """A thin wrapper around pyrealsense2 pipeline for RGB preview and frame access.

    Methods:
    - start(): start the pipeline
    - read(): return a BGR numpy array or None if no frame
    - stop(): stop the pipeline
    - show_preview(window_name='RealSense Camera - RGB', quit_key='q', callback=None):
        show an interactive preview until the quit_key is pressed. If callback is
        provided it is called as callback(color_image) for every frame.

    Context manager support ensures pipeline is stopped on exit.

    Constructor arguments:
    - width, height, fps: stream config
    - format: pyrealsense2 format (default rs.format.bgr8)
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 fmt=rs.format.bgr8):
        self.width = width
        self.height = height
        self.fps = fps
        self.fmt = fmt

        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self._started = False

    def start(self, timeout: float = 2.0):
        """Start the RealSense pipeline. Raises on failure."""
        if self._started:
            return

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, self.fmt, self.fps)

        # start may raise if device not present
        profile = self.pipeline.start(self.config)
        # Optional: wait a short moment for the camera to warm up
        start_time = time.time()
        while time.time() - start_time < timeout:
            frames = self.pipeline.poll_for_frames()
            if frames:
                break
            time.sleep(0.01)

        self._started = True

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read a single color frame as a BGR numpy array. Returns None on no frame or timeout."""
        if not self._started or self.pipeline is None:
            raise RuntimeError("RealSense pipeline is not started. Call start() first.")

        # 使用 poll_for_frames 替代 wait_for_frames 来避免无限阻塞
        start_time = time.time()
        while time.time() - start_time < timeout:
            frames = self.pipeline.poll_for_frames()
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    return color_image
            time.sleep(0.001)  # 短暂休眠避免CPU占用过高
        
        # 超时返回 None
        return None

    def stop(self):
        """Stop the pipeline and release resources (best-effort)."""
        if self.pipeline is None:
            self._started = False
            return
        try:
            self.pipeline.stop()
        except Exception:
            # best-effort cleanup; ignore errors during shutdown
            pass
        finally:
            self.pipeline = None
            self.config = None
            self._started = False

    def show_preview(self, window_name: str = "RealSense Camera - RGB", quit_key: str = 'q',
                     callback: Optional[Callable[[np.ndarray], None]] = None):
        """Show a simple preview window and return when quit_key is pressed.

        If callback is provided it will be invoked for each frame: callback(image).
        """
        try:
            if not self._started:
                self.start()

            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            while True:
                frame = self.read()
                if frame is None:
                    continue
                if callback is not None:
                    try:
                        callback(frame)
                    except Exception:
                        # don't let a callback crash the preview
                        pass
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break
        except KeyboardInterrupt:
            # allow Ctrl+C to exit preview
            pass
        finally:
            cv2.destroyWindow(window_name)
            # stop pipeline on exit
            try:
                self.stop()
            except Exception:
                pass

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        # don't suppress exceptions
        return False

"""Simple USB camera wrapper using OpenCV.

Provides a USBCamera class that encapsulates USB webcam access, frame
acquisition, preview and cleanup. Compatible interface with RealSenseCamera.

Usage example:

from usb_camera import USBCamera

with USBCamera(camera_id=0, width=640, height=480) as cam:
    cam.show_preview()

Or programmatically:

cam = USBCamera(camera_id=0)
cam.start()
frame = cam.read()
cam.stop()

"""
from typing import Optional, Callable
import time

import numpy as np
import cv2


class USBCamera:
    """A thin wrapper around cv2.VideoCapture for USB webcam access.

    Methods:
    - start(): open the camera device
    - read(): return a BGR numpy array or None if no frame
    - stop(): release the camera device
    - show_preview(window_name='USB Camera', quit_key='q', callback=None):
        show an interactive preview until the quit_key is pressed. If callback is
        provided it is called as callback(color_image) for every frame.

    Context manager support ensures camera is released on exit.

    Constructor arguments:
    - camera_id: Camera device index (0 for default camera, 1 for second, etc.)
    - width, height: Desired frame resolution (camera may not support all resolutions)
    - fps: Desired frames per second (camera may not support exact value)
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self.capture: Optional[cv2.VideoCapture] = None
        self._started = False

    def start(self, timeout: float = 2.0):
        """Open the USB camera device. Raises on failure."""
        if self._started:
            return

        self.capture = cv2.VideoCapture(self.camera_id)
        
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open USB camera with ID {self.camera_id}")

        # Set camera properties (note: not all cameras support all properties)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        # Optional: wait and read a few frames for camera to warm up
        start_time = time.time()
        while time.time() - start_time < timeout:
            ret, frame = self.capture.read()
            if ret and frame is not None:
                break
            time.sleep(0.001)

        # Verify actual resolution
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        
        if actual_width != self.width or actual_height != self.height:
            print(f"Warning: Requested {self.width}x{self.height}, but camera provides {actual_width}x{actual_height}")
        if actual_fps != self.fps:
            print(f"Warning: Requested {self.fps} FPS, but camera provides {actual_fps} FPS")

        self._started = True

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read a single frame as a BGR numpy array. Returns None on failure or timeout."""
        if not self._started or self.capture is None:
            raise RuntimeError("USB camera is not started. Call start() first.")

        # 设置读取超时 (以毫秒为单位)
        # 注意: 不是所有的OpenCV后端都支持超时设置
        try:
            # 尝试读取，如果失败则重试几次（处理USB带宽竞争）
            max_retries = 3
            for retry in range(max_retries):
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    return frame
                # 如果读取失败，短暂等待后重试（可能是USB带宽竞争）
                if retry < max_retries - 1:
                    time.sleep(0.01)
            return None
        except Exception as e:
            # 只在verbose模式下打印错误，避免刷屏
            # print(f"USB camera (ID={self.camera_id}) read error: {e}")
            return None

    def stop(self):
        """Release the camera device (best-effort)."""
        if self.capture is None:
            self._started = False
            return
        try:
            self.capture.release()
        except Exception:
            # best-effort cleanup; ignore errors during shutdown
            pass
        finally:
            self.capture = None
            self._started = False

    def show_preview(self, window_name: str = "USB Camera", quit_key: str = 'q',
                     callback: Optional[Callable[[np.ndarray], None]] = None):
        """Show a simple preview window and return when quit_key is pressed.

        If callback is provided it will be invoked for each frame: callback(image).
        """
        try:
            if not self._started:
                self.start()

            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            while True:
                frame = self.read()
                if frame is None:
                    continue
                if callback is not None:
                    try:
                        callback(frame)
                    except Exception:
                        # don't let a callback crash the preview
                        pass
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break
        except KeyboardInterrupt:
            # allow Ctrl+C to exit preview
            pass
        finally:
            cv2.destroyWindow(window_name)
            # stop camera on exit
            try:
                self.stop()
            except Exception:
                pass

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        # don't suppress exceptions
        return False


@dataclass
class FrameData:
    """帧数据"""
    frame: np.ndarray
    timestamp: float
    frame_idx: int


class GoproCamera:
    """
    GoPro 相机封装类，通过 Elgato Cam Link 采集卡读取视频流。
    使用 OpenCV 的 VideoCapture 访问 v4l 设备。
    
    与 USBCamera 类似，但增加了预热帧数配置，适合需要更长稳定时间的采集卡设备。
    支持多线程读取，使用队列存储最新帧。
    
    Usage example:

    with GoproCamera(camera_id=0, width=1920, height=1080) as cam:
        cam.show_preview()

    Or programmatically:

    cam = GoproCamera(camera_id=0)
    cam.start()
    frame = cam.read()  # 同步读取
    latest_frame = cam.get_latest_frame()  # 获取队列中的最新帧（多线程模式）
    cam.stop()
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        name: str = "gopro",
        warmup_time: float = 1.0,
        use_threading: bool = True,
        queue_size: int = 30,
        center_crop: bool = False
    ):
        """
        初始化 GoPro 相机。
        
        Args:
            camera_id: 相机设备 ID 或路径（如 0, "/dev/video0"）
            width: 图像宽度（如果 width == height，将启用 1:1 裁剪模式）
            height: 图像高度
            fps: 帧率
            name: 相机名称（用于日志）
            warmup_time: 预热时间（秒），启动时读取并丢弃帧的持续时间
            use_threading: 是否使用多线程读取（默认 True）
            queue_size: 帧队列大小（仅在使用多线程时有效）
            center_crop: 是否启用中心裁剪。如果为 True，在 read() 时会从原始帧的中心
                        裁剪到指定的 width x height 尺寸
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.warmup_time = warmup_time
        self.use_threading = use_threading
        self.queue_size = queue_size
        
        # 检测是否为 1:1 分辨率（正方形）
        self._is_square_mode = (width == height)
        
        # 如果是 1:1 模式，计算实际相机分辨率（4:3 比例）
        # GoPro 实际输出 640x480，但有效内容在中间的 480x480
        # 所以如果用户要 480x480，我们实际创建 640x480 的相机
        if self._is_square_mode:
            # 计算 4:3 比例的实际宽度（向上取整到最近的 4 的倍数）
            self._actual_camera_width = int((width * 4 / 3 + 3) // 4 * 4)
            self._actual_camera_height = height
            self._crop_left = (self._actual_camera_width - width) // 2
            self._crop_right = self._actual_camera_width - width - self._crop_left
            print(f"[{self.name}] 检测到 1:1 分辨率模式: 用户指定 {width}x{height}, "
                  f"实际相机分辨率 {self._actual_camera_width}x{self._actual_camera_height}, "
                  f"将裁剪左右各 {self._crop_left}/{self._crop_right} 像素")
        else:
            self._actual_camera_width = width
            self._actual_camera_height = height
            self._crop_left = 0
            self._crop_right = 0
        
        self.capture: Optional[cv2.VideoCapture] = None
        self._started = False
        
        # 多线程相关
        self._frame_queue: Optional[Queue] = None
        self._read_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_count = 0

        # 中心裁剪相关
        self._center_crop = center_crop
        self.crop_width = 800
        self.crop_height = 600
    
    def start(self, timeout: float = 5.0) -> bool:
        """启动相机"""
        if self._started:
            return True
        
        try:
            self.capture = cv2.VideoCapture(self.camera_id)
            
            if not self.capture.isOpened():
                print(f"[{self.name}] ❌ 无法打开相机 {self.camera_id}")
                return False
            
            # 设置相机属性（使用实际相机分辨率）
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._actual_camera_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._actual_camera_height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)            
            
            # 等待第一帧成功读取
            start_time = time.time()
            first_frame_read = False
            while time.time() - start_time < timeout:
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    first_frame_read = True
                    break
                time.sleep(0.01)
            
            if not first_frame_read:
                print(f"[{self.name}] ⚠️ 无法读取第一帧，预热可能不完整")
            
            # 预热：读取并丢弃帧一段时间以确保摄像头稳定
            if self.warmup_time > 0:
                print(f"[{self.name}] 正在预热摄像头（持续 {self.warmup_time} 秒）...")
                warmup_start = time.time()
                warmup_count = 0
                while time.time() - warmup_start < self.warmup_time:
                    ret, frame = self.capture.read()
                    if ret and frame is not None:
                        warmup_count += 1
                    # time.sleep(0.033)  # 约 30fps 的间隔
                print(f"[{self.name}] ✅ 预热完成（持续 {time.time() - warmup_start:.2f} 秒，成功读取 {warmup_count} 帧）")
            
            # 获取实际分辨率
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            
            # 更新实际相机分辨率（可能和请求的不同）
            self._actual_camera_width = actual_width
            self._actual_camera_height = actual_height
            
            # 如果是 1:1 模式，重新计算裁剪参数
            if self._is_square_mode:
                self._crop_left = (actual_width - self.width) // 2
                self._crop_right = actual_width - self.width - self._crop_left
            
            self._started = True
            
            # 如果启用多线程，启动读取线程
            if self.use_threading:
                self._frame_queue = Queue(maxsize=self.queue_size)
                self._running = True
                self._frame_count = 0
                self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
                self._read_thread.start()
                print(f"[{self.name}] ✅ 多线程读取已启动")
            
            crop_info = []
            if self._center_crop:
                crop_info.append(f"中心裁剪到 {self.crop_width}x{self.crop_height}")
            if self._is_square_mode:
                crop_info.append("1:1 裁剪模式")
            
            if crop_info:
                crop_str = ", ".join(crop_info)
                print(f"[{self.name}] ✅ 相机已启动: 相机分辨率 {actual_width}x{actual_height}@{actual_fps}fps, "
                      f"输出分辨率 {self.crop_width}x{self.crop_height} ({crop_str})")
            else:
                print(f"[{self.name}] ✅ 相机已启动: {actual_width}x{actual_height}@{actual_fps}fps")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ❌ 启动失败: {e}")
            return False
    
    def _read_loop(self):
        """后台读取循环（多线程模式）"""
        while self._running:
            try:
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    # 处理 1:1 裁剪
                    processed_frame = self._process_frame(frame)
                    if processed_frame is not None:
                        self._frame_count += 1
                        frame_data = FrameData(
                            frame=processed_frame,
                            timestamp=time.time(),
                            frame_idx=self._frame_count
                        )
                        
                        # 非阻塞放入队列，如果满了就丢弃最旧的
                        if self._frame_queue.full():
                            try:
                                self._frame_queue.get_nowait()
                            except:
                                pass
                        
                        self._frame_queue.put(frame_data)
            except Exception as e:
                print(f"[{self.name}] 读取循环错误: {e}")
                time.sleep(0.001)  # 避免错误时CPU占用过高
    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """处理帧（裁剪等）"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        
        # 如果是 1:1 模式，裁剪左右黑边（在中心裁剪之后）
        if self._is_square_mode:
            # 只有当实际宽度大于目标宽度时才需要裁剪
            if w > self.width and self._crop_left > 0:
                # 裁剪中间的正方形区域
                frame = frame[:, self._crop_left:w-self._crop_right]
            elif w != self.width:
                # 如果宽度不匹配，尝试居中裁剪
                crop = (w - self.width) // 2
                if crop > 0:
                    frame = frame[:, crop:w-crop]

        # 如果启用了中心裁剪，先从原始帧中心裁剪到指定宽高
        if self._center_crop:
            # 计算需要裁剪的像素数
            crop_w = max(0, (w - self.crop_width) // 2)
            crop_h = max(0, (h - self.crop_height) // 2)
            
            if crop_w > 0 or crop_h > 0:
                # 中心裁剪：从中间裁剪到指定尺寸
                # 如果原始尺寸小于目标尺寸，则不裁剪（保持原样或可能需要resize，这里只做裁剪）
                if w >= self.crop_width and h >= self.crop_height:
                    frame = frame[crop_h:h-crop_h, crop_w:w-crop_w]
                elif w >= self.crop_width:
                    # 只裁剪宽度
                    frame = frame[:, crop_w:w-crop_w]
                elif h >= self.crop_height:
                    # 只裁剪高度
                    frame = frame[crop_h:h-crop_h, :]
        
        return frame
    
    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """读取一帧图像（BGR 格式）
        
        如果启用了中心裁剪（center_crop=True），会从原始帧的中心裁剪到指定的 width x height 尺寸。
        如果是 1:1 分辨率模式，会自动裁剪左右黑边，返回正方形图像。
        
        注意：如果启用了多线程模式，建议使用 get_latest_frame() 获取最新帧。
        """
        if not self._started or self.capture is None:
            return None
        
        try:
            ret, frame = self.capture.read()
            if ret and frame is not None:
                return self._process_frame(frame)
            return None
        except Exception as e:
            print(f"[{self.name}] ❌ 读取错误: {e}")
            return None
    
    def get_latest_frame(self) -> Optional[FrameData]:
        """获取队列中最新的帧（多线程模式）
        
        清空队列，返回最后一个帧数据（包含帧、时间戳和帧索引）。
        如果未启用多线程或队列为空，立即返回 None（非阻塞）。
        
        注意：此方法设计为快速获取最新帧，不应该阻塞。
        如果需要等待新帧，请使用 get_frame(timeout=...) 方法。
        
        Returns:
            FrameData 对象，包含 frame, timestamp, frame_idx，或 None
        """
        if not self.use_threading or self._frame_queue is None:
            return None
        
        latest = None
        # 非阻塞地清空队列，保留最新的
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except:
                # 队列为空，退出循环
                break
        
        return latest
    
    def get_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        """从队列获取一帧（多线程模式）
        
        如果队列为空，等待最多 timeout 秒。
        
        Returns:
            FrameData 对象，包含 frame, timestamp, frame_idx
        """
        if not self.use_threading or self._frame_queue is None:
            return None
        
        try:
            return self._frame_queue.get(timeout=timeout)
        except:
            return None
    
    def stop(self):
        """停止相机"""
        # 停止多线程读取
        if self.use_threading:
            self._running = False
            if self._read_thread is not None:
                self._read_thread.join(timeout=2.0)
            self._read_thread = None
            self._frame_queue = None
        
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None
        self._started = False
        print(f"[{self.name}] 相机已停止")
    
    @property
    def is_started(self) -> bool:
        return self._started

    def show_preview(self, window_name: str = "GoPro Camera", quit_key: str = 'q',
                     callback: Optional[Callable[[np.ndarray], None]] = None):
        """Show a simple preview window and return when quit_key is pressed.

        If callback is provided it will be invoked for each frame: callback(image).
        """
        try:
            if not self._started:
                self.start()

            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            while True:
                frame = self.read()
                if frame is None:
                    continue
                if callback is not None:
                    try:
                        callback(frame)
                    except Exception:
                        # don't let a callback crash the preview
                        pass
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord(quit_key):
                    break
        except KeyboardInterrupt:
            # allow Ctrl+C to exit preview
            pass
        finally:
            cv2.destroyWindow(window_name)
            # stop camera on exit
            try:
                self.stop()
            except Exception:
                pass

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        # don't suppress exceptions
        return False


def test_gopro_fps(camera_id: int = 0, width: int = 480, height: int = 480, fps: int = 60, 
                   duration: float = 30.0):
    """
    测试 GoPro 相机的实际帧率，每秒打印一次统计信息。
    
    Args:
        camera_id: 相机设备 ID
        width: 图像宽度
        height: 图像高度
        fps: 期望的帧率
        duration: 测试持续时间（秒），0 表示无限运行
    """
    print(f"开始测试 GoPro 相机帧率...")
    print(f"配置: camera_id={camera_id}, 分辨率={width}x{height}, 期望FPS={fps}")
    print(f"按 Ctrl+C 退出测试\n")
    
    cam = GoproCamera(camera_id=camera_id, width=width, height=height, fps=fps)
    
    try:
        if not cam.start():
            print("❌ 相机启动失败")
            return
        
        frame_count = 0
        last_print_time = time.time()
        start_time = time.time()
        last_second_start = time.time()
        frames_in_last_second = 0
        
        print("开始采集帧...")
        print("-" * 60)
        
        while True:
            frame = cam.read()
            if frame is not None:
                frame_count += 1
                frames_in_last_second += 1
            
            current_time = time.time()
            elapsed = current_time - last_print_time
            
            # 每秒打印一次
            if elapsed >= 1.0:
                fps_this_second = frames_in_last_second / elapsed
                total_elapsed = current_time - start_time
                avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
                
                print(f"[{total_elapsed:6.1f}s] 当前帧率: {fps_this_second:5.1f} FPS | "
                      f"平均帧率: {avg_fps:5.1f} FPS | 总帧数: {frame_count}")
                
                # 重置计数器
                last_print_time = current_time
                frames_in_last_second = 0
                
                # 如果设置了持续时间，检查是否超时
                if duration > 0 and total_elapsed >= duration:
                    print(f"\n测试完成（已运行 {duration} 秒）")
                    break
            
            # 短暂休眠避免 CPU 占用过高
            # time.sleep(0.001)
            
    except KeyboardInterrupt:
        total_elapsed = time.time() - start_time
        avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
        print(f"\n\n测试中断")
        print(f"总运行时间: {total_elapsed:.2f} 秒")
        print(f"总帧数: {frame_count}")
        print(f"平均帧率: {avg_fps:.2f} FPS")
    finally:
        cam.stop()


def test_dual_gopro_fps(camera_id_0: int = 0, camera_id_1: int = 1, 
                        width: int = 480, height: int = 480, fps: int = 60,
                        layout: str = 'horizontal'):
    """
    同时测试两个 GoPro 相机的实际帧率，实时显示拼接图像，每秒打印各自的帧率。
    
    Args:
        camera_id_0: 第一个相机设备 ID
        camera_id_1: 第二个相机设备 ID
        width: 图像宽度
        height: 图像高度
        fps: 期望的帧率
        layout: 拼接方式，'horizontal' (水平) 或 'vertical' (垂直)
    """
    print(f"开始测试双 GoPro 相机帧率...")
    print(f"配置: 相机0={camera_id_0}, 相机1={camera_id_1}, 分辨率={width}x{height}, 期望FPS={fps}")
    print(f"拼接方式: {layout}")
    print(f"按 'q' 键或 Ctrl+C 退出测试\n")
    
    cam0 = GoproCamera(camera_id=camera_id_0, width=width, height=height, fps=fps, name="gopro_0")
    cam1 = GoproCamera(camera_id=camera_id_1, width=width, height=height, fps=fps, name="gopro_1")
    
    try:
        # 启动两个相机
        if not cam0.start():
            print("❌ 相机0启动失败")
            return
        if not cam1.start():
            print("❌ 相机1启动失败")
            cam0.stop()
            return
        
        # 帧率统计
        frame_count_0 = 0
        frame_count_1 = 0
        last_print_time = time.time()
        start_time = time.time()
        frames_in_last_second_0 = 0
        frames_in_last_second_1 = 0
        
        # 用于存储最后一帧（用于显示）
        last_frame_0 = None
        last_frame_1 = None
        
        window_name = "Dual GoPro Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        print("开始采集帧...")
        print("-" * 80)
        
        while True:
            # 读取两个相机的帧
            frame0 = cam0.read()
            frame1 = cam1.read()
            
            # 统计帧数
            if frame0 is not None:
                frame_count_0 += 1
                frames_in_last_second_0 += 1
                last_frame_0 = frame0.copy()
            
            if frame1 is not None:
                frame_count_1 += 1
                frames_in_last_second_1 += 1
                last_frame_1 = frame1.copy()
            
            # 拼接图像并显示
            if last_frame_0 is not None and last_frame_1 is not None:
                if layout == 'horizontal':
                    # 水平拼接
                    combined = np.hstack([last_frame_0, last_frame_1])
                else:
                    # 垂直拼接
                    combined = np.vstack([last_frame_0, last_frame_1])
                
                # 在图像上添加帧率信息
                total_elapsed = time.time() - start_time
                avg_fps_0 = frame_count_0 / total_elapsed if total_elapsed > 0 else 0
                avg_fps_1 = frame_count_1 / total_elapsed if total_elapsed > 0 else 0
                
                # 在图像上绘制文字
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                color = (0, 255, 0)
                
                if layout == 'horizontal':
                    # 在左图（相机0）上添加信息
                    text0 = f"Cam0: {avg_fps_0:.1f} FPS"
                    cv2.putText(combined, text0, (10, 30), font, font_scale, color, thickness)
                    # 在右图（相机1）上添加信息
                    text1 = f"Cam1: {avg_fps_1:.1f} FPS"
                    cv2.putText(combined, text1, (width + 10, 30), font, font_scale, color, thickness)
                else:
                    # 在上图（相机0）上添加信息
                    text0 = f"Cam0: {avg_fps_0:.1f} FPS"
                    cv2.putText(combined, text0, (10, 30), font, font_scale, color, thickness)
                    # 在下图（相机1）上添加信息
                    text1 = f"Cam1: {avg_fps_1:.1f} FPS"
                    cv2.putText(combined, text1, (10, height + 30), font, font_scale, color, thickness)
                
                cv2.imshow(window_name, combined)
            
            # 每秒打印一次帧率
            current_time = time.time()
            elapsed = current_time - last_print_time
            
            if elapsed >= 1.0:
                fps_this_second_0 = frames_in_last_second_0 / elapsed
                fps_this_second_1 = frames_in_last_second_1 / elapsed
                total_elapsed = current_time - start_time
                avg_fps_0 = frame_count_0 / total_elapsed if total_elapsed > 0 else 0
                avg_fps_1 = frame_count_1 / total_elapsed if total_elapsed > 0 else 0
                
                print(f"[{total_elapsed:6.1f}s] "
                      f"相机0: 当前={fps_this_second_0:5.1f} FPS, 平均={avg_fps_0:5.1f} FPS, 总帧数={frame_count_0:5d} | "
                      f"相机1: 当前={fps_this_second_1:5.1f} FPS, 平均={avg_fps_1:5.1f} FPS, 总帧数={frame_count_1:5d}")
                
                # 重置计数器
                last_print_time = current_time
                frames_in_last_second_0 = 0
                frames_in_last_second_1 = 0
            
            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 短暂休眠避免 CPU 占用过高
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        total_elapsed = time.time() - start_time
        avg_fps_0 = frame_count_0 / total_elapsed if total_elapsed > 0 else 0
        avg_fps_1 = frame_count_1 / total_elapsed if total_elapsed > 0 else 0
        print(f"\n\n测试中断")
        print(f"总运行时间: {total_elapsed:.2f} 秒")
        print(f"相机0: 总帧数={frame_count_0}, 平均帧率={avg_fps_0:.2f} FPS")
        print(f"相机1: 总帧数={frame_count_1}, 平均帧率={avg_fps_1:.2f} FPS")
    finally:
        cv2.destroyWindow(window_name)
        cam0.stop()
        cam1.stop()


if __name__ == '__main__':
    import sys
    
    # quick smoke test when run directly
    # cam = RealSenseCamera()
    # try:
    #     print("按 'q' 键或 Ctrl+C 退出预览...")
    #     cam.show_preview()
    # finally:
    #     cam.stop()

    # 根据命令行参数选择测试模式
    # if len(sys.argv) > 1 and sys.argv[1] == 'dual':
    #     # 双相机测试
    #     camera_id_0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    #     camera_id_1 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    #     layout = sys.argv[4] if len(sys.argv) > 4 else 'horizontal'
    #     test_dual_gopro_fps(camera_id_0=camera_id_0, camera_id_1=camera_id_1, 
    #                        width=480, height=480, fps=60, layout=layout)
    # else:
    #     # 单相机测试
    #     test_gopro_fps(camera_id=0, width=480, height=480, fps=60, duration=0)  # duration=0 表示无限运行

    # with GoproCamera(camera_id=6, width=1600, height=1200, fps=30) as cam:
    #     cam.show_preview()
    
    test_gopro_fps(camera_id=2, width=1600, height=1200, fps=30, duration=0)  # duration=0 表示无限运行