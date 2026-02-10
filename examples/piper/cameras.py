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
            # 尝试快速读取
            ret, frame = self.capture.read()
            if not ret or frame is None:
                return None
            return frame
        except Exception as e:
            print(f"USB camera read error: {e}")
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




if __name__ == '__main__':
    # quick smoke test when run directly
    # cam = RealSenseCamera()
    # try:
    #     print("按 'q' 键或 Ctrl+C 退出预览...")
    #     cam.show_preview()
    # finally:
    #     cam.stop()

    with USBCamera(camera_id=8, width=640, height=480, fps=30) as cam:
        cam.show_preview()