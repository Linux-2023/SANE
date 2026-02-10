#!/usr/bin/env python3
"""
可视化双臂上两个单目手腕相机的实时画面

本脚本可同时采集和显示左右两个手腕 USB 相机的画面和帧率。支持通过配置文件或命令行参数指定相机ID和分辨率。

用法:
    python test_wrist_cameras.py [--config config.yaml]
    或直接指定相机ID:
    python test_wrist_cameras.py --left_id 0 --right_id 2 --width 640 --height 480
"""

import argparse
import time
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# from env_dual_feetech import create_dual_arm_feetech_env_from_config, DualArmEnvironmentFeetech
from env_dual_feetech import create_dual_arm_feetech_env_from_config

# ---- MJPEG专用USB相机包装 ----
class USBMJPEGGrabber:
    """专用于MJPEG模式读取USB单目相机帧"""
    def __init__(self, camera_id, width=320, height=240, fps=30, camera_name=None):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_name = camera_name or f"Cam{camera_id}"
        self._capture = None

    def start(self):
        self._capture = cv2.VideoCapture(self.camera_id)
        if not self._capture.isOpened():
            raise RuntimeError(f"无法打开相机: {self.camera_id}")
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = int(self._capture.get(cv2.CAP_PROP_FPS))
        fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        print(f"[{self.camera_name}] {w}x{h}@{f}fps FOURCC:{fourcc_str}")

    def read(self):
        if self._capture is None:
            self.start()
        ret, frame = self._capture.read()
        if not ret or frame is None:
            return None
        return frame

    def release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

def show_wrist_monoculars(
    left_id=None, right_id=None,
    width=320, height=240, fps=30, config_path=None
):
    """
    可视化左右手腕上的两个 USB 单目相机，画面并排显示。
    """
    # 通过配置文件获得ID，否则用命令行参数
    if config_path and os.path.exists(config_path):
        print(f"\n从配置文件加载: {config_path}")
        env = create_dual_arm_feetech_env_from_config(config_path)
        env.reset()
        left_cam_id = getattr(getattr(env, "_left_wrist_camera", None), "camera_id", left_id)
        right_cam_id = getattr(getattr(env, "_right_wrist_camera", None), "camera_id", right_id)
        # 分辨率优先用参数（方便测试），否则尝试从配置拿
        cam_w = width or getattr(getattr(env, "_left_wrist_camera", None), "width", 320)
        cam_h = height or getattr(getattr(env, "_left_wrist_camera", None), "height", 240)
        cam_fps = fps or getattr(getattr(env, "_left_wrist_camera", None), "fps", 30)
    else:
        print("\n未提供配置文件或文件不存在，使用命令行参数指定相机ID和分辨率")
        left_cam_id = left_id
        right_cam_id = right_id
        cam_w, cam_h, cam_fps = width, height, fps
        env = None

    if left_cam_id is None or right_cam_id is None:
        print("❌ 请指定左右手腕相机ID（可通过--left_id/--right_id或配置文件）")
        return

    print(f"\n左腕相机ID: {left_cam_id}  |  右腕相机ID: {right_cam_id}")
    print(f"分辨率: {cam_w}x{cam_h}  |  帧率: {cam_fps}")

    grabber_left = USBMJPEGGrabber(left_cam_id, cam_w, cam_h, cam_fps, camera_name="Left")
    grabber_right = USBMJPEGGrabber(right_cam_id, cam_w, cam_h, cam_fps, camera_name="Right")

    try:
        grabber_left.start()
        grabber_right.start()
        print("\n按 q 退出。")
        win_name = "Wrist Cameras (Left | Right)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        last_time = time.time()
        fps_val = 0.0

        while True:
            f_left = grabber_left.read()
            f_right = grabber_right.read()
            if f_left is None or f_right is None:
                print("有相机读取失败, 将忽略该帧")
                continue

            # 尺寸可能不一致，强制resize到统一大小
            f_left_r = cv2.resize(f_left, (cam_w, cam_h))
            f_right_r = cv2.resize(f_right, (cam_w, cam_h))

            # 并排拼接显示
            shown = np.concatenate([f_left_r, f_right_r], axis=1)

            # 显示帧率文字
            frame_count += 1
            now = time.time()
            dt = now - last_time
            if dt > 0.5:
                fps_val = frame_count / dt
                frame_count = 0
                last_time = now

            try:
                fps_text = f"L[{left_cam_id}] | R[{right_cam_id}]  FPS: {fps_val:.1f}"
                cv2.putText(shown, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (20, 220, 20), 2, cv2.LINE_AA)
                cv2.imshow(win_name, shown)
            except Exception as e:
                print("画面显示失败:", e)

            key = cv2.waitKey(1)
            if key & 0xFF in (ord('q'), 27):  # q or ESC
                print("收到退出指令，结束显示")
                break

    except KeyboardInterrupt:
        print("\n手动中断。")
    except Exception as e:
        print(f"\n❌ 采集或显示过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        grabber_left.release()
        grabber_right.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if env:
            env.close()
        print("已释放相机，退出。")

def main():
    parser = argparse.ArgumentParser(
        description='可视化双臂上两个单目手腕相机（并排）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（优先获取相机ID、分辨率）'
    )
    parser.add_argument(
        '--left_id', type=int, default=None, help='左手腕相机ID'
    )
    parser.add_argument(
        '--right_id', type=int, default=None, help='右手腕相机ID'
    )
    parser.add_argument(
        '--width', type=int, default=320, help='画面宽度'
    )
    parser.add_argument(
        '--height', type=int, default=240, help='画面高度'
    )
    parser.add_argument(
        '--fps', type=int, default=30, help='采集帧率'
    )

    args = parser.parse_args()

    if args.config is not None and args.config.strip() != "":
        config_path = os.path.join(os.path.dirname(__file__), args.config)
    else:
        config_path = None

    show_wrist_monoculars(
        left_id=args.left_id,
        right_id=args.right_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        config_path=config_path
    )


if __name__ == "__main__":
    main()
