"""
摄像头测试程序

使用 env.py 中的 GoproCamera 接口，持续读取摄像头并实时预览。

使用方法:
    # 测试单个相机（默认 ID 0）
    python examples/flexiv/test_camera.py
    
    # 指定相机 ID
    python examples/flexiv/test_camera.py --camera_id 2
    
    # 测试多个相机
    python examples/flexiv/test_camera.py --camera_ids 0 2
    
    # 自定义分辨率
    python examples/flexiv/test_camera.py --width 1920 --height 1080
    
按 'q' 键退出预览，按 's' 键保存当前帧。
"""

import argparse
import time
import cv2
import numpy as np
from typing import List

# 导入 GoproCamera 类
from env import GoproCamera


def test_single_camera(
    camera_id: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    window_width: int = 960,
    window_height: int = 540,
):
    """测试单个摄像头"""
    print(f"\n{'='*50}")
    print(f"单摄像头测试 - ID: {camera_id}")
    print(f"{'='*50}")
    
    camera = GoproCamera(
        camera_id=camera_id,
        width=width,
        height=height,
        fps=fps,
        name=f"camera_{camera_id}"
    )
    
    if not camera.start():
        print(f"❌ 摄像头 {camera_id} 启动失败")
        return
    
    window_name = f"Camera {camera_id} Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    print(f"\n📷 预览中... 按 'q' 退出, 按 's' 保存当前帧")
    print(f"   分辨率: {camera.width}x{camera.height}")
    
    frame_count = 0
    start_time = time.time()
    save_count = 0
    
    try:
        import os

        save_dir = f"camera_{camera_id}_frames"
        os.makedirs(save_dir, exist_ok=True)

        while True:
            frame = camera.read()
            
            if frame is not None:
                frame_count += 1
                if frame_count % 1 == 0:
                    filename = os.path.join(save_dir, f"frame_{frame_count}.png")
                    cv2.imwrite(filename, frame)
                    print(f"💾 已保存: {filename}")
                
                # 计算并显示 FPS
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps_actual = frame_count / elapsed
                    
                    # 在图像上显示信息
                    info_text = f"FPS: {fps_actual:.1f} | Frame: {frame_count} | Resolution: {frame.shape[1]}x{frame.shape[0]}"
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️ 退出预览")
                break
            elif key == ord('s'):
                if frame is not None:
                    filename = f"camera_{camera_id}_frame_{save_count}.png"
                    cv2.imwrite(filename, frame)
                    save_count += 1
                    print(f"💾 已保存: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️ 中断")
    
    finally:
        camera.stop()
        cv2.destroyWindow(window_name)
        
        elapsed = time.time() - start_time
        print(f"\n📊 统计:")
        print(f"   - 总帧数: {frame_count}")
        print(f"   - 运行时间: {elapsed:.1f}s")
        print(f"   - 平均 FPS: {frame_count/elapsed:.1f}" if elapsed > 0 else "   - 平均 FPS: N/A")


def test_multiple_cameras(
    camera_ids: List[int],
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
):
    """测试多个摄像头"""
    print(f"\n{'='*50}")
    print(f"多摄像头测试 - IDs: {camera_ids}")
    print(f"{'='*50}")
    
    cameras = []
    for cam_id in camera_ids:
        camera = GoproCamera(
            camera_id=cam_id,
            width=width,
            height=height,
            fps=fps,
            name=f"camera_{cam_id}"
        )
        if camera.start():
            cameras.append((cam_id, camera))
        else:
            print(f"❌ 摄像头 {cam_id} 启动失败，跳过")
    
    if not cameras:
        print("❌ 没有可用的摄像头")
        return
    
    window_name = "Multi-Camera Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print(f"\n📷 预览中... 按 'q' 退出, 按 's' 保存当前帧")
    print(f"   已启动 {len(cameras)} 个摄像头")
    
    frame_count = 0
    start_time = time.time()
    save_count = 0
    
    try:
        while True:
            frames = []
            for cam_id, camera in cameras:
                frame = camera.read()
                if frame is not None:
                    # 调整每个帧大小以便拼接显示
                    display_height = 480
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    resized = cv2.resize(frame, (display_width, display_height))
                    
                    # 添加相机 ID 标签
                    cv2.putText(resized, f"Cam {cam_id}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frames.append(resized)
                else:
                    # 如果没有帧，显示黑色占位
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"Cam {cam_id} - No Signal", (10, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frames.append(placeholder)
            
            if frames:
                frame_count += 1
                
                # 水平拼接所有帧
                combined = np.hstack(frames)
                
                # 显示 FPS
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps_actual = frame_count / elapsed
                    cv2.putText(combined, f"FPS: {fps_actual:.1f}", 
                               (combined.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow(window_name, combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️ 退出预览")
                break
            elif key == ord('s'):
                if frames:
                    filename = f"multi_camera_frame_{save_count}.png"
                    cv2.imwrite(filename, combined)
                    save_count += 1
                    print(f"💾 已保存: {filename}")
    
    except KeyboardInterrupt:
        print("\n⏹️ 中断")
    
    finally:
        for cam_id, camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n📊 统计:")
        print(f"   - 总帧数: {frame_count}")
        print(f"   - 运行时间: {elapsed:.1f}s")
        print(f"   - 平均 FPS: {frame_count/elapsed:.1f}" if elapsed > 0 else "   - 平均 FPS: N/A")


def list_available_cameras(max_id: int = 10):
    """列出可用的摄像头"""
    print(f"\n{'='*50}")
    print("扫描可用摄像头...")
    print(f"{'='*50}")
    
    available = []
    for i in range(max_id):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available.append((i, width, height, fps))
                print(f"  ✅ Camera {i}: {width}x{height}@{fps}fps")
            cap.release()
    
    if not available:
        print("  ❌ 未找到可用摄像头")
    else:
        print(f"\n共找到 {len(available)} 个可用摄像头")
    
    return available


def main():
    parser = argparse.ArgumentParser(description="摄像头测试程序")
    parser.add_argument("--camera_id", type=int, default=None, help="单个摄像头 ID")
    parser.add_argument("--camera_ids", type=int, nargs="+", default=None, help="多个摄像头 ID 列表")
    parser.add_argument("--width", type=int, default=1920, help="采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="采集高度")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--list", action="store_true", help="列出可用摄像头")
    args = parser.parse_args()
    
    if args.list:
        list_available_cameras()
        return
    
    if args.camera_ids is not None:
        # 多摄像头模式
        test_multiple_cameras(
            camera_ids=args.camera_ids,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    else:
        # 单摄像头模式
        camera_id = args.camera_id if args.camera_id is not None else 0
        test_single_camera(
            camera_id=camera_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()

