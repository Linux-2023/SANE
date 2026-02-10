#!/usr/bin/env python3
"""
测试双 USB 摄像头实时可视化
使用 USBCamera 类实例化两个摄像头（ID=0 和 ID=2）
"""

import cv2
import numpy as np
import time
from cameras import USBCamera


def test_dual_usb_cameras(camera_id_0: int = 0, camera_id_1: int = 2,
                          width: int = 640, height: int = 480, fps: int = 30):
    """
    同时测试两个 USB 摄像头，实时显示拼接图像，每秒打印各自的帧率。
    
    Args:
        camera_id_0: 第一个相机设备 ID
        camera_id_1: 第二个相机设备 ID
        width: 图像宽度
        height: 图像高度
        fps: 期望的帧率
    """
    print(f"开始测试双 USB 摄像头...")
    print(f"配置: 相机0={camera_id_0}, 相机1={camera_id_1}, 分辨率={width}x{height}, 期望FPS={fps}")
    print(f"按 'q' 键或 Ctrl+C 退出测试\n")
    
    cam0 = USBCamera(camera_id=camera_id_0, width=width, height=height, fps=fps)
    cam1 = USBCamera(camera_id=camera_id_1, width=width, height=height, fps=fps)
    
    try:
        # 启动两个相机
        print(f"正在启动相机 {camera_id_0}...")
        try:
            cam0.start()
            # 检查实际分辨率
            actual_w0 = int(cam0.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h0 = int(cam0.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ 相机 {camera_id_0} 启动成功: 实际分辨率={actual_w0}x{actual_h0}")
        except Exception as e:
            print(f"❌ 相机 {camera_id_0} 启动失败: {e}")
            return
        
        print(f"正在启动相机 {camera_id_1}...")
        try:
            cam1.start()
            # 检查实际分辨率
            actual_w1 = int(cam1.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h1 = int(cam1.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✅ 相机 {camera_id_1} 启动成功: 实际分辨率={actual_w1}x{actual_h1}")
        except Exception as e:
            print(f"❌ 相机 {camera_id_1} 启动失败: {e}")
            cam0.stop()
            return
        
        # 测试读取第一帧（多次尝试）
        print(f"\n测试读取第一帧（多次尝试）...")
        test_frame0 = None
        test_frame1 = None
        for attempt in range(10):
            if test_frame0 is None:
                test_frame0 = cam0.read()
            if test_frame1 is None:
                test_frame1 = cam1.read()
            if test_frame0 is not None and test_frame1 is not None:
                break
            time.sleep(0.1)
        
        if test_frame0 is not None:
            print(f"✅ 相机 {camera_id_0} 能读取帧: shape={test_frame0.shape}, dtype={test_frame0.dtype}, "
                  f"min={test_frame0.min()}, max={test_frame0.max()}, mean={test_frame0.mean():.1f}")
        else:
            print(f"⚠️  相机 {camera_id_0} 无法读取帧（返回 None，可能是USB带宽不足或资源竞争）")
            # 检查相机状态
            if cam0.capture is not None:
                is_opened = cam0.capture.isOpened()
                print(f"   相机 {camera_id_0} 状态: isOpened={is_opened}")
        
        if test_frame1 is not None:
            print(f"✅ 相机 {camera_id_1} 能读取帧: shape={test_frame1.shape}, dtype={test_frame1.dtype}, "
                  f"min={test_frame1.min()}, max={test_frame1.max()}, mean={test_frame1.mean():.1f}")
        else:
            print(f"⚠️  相机 {camera_id_1} 无法读取帧（返回 None，可能是USB带宽不足或资源竞争）")
            # 检查相机状态
            if cam1.capture is not None:
                is_opened = cam1.capture.isOpened()
                print(f"   相机 {camera_id_1} 状态: isOpened={is_opened}")
        
        # USB带宽警告
        if test_frame0 is None or test_frame1 is None:
            print(f"\n💡 提示：两个相机同时工作在 {width}x{height}@{fps}fps 可能需要大量USB带宽。")
            print(f"   如果相机单独测试正常，但同时使用时失败，可能是：")
            print(f"   1. USB 2.0 带宽不足（建议使用 USB 3.0 端口）")
            print(f"   2. USB HUB 带宽限制（建议直接连接到主板USB端口）")
            print(f"   3. 尝试降低分辨率或帧率：--width 640 --height 480 --fps 30")
        print()
        
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
        
        window_name = "Dual USB Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # 初始化字体相关变量（在所有分支之前定义）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
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
                # 水平拼接
                combined = np.hstack([last_frame_0, last_frame_1])
                
                # 在图像上添加帧率信息
                total_elapsed = time.time() - start_time
                avg_fps_0 = frame_count_0 / total_elapsed if total_elapsed > 0 else 0
                avg_fps_1 = frame_count_1 / total_elapsed if total_elapsed > 0 else 0
                
                # 在图像上绘制文字
                color = (0, 255, 0)
                
                # 在左图（相机0）上添加信息
                text0 = f"Camera {camera_id_0}: {avg_fps_0:.1f} FPS"
                cv2.putText(combined, text0, (10, 30), font, font_scale, color, thickness)
                
                # 在右图（相机1）上添加信息
                h, w = last_frame_0.shape[:2]
                text1 = f"Camera {camera_id_1}: {avg_fps_1:.1f} FPS"
                cv2.putText(combined, text1, (w + 10, 30), font, font_scale, color, thickness)
                
                # 添加总帧数信息
                total_frames_text = f"Total: Cam0={frame_count_0}, Cam1={frame_count_1}"
                cv2.putText(combined, total_frames_text, (10, h - 10), font, font_scale, color, thickness)
                
                cv2.imshow(window_name, combined)
            elif last_frame_0 is not None:
                # 只有相机0有图像
                combined = last_frame_0.copy()
                text = f"Camera {camera_id_0} (Camera {camera_id_1} not available)"
                cv2.putText(combined, text, (10, 30), font, font_scale, (0, 0, 255), thickness)
                cv2.imshow(window_name, combined)
            elif last_frame_1 is not None:
                # 只有相机1有图像
                combined = last_frame_1.copy()
                text = f"Camera {camera_id_1} (Camera {camera_id_0} not available)"
                cv2.putText(combined, text, (10, 30), font, font_scale, (0, 0, 255), thickness)
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
                      f"相机{camera_id_0}: 当前={fps_this_second_0:5.1f} FPS, 平均={avg_fps_0:5.1f} FPS, 总帧数={frame_count_0:5d} | "
                      f"相机{camera_id_1}: 当前={fps_this_second_1:5.1f} FPS, 平均={avg_fps_1:5.1f} FPS, 总帧数={frame_count_1:5d}")
                
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
        print(f"相机{camera_id_0}: 总帧数={frame_count_0}, 平均帧率={avg_fps_0:.2f} FPS")
        print(f"相机{camera_id_1}: 总帧数={frame_count_1}, 平均帧率={avg_fps_1:.2f} FPS")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cv2.destroyWindow(window_name)
        except:
            pass
        cam0.stop()
        cam1.stop()
        print("\n✅ 测试完成，相机已关闭")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='测试双 USB 摄像头实时可视化')
    parser.add_argument('--camera0', type=int, default=2, help='第一个相机 ID (默认: 0)')
    parser.add_argument('--camera1', type=int, default=0, help='第二个相机 ID (默认: 2)')
    parser.add_argument('--width', type=int, default=1600, help='图像宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=1200, help='图像高度 (默认: 480)')
    parser.add_argument('--fps', type=int, default=30, help='期望帧率 (默认: 30)')
    
    args = parser.parse_args()
    
    test_dual_usb_cameras(
        camera_id_0=args.camera0,
        camera_id_1=args.camera1,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

