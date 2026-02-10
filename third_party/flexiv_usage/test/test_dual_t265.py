#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双 T265 相机测试程序
测试两个 T265 相机是否都能正常连接和读取数据
"""

import time
import threading
import numpy as np
import yaml
import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("未找到 pyrealsense2，请安装: pip install pyrealsense2")
    sys.exit(1)


def list_all_t265():
    """列出所有可用的 T265 设备"""
    ctx = rs.context()
    devices = ctx.query_devices()
    t265_list = []
    
    print("\n" + "=" * 60)
    print("检测到的 RealSense 设备:")
    print("=" * 60)
    
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        
        print(f"\n[设备 {i+1}]")
        print(f"  名称: {name}")
        print(f"  序列号: {serial}")
        print(f"  固件版本: {fw}")
        print(f"  USB 类型: {usb}")
        
        if "T265" in name:
            t265_list.append(serial)
    
    print("\n" + "-" * 60)
    print(f"T265 设备数量: {len(t265_list)}")
    if t265_list:
        print(f"T265 序列号列表: {t265_list}")
    print("=" * 60 + "\n")
    
    return t265_list


class T265Tester:
    """T265 测试器"""
    
    def __init__(self, serial_number: str, name: str = "T265"):
        self.serial = serial_number
        self.name = name
        self._pipe = None
        self._cfg = None
        self._connected = False
        self._lock = threading.Lock()
        
        # 数据
        self.position = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.confidence = 0
        self.frame_count = 0
        self.last_update_time = 0
    
    def connect(self) -> bool:
        """连接 T265"""
        try:
            print(f"[{self.name}] 正在连接 (SN: {self.serial})...")
            
            self._pipe = rs.pipeline()
            self._cfg = rs.config()
            self._cfg.enable_device(self.serial)
            self._cfg.enable_stream(rs.stream.pose)
            
            self._pipe.start(self._cfg)
            
            # 等待稳定
            for _ in range(10):
                self._pipe.wait_for_frames()
            
            self._connected = True
            print(f"[{self.name}] ✅ 连接成功!")
            return True
            
        except Exception as e:
            print(f"[{self.name}] ❌ 连接失败: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._pipe:
            try:
                self._pipe.stop()
            except:
                pass
            self._pipe = None
        self._connected = False
        print(f"[{self.name}] 已断开连接")
    
    def update(self) -> bool:
        """更新数据"""
        if not self._connected:
            return False
        
        try:
            frames = self._pipe.wait_for_frames(timeout_ms=100)
            pose_frame = frames.get_pose_frame()
            
            if pose_frame:
                data = pose_frame.get_pose_data()
                
                with self._lock:
                    self.position = np.array([
                        data.translation.x,
                        data.translation.y,
                        data.translation.z
                    ])
                    self.quaternion = np.array([
                        data.rotation.w,
                        data.rotation.x,
                        data.rotation.y,
                        data.rotation.z
                    ])
                    self.confidence = data.tracker_confidence
                    self.frame_count += 1
                    self.last_update_time = time.time()
                
                return True
        except Exception as e:
            pass
        return False
    
    def get_status_str(self) -> str:
        """获取状态字符串"""
        if not self._connected:
            return f"[{self.name}] ❌ 未连接"
        
        with self._lock:
            pos = self.position
            q = self.quaternion
            conf = self.confidence
            fps = self.frame_count / max(1, time.time() - self.last_update_time + 0.001) if self.frame_count > 0 else 0
        
        # 置信度说明
        conf_str = ["失败", "低", "中", "高"][min(conf, 3)]
        
        return (f"[{self.name}] "
                f"位置:[{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
                f"四元数:[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] "
                f"置信:{conf_str}({conf}) "
                f"帧:{self.frame_count}")
    
    @property
    def is_connected(self) -> bool:
        return self._connected


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="双 T265 相机测试")
    parser.add_argument('--config', default='dual_arm_config.yaml', help='配置文件路径')
    parser.add_argument('--list-only', action='store_true', help='仅列出设备，不进行测试')
    parser.add_argument('--duration', type=float, default=0, help='测试持续时间(秒)，0表示无限')
    args = parser.parse_args()
    
    # 首先列出所有设备
    available_serials = list_all_t265()
    
    if args.list_only:
        return
    
    if len(available_serials) < 2:
        print("⚠️  警告: 检测到的 T265 少于 2 个!")
        print("   请检查 USB 连接和供电")
        if len(available_serials) == 0:
            print("   没有可用的 T265，退出测试")
            return
    
    # 加载配置
    try:
        config = load_config(args.config)
        left_serial = config.get('left_arm', {}).get('t265_serial', '')
        right_serial = config.get('right_arm', {}).get('t265_serial', '')
        print(f"\n配置文件中的 T265 序列号:")
        print(f"  左臂 T265: {left_serial or '(未指定)'}")
        print(f"  右臂 T265: {right_serial or '(未指定)'}")
    except Exception as e:
        print(f"无法加载配置文件 {args.config}: {e}")
        print("将使用检测到的 T265 进行测试")
        left_serial = available_serials[0] if len(available_serials) > 0 else ""
        right_serial = available_serials[1] if len(available_serials) > 1 else ""
    
    # 验证序列号
    testers = []
    
    if left_serial:
        if left_serial in available_serials:
            testers.append(T265Tester(left_serial, "左臂 T265"))
        else:
            print(f"⚠️  左臂 T265 序列号 {left_serial} 未在可用设备中找到!")
    
    if right_serial:
        if right_serial in available_serials:
            testers.append(T265Tester(right_serial, "右臂 T265"))
        else:
            print(f"⚠️  右臂 T265 序列号 {right_serial} 未在可用设备中找到!")
    
    if not testers:
        print("没有有效的 T265 可以测试，退出")
        return
    
    # 连接
    print("\n" + "=" * 60)
    print("开始连接 T265...")
    print("=" * 60)
    
    connected_testers = []
    for tester in testers:
        if tester.connect():
            connected_testers.append(tester)
        time.sleep(0.5)  # 避免同时启动造成冲突
    
    if not connected_testers:
        print("没有 T265 连接成功，退出")
        return
    
    # 测试循环
    print("\n" + "=" * 60)
    print("开始读取数据 (按 Ctrl+C 停止)")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    try:
        while True:
            # 更新数据
            for tester in connected_testers:
                tester.update()
            
            # 显示状态
            status_lines = []
            for tester in connected_testers:
                status_lines.append(tester.get_status_str())
            
            # 清屏并打印
            print("\033[2J\033[H")  # 清屏
            print("=" * 80)
            print("双 T265 相机测试")
            print("=" * 80)
            print()
            for line in status_lines:
                print(line)
            print()
            print("-" * 80)
            print(f"运行时间: {time.time() - start_time:.1f} 秒")
            print("按 Ctrl+C 停止测试")
            
            # 检查持续时间
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                print(f"\n测试时间 {args.duration} 秒已到，自动停止")
                break
            
            time.sleep(0.05)  # 20Hz 刷新
            
    except KeyboardInterrupt:
        print("\n\n停止测试...")
    
    finally:
        # 断开连接
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        for tester in connected_testers:
            print(f"[{tester.name}] 总帧数: {tester.frame_count}")
            tester.disconnect()
        
        print("\n测试完成!")


if __name__ == "__main__":
    main()

