#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位姿遥操作控制器

将 T265 相机的位姿变化映射到 Flexiv 机器人的末端位姿控制。
"""

import time
import threading
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation

from t265_action_reader import ActionReader
from flexiv_robot import FlexivRobot


class PoseTeleopController:
    """
    位姿遥操作控制器
    
    将 T265 相机的位姿变化实时映射到机器人末端位姿控制。
    """
    
    def __init__(
        self,
        action_reader: ActionReader,
        flexiv_robot: FlexivRobot,
        control_frequency: float = 50.0,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        enable_gripper_control: bool = False,
        verbose: bool = True
    ):
        """
        初始化位姿遥操作控制器
        
        Args:
            action_reader: ActionReader 实例
            flexiv_robot: FlexivRobot 实例
            control_frequency: 控制频率 [Hz]
            position_scale: 位置缩放因子（T265位移 × scale = 机器人位移）
            rotation_scale: 旋转缩放因子（T265旋转 × scale = 机器人旋转）
            enable_gripper_control: 是否根据编码器角度控制夹爪
            verbose: 是否打印详细信息
        """
        self.action_reader = action_reader
        self.robot = flexiv_robot
        self.control_frequency = control_frequency
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.enable_gripper_control = enable_gripper_control
        self.verbose = verbose
        
        # 线程控制
        self._control_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        
        # 参考位姿（用于计算变化量）
        self._t265_reference_pose: Optional[np.ndarray] = None  # [x, y, z, qw, qx, qy, qz]
        self._robot_reference_pose: Optional[np.ndarray] = None  # [x, y, z, qw, qx, qy, qz]
        self._t265_reference_angle: Optional[float] = None  # 夹爪参考角度
        
        # 统计信息
        self._loop_count = 0
        self._last_loop_time = 0.0
        self._avg_loop_time = 0.0
    
    def start(self, reset_reference: bool = True):
        """
        启动控制线程
        
        Args:
            reset_reference: 是否重置参考位姿（使用当前位姿作为参考）
        """
        if self._running:
            self._log("控制线程已在运行", level="warning")
            return
        
        # 检查设备状态
        if not self.action_reader.is_connected:
            raise RuntimeError("ActionReader 未连接，请先调用 connect()")
        
        if not self.robot.is_ready():
            raise RuntimeError("FlexivRobot 未就绪，请先调用 initialize()")
        
        # 重置参考位姿
        if reset_reference:
            self.reset_reference()
        
        # 启动控制线程
        self._stop_flag.clear()
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        
        self._log("位姿遥操作控制已启动")
    
    def stop(self):
        """停止控制线程"""
        if not self._running:
            return
        
        self._stop_flag.set()
        self._running = False
        
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
        
        self._log("位姿遥操作控制已停止")
    
    def reset_reference(self):
        """
        重置参考位姿
        
        将当前 T265 位姿和机器人位姿设为参考，之后的变化量将基于此参考计算。
        """
        with self._lock:
            # 获取当前 T265 位姿
            action = self.action_reader.get_action()
            self._t265_reference_pose = action.pose.copy()
            self._t265_reference_angle = action.joint_angle
            
            # 获取当前机器人位姿
            self._robot_reference_pose = self.robot.get_tcp_pose().copy()
            
            self._log("参考位姿已重置")
            self._log(f"  T265 参考位置: {self._t265_reference_pose[:3]}")
            self._log(f"  机器人参考位置: {self._robot_reference_pose[:3]}")
            if self.enable_gripper_control:
                self._log(f"  夹爪参考角度: {self._t265_reference_angle:.2f}°")
    
    def is_running(self) -> bool:
        """检查控制线程是否正在运行"""
        return self._running
    
    def get_statistics(self) -> dict:
        """
        获取控制循环统计信息
        
        Returns:
            包含循环次数、平均循环时间等信息的字典
        """
        return {
            'loop_count': self._loop_count,
            'avg_loop_time': self._avg_loop_time,
            'control_frequency': self.control_frequency,
            'running': self._running
        }
    
    def _control_loop(self):
        """控制循环（在线程中运行）"""
        period = 1.0 / self.control_frequency
        loop_times = []
        
        self._log(f"控制循环开始，频率: {self.control_frequency} Hz")
        
        while not self._stop_flag.is_set():
            loop_start = time.time()
            
            try:
                # 检查参考位姿是否已设置
                if self._t265_reference_pose is None or self._robot_reference_pose is None:
                    self._log("参考位姿未设置，跳过本次循环", level="warning")
                    time.sleep(period)
                    continue
                
                # 读取 T265 位姿和夹爪角度
                action = self.action_reader.get_action()
                t265_current_pose = action.pose  # [x, y, z, qw, qx, qy, qz]
                
                # 计算位姿变化量
                delta_pose = self._compute_pose_delta(
                    self._t265_reference_pose,
                    t265_current_pose
                )
                
                # 获取机器人当前位姿
                robot_current_pose = self.robot.get_tcp_pose()
                
                # 应用变化量到机器人位姿
                target_pose = self._apply_pose_delta(
                    self._robot_reference_pose,
                    delta_pose
                )
                print(f"target_pose: {target_pose}")
                # 发送目标位姿
                self.robot.send_cartesian_pose(target_pose)
                
                # 控制夹爪（如果启用）
                if self.enable_gripper_control and self.robot.is_gripper_enabled():
                    angle_delta = action.joint_angle - self._t265_reference_angle
                    # 这里可以根据角度变化控制夹爪，例如：
                    # 将角度变化映射到夹爪宽度
                    # 具体实现取决于夹爪控制需求
                    pass
                
                # 更新统计信息
                self._loop_count += 1
                loop_time = time.time() - loop_start
                loop_times.append(loop_time)
                
                # 保持最近100次的循环时间用于计算平均值
                if len(loop_times) > 100:
                    loop_times.pop(0)
                self._avg_loop_time = np.mean(loop_times)
                
            except Exception as e:
                self._log(f"控制循环错误: {e}", level="error")
                import traceback
                traceback.print_exc()
            
            # 控制循环频率
            elapsed = time.time() - loop_start
            sleep_time = max(0, period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > period * 1.5:  # 如果循环时间超过预期1.5倍，警告
                self._log(f"控制循环超时: {elapsed*1000:.1f}ms (预期: {period*1000:.1f}ms)", 
                         level="warning")
    
    def _compute_pose_delta(
        self,
        reference_pose: np.ndarray,
        current_pose: np.ndarray
    ) -> dict:
        """
        计算位姿变化量
        
        Args:
            reference_pose: 参考位姿 [x, y, z, qw, qx, qy, qz]
            current_pose: 当前位姿 [x, y, z, qw, qx, qy, qz]
            
        Returns:
            包含位置差和旋转差的字典
        """
        # 位置差
        delta_position = (current_pose[:3] - reference_pose[:3]) * self.position_scale
        
        # 旋转差（四元数）
        # 参考旋转的逆 × 当前旋转 = 相对旋转
        ref_rot = Rotation.from_quat([
            reference_pose[4], reference_pose[5], 
            reference_pose[6], reference_pose[3]  # [qx, qy, qz, qw]
        ])
        curr_rot = Rotation.from_quat([
            current_pose[4], current_pose[5],
            current_pose[6], current_pose[3]  # [qx, qy, qz, qw]
        ])
        
        # 相对旋转 = 当前旋转 × 参考旋转的逆
        relative_rot = curr_rot * ref_rot.inv()
        
        # 应用旋转缩放（通过轴角表示）
        if self.rotation_scale != 1.0:
            axis_angle = relative_rot.as_rotvec()
            scaled_axis_angle = axis_angle * self.rotation_scale
            relative_rot = Rotation.from_rotvec(scaled_axis_angle)
        
        # 转换回四元数 [qw, qx, qy, qz]
        delta_quat = relative_rot.as_quat()  # [qx, qy, qz, qw]
        delta_quaternion = np.array([
            delta_quat[3],  # qw
            delta_quat[0],  # qx
            delta_quat[1],  # qy
            delta_quat[2]   # qz
        ])
        
        return {
            'position': delta_position,
            'quaternion': delta_quaternion
        }
    
    def _apply_pose_delta(
        self,
        reference_pose: np.ndarray,
        delta: dict
    ) -> np.ndarray:
        """
        将位姿变化量应用到参考位姿
        
        Args:
            reference_pose: 参考位姿 [x, y, z, qw, qx, qy, qz]
            delta: 位姿变化量字典，包含 'position' 和 'quaternion'
            
        Returns:
            目标位姿 [x, y, z, qw, qx, qy, qz]
        """
        # 位置：直接相加
        target_position = reference_pose[:3] + delta['position']
        
        # 旋转：四元数乘法
        ref_rot = Rotation.from_quat([
            reference_pose[4], reference_pose[5],
            reference_pose[6], reference_pose[3]  # [qx, qy, qz, qw]
        ])
        delta_rot = Rotation.from_quat([
            delta['quaternion'][1], delta['quaternion'][2],
            delta['quaternion'][3], delta['quaternion'][0]  # [qx, qy, qz, qw]
        ])
        
        # 目标旋转 = 参考旋转 × 变化旋转
        target_rot = ref_rot * delta_rot
        
        # 转换回四元数 [qw, qx, qy, qz]
        target_quat = target_rot.as_quat()  # [qx, qy, qz, qw]
        target_quaternion = np.array([
            target_quat[3],  # qw
            target_quat[0],  # qx
            target_quat[1],  # qy
            target_quat[2]   # qz
        ])
        
        # 组合目标位姿
        target_pose = np.concatenate([target_position, target_quaternion])
        
        return target_pose
    
    def _log(self, msg: str, level: str = "info"):
        """打印日志"""
        if self.verbose:
            prefix = {
                "info": "[PoseTeleop]",
                "warning": "[PoseTeleop WARNING]",
                "error": "[PoseTeleop ERROR]"
            }.get(level, "[PoseTeleop]")
            print(f"{prefix} {msg}")
    
    def __enter__(self):
        """支持 with 语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.stop()


# ============ 使用示例 ============

if __name__ == "__main__":
    import argparse
    from flexiv_robot import load_robot_sn_from_yaml
    
    parser = argparse.ArgumentParser(description="位姿遥操作控制器测试")
    parser.add_argument('--robot-config', default='robot.yaml', help='机器人配置文件')
    parser.add_argument('--encoder-port', default='/dev/ttyUSB0', help='编码器串口')
    parser.add_argument('--freq', type=float, default=10.0, help='控制频率 [Hz]')
    parser.add_argument('--pos-scale', type=float, default=1.0, help='位置缩放因子')
    parser.add_argument('--rot-scale', type=float, default=1.0, help='旋转缩放因子')
    parser.add_argument('--no-gripper', action='store_true', help='不控制夹爪')
    args = parser.parse_args()
    
    print("=" * 60)
    print("位姿遥操作控制器测试")
    print("=" * 60)
    
    # 初始化 ActionReader
    print("\n初始化 ActionReader...")
    action_reader = ActionReader(encoder_port=args.encoder_port)
    if not action_reader.connect():
        print("ActionReader 连接失败!")
        exit(1)
    
    # 校准编码器（如果需要）
    if action_reader.use_encoder:
        print("\n请将机构移动到初始位置（闭合状态）")
        input("按 Enter 键进行校准...")
        action_reader.calibrate()
    
    # 初始化 FlexivRobot
    print("\n初始化 FlexivRobot...")
    robot_sn = load_robot_sn_from_yaml(args.robot_config)
    robot = FlexivRobot(robot_sn, auto_init=True)
    
    # 创建控制器
    print("\n创建位姿遥操作控制器...")
    controller = PoseTeleopController(
        action_reader=action_reader,
        flexiv_robot=robot,
        control_frequency=args.freq,
        position_scale=args.pos_scale,
        rotation_scale=args.rot_scale,
        enable_gripper_control=not args.no_gripper
    )
    
    # 启动控制
    print("\n准备启动控制...")
    print("请将 T265 相机和机器人末端移动到初始对齐位置")
    input("按 Enter 键开始控制...")
    
    controller.start(reset_reference=True)
    
    try:
        print("\n控制已启动，按 Ctrl+C 停止...")
        print("-" * 60)
        
        # 主循环：显示统计信息
        while True:
            time.sleep(1.0)
            stats = controller.get_statistics()
            if stats['loop_count'] > 0:
                print(f"\r循环次数: {stats['loop_count']:5d} | "
                      f"平均循环时间: {stats['avg_loop_time']*1000:5.1f}ms | "
                      f"实际频率: {1.0/stats['avg_loop_time']:.1f}Hz", end='')
    
    except KeyboardInterrupt:
        print("\n\n停止控制...")
    finally:
        controller.stop()
        action_reader.disconnect()
        robot.stop()
        print("程序结束")

