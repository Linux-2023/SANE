#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Piper 位姿遥操作控制器

将 T265 相机的位姿变化映射到 Piper 机械臂的末端位姿控制。
"""

import time
import threading
import numpy as np
from typing import Optional
from scipy.spatial.transform import Rotation

from t265_action_reader import ActionReader
from piper_controller import PiperController


class PiperPoseTeleopController:
    """
    Piper 位姿遥操作控制器
    
    将 T265 相机的位姿变化实时映射到 Piper 机械臂末端位姿控制。
    """
    
    def __init__(
        self,
        action_reader: ActionReader,
        piper_robot: PiperController,
        control_frequency: float = 50.0,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        enable_gripper_control: bool = False,
        velocity: int = 100,
        verbose: bool = True
    ):
        """
        初始化 Piper 位姿遥操作控制器
        
        Args:
            action_reader: ActionReader 实例
            piper_robot: PiperController 实例
            control_frequency: 控制频率 [Hz]
            position_scale: 位置缩放因子（T265位移 × scale = 机器人位移）
            rotation_scale: 旋转缩放因子（T265旋转 × scale = 机器人旋转）
            enable_gripper_control: 是否根据编码器角度控制夹爪
            velocity: 机器人运动速度（Piper 的速度参数）
            verbose: 是否打印详细信息
        """
        self.action_reader = action_reader
        self.robot = piper_robot
        self.control_frequency = control_frequency
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.enable_gripper_control = enable_gripper_control
        self.velocity = velocity
        self.verbose = verbose
        
        # 线程控制
        self._control_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        
        # 参考位姿（用于计算变化量）
        # T265 参考位姿 [x, y, z, qw, qx, qy, qz]
        self._t265_reference_pose: Optional[np.ndarray] = None
        # Piper 参考位姿 [X, Y, Z, RX, RY, RZ] (米和度)
        self._robot_reference_pose: Optional[np.ndarray] = None
        self._t265_reference_angle: Optional[float] = None  # 夹爪参考角度
        self._last_target_pose: Optional[np.ndarray] = None
        self._last_t265_pose: Optional[np.ndarray] = None
        self._last_delta_pose: Optional[np.ndarray] = None
        self._last_delta_rot: Optional[np.ndarray] = None
        self.R_gripper_piper_pos = np.array([
            [0,0,-1],
            [-1,0,0],
            [0,1,0]
        ])
        self.R_gripper_piper_rot = np.array([
            [0,1,0],
            [1,0,0],
            [0,0,1]
        ])
        
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
            self._t265_reference_pose = action.pose.copy()  # [x, y, z, qw, qx, qy, qz]
            self._t265_reference_angle = action.joint_angle
            
            # 获取当前机器人位姿
            # get_end_pose 返回 [RX, RY, RZ, X, Y, Z]，需要转换为 [X, Y, Z, RX, RY, RZ]
            robot_pose_raw = self.robot.get_end_pose()  # [RX, RY, RZ, X, Y, Z]
            self._robot_reference_pose = np.array([
                robot_pose_raw[3],  # X
                robot_pose_raw[4],  # Y
                robot_pose_raw[5],  # Z
                robot_pose_raw[0],  # RX
                robot_pose_raw[1],  # RY
                robot_pose_raw[2]   # RZ
            ])  # [X, Y, Z, RX, RY, RZ]
            
            self._log("参考位姿已重置")
            self._log(f"  T265 参考位置: {self._t265_reference_pose[:3]}")
            self._log(f"  机器人参考位置: {self._robot_reference_pose[:3]}")
            self._log(f"  机器人参考姿态: {self._robot_reference_pose[3:]}°")
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
                self._last_t265_pose = t265_current_pose
                # 计算位姿变化量
                delta_pose = self._compute_pose_delta(
                    self._t265_reference_pose,
                    t265_current_pose
                )
                self._last_delta_pose = delta_pose['position']
                self._last_delta_rot = delta_pose['euler']

                # 获取机器人当前位姿
                robot_pose_raw = self.robot.get_end_pose()  # [RX, RY, RZ, X, Y, Z]
                robot_current_pose = np.array([
                    robot_pose_raw[3],  # X
                    robot_pose_raw[4],  # Y
                    robot_pose_raw[5],  # Z
                    robot_pose_raw[0],  # RX
                    robot_pose_raw[1],  # RY
                    robot_pose_raw[2]   # RZ
                ])  # [X, Y, Z, RX, RY, RZ]
                
                # 应用变化量到机器人位姿
                target_pose = self._apply_pose_delta(
                    self._robot_reference_pose,
                    delta_pose
                )
                
                # 发送目标位姿（Piper 需要 [X, Y, Z, RX, RY, RZ] 格式）
                self.robot.move_end_pose(target_pose.tolist(), velocity=self.velocity)
                self._last_target_pose = target_pose
                
                # 控制夹爪（如果启用）
                if self.enable_gripper_control:
                    angle_delta = action.joint_angle - self._t265_reference_angle
                    # 将角度变化映射到夹爪宽度
                    # 这里假设角度变化直接映射到夹爪宽度（需要根据实际情况调整）
                    # 例如：角度变化 1 度 = 夹爪宽度变化 0.001 米
                    gripper_width_delta = angle_delta * 0.001  # 可调整的映射系数
                    current_gripper_width = self.robot.get_gripper_status()
                    target_gripper_width = current_gripper_width + gripper_width_delta
                    # 限制夹爪宽度范围
                    target_gripper_width = max(0.0, min(0.07, target_gripper_width))
                    self.robot.control_gripper(target_gripper_width)
                
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
            包含位置差和旋转差（欧拉角，度）的字典
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
        
        # 转换为欧拉角（度）- 使用 'xyz' 顺序（Roll, Pitch, Yaw）
        delta_euler = relative_rot.as_euler('xyz', degrees=True)
        
        return {
            'position': delta_position,
            'euler': delta_euler  # [RX, RY, RZ] 单位：度
        }
    
    def _apply_pose_delta(
        self,
        reference_pose: np.ndarray,
        delta: dict
    ) -> np.ndarray:
        """
        将位姿变化量应用到参考位姿
        
        在应用前进行坐标系变换：
        1. 先绕x轴逆时针旋转90度
        2. 再绕z轴逆时针旋转90度
        
        Args:
            reference_pose: 参考位姿 [X, Y, Z, RX, RY, RZ] (米和度)
            delta: 位姿变化量字典，包含 'position' 和 'euler'
            
        Returns:
            目标位姿 [X, Y, Z, RX, RY, RZ] (米和度)
        """
        # 坐标系变换：先绕x轴逆时针旋转90度，再绕z轴逆时针旋转90度
        # R_transform = R_z(90°) * R_x(90°)
        #R_transform = Rotation.from_euler('y', 0, degrees=True)
        
        # 对位置变化量进行坐标系变换
        delta_position_transformed = self.R_gripper_piper_pos @ delta['position']
        
        # 对旋转变化量进行坐标系变换：先转矩阵，再用映射矩阵乘法
        delta_rot_matrix = Rotation.from_euler('xyz', delta['euler'], degrees=True).as_matrix()
        R_transform = self.R_gripper_piper_pos
        delta_rot_transformed_matrix = delta_rot_matrix #R_transform @ delta_rot_matrix @ R_transform.T

        
        # 位置：参考位置 + 变换后的位置变化量
        target_position = reference_pose[:3] + delta_position_transformed

        # 旋转：参考旋转 × 变换后的变化旋转
        # 将参考欧拉角转换为旋转对象
        ref_euler = reference_pose[3:6]  # [RX, RY, RZ] 度
        ref_rot_matrix = Rotation.from_euler('xyz', ref_euler, degrees=True).as_matrix()
        
        # 目标旋转矩阵 = 参考旋转矩阵 × 变换后的相对旋转矩阵
        target_rot_matrix = ref_rot_matrix @ delta_rot_transformed_matrix
        
        # 转换回欧拉角（度）
        target_euler = Rotation.from_matrix(target_rot_matrix).as_euler('xyz', degrees=True)
        
        # 组合目标位姿 [X, Y, Z, RX, RY, RZ]
        target_pose = np.concatenate([target_position, target_euler])
        
        return target_pose
    
    def _log(self, msg: str, level: str = "info"):
        """打印日志"""
        if self.verbose:
            prefix = {
                "info": "[PiperPoseTeleop]",
                "warning": "[PiperPoseTeleop WARNING]",
                "error": "[PiperPoseTeleop ERROR]"
            }.get(level, "[PiperPoseTeleop]")
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
    
    parser = argparse.ArgumentParser(description="Piper 位姿遥操作控制器测试")
    parser.add_argument('--can-port', default='can0', help='CAN 端口')
    parser.add_argument('--encoder-port', default='/dev/ttyUSB1', help='编码器串口')
    parser.add_argument('--freq', type=float, default=50.0, help='控制频率 [Hz]')
    parser.add_argument('--pos-scale', type=float, default=100.0, help='位置缩放因子')
    parser.add_argument('--rot-scale', type=float, default=1.0, help='旋转缩放因子')
    parser.add_argument('--velocity', type=int, default=100, help='机器人运动速度')
    parser.add_argument('--no-gripper', action='store_true', help='不控制夹爪')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Piper 位姿遥操作控制器测试")
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
    
    # 初始化 PiperController
    print("\n初始化 PiperController...")
    robot = PiperController(can_port=args.can_port)
    
    # 使能机器人
    robot.enable()
    
    # 创建控制器
    print("\n创建位姿遥操作控制器...")
    controller = PiperPoseTeleopController(
        action_reader=action_reader,
        piper_robot=robot,
        control_frequency=args.freq,
        position_scale=args.pos_scale,
        rotation_scale=args.rot_scale,
        enable_gripper_control=not args.no_gripper,
        velocity=args.velocity
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
                      f"实际频率: {1.0/stats['avg_loop_time']:.1f}Hz | "
                      f"self._last_delta_pose: {controller._last_delta_rot.tolist()} | ",
                      f"self._last_target_pose: {controller._last_target_pose.tolist()} | ")
    
    except KeyboardInterrupt:
        print("\n\n停止控制...")
    finally:
        controller.stop()
        robot.disable()
        action_reader.disconnect()
        print("程序结束")

