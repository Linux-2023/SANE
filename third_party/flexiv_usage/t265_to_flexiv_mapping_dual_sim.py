#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂 T265 -> Flexiv 映射控制器 (模拟版本)

用于测试碰撞检测效果，不让实机移动。
- 连接实机获取 IK 解算（使用 reachable 接口）
- 使用 MeshCat 可视化双臂姿态
- 实时显示碰撞检测结果
"""

import time
import threading
import numpy as np
import argparse
import yaml
from typing import Optional, Dict, Any, List
import sys
import subprocess

# 自动检查并安装依赖
def install_package(package):
    print(f"正在安装 {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"无法安装 {package}: {e}")
        print(f"请手动运行: pip install {package}")
        sys.exit(1)

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    print("未找到 scipy，尝试安装...")
    install_package("scipy")
    from scipy.spatial.transform import Rotation as R

try:
    from pynput import keyboard
except ImportError:
    print("未找到 pynput，尝试安装...")
    install_package("pynput")
    from pynput import keyboard

try:
    import pyrealsense2 as rs
except ImportError:
    print("未找到 pyrealsense2，尝试安装...")
    install_package("pyrealsense2")
    import pyrealsense2 as rs

import flexivrdk
from flexiv_robot import FlexivRobot
from collision_check.dual_arm_collision_checker import create_checker, DualArmCollisionChecker
from collision_check.dual_arm_visualizer import DualArmVisualizer


def list_t265_devices() -> List[str]:
    """列出所有可用的 T265 设备序列号"""
    ctx = rs.context()
    devices = ctx.query_devices()
    t265_serials = []
    for dev in devices:
        if dev.get_info(rs.camera_info.name) == "Intel RealSense T265":
            serial = dev.get_info(rs.camera_info.serial_number)
            t265_serials.append(serial)
    return t265_serials


class T265Reader:
    """支持指定序列号的 T265 读取器"""
    def __init__(self, serial_number: str = ""):
        self.serial_number = serial_number
        self._pipe = None
        self._cfg = None
        self._connected = False
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        try:
            self._pipe = rs.pipeline()
            self._cfg = rs.config()
            
            if self.serial_number:
                self._cfg.enable_device(self.serial_number)
                
            self._cfg.enable_stream(rs.stream.pose)
            self._pipe.start(self._cfg)
            
            for _ in range(10):
                self._pipe.wait_for_frames()
            
            if not self.serial_number:
                device = self._pipe.get_active_profile().get_device()
                self.serial_number = device.get_info(rs.camera_info.serial_number)
            
            self._connected = True
            print(f"[T265] ✅ 连接成功 (SN: {self.serial_number})")
            return True
        except Exception as e:
            print(f"[T265] ❌ 连接失败: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        if self._pipe:
            try:
                self._pipe.stop()
            except:
                pass
            self._pipe = None
        self._connected = False
    
    def get_pose(self):
        if not self._connected:
            return np.zeros(3), np.array([1, 0, 0, 0])
        
        with self._lock:
            try:
                frames = self._pipe.wait_for_frames()
                pose_frame = frames.get_pose_frame()
                
                if not pose_frame:
                    return np.zeros(3), np.array([1, 0, 0, 0])
                
                data = pose_frame.get_pose_data()
                
                position = np.array([
                    data.translation.x,
                    data.translation.y,
                    data.translation.z
                ])
                
                quaternion = np.array([
                    data.rotation.w,
                    data.rotation.x,
                    data.rotation.y,
                    data.rotation.z
                ])
                
                return position, quaternion
            except Exception as e:
                print(f"[T265] 读取错误: {e}")
                return np.zeros(3), np.array([1, 0, 0, 0])
    
    @property
    def is_connected(self) -> bool:
        return self._connected


class SingleArmSimController:
    """
    单臂模拟控制器
    使用实机进行 IK 解算，但不发送运动命令
    """
    def __init__(
        self,
        arm_name: str,
        t265_reader: T265Reader,
        flexiv_robot: FlexivRobot,
        control_frequency: float = 20.0,
        position_scale: float = 1.5,
        filter_alpha: float = 0.8,
        rot_deadband_deg: float = 1.0,
        rot_fullspeed_deg: float = 8.0,
        rot_alpha_min: float = 0.05,
        pos_deadband_m: float = 0.001,
        pos_fullspeed_m: float = 0.01,
        pos_alpha_min: float = 0.05,
        pos_alpha_max: float = 0.6,
        t265_rot_center_offset_m: Optional[np.ndarray] = None,
        verbose: bool = True,
        is_left_arm: bool = True
    ):
        self.arm_name = arm_name
        self.t265 = t265_reader
        self.robot = flexiv_robot
        self.dt = 1.0 / control_frequency
        self.pos_scale = position_scale
        self.verbose = verbose
        self.is_left_arm = is_left_arm
        
        # 滤波参数
        self.rot_deadband_rad = float(np.deg2rad(rot_deadband_deg))
        self.rot_fullspeed_rad = float(np.deg2rad(rot_fullspeed_deg))
        self.rot_alpha_min = float(rot_alpha_min)
        self.rot_alpha_max = float(filter_alpha)
        self.pos_deadband_m = float(pos_deadband_m)
        self.pos_fullspeed_m = float(pos_fullspeed_m)
        self.pos_alpha_min = float(pos_alpha_min)
        self.pos_alpha_max = float(pos_alpha_max)
        
        # T265 旋转中心偏移
        if t265_rot_center_offset_m is None:
            self.t265_rot_center_offset_m = np.array([0.0, -0.04, 0.0], dtype=float)
        else:
            self.t265_rot_center_offset_m = np.array(t265_rot_center_offset_m, dtype=float).reshape(3)

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # 离合状态
        self.is_following = True
        
        # 引用位姿
        self.t265_ref_pos: Optional[np.ndarray] = None
        self.t265_ref_rot: Optional[R] = None
        self.robot_ref_pos: Optional[np.ndarray] = None
        self.robot_ref_rot: Optional[R] = None
        
        # 滤波状态
        self.last_filtered_pos: Optional[np.ndarray] = None
        self.last_filtered_quat: Optional[np.ndarray] = None
        
        # 人与机器人同向时的旋转矩阵
        self.M_rotvec = np.array([
            [0, -1,  0],
            [-1, 0,  0],
            [0, 0, -1],
        ], dtype=float)
        
        # IK Model
        self._model: Optional[flexivrdk.Model] = None
        
        # 当前关节角度（用于可视化和碰撞检测）
        self._current_joints: Optional[np.ndarray] = None
        self._target_joints: Optional[np.ndarray] = None
        self._joints_lock = threading.Lock()
        
        # IK 状态
        self._ik_reachable = True
    
    def setup_model(self):
        """初始化 Model 用于 IK"""
        try:
            self._model = flexivrdk.Model(self.robot.robot)
            # 初始化当前关节角度
            self._current_joints = np.array(self.robot.get_joint_positions())
            self._target_joints = self._current_joints.copy()
            print(f"[{self.arm_name}] IK Model 已初始化")
        except Exception as e:
            print(f"[{self.arm_name}] ⚠️ 无法创建 Model: {e}")
            self._model = None
    
    def get_current_joints(self) -> np.ndarray:
        """获取当前目标关节角度（线程安全）"""
        with self._joints_lock:
            if self._target_joints is None:
                return np.zeros(7)
            return self._target_joints.copy()
    
    def _solve_ik(self, target_pose: np.ndarray) -> tuple:
        """使用 Model.reachable 求解 IK"""
        if self._model is None:
            return False, None
        
        try:
            seed_joints = list(self.get_current_joints())
            is_reachable, ik_solution = self._model.reachable(
                list(target_pose), 
                seed_joints, 
                False
            )
            
            if is_reachable:
                return True, np.array(ik_solution)
            else:
                return False, None
        except Exception as e:
            if self.verbose:
                print(f"[{self.arm_name}] IK 求解失败: {e}")
            return False, None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[{self.arm_name}] 模拟控制循环已启动")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        print(f"[{self.arm_name}] 模拟控制循环已停止")

    def set_following(self, following: bool):
        was_following = self.is_following
        self.is_following = following
        if following and not was_following:
            self.reset_reference()

    def reset_reference(self):
        with self._lock:
            pos, quat = self.t265.get_pose()
            q = quat
            self.t265_ref_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            self.t265_ref_pos = self._t265_center_pos(pos, self.t265_ref_rot)

            robot_pose = self.robot.get_tcp_pose()
            self.robot_ref_pos = robot_pose[:3].copy()
            rq = robot_pose[3:]
            self.robot_ref_rot = R.from_quat([rq[1], rq[2], rq[3], rq[0]])
            
            self.last_filtered_pos = self.robot_ref_pos.copy()
            self.last_filtered_quat = rq.copy()
            
            if self.verbose:
                print(f"[{self.arm_name}] 参考位姿已重置")

    def _t265_center_pos(self, pos: np.ndarray, rot: R) -> np.ndarray:
        return np.array(pos, dtype=float).reshape(3) + rot.apply(self.t265_rot_center_offset_m)

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _adaptive_alpha(self, err: float, deadband: float, fullspeed: float, alpha_min: float, alpha_max: float) -> float:
        if err <= deadband:
            return 0.0
        if fullspeed <= deadband:
            return alpha_max
        t = (err - deadband) / (fullspeed - deadband)
        k = self._smoothstep01(t)
        return alpha_min + k * (alpha_max - alpha_min)

    def _loop(self):
        self.reset_reference()

        while self._running:
            start_time = time.time()

            if self.is_following:
                try:
                    pos, quat = self.t265.get_pose()
                    q = quat
                    curr_t265_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                    curr_t265_pos = self._t265_center_pos(pos, curr_t265_rot)

                    with self._lock:
                        if self.t265_ref_pos is None or self.robot_ref_pos is None:
                            continue
                        
                        delta_p_t265 = curr_t265_pos - self.t265_ref_pos
                        delta_r_t265 = self.t265_ref_rot.inv() * curr_t265_rot

                        delta_p_robot = np.zeros(3)
                        delta_p_robot[0] = delta_p_t265[0]
                        delta_p_robot[1] = -delta_p_t265[2]
                        delta_p_robot[2] = delta_p_t265[1]
                        delta_p_robot *= self.pos_scale

                        rotvec_t265 = delta_r_t265.as_rotvec()
                        rotvec_tcp = self.M_rotvec @ rotvec_t265
                        delta_r_robot = R.from_rotvec(rotvec_tcp)
                        target_rot = self.robot_ref_rot * delta_r_robot
                        
                        target_pos = self.robot_ref_pos + delta_p_robot
                        target_quat = target_rot.as_quat()
                        target_quat_flexiv = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])

                        # 滤波
                        if self.last_filtered_pos is None:
                            self.last_filtered_pos = target_pos.copy()
                        if self.last_filtered_quat is None:
                            self.last_filtered_quat = target_quat_flexiv.copy()

                        # 位置滤波
                        pos_err = float(np.linalg.norm(target_pos - self.last_filtered_pos))
                        pos_alpha = self._adaptive_alpha(
                            err=pos_err,
                            deadband=self.pos_deadband_m,
                            fullspeed=self.pos_fullspeed_m,
                            alpha_min=self.pos_alpha_min,
                            alpha_max=self.pos_alpha_max,
                        )
                        if pos_alpha <= 0.0:
                            filtered_pos = self.last_filtered_pos
                        else:
                            filtered_pos = self.last_filtered_pos + pos_alpha * (target_pos - self.last_filtered_pos)

                        # 姿态滤波
                        if np.dot(self.last_filtered_quat, target_quat_flexiv) < 0:
                            target_quat_flexiv = -target_quat_flexiv

                        last_r = R.from_quat([
                            self.last_filtered_quat[1], self.last_filtered_quat[2], 
                            self.last_filtered_quat[3], self.last_filtered_quat[0]
                        ])
                        targ_r = R.from_quat([
                            target_quat_flexiv[1], target_quat_flexiv[2], 
                            target_quat_flexiv[3], target_quat_flexiv[0]
                        ])
                        r_err = last_r.inv() * targ_r
                        rot_err = float(np.linalg.norm(r_err.as_rotvec()))

                        rot_alpha = self._adaptive_alpha(
                            err=rot_err,
                            deadband=self.rot_deadband_rad,
                            fullspeed=self.rot_fullspeed_rad,
                            alpha_min=self.rot_alpha_min,
                            alpha_max=self.rot_alpha_max,
                        )
                        if rot_alpha <= 0.0:
                            filtered_quat = self.last_filtered_quat
                        else:
                            filtered_quat = (1.0 - rot_alpha) * self.last_filtered_quat + rot_alpha * target_quat_flexiv
                            n = float(np.linalg.norm(filtered_quat))
                            if n > 1e-6:
                                filtered_quat = filtered_quat / n
                            else:
                                filtered_quat = self.last_filtered_quat

                        self.last_filtered_pos = filtered_pos
                        self.last_filtered_quat = filtered_quat

                        cmd_pose = np.concatenate([filtered_pos, filtered_quat])
                        
                        # 使用 IK 将位姿转换为关节角度（不发送给实机）
                        if self._model is not None:
                            is_reachable, ik_solution = self._solve_ik(cmd_pose)
                            
                            with self._joints_lock:
                                self._ik_reachable = is_reachable
                                if is_reachable:
                                    self._target_joints = ik_solution
                        
                except Exception as e:
                    print(f"[{self.arm_name} Error] {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)


class DualArmSimController:
    """
    双臂模拟控制器
    管理可视化和碰撞检测
    """
    def __init__(
        self, 
        left_arm: SingleArmSimController, 
        right_arm: SingleArmSimController, 
        visualizer: DualArmVisualizer,
        collision_checker: DualArmCollisionChecker,
        collision_safety_threshold: float = 0.0,
        verbose: bool = True
    ):
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.visualizer = visualizer
        self.collision_checker = collision_checker
        self.collision_safety_threshold = collision_safety_threshold
        self.verbose = verbose
        
        self.kb_listener = None
        self._running = False
        self._viz_thread = None
        
        # 碰撞状态
        self._is_collision = False
        self._min_distance = float('inf')
        self._collision_lock = threading.Lock()
        
        # 初始化 MeshCat 碰撞可视化元素
        self._setup_collision_visualization()
    
    def _setup_collision_visualization(self):
        """初始化碰撞可视化元素"""
        import meshcat.geometry as g
        import meshcat.transformations as tf
        
        viewer = self.visualizer.viz.viewer
        
        # 创建状态指示球（放在场景上方）
        # 安全时绿色，碰撞时红色
        self._status_sphere_radius = 0.08
        sphere = g.Sphere(self._status_sphere_radius)
        
        # 初始为绿色（安全）
        self._safe_material = g.MeshLambertMaterial(
            color=0x00ff00,  # 绿色
            opacity=0.8,
            transparent=True
        )
        self._collision_material = g.MeshLambertMaterial(
            color=0xff0000,  # 红色
            opacity=0.8,
            transparent=True
        )
        
        viewer["collision_status/sphere"].set_object(sphere, self._safe_material)
        viewer["collision_status/sphere"].set_transform(
            tf.translation_matrix([0, 0, 1.2])  # 放在场景上方
        )
        
        # 创建状态文字背景板
        # MeshCat 不直接支持文字，用一个平面作为背景
        
        # 创建最近点连线（碰撞时显示）
        # 初始化为不可见的线
        self._line_material_safe = g.LineBasicMaterial(color=0x00ff00, linewidth=3)
        self._line_material_danger = g.LineBasicMaterial(color=0xff0000, linewidth=5)
        
        # 初始线段（两个点）
        initial_points = np.array([[0, 0, 0], [0, 0, 0.001]]).T.astype(np.float32)
        line_geom = g.Line(g.PointsGeometry(initial_points), self._line_material_safe)
        viewer["collision_status/closest_line"].set_object(line_geom)
        
        # 创建最近点标记球
        point_sphere = g.Sphere(0.02)
        point_material_left = g.MeshLambertMaterial(color=0x0000ff, opacity=0.9)  # 蓝色
        point_material_right = g.MeshLambertMaterial(color=0xffff00, opacity=0.9)  # 黄色
        
        viewer["collision_status/point_left"].set_object(point_sphere, point_material_left)
        viewer["collision_status/point_right"].set_object(point_sphere, point_material_right)
        
        # 初始位置（隐藏）
        viewer["collision_status/point_left"].set_transform(tf.translation_matrix([0, 0, -10]))
        viewer["collision_status/point_right"].set_transform(tf.translation_matrix([0, 0, -10]))
        
        print("[可视化] 碰撞状态显示已初始化")
    
    def _update_collision_visualization(self, q_left, q_right, is_collision, min_dist):
        """更新 MeshCat 中的碰撞可视化"""
        import meshcat.geometry as g
        import meshcat.transformations as tf
        
        viewer = self.visualizer.viz.viewer
        
        # 更新状态指示球颜色
        sphere = g.Sphere(self._status_sphere_radius)
        if is_collision:
            viewer["collision_status/sphere"].set_object(sphere, self._collision_material)
        else:
            viewer["collision_status/sphere"].set_object(sphere, self._safe_material)
        
        # 获取最近点并更新连线
        try:
            point_left, point_right, dist = self.collision_checker.get_closest_points(q_left, q_right)
            
            # 更新最近点标记位置
            viewer["collision_status/point_left"].set_transform(
                tf.translation_matrix(point_left)
            )
            viewer["collision_status/point_right"].set_transform(
                tf.translation_matrix(point_right)
            )
            
            # 更新连线
            points = np.array([point_left, point_right]).T.astype(np.float32)
            
            if is_collision:
                line_geom = g.Line(g.PointsGeometry(points), self._line_material_danger)
                viewer["collision_status/closest_line"].set_object(line_geom)
            else:
                line_geom = g.Line(g.PointsGeometry(points), self._line_material_safe)
                viewer["collision_status/closest_line"].set_object(line_geom)
                
        except Exception as e:
            pass  # 忽略获取最近点的错误
        
    def start(self):
        # 启动键盘监听
        self.kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.kb_listener.start()
        
        # 启动各臂控制循环
        self.left_arm.start()
        self.right_arm.start()
        
        # 启动可视化线程
        self._running = True
        self._viz_thread = threading.Thread(target=self._viz_loop, daemon=True)
        self._viz_thread.start()
        
        print("[DualArmSim] 模拟控制器已启动")
    
    def stop(self):
        self._running = False
        if self.kb_listener:
            self.kb_listener.stop()
        if self._viz_thread:
            self._viz_thread.join()
        self.left_arm.stop()
        self.right_arm.stop()
        print("[DualArmSim] 模拟控制器已停止")
    
    def _viz_loop(self):
        """可视化和碰撞检测循环"""
        last_print_time = 0
        print_interval = 0.5  # 每 0.5 秒打印一次状态
        
        while self._running:
            try:
                # 获取两臂的目标关节角度
                q_left = self.left_arm.get_current_joints()
                q_right = self.right_arm.get_current_joints()
                
                # 更新机器人可视化
                self.visualizer.display(q_left, q_right)
                
                # 碰撞检测
                if self.collision_safety_threshold <= 0:
                    is_collision = self.collision_checker.check_collision(q_left, q_right)
                    min_dist = self.collision_checker.compute_min_distance(q_left, q_right)
                else:
                    min_dist = self.collision_checker.compute_min_distance(q_left, q_right)
                    is_collision = min_dist <= self.collision_safety_threshold
                
                with self._collision_lock:
                    self._is_collision = is_collision
                    self._min_distance = min_dist
                
                # 更新碰撞可视化（状态球、最近点连线等）
                self._update_collision_visualization(q_left, q_right, is_collision, min_dist)
                
                # 定期打印状态
                current_time = time.time()
                if current_time - last_print_time > print_interval:
                    self._print_status(q_left, q_right, is_collision, min_dist)
                    last_print_time = current_time
                
            except Exception as e:
                print(f"[DualArmSim] 可视化错误: {e}")
            
            time.sleep(0.02)  # 50Hz
    
    def _print_status(self, q_left, q_right, is_collision, min_dist):
        """打印状态信息"""
        # 清除当前行并打印状态
        left_ik = "✅" if self.left_arm._ik_reachable else "❌"
        right_ik = "✅" if self.right_arm._ik_reachable else "❌"
        
        if is_collision:
            collision_status = "\033[91m⚠️ 碰撞!\033[0m"  # 红色
        else:
            collision_status = "\033[92m✅ 安全\033[0m"  # 绿色
        
        # 总是显示距离
        dist_info = f" | 距离: {min_dist*1000:.1f}mm"
        
        clutch_status = "跟随" if self.left_arm.is_following else "断开"
        
        print(f"\r[状态] 左臂IK:{left_ik} 右臂IK:{right_ik} | {collision_status}{dist_info} | 离合:{clutch_status}    ", end="", flush=True)

    def _on_press(self, key):
        if key == keyboard.Key.space:
            self.left_arm.set_following(False)
            self.right_arm.set_following(False)
            if self.verbose:
                print("\n[Clutch] 双臂断开控制 (Space Pressed)")

    def _on_release(self, key):
        if key == keyboard.Key.space:
            self.left_arm.set_following(True)
            self.right_arm.set_following(True)
            if self.verbose:
                print("\n[Clutch] 双臂恢复跟随 (Space Released)")


def load_dual_arm_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_joint_list_deg(data) -> Optional[np.ndarray]:
    if data is None:
        return None
    if isinstance(data, str):
        s = data.strip()
        if not s:
            return None
        s = s.replace('[', '').replace(']', '')
        parts = [p for p in s.replace(',', ' ').split() if p]
        try:
            vals = [float(p) for p in parts]
            return np.array(vals, dtype=float)
        except:
            return None
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return None


def setup_single_arm_sim(
    arm_name: str,
    arm_config: Dict[str, Any],
    control_config: Dict[str, Any],
    available_t265_serials: List[str],
    used_t265_serials: List[str],
    is_left_arm: bool = True
) -> tuple:
    """设置单臂模拟控制器"""
    robot_sn = arm_config['robot_sn']
    t265_serial = arm_config.get('t265_serial', '')
    init_joints_deg = parse_joint_list_deg(arm_config.get('init_joints_deg'))
    
    print(f"\n{'='*50}")
    print(f"正在初始化 [{arm_name}] (模拟模式)")
    print(f"{'='*50}")
    
    # 1. 连接 T265
    print(f"[{arm_name}] 连接 T265...")
    
    if not t265_serial and available_t265_serials:
        for serial in available_t265_serials:
            if serial not in used_t265_serials:
                t265_serial = serial
                break
    
    t265_reader = T265Reader(serial_number=t265_serial)
    if not t265_reader.connect():
        print(f"[{arm_name}] T265 连接失败!")
        sys.exit(1)
    
    used_t265_serials.append(t265_reader.serial_number)
    
    # 2. 连接机器人（仅用于 IK，不发送运动命令）
    print(f"[{arm_name}] 连接 Flexiv Robot ({robot_sn}) [仅用于 IK]...")
    robot = FlexivRobot(robot_sn)
    
    # 3. 创建模拟控制器
    controller = SingleArmSimController(
        arm_name=arm_name,
        t265_reader=t265_reader,
        flexiv_robot=robot,
        control_frequency=control_config.get('frequency', 20.0),
        position_scale=control_config.get('position_scale', 1.5),
        filter_alpha=control_config.get('alpha', 0.8),
        rot_deadband_deg=control_config.get('rot_deadband_deg', 1.0),
        rot_fullspeed_deg=control_config.get('rot_fullspeed_deg', 8.0),
        rot_alpha_min=control_config.get('rot_alpha_min', 0.05),
        pos_deadband_m=control_config.get('pos_deadband_mm', 1.0) / 1000.0,
        pos_fullspeed_m=control_config.get('pos_fullspeed_mm', 10.0) / 1000.0,
        pos_alpha_min=control_config.get('pos_alpha_min', 0.05),
        pos_alpha_max=control_config.get('pos_alpha_max', 0.6),
        is_left_arm=is_left_arm
    )
    
    # 4. 初始化 IK Model
    controller.setup_model()
    
    return controller, t265_reader, robot


def main():
    parser = argparse.ArgumentParser(description="双臂 T265 -> Flexiv 位姿映射控制 (模拟版本)")
    parser.add_argument('--config', default='dual_arm_config.yaml', help='双臂配置文件路径')
    parser.add_argument('--collision-threshold', type=float, default=None, 
                        help='碰撞安全距离阈值 (米)，0 表示仅检测碰撞')
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置文件: {args.config}")
    try:
        config = load_dual_arm_config(args.config)
    except Exception as e:
        print(f"无法加载配置文件: {e}")
        sys.exit(1)
    
    left_arm_config = config.get('left_arm', {})
    right_arm_config = config.get('right_arm', {})
    control_config = config.get('control', {})
    collision_config = config.get('collision', {})
    
    # 碰撞检测配置
    collision_safety_threshold = args.collision_threshold
    if collision_safety_threshold is None:
        collision_safety_threshold = collision_config.get('safety_threshold', 0.0)
    
    # 列出可用的 T265 设备
    print("\n检测 T265 设备...")
    available_t265 = list_t265_devices()
    print(f"发现 {len(available_t265)} 个 T265: {available_t265}")
    
    if len(available_t265) < 2:
        print("警告: 检测到的 T265 数量少于 2，双臂控制可能无法正常工作!")
    
    used_t265_serials = []
    
    # 创建可视化器
    print("\n初始化可视化器...")
    visualizer = DualArmVisualizer(open_browser=True)
    
    # 创建碰撞检测器
    print("\n初始化碰撞检测器...")
    collision_checker = create_checker(mode="capsule")
    print(f"碰撞检测器已创建，检测对数: {collision_checker.n_collision_pairs}")
    
    # 设置左臂
    left_controller, left_t265, left_robot = setup_single_arm_sim(
        arm_name="左臂",
        arm_config=left_arm_config,
        control_config=control_config,
        available_t265_serials=available_t265,
        used_t265_serials=used_t265_serials,
        is_left_arm=True
    )
    
    # 设置右臂
    right_controller, right_t265, right_robot = setup_single_arm_sim(
        arm_name="右臂",
        arm_config=right_arm_config,
        control_config=control_config,
        available_t265_serials=available_t265,
        used_t265_serials=used_t265_serials,
        is_left_arm=False
    )
    
    # 创建双臂模拟控制器
    dual_controller = DualArmSimController(
        left_controller, 
        right_controller,
        visualizer=visualizer,
        collision_checker=collision_checker,
        collision_safety_threshold=collision_safety_threshold
    )
    dual_controller.start()
    
    print("\n" + "=" * 60)
    print("  双臂模拟遥操作已就绪 (实机不会移动)")
    print("=" * 60)
    print(f"  可视化器 URL: {visualizer.url}")
    print("  [Space] 松开 (默认): 跟随模式 (Follow)")
    print("  [Space] 按住:        断开模式 (Hold)")
    if collision_safety_threshold <= 0:
        print("  碰撞检测: 仅检测碰撞（无安全距离）")
    else:
        print(f"  碰撞检测: 安全距离 {collision_safety_threshold*1000:.0f}mm")
    print("  [Ctrl+C] 退出程序")
    print("=" * 60 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n正在停止...")
    finally:
        dual_controller.stop()
        left_t265.disconnect()
        right_t265.disconnect()
        left_robot.stop()
        right_robot.stop()


if __name__ == "__main__":
    main()

