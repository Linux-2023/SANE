#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T265 -> Flexiv 映射控制器
根据 docs/mapping.md 实现位姿映射，并包含空格键离合功能。
"""

import time
import threading
import numpy as np
import argparse
from typing import Optional
import sys
import subprocess
import math

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

import flexivrdk
from t265_action_reader import ActionReader
from flexiv_robot import FlexivRobot, load_robot_config

class T265MappingController:
    def __init__(
        self,
        action_reader: ActionReader,
        flexiv_robot: FlexivRobot,
        control_frequency: float = 20.0,
        position_scale: float = 1,
        # 兼容参数：作为“姿态最大响应增益”(0~1)，越大越灵敏
        # filter_alpha: float = 0.8,
        filter_alpha: float = 1,
        # 自适应滤波参数（姿态）
        rot_deadband_deg: float = 1.0,
        rot_fullspeed_deg: float = 8.0,
        # rot_alpha_min: float = 0.05,
        rot_alpha_min: float = 1,
        # 自适应滤波参数（位置）
        pos_deadband_m: float = 0.001,
        pos_fullspeed_m: float = 0.01,
        # pos_alpha_min: float = 0.05,
        pos_alpha_min: float = 1,
        # pos_alpha_max: float = 0.6,
        pos_alpha_max: float = 1,
        # T265 旋转中心偏移（在 T265 自身坐标系内，单位 m）
        # 例如旋转中心在 T265 坐标系 y = -0.10m 处，则 offset = [0, -0.10, 0]
        t265_rot_center_offset_m: Optional[np.ndarray] = None,
        gripper_enabled: bool = False,
        verbose: bool = True
    ):
        self.reader = action_reader
        self.robot = flexiv_robot
        self.dt = 1.0 / control_frequency
        self.pos_scale = position_scale
        # 姿态自适应滤波参数
        self.rot_deadband_rad = float(np.deg2rad(rot_deadband_deg))
        self.rot_fullspeed_rad = float(np.deg2rad(rot_fullspeed_deg))
        self.rot_alpha_min = float(rot_alpha_min)
        self.rot_alpha_max = float(filter_alpha)
        # 位置自适应滤波参数
        self.pos_deadband_m = float(pos_deadband_m)
        self.pos_fullspeed_m = float(pos_fullspeed_m)
        self.pos_alpha_min = float(pos_alpha_min)
        self.pos_alpha_max = float(pos_alpha_max)

        self.gripper_enabled = gripper_enabled
        self.verbose = verbose

        # T265 旋转中心偏移（默认 y=-10cm）
        if t265_rot_center_offset_m is None:
            self.t265_rot_center_offset_m = np.array([0.0, -0.04, 0.0], dtype=float)
        else:
            self.t265_rot_center_offset_m = np.array(t265_rot_center_offset_m, dtype=float).reshape(3)

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # 离合状态：默认跟随 (True)，按下空格断开 (False)
        self.is_following = True
        
        # 引用位姿
        self.t265_ref_pos: Optional[np.ndarray] = None
        self.t265_ref_rot: Optional[R] = None
        self.robot_ref_pos: Optional[np.ndarray] = None
        self.robot_ref_rot: Optional[R] = None
        
        # 滤波状态
        self.last_filtered_pos: Optional[np.ndarray] = None
        # 存储上一次滤波后的四元数 [qw, qx, qy, qz]
        self.last_filtered_quat: Optional[np.ndarray] = None
        
        # 夹爪控制状态
        self.last_gripper_move_time = 0.0
        self.gripper_move_interval = 0.2  # 限制夹爪指令频率 (200ms)

        
        # 旋转轴映射矩阵（用于 rotvec 映射）: rotvec_tcp = M_rotvec @ rotvec_t265
        # - T265 绕 X 轴旋转 -> 机器人绕 TCP -Y 轴旋转
        # - T265 绕 Y 轴旋转 -> 机器人绕 TCP +X 轴旋转 
        # - T265 绕 Z 轴旋转 -> 机器人绕 TCP -Z 轴旋转
        #
        # 即：T265_X -> -TCP_Y,  T265_Y -> +TCP_X,  T265_Z -> -TCP_Z
        # 方向相反时，即人对着机械臂正面
        # self.M_rotvec = np.array([
        #     # 修正：T265 绕 Y/Z 轴旋转方向相反 -> 翻转第2/3列符号
        #     [0, 1,  0],
        #     [1, 0,  0],
        #     [0, 0, -1],
        # ], dtype=float)

        # 与机器人同向时的旋转矩阵
        self.M_rotvec = np.array([
            [0, -1,  0],
            [-1, 0,  0],
            [0, 0, -1],
        ], dtype=float)

    def start(self):
        if self._running:
            return
        
        # 启动键盘监听
        self.kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.kb_listener.start()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Mapping] 控制循环已启动")

    def stop(self):
        self._running = False
        if hasattr(self, 'kb_listener') and self.kb_listener:
            self.kb_listener.stop()
        if self._thread:
            self._thread.join()
        print("[Mapping] 控制循环已停止")

    def _on_press(self, key):
        if key == keyboard.Key.space:
            # 按下空格，断开控制
            if self.is_following:
                self.is_following = False
                if self.verbose:
                    print("\n[Clutch] 断开控制 (Space Pressed)")

    def _on_release(self, key):
        if key == keyboard.Key.space:
            # 松开空格，恢复跟随，并重置参考点
            self.is_following = True
            self.reset_reference()
            if self.verbose:
                print("\n[Clutch] 恢复跟随 (Space Released)")

    def reset_reference(self):
        """重置参考位姿，将当前状态设为零点"""
        with self._lock:
            # 获取 T265 当前位姿
            action = self.reader.get_action()
            # [qw, qx, qy, qz] -> [qx, qy, qz, qw] for scipy
            q = action.quaternion
            self.t265_ref_rot = R.from_quat([q[1], q[2], q[3], q[0]])
            # 旋转中心补偿后的参考位置
            self.t265_ref_pos = self._t265_center_pos(action.position, self.t265_ref_rot)

            # 获取 Robot 当前位姿
            robot_pose = self.robot.get_tcp_pose()
            self.robot_ref_pos = robot_pose[:3].copy()
            # [qw, qx, qy, qz] -> [qx, qy, qz, qw]
            rq = robot_pose[3:]
            self.robot_ref_rot = R.from_quat([rq[1], rq[2], rq[3], rq[0]])
            
            # 重置滤波器状态为当前机器人姿态/位置
            self.last_filtered_pos = self.robot_ref_pos.copy()
            self.last_filtered_quat = rq.copy()
            
            if self.verbose:
                print("[Mapping] 参考位姿已重置")

    def _t265_center_pos(self, pos: np.ndarray, rot: R) -> np.ndarray:
        """
        将 T265 报告的“坐标系原点位置”补偿到“旋转中心位置”。
        设偏移向量 o（在 T265 坐标系内）为“原点 -> 旋转中心”，则：
            p_center = p_origin + R * o
        """
        return np.array(pos, dtype=float).reshape(3) + rot.apply(self.t265_rot_center_offset_m)

    @staticmethod
    def _smoothstep01(x: float) -> float:
        """0..1 平滑插值：3x^2 - 2x^3"""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _adaptive_alpha(self, err: float, deadband: float, fullspeed: float, alpha_min: float, alpha_max: float) -> float:
        """
        自适应增益（用于“抖动冻结 + 大动作快跟随”）：
        - err <= deadband: alpha = 0 (完全不动)
        - err >= fullspeed: alpha = alpha_max (快速响应)
        - 中间：alpha 从 alpha_min 平滑过渡到 alpha_max
        """
        if err <= deadband:
            return 0.0
        if fullspeed <= deadband:
            return alpha_max
        t = (err - deadband) / (fullspeed - deadband)
        k = self._smoothstep01(t)
        return alpha_min + k * (alpha_max - alpha_min)

    def _loop(self):
        # 确保初始有参考值
        self.reset_reference()

        while self._running:
            start_time = time.time()

            if self.is_following:
                try:
                    # 1. 获取 T265 当前位姿
                    action = self.reader.get_action()
                    q = action.quaternion
                    curr_t265_rot = R.from_quat([q[1], q[2], q[3], q[0]])
                    # 使用旋转中心补偿后的 T265 位置参与映射
                    curr_t265_pos = self._t265_center_pos(action.position, curr_t265_rot)

                    with self._lock:
                        if self.t265_ref_pos is None or self.robot_ref_pos is None:
                            continue
                        
                        # 2. 计算 T265 增量
                        # 位置增量
                        delta_p_t265 = curr_t265_pos - self.t265_ref_pos
                        
                        # 旋转增量 (Local): R_delta = R_ref_inv * R_curr 
                        # 文档: r_delta_hand = r_t265_start.inv() * r_t265_curr
                        delta_r_t265 = self.t265_ref_rot.inv() * curr_t265_rot

                        # 3. 映射到 Robot 坐标系
                        
                        # A. 位置映射
                        # dX_r = dX_t
                        # dY_r = +dZ_t  
                        # dZ_r = dY_t
                        # 方向相反时，即人对着机械臂正面
                        # delta_p_robot = np.zeros(3)
                        # delta_p_robot[0] = -delta_p_t265[0]
                        # delta_p_robot[1] = delta_p_t265[2]
                        # delta_p_robot[2] = delta_p_t265[1]

                        delta_p_robot = np.zeros(3)
                        delta_p_robot[0] = delta_p_t265[0]
                        delta_p_robot[1] = -delta_p_t265[2]
                        delta_p_robot[2] = delta_p_t265[1]
                        
                        # 应用缩放
                        delta_p_robot *= self.pos_scale

                        # B. 姿态映射
                        # 先求 T265 相对旋转，再把旋转向量 rotvec 做轴映射（可独立翻转每个轴方向）
                        rotvec_t265 = delta_r_t265.as_rotvec()
                        rotvec_tcp = self.M_rotvec @ rotvec_t265
                        delta_r_robot = R.from_rotvec(rotvec_tcp)
                        target_rot = self.robot_ref_rot * delta_r_robot
                        
                        # 4. 计算目标 Robot 位姿
                        # Pos = Ref + Delta
                        target_pos = self.robot_ref_pos + delta_p_robot
                        
                        # 构造发送给 Robot 的指令 [x, y, z, qw, qx, qy, qz]
                        target_quat = target_rot.as_quat() # [qx, qy, qz, qw]
                        # 转换为 Flexiv 格式 [qw, qx, qy, qz]
                        target_quat_flexiv = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])

                        # === 自适应滤波（位置 + 姿态）：小抖动冻结，大变化快速跟随 ===
                        if self.last_filtered_pos is None:
                            self.last_filtered_pos = target_pos.copy()
                        if self.last_filtered_quat is None:
                            self.last_filtered_quat = target_quat_flexiv.copy()

                        # 1) 位置滤波：带死区 + 自适应一阶
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

                        # 2) 姿态滤波：用误差角作为“控制量”，自适应增益
                        # 2.1 连续性处理：防止 q 跳变到 -q（相对上次滤波结果）
                        if np.dot(self.last_filtered_quat, target_quat_flexiv) < 0:
                            target_quat_flexiv = -target_quat_flexiv

                        last_r = R.from_quat([
                            self.last_filtered_quat[1], self.last_filtered_quat[2], self.last_filtered_quat[3], self.last_filtered_quat[0]
                        ])
                        targ_r = R.from_quat([
                            target_quat_flexiv[1], target_quat_flexiv[2], target_quat_flexiv[3], target_quat_flexiv[0]
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
                            # NLERP + 归一化
                            filtered_quat = (1.0 - rot_alpha) * self.last_filtered_quat + rot_alpha * target_quat_flexiv
                            n = float(np.linalg.norm(filtered_quat))
                            if n > 1e-6:
                                filtered_quat = filtered_quat / n
                            else:
                                filtered_quat = self.last_filtered_quat

                        # 3) 更新滤波状态
                        self.last_filtered_pos = filtered_pos
                        self.last_filtered_quat = filtered_quat

                        cmd_pose = np.concatenate([filtered_pos, filtered_quat])

                        # 5. 发送指令
                        self.robot.send_cartesian_pose(cmd_pose)

                        # 6. 控制夹爪 (如果启用)
                        # ActionReader.joint_angle 是校准后的张开角度 (0度=闭合, 正值=张开)
                        # 我们将其线性映射到 夹爪宽度 (0 ~ max_width)
                        if self.gripper_enabled and self.robot.is_gripper_enabled():
                            # 简单的线性映射: 0度 -> min_width, 45度 -> max_width
                            # 假设 45度 是最大张开角度
                            MAX_ANGLE = 45.0
                            
                            angle = action.joint_angle
                            # 限制范围
                            angle = max(0.0, min(MAX_ANGLE, angle))
                            
                            # 获取夹爪参数
                            params = self.robot.get_gripper_params()
                            min_w = params.min_width
                            max_w = params.max_width
                            
                            # 计算目标宽度
                            ratio = angle / MAX_ANGLE
                            target_width = min_w + (max_w - min_w) * ratio
                            
                            # 限制发送频率，避免串口拥堵
                            if time.time() - self.last_gripper_move_time > self.gripper_move_interval:
                                # 使用非阻塞模式移动 (wait=False)
                                self.robot.gripper_move(target_width, velocity=0.1, wait=False)
                                self.last_gripper_move_time = time.time()

                except Exception as e:
                    print(f"[Error] {e}")
            else:
                # 断开控制期间，发送当前位姿以维持 Hold
                try:
                    curr_pose = self.robot.get_tcp_pose()
                    self.robot.send_cartesian_pose(curr_pose)
                except:
                    pass

            # 频率控制
            elapsed = time.time() - start_time
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T265 -> Flexiv 位姿映射控制")
    parser.add_argument('--robot-config', default='robot.yaml', help='机器人配置文件')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='编码器串口 (用于夹爪角度)，如 /dev/ttyUSB0')
    parser.add_argument('--scale', type=float, default=1.5, help='位置映射缩放比例')
    # 滤波参数：
    # --alpha 兼容旧用法：作为“姿态最大响应增益”(0~1)，越大越灵敏
    parser.add_argument('--alpha', type=float, default=0.8, help='姿态最大响应增益(0.0-1.0)，越大越灵敏')
    parser.add_argument('--rot-dead-deg', type=float, default=1.0, help='姿态死区阈值(度)：小于此值目标姿态保持不变')
    parser.add_argument('--rot-full-deg', type=float, default=8.0, help='姿态快速响应阈值(度)：大于此值用最大增益快速跟随')
    parser.add_argument('--rot-alpha-min', type=float, default=0.05, help='姿态最小响应增益(0.0-1.0)')
    parser.add_argument('--pos-dead-mm', type=float, default=1.0, help='位置死区阈值(mm)：小于此值目标位置保持不变')
    parser.add_argument('--pos-full-mm', type=float, default=10.0, help='位置快速响应阈值(mm)：大于此值用最大增益快速跟随')
    parser.add_argument('--pos-alpha-min', type=float, default=0.05, help='位置最小响应增益(0.0-1.0)')
    parser.add_argument('--pos-alpha-max', type=float, default=0.6, help='位置最大响应增益(0.0-1.0)')
    parser.add_argument('--freq', type=float, default=20.0, help='控制频率')
    parser.add_argument('--no-gripper', action='store_true', help='禁用夹爪控制')
    parser.add_argument('--reverse', action='store_true', help='反转编码器方向（张开=减少）')
    parser.add_argument('--angle-scale', type=float, default=2.0, help='编码器角度倍率 (参考 t265_action_reader.py), 默认 2.0')
    parser.add_argument(
        '--init-joints-deg',
        type=str,
        default='2.54, -35.28, -9.12, -79.65, 89.37, 6.04, 43.32',
        help='初始化后移动到目标关节角(度)，用逗号分隔，例如: "0,-30,0,60,0,30,0"；为空则不执行该步骤'
    )
    parser.add_argument(
        '--init-joints-timeout',
        type=float,
        default=60.0,
        help='初始化关节运动超时(s)'
    )
    args = parser.parse_args()

    def _parse_joint_list_deg(text: str) -> Optional[np.ndarray]:
        """解析逗号/空格分隔的关节角(度)字符串，返回 numpy 数组；解析失败返回 None。"""
        s = (text or '').strip()
        if not s:
            return None
        # 支持写成 [a,b,c] 或 a,b,c 或 a b c
        s = s.replace('[', '').replace(']', '')
        # 允许逗号与空格混用
        parts = [p for p in s.replace(',', ' ').split() if p]
        try:
            vals = [float(p) for p in parts]
            return np.array(vals, dtype=float)
        except Exception:
            return None

    # 1. 读取机器人配置
    print("正在连接 Flexiv Robot...")
    try:
        robot_config = load_robot_config(args.robot_config)
        robot_sn = robot_config['robot_sn']
        gripper_name = robot_config.get('gripper_name')
    except Exception:
        print(f"无法读取 {args.robot_config}，请确保文件存在。")
        sys.exit(1)
    
    # 2. 初始化 ActionReader (T265 + 可选编码器)
    want_gripper = (not args.no_gripper) and bool(gripper_name)
    print("正在连接 T265/编码器...")
    direction = -1 if args.reverse else 1
    reader = ActionReader(
        encoder_port=args.port,
        use_t265=True,
        use_encoder=want_gripper,
        angle_direction=direction,
        angle_scale=args.angle_scale,
    )
    reader.connect()
    if not reader.is_connected:
        print("T265/编码器均未连接成功，退出。")
        sys.exit(1)

    # 若需要夹爪但编码器未连上，则禁用夹爪控制
    if want_gripper and not reader.use_encoder:
        print("[Warning] 需要夹爪控制但编码器连接失败，已自动禁用夹爪控制。")
        want_gripper = False

    # 按文档建议：启用编码器时先校准零点（把当前状态定义为 0°）
    if reader.use_encoder:
        print(">>> 即将进行编码器零点校准：请确保夹爪机构处于【闭合/初始位置】")
        time.sleep(2.0)
        reader.calibrate(print_info=True)

    # 3. 初始化机器人
    robot = FlexivRobot(robot_sn)

    # 3.1 初始化后移动到固定目标关节角（如果提供）
    init_joints_deg = _parse_joint_list_deg(args.init_joints_deg)
    if init_joints_deg is not None:
        try:
            dof = robot.get_dof()
        except Exception:
            dof = len(init_joints_deg)

        if len(init_joints_deg) != dof:
            print(f"[Warning] init-joints-deg 长度={len(init_joints_deg)} 与机器人DoF={dof} 不一致，跳过初始化关节运动")
        else:
            target_joints_rad = np.deg2rad(init_joints_deg)
            print(f">>> 初始化关节运动：目标(度)={np.round(init_joints_deg, 3).tolist()}")
            try:
                ok = robot.move_joint(target_joints_rad, timeout=float(args.init_joints_timeout))
                if not ok:
                    print("[Warning] 初始化关节运动未在超时前到达目标（已继续后续流程）")
                else:
                    print(">>> 初始化关节运动完成")
            except Exception as e:
                print(f"[Warning] 初始化关节运动失败：{e}（已继续后续流程）")

    # 初始化夹爪
    gripper_enabled = False
    if want_gripper and gripper_name:
        print(f">>> 正在启用夹爪 [{gripper_name}]...")
        try:
            # 注意: switch_tool=False, 因为后面我们会强制切换到 Flange
            robot.gripper_enable(gripper_name, switch_tool=False)
            gripper_enabled = True
        except Exception as e:
            print(f"[Warning] 启用夹爪失败: {e}")
    
    # 关键步骤：切换工具为 Flange
    print(">>> 将工具坐标系切换为 Flange (TCP = Flange)...")
    try:
        tool = flexivrdk.Tool(robot.robot)
        tool.Switch("Flange")
        time.sleep(1) # 等待生效
        print(">>> 工具已切换为 Flange")
    except Exception as e:
        print(f"[Warning] 切换工具失败: {e}")

    # 3. 启动控制器
    controller = T265MappingController(
        action_reader=reader,
        flexiv_robot=robot,
        control_frequency=args.freq,
        position_scale=args.scale,
        filter_alpha=args.alpha,
        rot_deadband_deg=args.rot_dead_deg,
        rot_fullspeed_deg=args.rot_full_deg,
        rot_alpha_min=args.rot_alpha_min,
        pos_deadband_m=args.pos_dead_mm / 1000.0,
        pos_fullspeed_m=args.pos_full_mm / 1000.0,
        pos_alpha_min=args.pos_alpha_min,
        pos_alpha_max=args.pos_alpha_max,
        gripper_enabled=gripper_enabled
    )
    
    controller.start()
    
    print("\n" + "="*2)
    print("  映射控制已就绪")
    print("  [Space] 松开 (默认): 跟随模式 (Follow)")
    print("  [Space] 按住:        断开模式 (Hold)")
    print("  [Ctrl+C] 退出程序")
    print("="*2 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        controller.stop()
        reader.disconnect()
        robot.stop()
