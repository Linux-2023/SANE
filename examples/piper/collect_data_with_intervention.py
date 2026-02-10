#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
RECAP 人工干预数据采集脚本（基于 Runtime 架构）

基于 human_intervention.md 文档实现的完整数据采集流程，支持：
- 三状态机：AUTO（AI自主）-> PAUSE_ALIGN（暂停对齐）-> INTERVENTION（专家接管）
- 双 CAN 架构：CAN0 控制从臂，CAN1 读取主臂
- 软件偏差补偿（Soft Offset Compensation）
- HDF5 数据录制，包含 is_intervention 标签

本版本使用与 main.py 相同的 Runtime 架构，确保控制行为一致。

硬件要求：
- CAN0: 连接从臂（Slave），PC 发送控制指令
- CAN1: 连接主臂（Master），PC 仅读取关节状态（主臂需设置为示教/拖动模式）

按键说明：
- 's': 开始录制（进入 AUTO 模式）
- SPACE: 暂停（AUTO -> PAUSE_ALIGN）或结束干预（INTERVENTION -> AUTO）
- ENTER: 开始专家接管（PAUSE_ALIGN -> INTERVENTION，会计算偏差补偿）
- 'q': 停止录制并保存数据
- ESC: 退出程序

Usage:
    # AI 模式（需要先启动策略服务器）
    python examples/piper/collect_data_with_intervention.py --prompt "grasp anything"
    
    # 指定策略服务器地址
    python collect_data_with_intervention.py --prompt "pick up the red cube" \\
        --policy_host 192.168.1.100 --policy_port 8000
    
    # 测试模式（无需策略服务器，使用随机动作）
    python collect_data_with_intervention.py --prompt "test task" --test_mode
"""

import logging
import time
import numpy as np
import cv2
import h5py
from datetime import datetime
import os
import argparse
from enum import IntEnum
from typing import Optional, List, Dict, Any
from typing_extensions import override

from piper_sdk import C_PiperInterface_V2
from env import PiperEnvironment

# Runtime 架构相关（与 main.py 一致）
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import subscriber as _subscriber
from openpi_client.runtime.agents import policy_agent as _policy_agent


# ============================================================================
# 常量定义
# ============================================================================
PIPER_GRIPPER_MAX = 0.07  # Piper 夹爪物理最大开口（米）


# ============================================================================
# 控制模式枚举
# ============================================================================
class ControlMode(IntEnum):
    """三状态机的状态定义"""
    AUTO = 0           # AI 自主控制
    PAUSE_ALIGN = 1    # 暂停，等待操作员对齐主臂
    INTERVENTION = 2   # 专家接管（人类控制）


# ============================================================================
# 主臂读取器
# ============================================================================
class MasterArmReader:
    """
    用于读取主臂（Master Arm）关节状态的类。
    主臂通过 CAN1 连接，PC 仅读取其状态，不发送控制指令。
    主臂应设置为示教/拖动模式，以便操作员可以轻松移动它。
    """
    
    def __init__(self, can_port: str = "can1"):
        """
        初始化主臂读取器。
        
        Args:
            can_port: 主臂连接的 CAN 端口，默认为 "can1"
        """
        self.can_port = can_port
        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()
        print(f"[MasterArmReader] 已连接主臂，CAN 端口: {can_port}")
        
        # 单位换算系数（与 PiperController 保持一致）
        self._joint_factor = 57295.7795  # 1000 * 180 / pi
        self._gripper_factor = 1000 * 1000  # m -> μm
    
    def get_joint_states(self) -> List[float]:
        """获取主臂的 6 个关节角度（弧度）。"""
        js_msg = self.piper.GetArmJointCtrl()
        js = js_msg.joint_ctrl
        
        if js.joint_1 == 0 and js.joint_2 == 0:
            js_msg = self.piper.GetArmJointMsgs()
            js = js_msg.joint_state

        return [
            js.joint_1 / self._joint_factor,
            js.joint_2 / self._joint_factor,
            js.joint_3 / self._joint_factor,
            js.joint_4 / self._joint_factor,
            js.joint_5 / self._joint_factor,
            js.joint_6 / self._joint_factor,
        ]
    
    def get_gripper_state(self) -> float:
        """获取主臂夹爪开口宽度（米）。"""
        gs_msg = self.piper.GetArmGripperCtrl()
        gs = gs_msg.gripper_ctrl
        
        if gs.grippers_angle == 0:
            gs_msg = self.piper.GetArmGripperMsgs()
            gs = gs_msg.gripper_state

        return gs.grippers_angle / self._gripper_factor
    
    def get_full_state(self) -> List[float]:
        """获取主臂完整状态（6 关节 + 夹爪）。"""
        return self.get_joint_states() + [self.get_gripper_state()]


# ============================================================================
# 干预 Agent（继承自 Agent，支持模式切换）
# ============================================================================
class InterventionAgent(_agent.Agent):
    """
    支持人工干预的 Agent。
    
    内部包含一个 PolicyAgent 用于 AUTO 模式，同时支持 INTERVENTION 模式
    下从主臂读取动作。状态机由外部 Runtime 控制。
    """
    
    def __init__(
        self,
        policy_agent: _policy_agent.PolicyAgent,
        master_reader: MasterArmReader,
        max_offset: float = 0.5,
        initial_recording: bool = False,
    ):
        """
        初始化干预 Agent。
        
        Args:
            policy_agent: 用于 AUTO 模式的策略 Agent
            master_reader: 主臂读取器
            max_offset: 允许的最大主从偏差（弧度）
            initial_recording: 初始录制状态
        """
        self._policy_agent = policy_agent
        self._master_reader = master_reader
        self._max_offset = max_offset
        self._initial_recording = initial_recording
        
        # 状态机
        self._mode = ControlMode.AUTO
        self._recording = initial_recording
        
        # 记录模式切换时间，用于防止切回 AUTO 时立即触发静止结束
        self._last_mode_switch_time = time.time()
        
        # 偏差补偿
        self._joint_offset: Optional[np.ndarray] = None
        self._gripper_scale: float = 1.0
        self._use_gripper_mapping: bool = False
        
        # 最新观测（用于生成保持动作）
        self._last_obs: Optional[dict] = None
        
        # 标记当前步是否为干预
        self._is_intervention: bool = False
        
        # 干预模式下的初始动作和移动检测标志
        self._initial_intervention_action: Optional[np.ndarray] = None
        self._intervention_started: bool = False
    
    @property
    def mode(self) -> ControlMode:
        return self._mode
    
    @property
    def last_mode_switch_time(self) -> float:
        return self._last_mode_switch_time
    
    @property
    def recording(self) -> bool:
        return self._recording
    
    @property
    def is_intervention(self) -> bool:
        return self._is_intervention
    
    @property
    def joint_offset(self) -> Optional[np.ndarray]:
        return self._joint_offset
    
    def start_recording(self) -> None:
        """开始录制，进入 AUTO 模式。"""
        self._recording = True
        self._mode = ControlMode.AUTO
        self._last_mode_switch_time = time.time()
        print("\n" + "="*40)
        print(">>> 🔴 开始录制（AI 控制模式）")
        print("="*40)
    
    def stop_recording(self) -> None:
        """停止录制。"""
        self._recording = False
        self._mode = ControlMode.AUTO
        self._last_mode_switch_time = time.time()
        self._joint_offset = None
        self._gripper_scale = 1.0
        self._use_gripper_mapping = False
    
    def pause_for_alignment(self) -> None:
        """暂停，等待操作员对齐主臂。"""
        if self._mode == ControlMode.AUTO:
            print("\n\n")
            print(">>> 暂停！请将主臂移动到与从臂相似的姿态，然后按 ENTER 接管")
            self._mode = ControlMode.PAUSE_ALIGN
            self._last_mode_switch_time = time.time()
            self._joint_offset = None
            self._gripper_scale = 1.0
            self._use_gripper_mapping = False
    
    def resume_auto(self) -> None:
        """从暂停恢复 AUTO 模式，不重置策略。"""
        if self._mode == ControlMode.PAUSE_ALIGN:
            print("\n")
            print(">>> 恢复 AI 模式")
            self._mode = ControlMode.AUTO
            self._last_mode_switch_time = time.time()
            self._joint_offset = None
            self._gripper_scale = 1.0
            self._use_gripper_mapping = False

    def end_intervention(self) -> None:
        """结束干预，切回 AUTO 模式。"""
        if self._mode == ControlMode.INTERVENTION:
            print("\n")
            print(">>> 结束干预，切回 AI 模式")
            self._mode = ControlMode.AUTO
            self._last_mode_switch_time = time.time()
            self._joint_offset = None
            self._gripper_scale = 1.0
            self._use_gripper_mapping = False
            # 修改：结束干预后重置策略，确保重新开始生成动作，而不是使用旧的动作块
            self._policy_agent.reset()
    
    def try_takeover(self, slave_state: np.ndarray) -> bool:
        """
        尝试接管控制（计算偏差补偿）。
        
        Args:
            slave_state: 从臂当前状态 [j1, j2, j3, j4, j5, j6, gripper]
        
        Returns:
            是否成功接管
        """
        if self._mode != ControlMode.PAUSE_ALIGN:
            return False
        
        print("\n\n")
        
        # 读取主臂状态（物理单位：米）
        master_state = self._master_reader.get_full_state()
        master_q = np.array(master_state[:6])
        master_gripper = master_state[6]
        
        slave_q = slave_state[:6]
        # 修改：将归一化的 slave_gripper 转换为物理原始值（米）进行比较
        slave_gripper_phys = slave_state[6] * PIPER_GRIPPER_MAX
        
        # 计算关节偏差
        joint_offset = slave_q - master_q
        
        # 修改：计算夹爪偏差（使用物理单位）
        gripper_diff = abs(slave_gripper_phys - master_gripper)
        
        # 修改：计算夹爪映射（使用物理单位）
        if master_gripper > 0.001:
            gripper_scale = slave_gripper_phys / master_gripper
            use_gripper_mapping = True
            print(f"    夹爪映射已启用: k = {gripper_scale:.3f} (Master {master_gripper:.3f}m -> Slave {slave_gripper_phys:.3f}m)")
        else:
            gripper_scale = 1.0
            use_gripper_mapping = False
            print(f"    夹爪映射未启用 (主臂接近 0)，恢复正常映射")
        
        # 检查偏差是否过大
        max_diff = np.max(np.abs(joint_offset))
        if max_diff < self._max_offset and gripper_diff < 0.01:
            print(f"\n>>> 接管成功！偏差补偿已应用 (最大偏差: {max_diff:.3f} rad, 夹爪误差: {gripper_diff:.3f} m)")
            print(f"    Offset: {joint_offset}")
            self._joint_offset = joint_offset
            self._gripper_scale = gripper_scale
            self._use_gripper_mapping = use_gripper_mapping
            self._mode = ControlMode.INTERVENTION
            self._last_mode_switch_time = time.time()
            
            # 初始化移动检测
            self._initial_intervention_action = None
            self._intervention_started = False
            
            return True
        elif gripper_diff >= 0.01:
            print(f"\n!!! 夹爪误差过大 (当前误差: {gripper_diff:.3f} m > 0.01 m)")
            print(f"    请调整主臂夹爪，使其开口宽度与从臂接近")
            return False
        else:
            print(f"\n!!! 构型差异过大 (最大偏差: {max_diff:.3f} rad > {self._max_offset})")
            print(f"    请继续调整主臂姿态，使其更接近从臂")
            return False

    def print_alignment_status(self, slave_state: np.ndarray) -> None:
        """在 PAUSE_ALIGN 模式下打印对齐状态。"""
        if self._mode != ControlMode.PAUSE_ALIGN:
            return
        
        master_state = self._master_reader.get_full_state()
        master_q = np.array(master_state[:6])
        master_gripper = master_state[6]
        
        slave_q = slave_state[:6]
        # 修改：显示物理原始值
        slave_gripper_phys = slave_state[6] * PIPER_GRIPPER_MAX
        
        abs_diffs = np.abs(slave_q - master_q)
        max_diff = np.max(abs_diffs)
        max_idx = np.argmax(abs_diffs)
        
        diff_color = "\033[92m" if max_diff < self._max_offset else "\033[91m"
        reset_color = "\033[0m"
        highlight = "\033[1;37;41m"
        
        s_list = []
        m_list = []
        for i in range(6):
            s_val = f"{slave_q[i]:5.2f}"
            m_val = f"{master_q[i]:5.2f}"
            if i == max_idx:
                s_list.append(f"{highlight}{s_val}{reset_color}")
                m_list.append(f"{highlight}{m_val}{reset_color}")
            else:
                s_list.append(s_val)
                m_list.append(m_val)
        
        s_str = " ".join(s_list)
        m_str = " ".join(m_list)
        
        print(f"\r\033[K对齐中 | {diff_color}MaxDiff: {max_diff:.3f}{reset_color} (关节 {max_idx+1} 差异最大)")
        print(f"\r\033[K  Slave : [{s_str}] G:{slave_gripper_phys:.3f}m")
        print(f"\r\033[K  Master: [{m_str}] G:{master_gripper:.3f}m", end="", flush=True)
        print("\033[2A", end="")
    
    @override
    def get_action(self, observation: dict) -> dict:
        """
        根据当前模式获取动作。
        
        Args:
            observation: 观测字典
        
        Returns:
            动作字典，包含 'actions' 和 'is_intervention' 键
        """
        self._last_obs = observation
        state = observation['state']
        
        if self._mode == ControlMode.AUTO:
            if self._recording:
                # AI 控制 (只要 recording 为 True，无论是否开启 record_mode 都会执行 AI 控制)
                action_dict = self._policy_agent.get_action(observation)
                self._is_intervention = False
            else:
                # IDLE：保持不动
                action_dict = {'actions': np.array(state)}
                self._is_intervention = False
        
        elif self._mode == ControlMode.PAUSE_ALIGN:
            # 暂停：保持当前位置
            action_dict = {'actions': np.array(state)}
            self._is_intervention = False
        
        elif self._mode == ControlMode.INTERVENTION:
            # 人类控制（带偏差补偿）
            master_state = self._master_reader.get_full_state()
            master_q = np.array(master_state[:6])
            master_gripper = master_state[6]
            
            # 检测主臂夹爪归零
            if self._use_gripper_mapping and master_gripper < 0.002:
                print(">>> 主臂夹爪归零，恢复恒等映射 (Scale = 1.0)")
                self._gripper_scale = 1.0
                self._use_gripper_mapping = False
            
            # 应用关节偏差补偿
            target_joints = master_q + self._joint_offset
            
            # 应用夹爪映射
            target_gripper_phys = master_gripper * self._gripper_scale
            # 夹爪物理限幅（米）
            target_gripper_phys = np.clip(target_gripper_phys, 0.0, PIPER_GRIPPER_MAX)
            
            # 修改：将目标物理行程重新归一化（0-1），因为 apply_action 期望归一化值
            target_gripper_norm = target_gripper_phys / PIPER_GRIPPER_MAX
            
            target_action = np.concatenate([target_joints, [target_gripper_norm]])
            
            # 移动检测逻辑：
            # 如果是干预开始初期，记录初始动作。
            # 只有当当前动作与初始动作差异超过阈值时，才标记为已开始移动 (_intervention_started=True)。
            if not self._intervention_started:
                if self._initial_intervention_action is None:
                    self._initial_intervention_action = target_action.copy()
                
                # 分别计算关节和夹爪的差异
                action_diff = np.abs(target_action - self._initial_intervention_action)
                joint_max_diff = np.max(action_diff[:6])
                # 注意：target_action 中的夹爪已经是归一化后的值 (0-1)
                # 0.001m 对应的归一化阈值为 0.001 / 0.07 ≈ 0.014
                # 但为了直观和统一，我们在这里将其还原为物理单位进行判断，或者直接使用归一化阈值
                # 这里我们使用归一化后的比较： 0.001m / 0.07m ≈ 0.0142
                gripper_diff = action_diff[6]
                
                # 阈值设定：关节 > 0.01 rad (约0.57度)，夹爪 > 0.001m (约 0.0142 归一化值)
                gripper_threshold_norm = 0.001 / PIPER_GRIPPER_MAX
                
                if joint_max_diff > 0.01 or gripper_diff > gripper_threshold_norm:
                    self._intervention_started = True
                    print(f">>> 检测到主臂移动 (JointMax: {joint_max_diff:.4f}, Gripper: {gripper_diff:.4f})，开始录制干预数据")
            
            action_dict = {'actions': target_action}
            self._is_intervention = True
            
            # 如果干预尚未真正开始（未移动），添加 skip_recording 标志通知 Saver
            if not self._intervention_started:
                action_dict['skip_recording'] = True
        
        else:
            action_dict = {'actions': np.array(state)}
            self._is_intervention = False
        
        # 添加 is_intervention 标签到返回值
        action_dict['is_intervention'] = self._is_intervention
        return action_dict
    
    @override
    def reset(self) -> None:
        """重置 Agent 状态。"""
        self._policy_agent.reset()
        self._mode = ControlMode.AUTO
        self._recording = self._initial_recording
        self._last_mode_switch_time = time.time()
        self._joint_offset = None
        self._gripper_scale = 1.0
        self._use_gripper_mapping = False
        self._is_intervention = False


# ============================================================================
# 干预数据记录器（继承自 Subscriber）
# ============================================================================
class InterventionDataSaver(_subscriber.Subscriber):
    """
    记录包含 is_intervention 标签的数据。
    """
    
    def __init__(self, output_dir: str, prompt: str, agent: InterventionAgent, record_mode: bool = False):
        """
        初始化数据记录器。
        
        Args:
            output_dir: 保存目录
            prompt: 任务描述
            agent: InterventionAgent 实例，用于获取 is_intervention 状态
            record_mode: 是否启用录制模式。若为 False，则不保存任何数据。
        """
        self._output_dir = output_dir
        self._prompt = prompt
        self._agent = agent
        self._record_mode = record_mode
        self._episode_data: List[Dict[str, Any]] = []
    
    @override
    def on_episode_start(self) -> None:
        """Episode 开始时清空数据。"""
        self._episode_data = []
    
    @override
    def on_step(self, observation: dict, action: dict) -> None:
        """记录每一步的数据。"""
        # 只有在 record_mode 开启且 agent 正在录制时才记录
        if not self._record_mode or not self._agent.recording:
            return
        
        # 在 PAUSE_ALIGN 模式下不录制
        if self._agent.mode == ControlMode.PAUSE_ALIGN:
            return
        
        # 检查是否跳过录制（例如干预初期未移动）
        if action.get('skip_recording', False):
            return
        
        # 提取动作
        actions = action.get('actions', observation['state'])
        if hasattr(actions, 'tolist'):
            actions = actions.tolist()
        
        frame = {
            'obs': {
                'state': observation['state'].copy(),
                'images': {k: v.copy() for k, v in observation['images'].items()},
                'prompt': observation.get('prompt', ''),
            },
            'action': actions,
            'is_intervention': action.get('is_intervention', False),
            'timestamp': time.time(),
        }
        self._episode_data.append(frame)
    
    @override
    def on_episode_end(self) -> None:
        """Episode 结束时保存数据。"""
        if self._record_mode:
            self._save_to_hdf5()
    
    def save_now(self) -> None:
        """立即保存当前数据（用于中途保存）。"""
        if self._record_mode:
            self._save_to_hdf5()
            self._episode_data = []
    
    @property
    def step_count(self) -> int:
        """返回已记录的步数。"""
        return len(self._episode_data)
    
    def prune_static_intervention_frames(self) -> None:
        """
        剪枝：结束人工干预时触发。删除掉切换回AI控制前的静止帧（结束遥操作到按下space之间的时间）。
        
        仅检查带有 is_intervention=True 的帧。
        一旦遇到动作变化（与最后一帧不一致）或非干预帧，即停止。
        """
        if not self._episode_data:
            return
            
        print("正在检查并修剪干预结束时的静止帧...")
        
        # 从倒数第二帧开始检查，对比其与最后一帧的动作
        
        pruned_count = 0
        while len(self._episode_data) >= 2:
            last_frame = self._episode_data[-1]
            prev_frame = self._episode_data[-2]
            
            # 如果最后一帧不是干预帧，说明干预早已结束（或者混入了 AUTO 帧），停止剪枝
            # 注意：这里的逻辑是“退出干预进入 AUTO 时”立即调用，所以最后几帧应该是干预帧
            if not last_frame['is_intervention']:
                break
            
            # 对比动作
            last_action = np.array(last_frame['action'])
            prev_action = np.array(prev_frame['action'])
            
            # 分别计算关节和夹爪的差异
            action_diff = np.abs(last_action - prev_action)
            joint_max_diff = np.max(action_diff[:6])
            gripper_diff = action_diff[6]
            
            # 阈值设定：关节 < 0.01 rad，夹爪 < 0.001m (归一化后约 0.014)
            # 只有当两者都小于阈值时，才认为动作一致（静止）
            gripper_threshold_norm = 0.001 / PIPER_GRIPPER_MAX
            
            if joint_max_diff < 0.005 and gripper_diff < gripper_threshold_norm:
                # 删除上一帧
                self._episode_data.pop(-2)
                pruned_count += 1
            else:
                # 动作不一致，停止
                break
        
        if pruned_count > 0:
            print(f"  - 已修剪 {pruned_count} 帧静止数据")
    
    def _save_to_hdf5(self) -> None:
        """将数据保存为 HDF5 文件。"""
        if not self._episode_data:
            if self._record_mode:
                print("没有数据需要保存。")
            return
        
        os.makedirs(self._output_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self._prompt:
            safe_prompt = self._prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
            safe_prompt = safe_prompt.replace(",", "").replace(".", "")[:80]
            filename = os.path.join(self._output_dir, f"episode_{timestamp_str}_{safe_prompt}.hdf5")
        else:
            filename = os.path.join(self._output_dir, f"episode_{timestamp_str}.hdf5")
        
        print(f"正在保存数据到 {filename}...")
        
        intervention_count = sum(1 for frame in self._episode_data if frame['is_intervention'])
        total_count = len(self._episode_data)
        print(f"  - 总帧数: {total_count}")
        print(f"  - 干预帧数: {intervention_count} ({100*intervention_count/total_count:.1f}%)")
        
        with h5py.File(filename, "w") as f:
            first_frame = self._episode_data[0]
            num_steps = len(self._episode_data)
            
            obs_group = f.create_group("observations")
            img_group = obs_group.create_group("images")
            
            state_shape = first_frame['obs']['state'].shape
            obs_group.create_dataset("qpos", (num_steps,) + state_shape, 
                                      dtype=first_frame['obs']['state'].dtype)
            
            action_shape = (len(first_frame['action']),)
            f.create_dataset("action", (num_steps,) + action_shape, dtype=np.float32)
            f.create_dataset("is_intervention", (num_steps,), dtype=np.uint8)
            
            for cam_name, img in first_frame['obs']['images'].items():
                img_group.create_dataset(cam_name, (num_steps,) + img.shape, dtype=img.dtype)
            
            if self._prompt:
                prompt_bytes = self._prompt.encode('utf-8')
                f.create_dataset("task", (num_steps,), dtype=f'S{len(prompt_bytes)}')
            
            qpos_ds: Any = obs_group["qpos"]
            action_ds: Any = f["action"]
            is_intervention_ds: Any = f["is_intervention"]
            task_ds: Any = f["task"] if self._prompt else None
            
            for i, frame in enumerate(self._episode_data):
                qpos_ds[i] = frame['obs']['state']
                action_ds[i] = np.array(frame['action'], dtype=np.float32)
                is_intervention_ds[i] = 1 if frame['is_intervention'] else 0
                
                for cam_name, img in frame['obs']['images'].items():
                    img_ds: Any = img_group[cam_name]
                    img_ds[i] = img
                
                if self._prompt:
                    task_ds[i] = prompt_bytes
        
        print(f"✅ 成功保存 {num_steps} 帧数据。")


# ============================================================================
# 干预 Runtime（继承自 Runtime，支持键盘输入和 UI）
# ============================================================================
class InterventionRuntime(_runtime.Runtime):
    """
    支持人工干预的 Runtime。
    
    继承自标准 Runtime，重写 _step() 和 _run_episode() 来处理键盘输入和 UI 显示。
    """
    
    def __init__(
        self,
        environment: PiperEnvironment,
        agent: InterventionAgent,
        subscribers: list[_subscriber.Subscriber],
        data_saver: InterventionDataSaver,
        max_hz: float = 30,
        num_episodes: int = 1,
    ):
        super().__init__(
            environment=environment,
            agent=agent,
            subscribers=subscribers,
            max_hz=max_hz,
            num_episodes=num_episodes,
        )
        self._intervention_agent = agent
        self._data_saver = data_saver
        self._should_exit = False
        self._window_name = "RECAP Data Collector"
    
    def _display_observations(self, obs: dict) -> int:
        """显示相机图像和状态信息，返回按键值。"""
        images_dict = obs.get('images', {})
        if not images_dict:
            return cv2.waitKey(1) & 0xFF
        
        display_images = []
        mode = self._intervention_agent.mode
        recording = self._intervention_agent.recording
        
        for cam_name, img in images_dict.items():
            img_hwc = np.transpose(img, (1, 2, 0))
            img_bgr = img_hwc[:, :, ::-1].copy()
            
            if mode == ControlMode.AUTO:
                label_color = (0, 255, 0)
                mode_text = "AUTO (AI)"
            elif mode == ControlMode.PAUSE_ALIGN:
                label_color = (0, 165, 255)
                mode_text = "PAUSE (manually aligning)"
            else:
                label_color = (0, 0, 255)
                mode_text = "INTERVENTION (human)"
            
            cv2.putText(img_bgr, cam_name, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            display_images.append(img_bgr)
        
        combined = np.hstack(display_images) if len(display_images) > 1 else display_images[0]
        
        panel_height = 120
        panel = np.zeros((panel_height, combined.shape[1], 3), dtype=np.uint8)
        
        y_offset = 25
        if mode == ControlMode.PAUSE_ALIGN:
            rec_status = ""
            rec_color = (0, 0, 0)
        else:
            rec_status = "REC" if recording else "IDLE"
            rec_color = (0, 0, 255) if recording else (128, 128, 128)
        
        cv2.putText(panel, f"Mode: {mode_text}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
        cv2.putText(panel, rec_status, (300, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)
        cv2.putText(panel, f"Steps: {self._data_saver.step_count}", (400, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        prompt = obs.get('prompt', '')
        y_offset += 30
        cv2.putText(panel, f"Task: {prompt[:60]}{'...' if len(prompt) > 60 else ''}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        state = obs.get('state', [])
        y_offset += 25
        # 修改：在 UI 显示中也将归一化的夹爪值转回物理单位（米）
        state_str = "State: " + " ".join([f"{x:.2f}" for x in state[:6]])
        if len(state) >= 7:
            state_str += f" | G:{state[6] * PIPER_GRIPPER_MAX:.3f}m"
        
        cv2.putText(panel, state_str, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        joint_offset = self._intervention_agent.joint_offset
        if joint_offset is not None and mode == ControlMode.INTERVENTION:
            y_offset += 20
            offset_str = "Offset: " + " ".join([f"{x:.2f}" for x in joint_offset])
            cv2.putText(panel, offset_str, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
        
        y_offset += 20
        if not recording:
            help_text = "[s] Start | [ESC] Quit"
        elif mode == ControlMode.AUTO:
            help_text = "[SPACE] Pause | [q] Save & Stop"
        elif mode == ControlMode.PAUSE_ALIGN:
            help_text = "[ENTER] Takeover | [q] Save & Stop"
        else:
            help_text = "[SPACE] End Intervention | [q] Save & Stop"
        cv2.putText(panel, help_text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        
        final = np.vstack([combined, panel])
        cv2.imshow(self._window_name, final)
        
        return cv2.waitKey(1) & 0xFF
    
    def _handle_key(self, key: int, state: np.ndarray) -> None:
        """处理键盘输入。"""
        if key == 255:
            return
        
        if key == 27:  # ESC
            print("\n>>> 退出程序")
            self._should_exit = True
            self.mark_episode_complete()
        
        elif key == ord('s') or key == ord('S'):
            if not self._intervention_agent.recording:
                self._intervention_agent.start_recording()
        
        elif key == ord('q') or key == ord('Q'):
            if self._intervention_agent.recording:
                print("\n>>> 停止录制，保存数据...")
                self._intervention_agent.stop_recording()
                self._data_saver.save_now()
        
        elif key == 32:  # SPACE
            if self._intervention_agent.mode == ControlMode.AUTO:
                self._intervention_agent.pause_for_alignment()
            elif self._intervention_agent.mode == ControlMode.PAUSE_ALIGN:
                self._intervention_agent.resume_auto()
            elif self._intervention_agent.mode == ControlMode.INTERVENTION:
                self._intervention_agent.end_intervention()
                # 触发 Saver 进行静止帧剪枝
                self._data_saver.prune_static_intervention_frames()
        
        elif key == 13:  # ENTER
            if self._intervention_agent.mode == ControlMode.PAUSE_ALIGN:
                self._intervention_agent.try_takeover(state)
    
    @override
    def _step(self) -> None:
        """重写单步循环，添加键盘处理和 UI 显示。"""
        observation = self._environment.get_observation()
        state = np.array(observation['state'])
        
        # 显示 UI 并获取按键
        key = self._display_observations(observation)
        if key == 255:
            key = cv2.waitKey(1) & 0xFF
        
        # 处理键盘输入
        self._handle_key(key, state)
        
        # 在 PAUSE_ALIGN 模式下显示对齐状态
        if self._intervention_agent.mode == ControlMode.PAUSE_ALIGN:
            self._intervention_agent.print_alignment_status(state)
        
        # 获取动作
        action = self._intervention_agent.get_action(observation)
        
        # 执行动作
        self._environment.apply_action(action)
        
        # 通知订阅者
        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)
        
        # 检查是否完成
        # 注意：仅在AI控制模式且正在录制时，响应环境的自动静止检测（is_episode_complete）
        # 增加条件：距离上次模式切换超过 2 秒，防止人工干预结束切回 AUTO 时误触发
        time_since_switch = time.time() - self._intervention_agent.last_mode_switch_time
        
        if self._should_exit:
            self.mark_episode_complete()
        elif self._intervention_agent.mode == ControlMode.AUTO and \
             self._intervention_agent.recording and \
             time_since_switch > 2.0 and \
             self._environment.is_episode_complete():
            print("\n>>> [AUTO] 检测到任务可能已完成，自动结束当前 Episode")
            self.mark_episode_complete()
    
    @override
    def _run_episode(self) -> None:
        """重写 episode 运行逻辑。"""
        logging.info("Starting episode...")
        self._environment.reset()
        
        # 预热相机：无论是否手动开始，都先读取并丢弃一些帧，确保图像稳定（避免首帧偏绿）
        # 读取 30 帧或等待 1 秒
        print("📷 正在预热相机...")
        for _ in range(30):
            self._environment.get_observation()
            time.sleep(1/30) # 模拟 30FPS
            
        self._intervention_agent.reset()
        
        for subscriber in self._subscribers:
            subscriber.on_episode_start()
        
        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()
        
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        
        print("\n" + "=" * 60)
        print("  准备就绪！")
        print("  按 's' 开始录制（AI 控制），按 SPACE 暂停并准备接管")
        print("=" * 60 + "\n")
        
        while self._in_episode and not self._should_exit:
            self._step()
            
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
            
            self._episode_steps += 1
        
        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()
        
        cv2.destroyAllWindows()
    
    @override
    def run(self) -> None:
        """运行 Runtime。"""
        try:
            for _ in range(self._num_episodes):
                if self._should_exit:
                    break
                self._run_episode()
        except KeyboardInterrupt:
            print("\n\n>>> 用户中断")
            if self._intervention_agent.recording:
                print("正在保存已录制的数据...")
                self._data_saver.save_now()
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()
            if self._intervention_agent.recording:
                print("尝试保存已录制的数据...")
                self._data_saver.save_now()
        finally:
            print("\n正在清理资源...")
            cv2.destroyAllWindows()
            try:
                self._environment.close()
            except Exception:
                pass
            print("程序结束。")


# ============================================================================
# 测试策略（随机动作）
# ============================================================================
class TestPolicy:
    """测试策略：从预定义的动作集中随机选择动作。"""
    
    DEFAULT_ACTION_SET = [
        [0.0, -1.5, 1.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.23, -0.59, -0.04, 0.71, 0.028, 0.05],
        [0.0, 0.8, -0.89, -0.08, 0.24, 0.07, 0.025],
    ]
    
    def __init__(
        self,
        action_set: Optional[List[List[float]]] = None,
        hold_steps: int = 50,
        add_noise: bool = False,
        noise_scale: float = 0.01,
    ):
        import random
        self._random = random
        
        self._action_set = action_set if action_set is not None else self.DEFAULT_ACTION_SET
        self._hold_steps = hold_steps
        self._add_noise = add_noise
        self._noise_scale = noise_scale
        
        self._current_action_idx = 0
        self._steps_since_change = 0
        self._current_action = self._action_set[0]
        
        print(f"[TestPolicy] 初始化测试策略")
        print(f"[TestPolicy]   动作集大小: {len(self._action_set)}")
        print(f"[TestPolicy]   保持步数: {self._hold_steps}")
    
    def infer(self, obs: dict) -> dict:
        """返回动作字典（兼容 ActionChunkBroker 接口）。"""
        self._steps_since_change += 1
        
        if self._steps_since_change >= self._hold_steps:
            self._current_action_idx = self._random.randint(0, len(self._action_set) - 1)
            self._current_action = list(self._action_set[self._current_action_idx])
            self._steps_since_change = 0
            print(f"[TestPolicy] 切换到动作 {self._current_action_idx}")
        
        action = list(self._current_action)
        
        if self._add_noise:
            noise = np.random.normal(0, self._noise_scale, len(action))
            action = [a + n for a, n in zip(action, noise)]
        
        return {'actions': np.array(action)}
    
    def reset(self) -> None:
        """重置策略状态。"""
        self._current_action_idx = 0
        self._steps_since_change = 0
        self._current_action = self._action_set[0]


# ============================================================================
# 命令行参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="RECAP 人工干预数据采集脚本（基于 Runtime 架构）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法（需要先启动策略服务器）
    python collect_data_with_intervention.py --prompt "pick up the red cube"
    
    # 测试模式（无需策略服务器）
    python collect_data_with_intervention.py --prompt "test task" --test_mode
        """
    )
    
    parser.add_argument("--prompt", type=str, required=True, help="任务描述（必填）")
    parser.add_argument("--slave_can", type=str, default="can0", help="从臂 CAN 端口")
    parser.add_argument("--master_can", type=str, default="can1", help="主臂 CAN 端口")
    parser.add_argument("--left_wrist_camera_id", type=int, default=4, help="手眼相机 ID")
    parser.add_argument("--high_camera_id", type=int, default=6, help="全局相机 ID")
    parser.add_argument("--camera_fps", type=int, default=30, help="相机帧率")
    parser.add_argument("--max_offset", type=float, default=0.5, help="允许的最大主从偏差（弧度）")
    parser.add_argument("--output_dir", type=str, default="recorded_data_intervention", help="数据保存目录")
    
    # AI 策略相关参数（与 main.py 一致）
    parser.add_argument("--policy_host", type=str, default="0.0.0.0", help="策略服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="策略服务器端口 (对齐 main.py)")
    parser.add_argument("--policy_port", type=int, default=8000, help="策略服务器端口 (兼容旧参数)")
    parser.add_argument("--action_horizon", type=int, default=15, help="动作块大小")
    parser.add_argument("--use_rtc", action="store_true", default=True, help="使用 RTC")
    parser.add_argument("--actions_during_latency", type=int, default=5, help="延迟补偿步数")
    
    # 录制与启动模式
    parser.add_argument("--record_mode", action="store_true", default=False, 
                        help="是否启用录制模式。如果不输入此参数，将不会保存任何数据。")
    parser.add_argument("--manual_start", action="store_true", default=False,
                        help="是否手动开始。如果输入此参数，启动后需要按 's' 键才开始 AI 控制和数据录制。")
    
    # 测试模式参数
    parser.add_argument("--test_mode", action="store_true", help="测试模式")
    parser.add_argument("--test_hold_steps", type=int, default=50, help="测试模式每个动作保持步数")
    parser.add_argument("--test_add_noise", action="store_true", help="测试模式添加噪声")
    
    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    
    print("=" * 60)
    print("  RECAP 人工干预数据采集器（Runtime 架构）")
    print("=" * 60)
    print(f"  任务: {args.prompt}")
    print(f"  从臂 CAN: {args.slave_can}")
    print(f"  主臂 CAN: {args.master_can}")
    if args.test_mode:
        print(f"  模式: 🧪 测试模式（随机动作）")
    else:
        print(f"  模式: 🤖 AI 策略")
        print(f"    策略服务器: {args.policy_host}:{args.policy_port}")
        print(f"    Action Horizon: {args.action_horizon}")
    print("=" * 60)
    
    # ========== 初始化硬件 ==========
    print("\n[1/4] 初始化从臂环境...")
    environment = PiperEnvironment(
        can_port=args.slave_can,
        camera_fps=args.camera_fps,
        high_camera_id=args.high_camera_id,
        left_wrist_camera_id=args.left_wrist_camera_id,
        max_episode_steps=10000000,
        record_mode=False,
        prompt=args.prompt,
        gripper_norm=True,
    )
    
    print("\n[2/4] 初始化主臂读取器...")
    try:
        master_reader = MasterArmReader(can_port=args.master_can)
        test_state = master_reader.get_full_state()
        if all(v == 0 for v in test_state):
            print("⚠️ 警告: 主臂读数为全 0。")
    except Exception as e:
        print(f"❌ 无法连接主臂: {e}")
        environment.close()
        return
    
    print("\n[3/4] 初始化策略...")
    if args.test_mode:
        # 测试模式：使用 TestPolicy
        inner_policy = TestPolicy(
            hold_steps=args.test_hold_steps,
            add_noise=args.test_add_noise,
        )
    else:
        # 正常模式：与 main.py 完全一致的策略配置
        try:
            # 优先使用 --port，如果 --port 是默认值而 --policy_port 不是，则使用 --policy_port
            final_port = args.port
            if args.port == 8000 and args.policy_port != 8000:
                final_port = args.policy_port
                
            websocket_policy = _websocket_client_policy.WebsocketClientPolicy(
                host=args.policy_host,
                port=final_port,
            )
            
            inner_policy = action_chunk_broker.ActionChunkBroker_RTC(
                policy=websocket_policy,
                action_horizon=args.action_horizon,
                fps=args.camera_fps,
                actions_during_latency=args.actions_during_latency,
                use_rtc=args.use_rtc,
            )
            print(f"✅ 已连接到策略服务器 {args.policy_host}:{final_port} (RTC: {args.use_rtc})")
        except Exception as e:
            print(f"❌ 无法连接到策略服务器: {e}")
            print("请确保策略服务器已启动，或使用 --test_mode 进入测试模式")
            environment.close()
            return
    
    # 创建 PolicyAgent（与 main.py 一致）
    policy_agent = _policy_agent.PolicyAgent(policy=inner_policy)
    
    # 创建 InterventionAgent
    intervention_agent = InterventionAgent(
        policy_agent=policy_agent,
        master_reader=master_reader,
        max_offset=args.max_offset,
        initial_recording=not args.manual_start,
    )
    
    # 创建数据记录器
    data_saver = InterventionDataSaver(
        output_dir=args.output_dir,
        prompt=args.prompt,
        agent=intervention_agent,
        record_mode=args.record_mode,
    )
    
    print("\n[4/4] 创建 Runtime...")
    runtime = InterventionRuntime(
        environment=environment,
        agent=intervention_agent,
        subscribers=[data_saver],
        data_saver=data_saver,
        max_hz=args.camera_fps,
        num_episodes=1,
    )
    
    # 运行
    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
