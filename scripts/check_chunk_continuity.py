import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tyro
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging
import os
from tqdm import tqdm
import jax
from scipy.signal import butter, filtfilt

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.policies import policy_config as _policy_config
from lerobot.common.datasets import lerobot_dataset
from openpi.models_pytorch import pi0_pytorch as _pytorch_model
from openpi.models import model as _model
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


# ==========================================
# 零相位低通滤波（替代多项式平滑）
# ==========================================

def apply_zero_phase_filter(trajectory: np.ndarray, cutoff_freq: float = 2.0, 
                            fs: float = 30.0, order: int = 4) -> np.ndarray:
    """
    应用零相位低通滤波（Butterworth双向滤波）。
    
    优点：
    1. 全局一致性：不需要检测拐点，对全程动作一视同仁
    2. 能量守恒：只是把能量"抹匀"，不会创造过冲
    3. 零相位：双向滤波消除相位延迟，不会让机器人变迟钝
    
    Args:
        trajectory: [T, D] 动作轨迹
        cutoff_freq: 截止频率 (Hz)。低于此频率的动作保留，高于此频率的抖动被滤除。
                     - 5.0 Hz: 轻度平滑，保留大部分细节
                     - 2.0 Hz: 中度平滑（推荐），柔和减速
                     - 1.0 Hz: 重度平滑，动作变"肉"
        fs: 数据的采样频率 (Hz)，LeRobot 通常是 30Hz 或 50Hz
        order: 滤波器阶数，4阶通常足够平滑且不失真
        
    Returns:
        smoothed_trajectory: [T, D]
    """
    # 计算归一化截止频率
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    
    # 确保截止频率在有效范围内
    if normal_cutoff >= 1.0:
        print(f"    WARNING: cutoff_freq ({cutoff_freq}) >= Nyquist ({nyquist}), returning original")
        return trajectory.copy()
    
    # 设计巴特沃斯滤波器
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # 检查轨迹长度是否足够（filtfilt 需要至少 3*max(len(a),len(b)) 的数据点）
    min_length = 3 * max(len(a), len(b))
    if trajectory.shape[0] < min_length:
        print(f"    WARNING: Trajectory too short ({trajectory.shape[0]} < {min_length}), returning original")
        return trajectory.copy()
    
    # 使用 filtfilt 进行双向滤波（关键！消除相位延迟）
    # axis=0 表示沿时间轴滤波
    # padtype='odd' 帮助减少边界效应
    smoothed_trajectory = filtfilt(b, a, trajectory, axis=0, padtype='odd')
    
    return smoothed_trajectory

def detect_stop_points(trajectory: np.ndarray, velocity_threshold: float = 0.01, 
                       accel_threshold: float = 0.05, min_stop_frames: int = 3) -> List[int]:
    """
    检测轨迹中的停止点（拐点）。
    
    判断标准：
    1. 速度接近零
    2. 速度符号变化（方向反转）
    3. 加速度突变
    
    Args:
        trajectory: [T, D] 动作轨迹
        velocity_threshold: 速度阈值，低于此值认为静止
        accel_threshold: 加速度突变阈值
        min_stop_frames: 最小连续静止帧数
        
    Returns:
        stop_indices: 停止点的索引列表
    """
    T, D = trajectory.shape
    
    # 计算速度和加速度
    velocity = np.diff(trajectory, axis=0)  # [T-1, D]
    accel = np.diff(velocity, axis=0)  # [T-2, D]
    
    # 速度的模
    vel_norm = np.linalg.norm(velocity, axis=1)  # [T-1]
    accel_norm = np.linalg.norm(accel, axis=1)  # [T-2]
    
    stop_indices = []
    
    # 方法1：检测速度接近零的点
    low_vel_mask = vel_norm < velocity_threshold
    
    # 方法2：检测速度符号变化（方向反转）
    for d in range(D):
        vel_d = velocity[:, d]
        sign_change = np.where(np.diff(np.sign(vel_d)) != 0)[0]
        for idx in sign_change:
            if idx + 1 not in stop_indices:
                stop_indices.append(idx + 1)
    
    # 方法3：检测加速度突变
    accel_jump = np.abs(np.diff(accel_norm))
    jump_indices = np.where(accel_jump > accel_threshold)[0]
    for idx in jump_indices:
        if idx + 2 not in stop_indices:
            stop_indices.append(idx + 2)
    
    # 添加低速点
    low_vel_indices = np.where(low_vel_mask)[0]
    for idx in low_vel_indices:
        if idx + 1 not in stop_indices:
            stop_indices.append(idx + 1)
    
    return sorted(set(stop_indices))


def quintic_polynomial(t: np.ndarray, p0: float, v0: float, a0: float,
                       pf: float, vf: float, af: float, T: float) -> np.ndarray:
    """
    五次多项式插值：满足边界条件的位置、速度、加速度约束。
    
    p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    
    边界条件：
    - p(0) = p0, p'(0) = v0, p''(0) = a0
    - p(T) = pf, p'(T) = vf, p''(T) = af
    
    Args:
        t: 时间数组 [0, T]
        p0, v0, a0: 起始位置、速度、加速度
        pf, vf, af: 终止位置、速度、加速度
        T: 总时间
        
    Returns:
        p: 位置数组
    """
    if T <= 0:
        return np.full_like(t, pf)
    
    # 归一化时间 τ = t / T
    tau = t / T
    
    # 五次多项式系数（通过边界条件求解）
    # p(τ) = p0 + v0*T*τ + 0.5*a0*T²*τ² + c3*τ³ + c4*τ⁴ + c5*τ⁵
    # 
    # 令 Δp = pf - p0, Δv = vf - v0, Δa = af - a0
    # 求解线性方程组得到 c3, c4, c5
    
    dp = pf - p0
    dv_T = (vf - v0) * T
    da_T2 = (af - a0) * T * T
    
    # 系数（通过 Vandermonde 矩阵求解）
    a0_coef = p0
    a1_coef = v0 * T
    a2_coef = 0.5 * a0 * T * T
    
    # 剩余约束求解 c3, c4, c5
    # p(1) = p0 + v0*T + 0.5*a0*T² + c3 + c4 + c5 = pf
    # p'(1)/T = v0 + a0*T + 3*c3 + 4*c4 + 5*c5 = vf
    # p''(1)/T² = a0 + 6*c3 + 12*c4 + 20*c5 = af
    
    # 矩阵形式 A @ [c3, c4, c5].T = b
    A = np.array([
        [1, 1, 1],
        [3, 4, 5],
        [6, 12, 20]
    ], dtype=np.float64)
    
    b = np.array([
        dp - v0 * T - 0.5 * a0 * T * T,
        vf * T - v0 * T - a0 * T * T,
        af * T * T - a0 * T * T
    ], dtype=np.float64)
    
    try:
        c345 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 奇异矩阵，退化为线性插值
        return p0 + (pf - p0) * tau
    
    c3, c4, c5 = c345
    
    # 计算位置
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau
    
    p = a0_coef + a1_coef * tau + a2_coef * tau2 + c3 * tau3 + c4 * tau4 + c5 * tau5
    
    return p


def monotonic_quintic_deceleration(trajectory: np.ndarray, window_size: int = 15, 
                                    velocity_threshold: float = 0.02,
                                    use_cubic: bool = True) -> np.ndarray:
    """
    对轨迹进行单调减速平滑（三次或五次多项式）。
    
    只处理从"动"到"静"的过渡段，保证单调性（不产生反向弹跳）。
    
    原理：
    1. 检测停止点（速度突然变为0的位置）
    2. 根据初始速度自适应调整窗口大小
    3. 用多项式替换减速段
    4. 单调性约束：如果多项式产生反向运动，则退化为线性插值
    
    Args:
        trajectory: [T, D] 动作轨迹
        window_size: 基础减速窗口大小（帧数），实际窗口会根据速度自适应
        velocity_threshold: 速度阈值，用于检测停止点
        use_cubic: 是否使用三次多项式（更稳定），否则使用五次
        
    Returns:
        smoothed: [T, D] 平滑后的轨迹
    """
    T, D = trajectory.shape
    smoothed = trajectory.copy()
    
    # 检测停止点
    stop_indices = detect_stop_points(trajectory, velocity_threshold=velocity_threshold)
    
    if not stop_indices:
        print("    No stop points detected, returning original trajectory")
        return smoothed
    
    print(f"    Detected {len(stop_indices)} stop points, using {'cubic' if use_cubic else 'quintic'} polynomial")
    
    modified_count = 0
    total_jump_before = 0
    total_jump_after = 0
    
    for stop_idx in stop_indices:
        # 对每个维度单独处理
        for d in range(D):
            pf = trajectory[stop_idx, d]   # 终点位置
            
            # 计算停止点前的速度，找到速度开始减小的位置
            # 自适应窗口：根据速度大小调整
            if stop_idx > 0:
                v_at_stop = abs(trajectory[stop_idx, d] - trajectory[stop_idx - 1, d])
            else:
                continue
            
            # 向前搜索，找到速度开始明显减小的位置
            # 自适应窗口大小：速度越大，需要越长的减速距离
            adaptive_window = max(window_size, int(v_at_stop * 50))  # 速度越大窗口越大
            adaptive_window = min(adaptive_window, stop_idx, 30)  # 上限30帧
            
            start_idx = stop_idx - adaptive_window
            if start_idx < 0:
                start_idx = 0
            
            if start_idx >= stop_idx:
                continue
            
            p0 = trajectory[start_idx, d]  # 起点位置
            
            # 计算起点速度（使用有限差分）
            if start_idx > 0:
                v0 = trajectory[start_idx, d] - trajectory[start_idx - 1, d]
            else:
                v0 = 0.0
            
            # 窗口内的帧数
            n_frames = stop_idx - start_idx + 1
            
            # 记录修改前的最大跳变
            original_segment = trajectory[start_idx:stop_idx+1, d]
            if len(original_segment) > 1:
                jump_before = np.max(np.abs(np.diff(original_segment)))
                total_jump_before = max(total_jump_before, jump_before)
            
            # 生成多项式
            t_normalized = np.linspace(0, 1, n_frames)
            
            if use_cubic:
                poly_values = _cubic_with_monotonicity(t_normalized, p0, v0, pf, n_frames)
            else:
                poly_values = _quintic_with_monotonicity(t_normalized, p0, v0, pf, 0.0, n_frames)
            
            # 检查单调性并修正
            if poly_values is not None:
                smoothed[start_idx:stop_idx+1, d] = poly_values
                # 记录修改后的最大跳变
                jump_after = np.max(np.abs(np.diff(poly_values)))
                total_jump_after = max(total_jump_after, jump_after)
            else:
                # 退化为线性插值
                linear_values = np.linspace(p0, pf, n_frames)
                smoothed[start_idx:stop_idx+1, d] = linear_values
                jump_after = np.max(np.abs(np.diff(linear_values)))
                total_jump_after = max(total_jump_after, jump_after)
        
        modified_count += 1
    
    print(f"    Modified {modified_count} deceleration regions")
    print(f"    Max jump before: {total_jump_before:.6f}, after: {total_jump_after:.6f}")
    return smoothed


def _cubic_with_monotonicity(t: np.ndarray, p0: float, v0: float, pf: float, 
                              n_frames: int) -> Optional[np.ndarray]:
    """
    生成带单调性约束的三次多项式（Hermite插值）。
    
    边界条件（4个，正好匹配三次多项式的4个系数）：
    - p(0) = p0, p'(0) = v0
    - p(1) = pf, p'(1) = 0
    
    三次多项式比五次更稳定，不容易产生过冲。
    """
    # 归一化速度（因为t在[0,1]范围内）
    v0_norm = v0 * n_frames
    
    # Hermite 三次多项式
    # p(t) = (2t³ - 3t² + 1)p0 + (t³ - 2t² + t)v0 + (-2t³ + 3t²)pf + (t³ - t²)vf
    # 这里 vf = 0
    
    t2 = t * t
    t3 = t2 * t
    
    h00 = 2*t3 - 3*t2 + 1  # p0 系数
    h10 = t3 - 2*t2 + t     # v0 系数
    h01 = -2*t3 + 3*t2      # pf 系数
    # h11 = t3 - t2          # vf 系数 (vf=0, 不需要)
    
    values = h00 * p0 + h10 * v0_norm + h01 * pf
    
    # 检查单调性：所有中间值应该在合理范围内
    lower = min(p0, pf)
    upper = max(p0, pf)
    
    # 三次多项式可能会轻微超出，给予较大的容差
    # 但我们使用更智能的检查：检查是否有明显的"回弹"
    range_size = abs(pf - p0)
    if range_size < 1e-6:
        return np.full_like(t, pf)
    
    tolerance = 0.2 * range_size  # 允许20%的超出
    
    if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
        # 单调性严重违反，返回 None 以使用线性插值
        return None
    
    # 轻微超出时，使用 soft clamp 而不是硬裁剪
    # 这样可以保持曲线的平滑性
    overshoot_mask = (values < lower) | (values > upper)
    if np.any(overshoot_mask):
        # 使用 soft sigmoid 来平滑地拉回范围内
        values = np.clip(values, lower - 0.1 * range_size, upper + 0.1 * range_size)
    
    return values


def _quintic_with_monotonicity(t: np.ndarray, p0: float, v0: float, pf: float, vf: float, 
                                n_frames: int) -> Optional[np.ndarray]:
    """
    生成带单调性约束的五次多项式。
    
    边界条件：
    - p(0) = p0, p'(0) = v0, p''(0) = 0
    - p(1) = pf, p'(1) = 0, p''(1) = 0
    
    如果生成的多项式不满足单调性，返回 None。
    """
    # 归一化速度（因为t在[0,1]范围内）
    v0_norm = v0 * n_frames  # 将帧速度转换为归一化速度
    
    # 五次多项式系数求解
    # p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    # 6个边界条件，6个未知数
    
    # 边界条件矩阵形式 A * coeffs = b
    # t=0: p(0)=p0, p'(0)=v0_norm, p''(0)=0
    # t=1: p(1)=pf, p'(1)=0, p''(1)=0
    
    A = np.array([
        [1, 0, 0, 0, 0, 0],      # p(0) = a0
        [0, 1, 0, 0, 0, 0],      # p'(0) = a1
        [0, 0, 2, 0, 0, 0],      # p''(0) = 2*a2
        [1, 1, 1, 1, 1, 1],      # p(1)
        [0, 1, 2, 3, 4, 5],      # p'(1)
        [0, 0, 2, 6, 12, 20],    # p''(1)
    ], dtype=np.float64)
    
    b = np.array([p0, v0_norm, 0, pf, 0, 0], dtype=np.float64)
    
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    # 计算多项式值
    a0, a1, a2, a3, a4, a5 = coeffs
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    
    values = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
    
    # 检查单调性：所有中间值应该在 [min(p0,pf), max(p0,pf)] 范围内
    lower = min(p0, pf)
    upper = max(p0, pf)
    
    # 允许小量超出（数值误差）
    tolerance = 0.1 * abs(pf - p0) if abs(pf - p0) > 1e-6 else 0.01
    
    if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
        # 单调性违反，返回 None 以使用线性插值
        return None
    
    # Clamp 到范围内（处理微小的数值误差）
    values = np.clip(values, lower, upper)
    
    return values


def local_spline_smoothing(trajectory: np.ndarray, window_before: int = 5, 
                           window_after: int = 5, velocity_threshold: float = 0.02) -> np.ndarray:
    """
    对轨迹进行局部样条平滑（微创手术）。
    
    只在检测到的拐点附近进行五次多项式替换，保留其他区域的原始数据。
    
    Args:
        trajectory: [T, D] 动作轨迹
        window_before: 拐点前的替换窗口大小
        window_after: 拐点后的替换窗口大小
        velocity_threshold: 速度阈值，用于检测停止点
        
    Returns:
        smoothed: [T, D] 局部平滑后的轨迹
    """
    T, D = trajectory.shape
    smoothed = trajectory.copy()
    
    # 检测停止点
    stop_indices = detect_stop_points(trajectory, velocity_threshold=velocity_threshold)
    
    if not stop_indices:
        print("    No stop points detected, returning original trajectory")
        return smoothed
    
    print(f"    Detected {len(stop_indices)} stop points at indices: {stop_indices[:10]}{'...' if len(stop_indices) > 10 else ''}")
    
    # 对每个停止点进行局部平滑
    modified_regions = []
    
    for stop_idx in stop_indices:
        # 确定替换窗口
        start_idx = max(0, stop_idx - window_before)
        end_idx = min(T - 1, stop_idx + window_after)
        
        # 检查是否与已修改区域重叠
        overlap = False
        for (ms, me) in modified_regions:
            if not (end_idx < ms or start_idx > me):
                overlap = True
                break
        
        if overlap:
            continue
        
        window_len = end_idx - start_idx + 1
        if window_len < 3:
            continue
        
        # 对每个维度进行五次多项式插值
        t_local = np.arange(window_len, dtype=np.float64)
        T_total = float(window_len - 1)
        
        for d in range(D):
            # 边界条件
            p0 = trajectory[start_idx, d]
            pf = trajectory[end_idx, d]
            
            # 计算起始速度和加速度（使用有限差分）
            if start_idx > 0:
                v0 = trajectory[start_idx, d] - trajectory[start_idx - 1, d]
            else:
                v0 = trajectory[start_idx + 1, d] - trajectory[start_idx, d] if start_idx + 1 < T else 0
            
            if start_idx > 1:
                a0 = (trajectory[start_idx, d] - 2 * trajectory[start_idx - 1, d] + trajectory[start_idx - 2, d])
            else:
                a0 = 0
            
            # 终止条件：目标是平滑停止
            # 如果停止点在窗口中间，设置终止速度和加速度接近零
            if stop_idx <= end_idx:
                vf = 0.0  # 目标速度为零
                af = 0.0  # 目标加速度为零
            else:
                # 使用原始数据的速度/加速度
                if end_idx < T - 1:
                    vf = trajectory[end_idx + 1, d] - trajectory[end_idx, d]
                else:
                    vf = 0
                af = 0
            
            # 五次多项式插值
            p_smooth = quintic_polynomial(t_local, p0, v0, a0, pf, vf, af, T_total)
            
            # 替换局部区域
            smoothed[start_idx:end_idx + 1, d] = p_smooth
        
        modified_regions.append((start_idx, end_idx))
    
    print(f"    Modified {len(modified_regions)} regions")
    return smoothed


# ==========================================
# ResidualTransitionNet 模型定义 (从 train_residual_net.py 复制)
# ==========================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta"""
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, num_features * 2)
        self.cond_proj.weight.data.zero_()
        self.cond_proj.bias.data[:num_features] = 1.0  # gamma = 1
        self.cond_proj.bias.data[num_features:] = 0.0  # beta = 0

    def forward(self, x, cond):
        style = self.cond_proj(cond).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return gamma * x + beta


class ResBlock1D(nn.Module):
    """conv → act → film → conv → act → film + residual"""
    def __init__(self, channels, cond_dim, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.act1 = nn.SiLU()
        self.film1 = FiLM(cond_dim, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.act2 = nn.SiLU()
        self.film2 = FiLM(cond_dim, channels)

    def forward(self, x, cond):
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.film1(out, cond)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.film2(out, cond)
        return out + residual


class ResidualTransitionNet(nn.Module):
    def __init__(
        self, 
        latent_dim=32, 
        horizon=50, 
        cond_dim=2048,
        hidden_dim=256,
        num_blocks=4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.cond_dim = cond_dim
        
        # 输入归一化
        self.z_norm = nn.LayerNorm(latent_dim)
        self.cond_norm = nn.LayerNorm(cond_dim)
        
        # 输入投影
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        
        # 残差主干
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])
        
        # 输出头
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        
        # Zero-Init Output
        self.output_proj.weight.data.zero_()
        self.output_proj.bias.data.zero_()
        
        # Delta 归一化统计量
        self.register_buffer('delta_mean', torch.zeros(latent_dim))
        self.register_buffer('delta_std', torch.ones(latent_dim))

    def set_delta_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.delta_mean = mean.to(self.delta_mean.device)
        self.delta_std = std.to(self.delta_std.device)

    def forward(self, z_t, prefix_embs, normalize_output=True):
        """
        Args:
            z_t: [Batch, Horizon, Latent_Dim]
            prefix_embs: [Batch, Seq_Len, Embed_Dim] or [Batch, Embed_Dim]
            normalize_output: 如果 True，输出归一化后的 delta；推理时设为 False
        """
        z_t = self.z_norm(z_t)
        x = z_t.transpose(1, 2)
        
        if prefix_embs.ndim == 3:
            cond = prefix_embs.mean(dim=1)
        else:
            cond = prefix_embs
        cond = self.cond_norm(cond)
        
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        
        x = x.transpose(1, 2)
        x = self.output_norm(x)
        x = x.transpose(1, 2)
        
        delta = self.output_proj(x)
        delta = delta.transpose(1, 2)
        
        # 反归一化（推理时）
        if not normalize_output and self.delta_std.sum() > 0:
            delta = delta * self.delta_std + self.delta_mean
        
        return delta


# ==========================================
# StochasticMotionPrior 模型定义 (从 train_stochastic_prior.py 复制)
# ==========================================

class StochasticMotionPrior(nn.Module):
    """
    随机运动先验网络 (Stochastic Motion Prior)
    
    结构：
    - 输入: z_t (32 dim) + pooled prefix_embs (2048 -> cond_proj_dim)
    - 核心: GRU (hidden_dim, num_layers)
    - 输出: μ (32 dim) + log σ (32 dim)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        cond_dim: int = 2048,
        cond_proj_dim: int = 64,
        dropout: float = 0.1,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # 条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_proj_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_proj_dim * 2, cond_proj_dim),
            nn.LayerNorm(cond_proj_dim),
        )
        
        # 输入投影
        gru_input_dim = latent_dim + cond_proj_dim
        self.input_proj = nn.Sequential(
            nn.Linear(gru_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # GRU 核心
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # 简化的输出头（单层 Linear）
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, latent_dim)
        
        # 初始化输出头（小方差初始化）
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.1)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.1)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # 初始 std ≈ 0.37
    
    def forward(
        self, 
        z_seq: torch.Tensor,
        prefix_embs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            z_seq: 潜变量序列 [B, T, latent_dim]
            prefix_embs: 条件特征 [B, cond_dim] 或 [B, seq_len, cond_dim]
            hidden: GRU 隐状态 [num_layers, B, hidden_dim]
            return_hidden: 是否返回最终隐状态
            
        Returns:
            mu: 均值 [B, T, latent_dim]
            log_std: 对数标准差 [B, T, latent_dim]
            hidden: (可选) 最终隐状态
        """
        B, T, D = z_seq.shape
        
        # 条件处理
        if prefix_embs.ndim == 3:
            cond = prefix_embs.mean(dim=1)
        else:
            cond = prefix_embs
        
        cond = self.cond_proj(cond)
        cond_expanded = cond.unsqueeze(1).expand(-1, T, -1)
        
        # 拼接输入
        gru_input = torch.cat([z_seq, cond_expanded], dim=-1)
        gru_input = self.input_proj(gru_input)
        
        # GRU 前向
        gru_out, hidden_out = self.gru(gru_input, hidden)
        
        # 输出归一化
        gru_out = self.output_norm(gru_out)
        
        # 预测分布参数
        mu = self.mu_head(gru_out)
        log_std = self.log_std_head(gru_out)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        if return_hidden:
            return mu, log_std, hidden_out
        return mu, log_std, None
    
    def sample(
        self, 
        mu: torch.Tensor, 
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """从预测的分布中采样"""
        if deterministic:
            return mu
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    @torch.no_grad()
    def generate_step(
        self,
        z_t: torch.Tensor,
        prefix_embs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步生成（用于自回归推理）
        
        Args:
            z_t: 当前潜变量 [B, 1, D] 或 [B, D]
            prefix_embs: 条件特征 [B, cond_dim]
            hidden: GRU 隐状态
            deterministic: 是否确定性生成
            
        Returns:
            z_next: 下一步潜变量 [B, 1, D]
            hidden: 更新后的隐状态
        """
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(1)
        
        mu, log_std, hidden_out = self.forward(z_t, prefix_embs, hidden, return_hidden=True)
        z_next = self.sample(mu, log_std, deterministic)
        
        return z_next, hidden_out


@dataclass
class Args:
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    dataset_path: str = "/home/ztlab/lgd/openpi/latent_dataset.h5"
    start_idx: int = 0
    num_steps_to_sim: int = 100 # How many simulation steps
    execution_horizon: int = 15 # Execute 15 steps from each chunk
    noise_scale: float = 0.05 # Scale for perturbed noise
    overlap_steps: int = 5  # 噪声重叠步数：上一个 chunk 末尾的噪声覆盖下一个 chunk 开头
    device: str = "cuda"
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    
    # 残差网络配置
    residual_net_dir: str = "/home/ztlab/lgd/openpi/checkpoints/residual_net"
    residual_net_latent_dim: int = 32
    residual_net_horizon: int = 50
    residual_net_cond_dim: int = 2048
    residual_net_hidden_dim: int = 256
    residual_net_num_blocks: int = 4
    
    # Stochastic Motion Prior 配置
    stochastic_prior_dir: str = "/home/ztlab/lgd/openpi/checkpoints/stochastic_prior"
    stochastic_prior_latent_dim: int = 32
    stochastic_prior_hidden_dim: int = 128
    stochastic_prior_num_layers: int = 2
    stochastic_prior_cond_dim: int = 2048
    stochastic_prior_cond_proj_dim: int = 64
    stochastic_prior_deterministic: bool = False  # 是否使用确定性采样（只用 μ）

@torch.no_grad()
def decode_chunk(model, latents, prefix_embs, prefix_pad_masks, prefix_att_masks, state, num_steps=10):
    """
    Run the flow matching ODE solver in forward direction (Noise t=1 -> Data t=0).
    """
    bsize = latents.shape[0]
    device = latents.device
    
    # 1. Prepare KV Cache from Prefix Embeddings
    # This logic mirrors invert_actions but doesn't re-encode images
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    
    # Enable caching
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    # Forward pass to get KV cache for the prefix
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    # 2. ODE Solver (Euler)
    # t goes from 1.0 to 0.0
    dt = -1.0 / num_steps
    x_t = latents.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    for _ in range(num_steps):
        expanded_time = time.expand(bsize)
        
        # Compute velocity v_t
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time
        )
        
        # Euler update
        x_t = x_t + dt * v_t
        time = time + dt
        
    return x_t


@torch.no_grad()
def inverse_ode_online(
    model: PI0Pytorch,
    actions: torch.Tensor,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    num_steps: int = 10
) -> torch.Tensor:
    """
    实时反推 ODE (Euler 方法)：从 actions (t=0) 反推到 noise (t=1)。
    
    这是方案B的实现：重新计算速度场进行反向积分。
    
    Args:
        model: PI0Pytorch 模型
        actions: 归一化后的 actions [B, H, D]
        state: 状态 [B, state_dim]
        prefix_pad_masks: prefix padding masks
        past_key_values: KV cache
        num_steps: ODE 步数
        
    Returns:
        noise: 反推得到的 noise [B, H, D]
    """
    device = actions.device
    bsize = actions.shape[0]
    
    # 确保 actions 是 float32
    x_t = actions.clone().float()
    
    dt = 1.0 / num_steps  # 正的步长，从 t=0 到 t=1
    time = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        # 在当前位置计算速度场
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        # Euler 更新
        x_t = x_t + dt * v_t
        time = time + dt
    
    return x_t


@torch.no_grad()
def inverse_ode_rk4(
    model: PI0Pytorch,
    actions: torch.Tensor,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    num_steps: int = 10
) -> torch.Tensor:
    """
    实时反推 ODE (RK4 方法)：从 actions (t=0) 反推到 noise (t=1)。
    
    RK4 (4阶 Runge-Kutta) 方法比 Euler 更精确。
    
    Args:
        model: PI0Pytorch 模型
        actions: 归一化后的 actions [B, H, D]
        state: 状态 [B, state_dim]
        prefix_pad_masks: prefix padding masks
        past_key_values: KV cache
        num_steps: ODE 步数
        
    Returns:
        noise: 反推得到的 noise [B, H, D]
    """
    device = actions.device
    bsize = actions.shape[0]
    
    # 确保 actions 是 float32
    x_t = actions.clone().float()
    
    dt = 1.0 / num_steps  # 正的步长，从 t=0 到 t=1
    time = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    def velocity_fn(x, t):
        """计算速度场 v(x, t)"""
        expanded_t = t.expand(bsize) if t.dim() == 0 else t
        return model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x,
            timestep=expanded_t,
        )
    
    for step in range(num_steps):
        # RK4 四个斜率
        k1 = velocity_fn(x_t, time)
        k2 = velocity_fn(x_t + 0.5 * dt * k1, time + 0.5 * dt)
        k3 = velocity_fn(x_t + 0.5 * dt * k2, time + 0.5 * dt)
        k4 = velocity_fn(x_t + dt * k3, time + dt)
        
        # RK4 更新
        x_t = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        time = time + dt
    
    return x_t


@torch.no_grad()
def prepare_observation_and_kv_cache(
    model: PI0Pytorch,
    item: dict,
    device: str,
    camera_names: List[str] = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
) -> Tuple[torch.Tensor, torch.Tensor, any]:
    """
    从 transformed dataset item 准备 observation 和 KV cache。
    
    Returns:
        state: [1, state_dim]
        prefix_pad_masks: [1, seq_len]
        past_key_values: KV cache
    """
    def to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        elif isinstance(v, np.ndarray):
            return torch.from_numpy(v).to(device)
        else:
            return torch.tensor(v, device=device)
    
    images_raw = item['image']
    images_dict = {}
    image_masks_dict = {}
    
    if isinstance(images_raw, dict):
        for cam_name in camera_names:
            if cam_name in images_raw:
                img = to_tensor(images_raw[cam_name])
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                images_dict[cam_name] = img[None, ...]
                image_masks_dict[cam_name] = torch.ones(1, dtype=torch.bool, device=device)
    else:
        for i, cam_name in enumerate(camera_names):
            if i < images_raw.shape[0]:
                img = to_tensor(images_raw[i])
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                images_dict[cam_name] = img[None, ...]
                image_masks_dict[cam_name] = torch.ones(1, dtype=torch.bool, device=device)
    
    state_raw = to_tensor(item['state'])
    if state_raw.shape[-1] < 32:
        state_padded = torch.zeros(32, device=device)
        state_padded[:state_raw.shape[-1]] = state_raw
        state_tensor = state_padded[None, ...]
    else:
        state_tensor = state_raw[None, ...]
    
    tokenized_prompt = to_tensor(item['tokenized_prompt'])[None, ...]
    tokenized_prompt_mask = to_tensor(item['tokenized_prompt_mask'])[None, ...]
    
    observation = _model.Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        state=state_tensor,
        tokenized_prompt=tokenized_prompt.long(),
        tokenized_prompt_mask=tokenized_prompt_mask.bool(),
    )
    
    # 预处理观测
    obs_preprocessed = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    
    images = list(obs_preprocessed.images.values())
    img_masks = list(obs_preprocessed.image_masks.values())
    lang_tokens = obs_preprocessed.tokenized_prompt
    lang_masks = obs_preprocessed.tokenized_prompt_mask
    state = obs_preprocessed.state
    
    # 嵌入前缀
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    
    # 准备 KV Cache
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    
    return state, prefix_pad_masks, past_key_values


def simple_collate(batch):
    # Retrieve state from the batch (list of dicts)
    # We assume batch size 1 for loop usually, but here we can stack
    states = []
    for item in batch:
        s = item.get("state")
        if s is None:
            s = item.get("observation.state")
        states.append(s)
    return torch.stack(states)

def run_simulation_with_policy(args, policy, raw_dataset, f_h5, mode="inverse", residual_net=None, stochastic_prior=None):
    """
    使用 policy.infer() 进行推理，确保所有预处理/后处理正确执行。
    只替换噪声来源进行对比实验。
    
    Modes:
        - "inverse": 使用反推的隐变量（从 HDF5 加载）
        - "random": 使用独立高斯噪声
        - "perturbed": 反推隐变量 + 高斯扰动
        - "residual": 使用残差网络预测噪声
        - "stochastic": 使用 StochasticMotionPrior 网络生成噪声
    """
    print(f"\n>>> Running Simulation Mode: {mode}")
    
    current_idx = args.start_idx
    executed_trajectory = [] 
    chunk_boundaries = [] 
    all_decoded_chunks = []
    
    sim_indices = []
    for _ in range(args.num_steps_to_sim):
        sim_indices.append(current_idx)
        current_idx += args.execution_horizon

    device = args.device
    
    # 对于 residual/stochastic 模式，需要追踪上一步的噪声
    prev_noise = None
    gru_hidden = None  # GRU 隐状态（stochastic 模式使用）
    
    # 获取 PI0 模型用于特征提取 (residual/stochastic 模式需要)
    model = policy._model if hasattr(policy, '_model') else None

    for step_i, idx in enumerate(tqdm(sim_indices, desc=mode)):
        # 1. 检查边界
        if idx >= len(raw_dataset) or idx >= f_h5['latents'].shape[0]:
            print(f"Index {idx} out of bounds. Stopping.")
            break
        
        # 2. 从 HDF5 获取反推的隐变量
        h5_latent = torch.from_numpy(f_h5['latents'][idx]).unsqueeze(0).to(device)
        
        # 3. 根据模式确定噪声
        if mode == "inverse":
            noise = h5_latent
        elif mode == "random":
            noise = torch.randn_like(h5_latent)
        elif mode == "perturbed":
            noise = h5_latent + args.noise_scale * torch.randn_like(h5_latent)
        elif mode == "residual":
            if residual_net is None:
                raise ValueError("Residual mode requires residual_net to be provided!")
            
            if step_i == 0:
                # 第一步：使用 HDF5 中的反推噪声作为初始值
                noise = h5_latent
            else:
                # 后续步：使用残差网络预测
                # 需要提取当前观测的 prefix features
                
                # 准备观测数据用于特征提取
                item = raw_dataset[idx]
                obs_for_feat = {}
                
                state_val = item.get("observation.state")
                if state_val is None:
                    state_val = item.get("state")
                if isinstance(state_val, torch.Tensor):
                    state_val = state_val.numpy()
                obs_for_feat["state"] = state_val
                
                images_dict = {}
                for key in item.keys():
                    if key.startswith("observation.images."):
                        cam_name = key.replace("observation.images.", "")
                        img_val = item[key]
                        if isinstance(img_val, torch.Tensor):
                            img_val = img_val.numpy()
                        images_dict[cam_name] = img_val
                obs_for_feat["images"] = images_dict
                
                prompt_val = item.get("prompt", "")
                if prompt_val is None:
                    prompt_val = ""
                obs_for_feat["prompt"] = prompt_val
                
                # 通过 policy 的 input transform 处理观测
                inputs = jax.tree.map(lambda x: x, obs_for_feat)
                inputs = policy._input_transform(inputs)
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
                
                # 创建 Observation 对象并提取特征
                observation = _model.Observation.from_dict(inputs)
                
                with torch.no_grad():
                    prefix_info = model.get_features(observation)
                    prefix_embs = prefix_info[0]  # [1, seq_len, embed_dim]
                    
                    # 使用残差网络预测 delta
                    noise_in = prev_noise.to(dtype=residual_net.output_proj.weight.dtype)
                    prefix_in = prefix_embs.to(dtype=residual_net.output_proj.weight.dtype)
                    
                    delta = residual_net(noise_in, prefix_in)
                    noise = prev_noise + delta.float()
                    
                    # Clamp 防止爆炸
                    noise = torch.clamp(noise, -3.0, 3.0)
        elif mode == "stochastic":
            if stochastic_prior is None:
                raise ValueError("Stochastic mode requires stochastic_prior to be provided!")
            
            if step_i == 0:
                # 第一步：使用 HDF5 中的反推噪声作为初始值
                noise = h5_latent
            else:
                # 后续步：使用 StochasticMotionPrior 网络生成
                # 需要提取当前观测的 prefix features
                
                # 准备观测数据用于特征提取
                item = raw_dataset[idx]
                obs_for_feat = {}
                
                state_val = item.get("observation.state")
                if state_val is None:
                    state_val = item.get("state")
                if isinstance(state_val, torch.Tensor):
                    state_val = state_val.numpy()
                obs_for_feat["state"] = state_val
                
                images_dict = {}
                for key in item.keys():
                    if key.startswith("observation.images."):
                        cam_name = key.replace("observation.images.", "")
                        img_val = item[key]
                        if isinstance(img_val, torch.Tensor):
                            img_val = img_val.numpy()
                        images_dict[cam_name] = img_val
                obs_for_feat["images"] = images_dict
                
                prompt_val = item.get("prompt", "")
                if prompt_val is None:
                    prompt_val = ""
                obs_for_feat["prompt"] = prompt_val
                
                # 通过 policy 的 input transform 处理观测
                inputs = jax.tree.map(lambda x: x, obs_for_feat)
                inputs = policy._input_transform(inputs)
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
                
                # 创建 Observation 对象并提取特征
                observation = _model.Observation.from_dict(inputs)
                
                with torch.no_grad():
                    prefix_info = model.get_features(observation)
                    prefix_embs = prefix_info[0]  # [1, seq_len, embed_dim]
                    
                    # 使用 StochasticMotionPrior 生成噪声
                    # prev_noise: [1, horizon, latent_dim]
                    # 我们使用整个 horizon 序列作为输入
                    z_input = prev_noise.float()
                    cond_input = prefix_embs.float()
                    
                    # 前向传播获取分布参数
                    mu, log_std, gru_hidden = stochastic_prior(
                        z_input, cond_input, gru_hidden, return_hidden=True
                    )
                    
                    # 从分布中采样（或使用确定性模式）
                    noise = stochastic_prior.sample(
                        mu, log_std, 
                        deterministic=args.stochastic_prior_deterministic
                    )
                    
                    # Clamp 防止爆炸
                    noise = torch.clamp(noise, -3.0, 3.0)
        else:
            raise ValueError(f"Unknown mode {mode}")
        
        # 4. 从原始数据集获取观测，并转换为 policy 期望的格式
        item = raw_dataset[idx]
        
        # 转换 LeRobot 格式 -> Policy 格式
        # LeRobot: {"observation.state": ..., "observation.images.cam_high": ..., ...}
        # Policy:  {"state": ..., "images": {"cam_high": ..., ...}, "prompt": ...}
        obs = {}
        
        # State
        state_val = item.get("observation.state")
        if state_val is None:
            state_val = item.get("state")
        if isinstance(state_val, torch.Tensor):
            state_val = state_val.numpy()
        obs["state"] = state_val
        
        # Images
        images_dict = {}
        for key in item.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                img_val = item[key]
                if isinstance(img_val, torch.Tensor):
                    img_val = img_val.numpy()
                images_dict[cam_name] = img_val
        obs["images"] = images_dict
        
        # Prompt
        prompt_val = item.get("prompt", "")
        if prompt_val is None:
            prompt_val = ""
        obs["prompt"] = prompt_val
        
        # 5. 使用 policy.infer() 进行推理，传入自定义噪声
        with torch.no_grad():
            result = policy.infer(obs, noise=noise)
        
        # 6. 提取动作 (已经经过反归一化)
        actions = result['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        else:
            actions = np.array(actions)
        
        # 确保是 [Horizon, Action_Dim] 形状
        if actions.ndim == 3:
            actions = actions[0]  # [1, H, D] -> [H, D]
        
        all_decoded_chunks.append((idx, actions))
        
        # 7. Execute
        to_exec = min(args.execution_horizon, actions.shape[0])
        for t in range(to_exec):
            executed_trajectory.append(actions[t])
        
        chunk_boundaries.append(len(executed_trajectory))
        
        # 8. 保存当前噪声用于 residual/stochastic 模式的下一步预测
        if mode in ["residual", "stochastic"]:
            prev_noise = noise

    return np.array(executed_trajectory), chunk_boundaries, all_decoded_chunks


def run_simulation_inverse_online(args, policy, model, raw_dataset, transformed_dataset, action_mean, action_std, ode_method="euler"):
    """
    实时反推模式：从 transformed_dataset 获取归一化后的 actions，
    使用 inverse ODE 反推 noise，然后解码检验。
    
    Args:
        ode_method: "euler" 或 "rk4"，选择 ODE 求解方法
    """
    method_name = f"inverse_{ode_method}"
    print(f"\n>>> Running Simulation Mode: {method_name} (实时反推, {ode_method.upper()} 方法)")
    
    current_idx = args.start_idx
    executed_trajectory = [] 
    chunk_boundaries = [] 
    all_decoded_chunks = []
    
    sim_indices = []
    for _ in range(args.num_steps_to_sim):
        sim_indices.append(current_idx)
        current_idx += args.execution_horizon

    device = args.device
    num_ode_steps = 10  # 与生成脚本一致
    
    for step_i, idx in enumerate(tqdm(sim_indices, desc=method_name)):
        # 1. 检查边界
        if idx >= len(transformed_dataset):
            print(f"Index {idx} out of bounds. Stopping.")
            break
        
        # 2. 从 transformed_dataset 获取归一化后的 actions
        item = transformed_dataset[idx]
        actions_normalized = item.get('actions')
        if actions_normalized is None:
            actions_normalized = item.get('action')
        
        if isinstance(actions_normalized, np.ndarray):
            actions_normalized = torch.from_numpy(actions_normalized)
        actions_normalized = actions_normalized.to(device).unsqueeze(0)  # [1, H, D]
        
        # 3. 准备 observation 和 KV cache
        state, prefix_pad_masks, past_key_values = prepare_observation_and_kv_cache(
            model, item, device
        )
        
        # 4. 使用 inverse ODE 反推 noise (根据方法选择)
        if ode_method == "rk4":
            noise = inverse_ode_rk4(
                model, actions_normalized, state, prefix_pad_masks, past_key_values, num_ode_steps
            )
        else:  # euler
            noise = inverse_ode_online(
                model, actions_normalized, state, prefix_pad_masks, past_key_values, num_ode_steps
            )
        
        # 5. 从 raw_dataset 获取观测用于 policy.infer()
        raw_item = raw_dataset[idx]
        obs = {}
        
        state_val = raw_item.get("observation.state")
        if state_val is None:
            state_val = raw_item.get("state")
        if isinstance(state_val, torch.Tensor):
            state_val = state_val.numpy()
        obs["state"] = state_val
        
        images_dict = {}
        for key in raw_item.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                img_val = raw_item[key]
                if isinstance(img_val, torch.Tensor):
                    img_val = img_val.numpy()
                images_dict[cam_name] = img_val
        obs["images"] = images_dict
        
        prompt_val = raw_item.get("prompt", "")
        if prompt_val is None:
            prompt_val = ""
        obs["prompt"] = prompt_val
        
        # 6. 使用 policy.infer() 解码 (会自动反归一化)
        with torch.no_grad():
            result = policy.infer(obs, noise=noise)
        
        # 7. 提取动作 (已经经过反归一化)
        actions = result['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        else:
            actions = np.array(actions)
        
        if actions.ndim == 3:
            actions = actions[0]
        
        all_decoded_chunks.append((idx, actions))
        
        # 8. Execute
        to_exec = min(args.execution_horizon, actions.shape[0])
        for t in range(to_exec):
            executed_trajectory.append(actions[t])
        
        chunk_boundaries.append(len(executed_trajectory))

    return np.array(executed_trajectory), chunk_boundaries, all_decoded_chunks


def run_simulation_random(args, policy, raw_dataset, transformed_dataset, action_mean, action_std):
    """
    使用随机高斯噪声进行解码的模拟。
    
    不需要 HDF5 文件，直接使用随机噪声。
    """
    print(f"\n>>> Running Simulation Mode: random (independent Gaussian)")
    
    current_idx = args.start_idx
    executed_trajectory = [] 
    chunk_boundaries = [] 
    all_decoded_chunks = []
    
    sim_indices = []
    for _ in range(args.num_steps_to_sim):
        sim_indices.append(current_idx)
        current_idx += args.execution_horizon

    device = args.device
    model = policy._model
    
    for step_i, idx in enumerate(tqdm(sim_indices, desc="random")):
        # 1. 检查边界
        if idx >= len(raw_dataset):
            print(f"Index {idx} out of bounds. Stopping.")
            break
        
        # 2. 生成随机噪声
        noise_shape = (1, model.config.action_horizon, model.config.action_dim)
        noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
        
        # 3. 从原始数据集获取观测
        raw_item = raw_dataset[idx]
        
        obs = {}
        state_val = raw_item.get("observation.state")
        if state_val is None:
            state_val = raw_item.get("state")
        if isinstance(state_val, torch.Tensor):
            state_val = state_val.numpy()
        obs["state"] = state_val
        
        images_dict = {}
        for key in raw_item.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                img_val = raw_item[key]
                if isinstance(img_val, torch.Tensor):
                    img_val = img_val.numpy()
                images_dict[cam_name] = img_val
        obs["images"] = images_dict
        
        prompt_val = raw_item.get("prompt", "")
        if prompt_val is None:
            prompt_val = ""
        obs["prompt"] = prompt_val
        
        # 4. 使用 policy.infer() 解码 (会自动反归一化)
        with torch.no_grad():
            result = policy.infer(obs, noise=noise)
        
        # 5. 提取动作
        actions = result['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        else:
            actions = np.array(actions)
        
        if actions.ndim == 3:
            actions = actions[0]
        
        all_decoded_chunks.append((idx, actions))
        
        # 6. Execute
        to_exec = min(args.execution_horizon, actions.shape[0])
        for t in range(to_exec):
            executed_trajectory.append(actions[t])
        
        chunk_boundaries.append(len(executed_trajectory))

    return np.array(executed_trajectory), chunk_boundaries, all_decoded_chunks


def run_simulation_with_overlap(args, policy, raw_dataset, f_h5, mode="random_overlap"):
    """
    带噪声重叠的模拟：上一个 chunk 末尾的噪声覆盖下一个 chunk 开头的噪声。
    
    核心思想：
    - 每个 chunk 的噪声 shape 是 [1, horizon, dim] = [1, 50, 32]
    - 执行 execution_horizon 步 (如 15 步)
    - 在生成下一个 chunk 时，用上一个 chunk 中位置 [execution_horizon - overlap : execution_horizon] 的噪声
      覆盖新 chunk 的位置 [0 : overlap]
    
    Modes:
        - "random_overlap": 基础噪声是随机高斯，但带重叠
        - "inverse_overlap": 基础噪声从 HDF5 加载，但带重叠
    """
    print(f"\n>>> Running Simulation Mode: {mode} (overlap_steps={args.overlap_steps})")
    
    current_idx = args.start_idx
    executed_trajectory = [] 
    chunk_boundaries = [] 
    all_decoded_chunks = []
    
    sim_indices = []
    for _ in range(args.num_steps_to_sim):
        sim_indices.append(current_idx)
        current_idx += args.execution_horizon

    device = args.device
    model = policy._model
    
    # 保存上一个 chunk 的噪声，用于覆盖下一个 chunk 的开头
    prev_chunk_noise = None
    
    for step_i, idx in enumerate(tqdm(sim_indices, desc=mode)):
        # 1. 检查边界
        if idx >= len(raw_dataset):
            print(f"Index {idx} out of bounds. Stopping.")
            break
        
        # 2. 生成基础噪声
        noise_shape = (1, model.config.action_horizon, model.config.action_dim)
        
        if mode == "random_overlap":
            # 随机高斯噪声
            noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
        elif mode == "inverse_overlap":
            # 从 HDF5 加载反推噪声
            if f_h5 is None or idx >= f_h5['latents'].shape[0]:
                print(f"HDF5 index {idx} out of bounds. Using random noise.")
                noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
            else:
                noise = torch.from_numpy(f_h5['latents'][idx]).unsqueeze(0).to(device)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 3. 关键步骤：用上一个 chunk 的末尾噪声覆盖当前 chunk 的开头
        if prev_chunk_noise is not None and args.overlap_steps > 0:
            # prev_chunk_noise: [1, horizon, dim]
            # 上一个 chunk 执行了 execution_horizon 步
            # 我们取上一个 chunk 的位置 [exec_horizon - overlap : exec_horizon] 的噪声
            # 覆盖当前 chunk 的位置 [0 : overlap]
            
            overlap = args.overlap_steps
            exec_h = args.execution_horizon
            
            # 确保索引有效
            src_start = exec_h - overlap
            src_end = exec_h
            
            if src_start >= 0 and src_end <= prev_chunk_noise.shape[1]:
                # 覆盖当前噪声的前 overlap 个位置
                noise[:, :overlap, :] = prev_chunk_noise[:, src_start:src_end, :].clone()
        
        # 4. 保存当前噪声用于下一个 chunk
        prev_chunk_noise = noise.clone()
        
        # 5. 从原始数据集获取观测
        raw_item = raw_dataset[idx]
        
        obs = {}
        state_val = raw_item.get("observation.state")
        if state_val is None:
            state_val = raw_item.get("state")
        if isinstance(state_val, torch.Tensor):
            state_val = state_val.numpy()
        obs["state"] = state_val
        
        images_dict = {}
        for key in raw_item.keys():
            if key.startswith("observation.images."):
                cam_name = key.replace("observation.images.", "")
                img_val = raw_item[key]
                if isinstance(img_val, torch.Tensor):
                    img_val = img_val.numpy()
                images_dict[cam_name] = img_val
        obs["images"] = images_dict
        
        prompt_val = raw_item.get("prompt", "")
        if prompt_val is None:
            prompt_val = ""
        obs["prompt"] = prompt_val
        
        # 6. 使用 policy.infer() 解码
        with torch.no_grad():
            result = policy.infer(obs, noise=noise)
        
        # 7. 提取动作
        actions = result['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        else:
            actions = np.array(actions)
        
        if actions.ndim == 3:
            actions = actions[0]
        
        all_decoded_chunks.append((idx, actions))
        
        # 8. Execute
        to_exec = min(args.execution_horizon, actions.shape[0])
        for t in range(to_exec):
            executed_trajectory.append(actions[t])
        
        chunk_boundaries.append(len(executed_trajectory))

    return np.array(executed_trajectory), chunk_boundaries, all_decoded_chunks


def main(args: Args):
    # 1. Load Model Config
    print(f"Loading policy config {args.policy_config}...")
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
    # 2. Init PI0 Model (训练好的)
    print("Initializing PI0 Model...")
    # Use create_trained_policy to correctly load weights AND normalizers
    policy = _policy_config.create_trained_policy(
        server_config, 
        args.policy_dir, 
        pytorch_device=args.device
    )

    # --- DEBUG: Verify Loading ---
    print("--- POLICY VERIFICATION ---")
    if hasattr(policy, "_model") and hasattr(policy._model, "paligemma_with_expert"):
        # Check a specific parameter to verify weights are non-random (or at least loaded)
        w = next(policy._model.paligemma_with_expert.parameters())
        print(f"DEBUG: Model Weight Sample (Mean): {w.float().mean().item():.6f}")
        print(f"DEBUG: Model Weight Sample (Std):  {w.float().std().item():.6f}")
    
    # Check normalization and extract action stats for unnormalization
    has_norm = False
    action_mean = None
    action_std = None
    
    if hasattr(policy, "_input_transform"):
        for i, t in enumerate(policy._input_transform.transforms):
            if "Normalize" in str(type(t)) and "Unnormalize" not in str(type(t)):
                has_norm = True
                print(f"DEBUG: Found Normalize transform at index {i}")
                # property is norm_stats, not stats
                if hasattr(t, "norm_stats") and t.norm_stats:
                    print(f"DEBUG: Norm Stats Loaded for keys: {list(t.norm_stats.keys())}")
                    if "action" in t.norm_stats:
                         action_mean = t.norm_stats['action'].mean
                         action_std = t.norm_stats['action'].std
                         print(f"DEBUG: Action Norm Mean (first 5): {action_mean[:5]}")
                         print(f"DEBUG: Action Norm Std  (first 5): {action_std[:5]}")
                else:
                    print("DEBUG: norm_stats attribute is empty or None!")
    
    if not has_norm:
        print("WARNING: No Normalize transform found within policy!")
    if action_mean is None:
        print("WARNING: Could not extract action normalization stats!")
    print("-------------------------")
    
    # Extract the underlying PI0Pytorch model
    if not hasattr(policy, "_model") or not isinstance(policy._model, PI0Pytorch):
        raise ValueError("Policy does not contain a PI0Pytorch model!")
        
    model = policy._model
    model.eval()
    model.to(args.device)
    
    # 3. 加载残差网络
    print("--- RESIDUAL NET LOADING ---")
    residual_net = None
    residual_model_path = os.path.join(args.residual_net_dir, "best_model.pth")
    delta_stats_path = os.path.join(args.residual_net_dir, "delta_stats.pt")
    
    if os.path.exists(residual_model_path):
        print(f"Loading ResidualTransitionNet from {residual_model_path}...")
        
        # 初始化模型结构
        residual_net = ResidualTransitionNet(
            latent_dim=args.residual_net_latent_dim,
            horizon=args.residual_net_horizon,
            cond_dim=args.residual_net_cond_dim,
            hidden_dim=args.residual_net_hidden_dim,
            num_blocks=args.residual_net_num_blocks
        ).to(args.device)
        
        # 加载权重
        state_dict = torch.load(residual_model_path, map_location=args.device, weights_only=False)
        residual_net.load_state_dict(state_dict)
        residual_net.eval()
        
        # 加载 delta 统计量
        if os.path.exists(delta_stats_path):
            delta_stats = torch.load(delta_stats_path, map_location=args.device, weights_only=False)
            residual_net.set_delta_stats(delta_stats['mean'], delta_stats['std'])
            print(f"  Delta Mean range: [{delta_stats['mean'].min():.4f}, {delta_stats['mean'].max():.4f}]")
            print(f"  Delta Std range:  [{delta_stats['std'].min():.4f}, {delta_stats['std'].max():.4f}]")
        else:
            print(f"  WARNING: delta_stats.pt not found, using unit stats")
        
        print(f"  Loaded successfully! Params: {sum(p.numel() for p in residual_net.parameters())/1e6:.2f}M")
    else:
        print(f"WARNING: Residual model not found at {residual_model_path}")
        print("  Residual mode will be skipped!")
    print("-------------------------")
    
    # 3.5 加载 Stochastic Motion Prior 网络
    print("--- STOCHASTIC PRIOR LOADING ---")
    stochastic_prior = None
    stochastic_model_path = os.path.join(args.stochastic_prior_dir, "best_model.pth")
    
    if os.path.exists(stochastic_model_path):
        print(f"Loading StochasticMotionPrior from {stochastic_model_path}...")
        
        # 加载 checkpoint（包含 config）
        checkpoint = torch.load(stochastic_model_path, map_location=args.device, weights_only=False)
        
        # 如果 checkpoint 中有 config，使用它；否则使用命令行参数
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            stochastic_prior = StochasticMotionPrior(
                latent_dim=cfg.get('latent_dim', args.stochastic_prior_latent_dim),
                hidden_dim=cfg.get('hidden_dim', args.stochastic_prior_hidden_dim),
                num_layers=cfg.get('num_layers', args.stochastic_prior_num_layers),
                cond_dim=cfg.get('cond_dim', args.stochastic_prior_cond_dim) if 'cond_dim' in cfg else 2048,
                cond_proj_dim=cfg.get('cond_proj_dim', args.stochastic_prior_cond_proj_dim),
            ).to(args.device)
            print(f"  Using config from checkpoint")
        else:
            stochastic_prior = StochasticMotionPrior(
                latent_dim=args.stochastic_prior_latent_dim,
                hidden_dim=args.stochastic_prior_hidden_dim,
                num_layers=args.stochastic_prior_num_layers,
                cond_dim=args.stochastic_prior_cond_dim,
                cond_proj_dim=args.stochastic_prior_cond_proj_dim,
            ).to(args.device)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            stochastic_prior.load_state_dict(checkpoint['model_state_dict'])
        else:
            stochastic_prior.load_state_dict(checkpoint)
        stochastic_prior.eval()
        
        print(f"  Loaded successfully! Params: {sum(p.numel() for p in stochastic_prior.parameters())/1e6:.2f}M")
        print(f"  Deterministic mode: {args.stochastic_prior_deterministic}")
    else:
        print(f"WARNING: Stochastic prior model not found at {stochastic_model_path}")
        print("  Stochastic mode will be skipped!")
    print("-------------------------")
    
    # 4. Load HDF5 Data (optional - may be locked if being generated)
    f_h5 = None
    h5_available = False
    print(f"Trying to load HDF5 from {args.dataset_path}...")
    try:
        f_h5 = h5py.File(args.dataset_path, 'r')
        h5_available = True
        print("  HDF5 loaded successfully!")
    except (BlockingIOError, OSError) as e:
        print(f"  WARNING: Could not open HDF5 file: {e}")
        print("  Skipping modes that require HDF5 (inverse, perturbed)")
        print("  Will only run: inverse_euler, inverse_rk4, random, ground_truth")
    
    # 5. Load Dataset (Need State + Masks)
    print("Loading original dataset for state/masks recovery...")
    data_config_factory = server_config.data
    data_config = data_config_factory.create(server_config.assets_dirs, server_config.model)
    repo_id = args.dataset_repo_id or data_config.repo_id
    
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=args.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)] 
            for key in data_config.action_sequence_keys
        },
    )
    from openpi import transforms as _transforms
    if data_config.prompt_from_task:
         dataset = data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    
    # 保存未归一化的原始数据集用于提取 Ground Truth 和 policy.infer()
    raw_dataset = dataset  # 在归一化之前保存引用
    
    # 创建 transformed_dataset 用于 inverse_online 模式（包含归一化后的 actions）
    transformed_dataset = data_loader.transform_dataset(dataset, data_config)
    print(f"Transformed dataset size: {len(transformed_dataset)}")

    # Run Experiments
    # 对比：ground_truth, inverse (HDF5 反推噪声), random (独立随机噪声), stochastic (训练的随机先验)
    if h5_available:
        modes = ["inverse", "random", "stochastic"]
    else:
        modes = ["random", "stochastic"]
    
    results = {}
    
    # Add Ground Truth Extraction - 从原始未归一化数据集提取
    print("Extracting Ground Truth Trajectory (from raw dataset)...")
    gt_trajectory = []
    
    # We follow the same indexing logic used in run_simulation
    current_idx = args.start_idx
    gt_indices = []
    for _ in range(args.num_steps_to_sim):
        gt_indices.append(current_idx)
        current_idx += args.execution_horizon
    
    # 确定 action key
    action_key = None
    sample_item = raw_dataset[0]
    for key in ['action', 'actions']:
        if key in sample_item:
            action_key = key
            break
    # LeRobot 格式通常是 action.xxx
    if action_key is None:
        for key in sample_item.keys():
            if key.startswith('action'):
                action_key = key
                break
    
    print(f"  Using action key: {action_key}")
    print(f"  Sample action shape: {sample_item[action_key].shape if action_key else 'N/A'}")
        
    for idx in gt_indices:
        if idx >= len(raw_dataset): break
        item = raw_dataset[idx]
        
        # 从原始数据集获取动作 (未归一化)
        actions = item.get(action_key)
        if actions is None:
            # Fallback: 尝试其他可能的 key
            actions = item.get('actions', item.get('action'))
        
        if isinstance(actions, torch.Tensor): 
            actions = actions.numpy()
        
        # Take the executed part
        to_exec = min(args.execution_horizon, actions.shape[0])
        for t in range(to_exec):
            act_t = actions[t]
            gt_trajectory.append(act_t)
            
    gt_trajectory = np.array(gt_trajectory)
    print(f"  Ground Truth shape: {gt_trajectory.shape}")
    print(f"  Ground Truth range: [{gt_trajectory.min():.4f}, {gt_trajectory.max():.4f}]")
    results["ground_truth"] = (gt_trajectory, [], []) # No boundaries/chunks needed for GT plotting
    
    # 计算原始 GT 的最大跳变
    gt_diffs = np.abs(np.diff(gt_trajectory, axis=0))
    gt_max_jump = gt_diffs.max()
    print(f"  Ground Truth max jump: {gt_max_jump:.6f}")
    
    # 运行各种模式
    for mode in modes:
        if mode == "inverse_euler":
            # 实时反推模式 (Euler)
            traj, bounds, chunks = run_simulation_inverse_online(
                args, policy, model, raw_dataset, transformed_dataset, action_mean, action_std, ode_method="euler"
            )
        elif mode == "inverse_rk4":
            # 实时反推模式 (RK4)
            traj, bounds, chunks = run_simulation_inverse_online(
                args, policy, model, raw_dataset, transformed_dataset, action_mean, action_std, ode_method="rk4"
            )
        elif mode == "random":
            # Random 模式不需要 HDF5，直接使用随机噪声
            traj, bounds, chunks = run_simulation_random(
                args, policy, raw_dataset, transformed_dataset, action_mean, action_std
            )
        elif mode in ["random_overlap", "inverse_overlap"]:
            # 带噪声重叠的模式
            traj, bounds, chunks = run_simulation_with_overlap(
                args, policy, raw_dataset, f_h5, mode=mode
            )
        else:
            # 其他模式（inverse, perturbed, stochastic）使用 HDF5
            if not h5_available:
                print(f"  Skipping mode '{mode}' - HDF5 not available")
                continue
            traj, bounds, chunks = run_simulation_with_policy(
                args, policy, raw_dataset, f_h5, mode=mode, 
                residual_net=residual_net, stochastic_prior=stochastic_prior
            )
        results[mode] = (traj, bounds, chunks)
        
    if f_h5 is not None:
        f_h5.close()
    
    print("Plotting combined results...")
    plot_combined_simulation(results, output_prefix="comparison")
    
    # 绘制 ΔAction 分布图
    print("Plotting Delta Action distributions...")
    plot_delta_action_distribution(results, output_prefix="delta_action")


def plot_gt_smoothing_comparison(results, dims=[0, 1, 2, 3, 4, 5], output_prefix="gt_smoothing"):
    """
    专门对比 Ground Truth 原始版本和不同低通滤波版本的效果。
    
    Args:
        results: dict {mode: (trajectory, boundaries, all_chunks)}
        dims: 要绘制的维度列表
        output_prefix: 输出文件前缀
    """
    gt_modes = ['ground_truth', 'ground_truth_lpf_5hz', 'ground_truth_lpf_2hz', 'ground_truth_lpf_1hz']
    colors = {
        'ground_truth': 'black',
        'ground_truth_lpf_5hz': '#e41a1c',  # 红
        'ground_truth_lpf_2hz': '#377eb8',  # 蓝
        'ground_truth_lpf_1hz': '#4daf4a',  # 绿
    }
    labels = {
        'ground_truth': 'Raw GT',
        'ground_truth_lpf_5hz': 'LPF 5Hz (light)',
        'ground_truth_lpf_2hz': 'LPF 2Hz (medium)',
        'ground_truth_lpf_1hz': 'LPF 1Hz (heavy)',
    }
    
    # 1. 轨迹对比图
    fig, axes = plt.subplots(len(dims), 1, figsize=(16, 3*len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]
    
    for i, dim in enumerate(dims):
        ax = axes[i]
        for mode in gt_modes:
            if mode not in results:
                continue
            trajectory, _, _ = results[mode]
            ax.plot(trajectory[:, dim], label=labels.get(mode, mode), 
                   color=colors.get(mode, 'gray'), 
                   linewidth=2.0 if mode == 'ground_truth' else 1.5,
                   alpha=0.9)
        
        ax.set_ylabel(f"Dim {dim}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Ground Truth: Raw vs Zero-Phase Low-Pass Filtered", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_trajectory.png", dpi=150)
    plt.close()
    print(f"Saved {output_prefix}_trajectory.png")
    
    # 2. ΔAction 对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, mode in enumerate(gt_modes):
        if mode not in results:
            continue
        trajectory, _, _ = results[mode]
        if len(trajectory) < 2:
            continue
        
        delta = np.diff(trajectory, axis=0).flatten()
        
        ax = axes[i]
        ax.hist(delta, bins=100, alpha=0.7, color=colors.get(mode, 'gray'), density=True)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f"{labels.get(mode, mode)}\nstd={delta.std():.4f}, |max|={np.abs(delta).max():.4f}")
        ax.set_xlabel("ΔAction")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("ΔAction Distribution: Raw vs Spline Smoothed GT", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_delta_dist.png", dpi=150)
    plt.close()
    print(f"Saved {output_prefix}_delta_dist.png")
    
    # 3. 局部放大对比（找到最大跳变的区域）
    if 'ground_truth' in results:
        gt_traj, _, _ = results['ground_truth']
        delta_gt = np.diff(gt_traj, axis=0)
        delta_abs = np.abs(delta_gt).sum(axis=1)  # 所有维度的绝对跳变和
        
        # 找到最大跳变点
        max_jump_idx = np.argmax(delta_abs)
        
        # 以该点为中心，取前后 50 帧
        start_idx = max(0, max_jump_idx - 50)
        end_idx = min(len(gt_traj), max_jump_idx + 50)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        
        for i, dim in enumerate(dims[:6]):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            for mode in gt_modes:
                if mode not in results:
                    continue
                trajectory, _, _ = results[mode]
                ax.plot(range(start_idx, end_idx), trajectory[start_idx:end_idx, dim], 
                       label=labels.get(mode, mode), color=colors.get(mode, 'gray'),
                       linewidth=2.0 if mode == 'ground_truth' else 1.5)
            
            ax.axvline(x=max_jump_idx, color='red', linestyle=':', alpha=0.5, label='Max Jump')
            ax.set_title(f"Dim {dim}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Zoom: Around Max Jump Point (idx={max_jump_idx})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_zoom.png", dpi=150)
        plt.close()
        print(f"Saved {output_prefix}_zoom.png")
    
    # 4. 打印统计对比
    print("\n=== GT Smoothing Statistics ===")
    for mode in gt_modes:
        if mode not in results:
            continue
        trajectory, _, _ = results[mode]
        if len(trajectory) < 2:
            continue
        delta = np.diff(trajectory, axis=0)
        print(f"\n{labels.get(mode, mode)}:")
        print(f"  ΔAction std: {delta.std():.6f}")
        print(f"  ΔAction |max|: {np.abs(delta).max():.6f}")
        print(f"  ΔAction |mean|: {np.abs(delta).mean():.6f}")


def plot_delta_action_distribution(results, dims=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], output_prefix="delta_action", unified_ylim=True):
    """
    绘制 ΔAction = action[t+1] - action[t] 的分布图。
    对比：ground_truth, inverse, random, stochastic
    
    Args:
        unified_ylim: 如果为 True，按维度绘制时使用统一的 X 轴范围（ΔAction 范围）
    """
    colors = {
        'ground_truth': 'black',         # 黑色 - Ground Truth
        'inverse': '#1f77b4',            # 蓝色 - HDF5 反推噪声
        'random': '#ff7f0e',             # 橙色 - 独立随机噪声
        'stochastic': '#2ca02c',         # 绿色 - Stochastic Prior
    }
    
    # 1. 计算每个模式的 ΔAction
    delta_actions = {}
    for mode, (trajectory, boundaries, _) in results.items():
        if len(trajectory) < 2:
            continue
        # ΔAction = action[t+1] - action[t]
        delta = np.diff(trajectory, axis=0)  # [T-1, D]
        delta_actions[mode] = delta
    
    # 2. 绘制所有维度汇总的分布图
    plt.figure(figsize=(16, 6))
    
    # 左图：所有维度合并的直方图
    plt.subplot(1, 2, 1)
    plot_modes = ['ground_truth', 'inverse', 'random', 'stochastic']
    for mode in plot_modes:
        if mode not in delta_actions:
            continue
        delta = delta_actions[mode]
        # 展平所有维度
        delta_flat = delta.flatten()
        plt.hist(delta_flat, bins=100, alpha=0.5, label=f"{mode} (std={delta_flat.std():.4f})", 
                 color=colors.get(mode, 'gray'), density=True)
    
    plt.xlabel("ΔAction")
    plt.ylabel("Density")
    plt.title("ΔAction Distribution (All Dimensions)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：边界处 vs 非边界处的 ΔAction 对比
    plt.subplot(1, 2, 2)
    for mode in ['inverse', 'random']:
        if mode not in results:
            continue
        trajectory, boundaries, _ = results[mode]
        if len(trajectory) < 2:
            continue
        
        delta = np.diff(trajectory, axis=0)
        
        # 分离边界处和非边界处的 delta
        boundary_set = set(b - 1 for b in boundaries if 0 < b <= len(delta))  # b-1 因为 diff 后索引偏移
        
        boundary_deltas = []
        non_boundary_deltas = []
        
        for t in range(len(delta)):
            if t in boundary_set:
                boundary_deltas.append(delta[t])
            else:
                non_boundary_deltas.append(delta[t])
        
        if boundary_deltas:
            boundary_deltas = np.array(boundary_deltas).flatten()
            plt.hist(boundary_deltas, bins=50, alpha=0.5, 
                     label=f"{mode} boundary (std={boundary_deltas.std():.4f})",
                     color=colors.get(mode, 'gray'), density=True, linestyle='--', histtype='step', linewidth=2)
        
        if non_boundary_deltas:
            non_boundary_deltas = np.array(non_boundary_deltas).flatten()
            plt.hist(non_boundary_deltas, bins=50, alpha=0.3,
                     label=f"{mode} non-boundary (std={non_boundary_deltas.std():.4f})",
                     color=colors.get(mode, 'gray'), density=True)
    
    plt.xlabel("ΔAction")
    plt.ylabel("Density")
    plt.title("ΔAction: Boundary vs Non-Boundary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_overview.png", dpi=150)
    plt.close()
    print(f"Saved {output_prefix}_overview.png")
    
    # 3. 按维度绘制分布图
    n_dims = len(dims)
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    # 计算全局 X 轴范围（ΔAction 范围）用于统一坐标轴
    global_delta_min, global_delta_max = float('inf'), float('-inf')
    if unified_ylim:
        for mode, delta in delta_actions.items():
            for dim in dims:
                if dim < delta.shape[1]:
                    dim_data = delta[:, dim]
                    global_delta_min = min(global_delta_min, dim_data.min())
                    global_delta_max = max(global_delta_max, dim_data.max())
        # 添加 5% 边距
        margin = (global_delta_max - global_delta_min) * 0.05
        global_delta_min -= margin
        global_delta_max += margin
        print(f"Unified ΔAction X-axis range: [{global_delta_min:.4f}, {global_delta_max:.4f}] rad")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
    
    for i, dim in enumerate(dims):
        ax = axes[i]
        
        for mode in ['ground_truth', 'inverse', 'random', 'stochastic']:
            if mode not in delta_actions:
                continue
            delta = delta_actions[mode]
            if dim >= delta.shape[1]:
                continue
            
            delta_dim = delta[:, dim]
            ax.hist(delta_dim, bins=50, alpha=0.5, label=f"{mode}", 
                    color=colors.get(mode, 'gray'), density=True)
        
        # 设置统一的 X 轴范围
        if unified_ylim:
            ax.set_xlim(global_delta_min, global_delta_max)
        
        ax.set_xlabel("ΔAction (rad)")
        ax.set_ylabel("Density")
        ax.set_title(f"Dim {dim}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_by_dim.png", dpi=150)
    plt.close()
    print(f"Saved {output_prefix}_by_dim.png")
    
    # 4. 打印统计信息
    print("\n=== ΔAction Statistics ===")
    for mode, delta in delta_actions.items():
        print(f"\n{mode}:")
        print(f"  Overall: mean={delta.mean():.6f}, std={delta.std():.6f}, "
              f"min={delta.min():.6f}, max={delta.max():.6f}")
        
        # 边界处统计
        if mode in results and mode != 'ground_truth':
            _, boundaries, _ = results[mode]
            boundary_set = set(b - 1 for b in boundaries if 0 < b <= len(delta))
            
            boundary_deltas = [delta[t] for t in range(len(delta)) if t in boundary_set]
            non_boundary_deltas = [delta[t] for t in range(len(delta)) if t not in boundary_set]
            
            if boundary_deltas:
                bd = np.array(boundary_deltas)
                print(f"  Boundary:     mean={bd.mean():.6f}, std={bd.std():.6f}")
            if non_boundary_deltas:
                nbd = np.array(non_boundary_deltas)
                print(f"  Non-Boundary: mean={nbd.mean():.6f}, std={nbd.std():.6f}")


def plot_combined_simulation(results, dims=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12 ], output_prefix="comparison", unified_ylim=True):
    """
    results: dict {mode: (trajectory, boundaries, all_chunks)}
    Generates one plot per dimension, comparing all modes.
    对比：ground_truth, inverse, random, stochastic
    
    Args:
        unified_ylim: 如果为 True，所有维度使用统一的 Y 轴范围，便于跨维度比较
    """
    colors = {
        'ground_truth': 'black',        # 黑色 - Ground Truth
        'inverse': '#1f77b4',           # 蓝色 - HDF5 反推噪声
        'random': '#ff7f0e',            # 橙色 - 独立随机噪声
        'stochastic': '#2ca02c',        # 绿色 - Stochastic Prior
    } 
    linestyles = {
        'ground_truth': '-',            # 实线
        'inverse': '-',                 # 实线
        'random': '--',                 # 虚线
        'stochastic': '-.',             # 点划线 - Stochastic Prior
    }
    widths = {
        'ground_truth': 2.5,            # 加粗
        'inverse': 2.0,
        'random': 1.5,
        'stochastic': 1.8,              # Stochastic Prior
    }
    
    # 计算全局 Y 轴范围（跨所有模式和指定维度）
    global_min, global_max = float('inf'), float('-inf')
    if unified_ylim:
        for mode, (trajectory, _, _) in results.items():
            for dim in dims:
                if dim < trajectory.shape[1]:
                    dim_data = trajectory[:, dim]
                    global_min = min(global_min, dim_data.min())
                    global_max = max(global_max, dim_data.max())
        # 添加 5% 边距
        margin = (global_max - global_min) * 0.05
        global_min -= margin
        global_max += margin
        print(f"Unified Y-axis range: [{global_min:.4f}, {global_max:.4f}] rad")
    
    for dim in dims:
        plt.figure(figsize=(18, 6))
        
        # 绘图顺序：先画 random，再画 stochastic，再画 inverse，最后画 ground_truth
        plot_order = ["random", "stochastic", "inverse", "ground_truth"]
        
        for mode in plot_order:
            if mode not in results: continue
            
            trajectory, boundaries, _ = results[mode]
            
            # Calculate metrics (Average Boundary Jump)
            avg_jump = 0
            if mode == "ground_truth":
                label = "Ground Truth"
            else:
                diffs = []
                for b in boundaries[:-1]:
                    if b < len(trajectory):
                        jump = trajectory[b, dim] - trajectory[b-1, dim]
                        diffs.append(abs(jump))
                avg_jump = np.mean(diffs) if diffs else 0
                # 更友好的模式名称
                mode_names = {
                    'inverse': 'Inverse (HDF5 反推噪声)',
                    'random': 'Random (独立随机噪声)',
                    'stochastic': 'Stochastic (训练的随机先验)',
                }
                display_name = mode_names.get(mode, mode.title())
                label = f"{display_name} (Avg Jump: {avg_jump:.4f} rad)"
            
            plt.plot(
                trajectory[:, dim], 
                label=label, 
                color=colors.get(mode, 'gray'), 
                linestyle=linestyles.get(mode, '-'),
                alpha=0.9,  # 统一透明度
                linewidth=widths.get(mode, 1.5)
            )
        
        # 设置统一的 Y 轴范围
        if unified_ylim:
            plt.ylim(global_min, global_max)
        
        plt.title(f"Action Trajectory Continuity Comparison - Dimension {dim}")
        plt.ylabel("Action Value (rad)")
        plt.xlabel("Simulation Steps")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{output_prefix}_dim_{dim}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved comparison to {filename}")



def plot_simulation_continuity(trajectory, boundaries, all_chunks, dims=[0, 1, 2], output_file="sim_continuity.png", unified_ylim=True):
    """
    trajectory: [Total_Steps, D] - The concatenated executed path
    boundaries: List[int] - Step indices where chunk switching occurred
    all_chunks: List of (start_idx, chunk_data) - The full predicted chunks
    
    Args:
        unified_ylim: 如果为 True，所有维度使用统一的 Y 轴范围
    """
    # 计算全局 Y 轴范围
    global_min, global_max = float('inf'), float('-inf')
    if unified_ylim:
        for dim in dims:
            if dim < trajectory.shape[1]:
                global_min = min(global_min, trajectory[:, dim].min())
                global_max = max(global_max, trajectory[:, dim].max())
            for _, chunk_data in all_chunks:
                if dim < chunk_data.shape[1]:
                    global_min = min(global_min, chunk_data[:, dim].min())
                    global_max = max(global_max, chunk_data[:, dim].max())
        margin = (global_max - global_min) * 0.05
        global_min -= margin
        global_max += margin
    
    fig, axes = plt.subplots(len(dims), 1, figsize=(15, 5 * len(dims)), sharex=True)
    if len(dims) == 1: axes = [axes]
    
    colors_cmap = plt.cm.viridis(np.linspace(0, 1, len(all_chunks)))
    
    for i, dim in enumerate(dims):
        ax = axes[i]
        
        # 1. Plot Full Decoded Chunks (Ghost lines) to see what the model PLANNED
        for chunk_idx, (start_time, chunk_data) in enumerate(all_chunks):
            # Map chunk time to relative global time for visualization 
            # Note: start_time is the dataset index.
            # But here we want to visualize the *composition*.
            # The i-th chunk starts execution at boundaries[i-1] (or 0).
            
            exec_start_t = 0 if chunk_idx == 0 else boundaries[chunk_idx-1]
            time_axis = np.arange(exec_start_t, exec_start_t + chunk_data.shape[0])
            
            ax.plot(time_axis, chunk_data[:, dim], color=colors_cmap[chunk_idx], alpha=0.3, linestyle='--', linewidth=1)
            
        # 2. Plot Executed Trajectory (Solid line)
        ax.plot(trajectory[:, dim], color='black', linewidth=2, label="Executed")
        
        # 设置统一的 Y 轴范围
        if unified_ylim:
            ax.set_ylim(global_min, global_max)
        
        # 3. Mark Boundaries and Calculate Jumps
        diffs = []
        for b_idx, b in enumerate(boundaries[:-1]): # Skip last boundary (end of time)
            ax.axvline(x=b - 0.5, color='red', linestyle=':', alpha=0.5)
            
            # Jump calculation: actions[b] - actions[b-1]
            if b < len(trajectory):
                jump = trajectory[b, dim] - trajectory[b-1, dim]
                diffs.append(abs(jump))
                # Annotate jump
                # ax.text(b, trajectory[b, dim], f"{jump:.3f}", fontsize=8, color='red')
        
        avg_jump = np.mean(diffs) if diffs else 0
        ax.set_title(f"Dimension {dim} (Avg Boundary Jump: {avg_jump:.4f} rad)")
        ax.set_ylabel("Action Value (rad)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved simulation visualization to {output_file}")


def plot_chunk_continuity(chunks, start_idx=0, dims=[0, 1, 2], output_file="chunk_continuity.png"):
    """
    chunks: [N_frames, Horizon, Action_Dim]
    """
    N, H, D = chunks.shape
    
    fig, axes = plt.subplots(len(dims), 1, figsize=(15, 4 * len(dims)), sharex=True)
    if len(dims) == 1: axes = [axes]
    
    # Colors for different chunks to distinguish them
    colors = plt.cm.jet(np.linspace(0, 1, N))
    
    for i, dim in enumerate(dims):
        ax = axes[i]
        ax.set_title(f"Action Dimension {dim}")
        ax.set_ylabel("Value")
        
        for t in range(N):
            # The chunk at frame t covers time [t + start_idx, t + start_idx + H]
            # (Assuming 1Hz or aligned steps)
            time_axis = np.arange(t, t + H) + start_idx
            chunk_data = chunks[t, :, dim]
            
            ax.plot(time_axis, chunk_data, color=colors[t], alpha=0.7, linewidth=1.5)
            # Mark the start of the chunk
            ax.scatter(time_axis[0], chunk_data[0], color=colors[t], s=20)

    plt.xlabel("Global Time Step")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved visualization to {output_file}")


if __name__ == "__main__":
    tyro.cli(main)
