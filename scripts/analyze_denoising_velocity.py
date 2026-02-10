"""
分析 PI0 去噪过程中每一步速度的贡献

目的：
1. 记录正向 ODE 去噪过程中的每一步速度场
2. 计算每一步速度对最终动作变化的贡献
3. 可视化各步的贡献比例

关键概念：
- 总变化量：Δx_total = actions - noise
- 每步变化量：Δx_step = dt * v_t
- 贡献度：可以从多个角度衡量
  1. 范数贡献：||Δx_step|| / sum(||Δx_step||)
  2. 方向贡献：Δx_step · Δx_total / ||Δx_total||² (投影到总变化方向的贡献)
  3. 累积进度：||x_t - noise|| / ||actions - noise|| (朝目标移动的进度)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import logging
import tyro
import os

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.policies import policy_config as _policy_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models import model as _model
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from lerobot.common.datasets import lerobot_dataset

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    device: str = "cuda"
    num_ode_steps: int = 10
    test_idx: int = 100
    output_dir: str = "velocity_contribution_analysis"
    num_samples: int = 5  # 分析多少个不同的样本


@dataclass
class DenoiseRecord:
    """记录去噪过程的所有中间状态"""
    noise: torch.Tensor                    # 初始噪声
    actions: torch.Tensor                  # 最终动作
    trajectories: List[torch.Tensor]       # 每一步的 x_t
    velocities: List[torch.Tensor]         # 每一步的速度 v_t
    times: List[float]                     # 每一步的时间 t
    step_changes: List[torch.Tensor]       # 每一步的变化量 dt * v_t
    direction: str = "forward"             # "forward" (noise->actions) 或 "backward" (actions->noise)


@torch.no_grad()
def forward_ode_with_full_recording(
    model: PI0Pytorch,
    noise: torch.Tensor,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    num_steps: int
) -> DenoiseRecord:
    """
    正向 ODE 同时记录完整的去噪过程。
    
    ODE: dx/dt = v(x, t)
    离散化: x_{t+1} = x_t + dt * v(x_t, t)
    
    正向 ODE: t 从 1 到 0，dt = -1/N
    """
    device = noise.device
    bsize = noise.shape[0]
    
    dt = -1.0 / num_steps  # 负步长，从 t=1 到 t=0
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    trajectories = [noise.clone()]  # 包含初始点
    velocities = []
    times = []
    step_changes = []
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        # 计算速度场
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        # 保存当前状态
        velocities.append(v_t.clone())
        times.append(time.item())
        
        # Euler 更新
        delta_x = dt * v_t
        step_changes.append(delta_x.clone())
        
        x_t = x_t + delta_x
        trajectories.append(x_t.clone())
        
        time = time + dt
    
    return DenoiseRecord(
        noise=noise,
        actions=x_t,
        trajectories=trajectories,
        velocities=velocities,
        times=times,
        step_changes=step_changes,
        direction="forward",
    )


@torch.no_grad()
def backward_ode_with_full_recording(
    model: PI0Pytorch,
    actions: torch.Tensor,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    num_steps: int
) -> DenoiseRecord:
    """
    反向 ODE 记录完整的加噪过程（从 actions 到 noise）。
    
    ODE: dx/dt = v(x, t)
    离散化: x_{t+1} = x_t + dt * v(x_t, t)
    
    反向 ODE: t 从 0 到 1，dt = +1/N
    """
    device = actions.device
    bsize = actions.shape[0]
    
    dt = 1.0 / num_steps  # 正步长，从 t=0 到 t=1
    x_t = actions.clone()
    time = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    trajectories = [actions.clone()]  # 包含初始点
    velocities = []
    times = []
    step_changes = []
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        # 计算速度场
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        # 保存当前状态
        velocities.append(v_t.clone())
        times.append(time.item())
        
        # Euler 更新
        delta_x = dt * v_t
        step_changes.append(delta_x.clone())
        
        x_t = x_t + delta_x
        trajectories.append(x_t.clone())
        
        time = time + dt
    
    return DenoiseRecord(
        noise=x_t,       # 反向 ODE 的终点是噪声
        actions=actions, # 反向 ODE 的起点是动作
        trajectories=trajectories,
        velocities=velocities,
        times=times,
        step_changes=step_changes,
        direction="backward",
    )


def compare_forward_backward_velocities(
    forward_record: DenoiseRecord,
    backward_record: DenoiseRecord,
) -> Dict[str, np.ndarray]:
    """
    比较正向和反向 ODE 过程的速度场。
    
    关键点：在同一个 (x, t) 位置，正向和反向的速度场应该相同。
    但由于轨迹不同，我们比较在相同 t 下的速度差异。
    """
    assert len(forward_record.velocities) == len(backward_record.velocities)
    num_steps = len(forward_record.velocities)
    
    # 正向时间: 1.0, 0.9, 0.8, ..., 0.1
    # 反向时间: 0.0, 0.1, 0.2, ..., 0.9
    # 对应关系: forward[i] 时间 = 1 - i/N, backward[i] 时间 = i/N
    # 要比较相同 t 下的速度，需要反转其中一个
    
    velocity_errors = []      # 速度场误差的范数
    velocity_rel_errors = []  # 相对误差
    position_diffs = []       # 位置差异
    
    for i in range(num_steps):
        # 正向第 i 步对应时间 t = 1 - i/N
        # 反向第 (N-1-i) 步对应时间 t = (N-1-i)/N ≈ 1 - i/N (近似)
        # 但更精确的是：正向 t=0.9 对应反向 t=0.9
        
        # 为了简单，我们比较相同步数索引下的速度
        # (虽然时间不同，但可以看 t 接近时速度场的变化规律)
        
        v_fwd = forward_record.velocities[i].cpu().numpy()
        # 反向过程的对应步: 时间相近的步
        j = num_steps - 1 - i  # 对应相近时间的反向步
        if j >= 0 and j < num_steps:
            v_bwd = backward_record.velocities[j].cpu().numpy()
            
            # 位置
            x_fwd = forward_record.trajectories[i].cpu().numpy()
            x_bwd = backward_record.trajectories[j].cpu().numpy()
            
            # 速度误差
            v_error = np.linalg.norm(v_fwd - v_bwd)
            v_norm = (np.linalg.norm(v_fwd) + np.linalg.norm(v_bwd)) / 2 + 1e-8
            v_rel_error = v_error / v_norm
            
            velocity_errors.append(v_error)
            velocity_rel_errors.append(v_rel_error)
            
            # 位置差异
            pos_diff = np.linalg.norm(x_fwd - x_bwd)
            position_diffs.append(pos_diff)
    
    # 计算轨迹重合度
    # 理想情况下：forward[0] = backward[-1], forward[-1] = backward[0]
    fwd_start = forward_record.trajectories[0].cpu().numpy()
    fwd_end = forward_record.trajectories[-1].cpu().numpy()
    bwd_start = backward_record.trajectories[0].cpu().numpy()
    bwd_end = backward_record.trajectories[-1].cpu().numpy()
    
    # 循环误差: 正向起点 vs 反向终点, 正向终点 vs 反向起点
    start_match_error = np.linalg.norm(fwd_start - bwd_end)  # 应该接近 0
    end_match_error = np.linalg.norm(fwd_end - bwd_start)    # 应该接近 0
    
    return {
        'velocity_errors': np.array(velocity_errors),
        'velocity_rel_errors': np.array(velocity_rel_errors),
        'position_diffs': np.array(position_diffs),
        'start_match_error': start_match_error,
        'end_match_error': end_match_error,
        'forward_times': np.array(forward_record.times),
        'backward_times': np.array(backward_record.times),
    }


def compute_velocity_contributions(record: DenoiseRecord) -> Dict[str, np.ndarray]:
    """
    计算每一步速度的各种贡献指标
    
    Returns:
        dict with:
        - norm_contribution: 每步变化的范数贡献比例
        - direction_contribution: 朝目标方向的贡献
        - cumulative_progress: 累积进度
        - velocity_norms: 速度范数
        - step_norms: 每步变化量的范数
    """
    noise = record.noise.squeeze(0).cpu().numpy()  # [H, D]
    actions = record.actions.squeeze(0).cpu().numpy()  # [H, D]
    
    total_change = actions - noise  # 总变化量
    total_change_norm = np.linalg.norm(total_change)
    
    num_steps = len(record.velocities)
    H, D = noise.shape
    
    # 各种指标
    norm_contribution = []      # 范数贡献比例
    direction_contribution = [] # 方向贡献
    cumulative_progress = []    # 累积进度
    velocity_norms = []         # 速度范数
    step_norms = []             # 每步变化范数
    velocity_magnitudes_per_dim = []  # 每维度的速度分量
    
    current_pos = noise.copy()
    
    for i, (delta_x, v_t) in enumerate(zip(record.step_changes, record.velocities)):
        delta_x_np = delta_x.squeeze(0).cpu().numpy()  # [H, D]
        v_t_np = v_t.squeeze(0).cpu().numpy()
        
        # 1. 每步变化的范数
        step_norm = np.linalg.norm(delta_x_np)
        step_norms.append(step_norm)
        
        # 2. 速度范数
        v_norm = np.linalg.norm(v_t_np)
        velocity_norms.append(v_norm)
        
        # 3. 方向贡献：投影到总变化方向
        # projection = (delta_x · total_change) / ||total_change||²
        dot_product = np.sum(delta_x_np * total_change)
        dir_contrib = dot_product / (total_change_norm ** 2 + 1e-8)
        direction_contribution.append(dir_contrib)
        
        # 4. 累积进度：当前位置相对于目标的距离比例
        current_pos = current_pos + delta_x_np
        distance_from_noise = np.linalg.norm(current_pos - noise)
        progress = distance_from_noise / (total_change_norm + 1e-8)
        cumulative_progress.append(progress)
        
        # 5. 每维度的速度幅度
        velocity_magnitudes_per_dim.append(np.abs(v_t_np))
    
    # 范数贡献比例（归一化）
    total_step_norms = sum(step_norms)
    norm_contribution = [s / (total_step_norms + 1e-8) for s in step_norms]
    
    return {
        'norm_contribution': np.array(norm_contribution),
        'direction_contribution': np.array(direction_contribution),
        'cumulative_progress': np.array(cumulative_progress),
        'velocity_norms': np.array(velocity_norms),
        'step_norms': np.array(step_norms),
        'velocity_magnitudes_per_dim': np.stack(velocity_magnitudes_per_dim),  # [steps, H, D]
        'times': np.array(record.times),
        'total_change_norm': total_change_norm,
    }


def plot_contribution_analysis(
    contributions: Dict[str, np.ndarray],
    sample_idx: int,
    output_dir: str
):
    """绘制单个样本的贡献分析图"""
    
    times = contributions['times']
    num_steps = len(times)
    steps = np.arange(num_steps)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Denoising Velocity Contribution Analysis (Sample {sample_idx})', fontsize=14)
    
    # 1. 范数贡献比例（饼图）
    ax1 = axes[0, 0]
    norm_contrib = contributions['norm_contribution']
    colors = plt.cm.viridis(np.linspace(0, 1, num_steps))
    wedges, texts, autotexts = ax1.pie(
        norm_contrib, 
        labels=[f't={t:.2f}' for t in times],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax1.set_title('Step Contribution by Norm')
    
    # 2. 范数贡献（条形图）
    ax2 = axes[0, 1]
    bars = ax2.bar(steps, norm_contrib * 100, color=colors, edgecolor='black')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Contribution (%)')
    ax2.set_title('Norm Contribution per Step')
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f'{t:.2f}' for t in times], rotation=45)
    
    # 添加百分比标签
    for bar, val in zip(bars, norm_contrib * 100):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. 方向贡献（朝目标方向的投影）
    ax3 = axes[0, 2]
    dir_contrib = contributions['direction_contribution']
    ax3.bar(steps, dir_contrib * 100, color=colors, edgecolor='black')
    ax3.axhline(y=100/num_steps, color='red', linestyle='--', label=f'Uniform ({100/num_steps:.1f}%)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Direction Contribution (%)')
    ax3.set_title('Contribution Along Total Change Direction')
    ax3.set_xticks(steps)
    ax3.set_xticklabels([f'{t:.2f}' for t in times], rotation=45)
    ax3.legend()
    
    # 4. 累积进度
    ax4 = axes[1, 0]
    ax4.plot(steps, contributions['cumulative_progress'], 'b-o', linewidth=2, markersize=8)
    ax4.fill_between(steps, 0, contributions['cumulative_progress'], alpha=0.3)
    ax4.axhline(y=1.0, color='green', linestyle='--', label='Target (100%)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Progress toward Actions')
    ax4.set_title('Cumulative Denoising Progress')
    ax4.set_xticks(steps)
    ax4.set_xticklabels([f'{t:.2f}' for t in times], rotation=45)
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 速度范数随时间变化
    ax5 = axes[1, 1]
    ax5.plot(times[::-1], contributions['velocity_norms'][::-1], 'r-o', linewidth=2, markersize=8)
    ax5.set_xlabel('Time t (noise → action)')
    ax5.set_ylabel('Velocity Norm ||v_t||')
    ax5.set_title('Velocity Magnitude vs Time')
    ax5.invert_xaxis()  # t 从 1 到 0
    ax5.grid(True, alpha=0.3)
    
    # 6. 每步变化量范数
    ax6 = axes[1, 2]
    ax6.bar(steps, contributions['step_norms'], color='purple', edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Step Change Norm ||Δx||')
    ax6.set_title(f'Change Magnitude per Step (Total: {contributions["total_change_norm"]:.2f})')
    ax6.set_xticks(steps)
    ax6.set_xticklabels([f'{t:.2f}' for t in times], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'contribution_analysis_sample{sample_idx}.png'), dpi=150)
    plt.close()


def plot_velocity_heatmap(
    contributions: Dict[str, np.ndarray],
    sample_idx: int,
    output_dir: str,
    dims_to_show: int = 14  # 只显示有效动作维度
):
    """绘制速度场的热力图（按维度和时间步）"""
    
    vel_mags = contributions['velocity_magnitudes_per_dim']  # [steps, H, D]
    times = contributions['times']
    num_steps, H, D = vel_mags.shape
    
    # 只取第一个 horizon 位置和前 dims_to_show 维度
    vel_slice = vel_mags[:, 0, :dims_to_show]  # [steps, dims_to_show]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 速度幅度热力图
    ax1 = axes[0]
    im1 = ax1.imshow(vel_slice.T, aspect='auto', cmap='viridis')
    ax1.set_xlabel('ODE Step')
    ax1.set_ylabel('Action Dimension')
    ax1.set_title(f'Velocity Magnitude per Step (Sample {sample_idx})')
    ax1.set_xticks(range(num_steps))
    ax1.set_xticklabels([f't={t:.2f}' for t in times], rotation=45)
    plt.colorbar(im1, ax=ax1, label='|v_t|')
    
    # 2. 归一化的速度幅度（每步归一化到 0-1）
    ax2 = axes[1]
    vel_normalized = vel_slice / (vel_slice.max(axis=1, keepdims=True) + 1e-8)
    im2 = ax2.imshow(vel_normalized.T, aspect='auto', cmap='plasma')
    ax2.set_xlabel('ODE Step')
    ax2.set_ylabel('Action Dimension')
    ax2.set_title(f'Normalized Velocity per Step (Sample {sample_idx})')
    ax2.set_xticks(range(num_steps))
    ax2.set_xticklabels([f't={t:.2f}' for t in times], rotation=45)
    plt.colorbar(im2, ax=ax2, label='Normalized |v_t|')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'velocity_heatmap_sample{sample_idx}.png'), dpi=150)
    plt.close()


def plot_aggregated_analysis(
    all_contributions: List[Dict[str, np.ndarray]],
    output_dir: str
):
    """聚合多个样本的分析结果"""
    
    num_samples = len(all_contributions)
    num_steps = len(all_contributions[0]['times'])
    times = all_contributions[0]['times']
    
    # 聚合数据
    norm_contribs = np.array([c['norm_contribution'] for c in all_contributions])
    dir_contribs = np.array([c['direction_contribution'] for c in all_contributions])
    progress = np.array([c['cumulative_progress'] for c in all_contributions])
    vel_norms = np.array([c['velocity_norms'] for c in all_contributions])
    
    # 计算均值和标准差
    norm_mean, norm_std = norm_contribs.mean(axis=0), norm_contribs.std(axis=0)
    dir_mean, dir_std = dir_contribs.mean(axis=0), dir_contribs.std(axis=0)
    prog_mean, prog_std = progress.mean(axis=0), progress.std(axis=0)
    vel_mean, vel_std = vel_norms.mean(axis=0), vel_norms.std(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Aggregated Denoising Analysis ({num_samples} samples)', fontsize=14)
    
    steps = np.arange(num_steps)
    
    # 1. 平均范数贡献
    ax1 = axes[0, 0]
    ax1.bar(steps, norm_mean * 100, yerr=norm_std * 100, capsize=5, color='steelblue', edgecolor='black')
    ax1.axhline(y=100/num_steps, color='red', linestyle='--', label=f'Uniform ({100/num_steps:.1f}%)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Contribution (%)')
    ax1.set_title('Average Norm Contribution per Step')
    ax1.set_xticks(steps)
    ax1.set_xticklabels([f't={t:.2f}' for t in times], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均方向贡献
    ax2 = axes[0, 1]
    ax2.bar(steps, dir_mean * 100, yerr=dir_std * 100, capsize=5, color='coral', edgecolor='black')
    ax2.axhline(y=100/num_steps, color='red', linestyle='--', label=f'Uniform ({100/num_steps:.1f}%)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Direction Contribution (%)')
    ax2.set_title('Average Contribution Along Total Change')
    ax2.set_xticks(steps)
    ax2.set_xticklabels([f't={t:.2f}' for t in times], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均累积进度
    ax3 = axes[1, 0]
    ax3.plot(steps, prog_mean, 'b-o', linewidth=2, markersize=8, label='Mean')
    ax3.fill_between(steps, prog_mean - prog_std, prog_mean + prog_std, alpha=0.3)
    ax3.axhline(y=1.0, color='green', linestyle='--', label='Target')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Progress toward Actions')
    ax3.set_title('Average Cumulative Progress')
    ax3.set_xticks(steps)
    ax3.set_xticklabels([f't={t:.2f}' for t in times], rotation=45)
    ax3.set_ylim(0, 1.2)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 平均速度范数
    ax4 = axes[1, 1]
    ax4.errorbar(times[::-1], vel_mean[::-1], yerr=vel_std[::-1], fmt='r-o', 
                 linewidth=2, markersize=8, capsize=5, label='Mean ± Std')
    ax4.set_xlabel('Time t (noise → action)')
    ax4.set_ylabel('Velocity Norm ||v_t||')
    ax4.set_title('Average Velocity Magnitude vs Time')
    ax4.invert_xaxis()
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_analysis.png'), dpi=150)
    plt.close()
    
    # 打印统计摘要
    print("\n" + "=" * 60)
    print("CONTRIBUTION SUMMARY (Averaged over all samples)")
    print("=" * 60)
    print(f"\nStep-by-step contribution breakdown:")
    print("-" * 50)
    print(f"{'Step':<6} {'Time':<8} {'Norm%':<12} {'Direction%':<12} {'Progress':<12}")
    print("-" * 50)
    
    for i, t in enumerate(times):
        print(f"{i:<6} t={t:.2f}   {norm_mean[i]*100:>6.1f}±{norm_std[i]*100:.1f}   "
              f"{dir_mean[i]*100:>6.1f}±{dir_std[i]*100:.1f}   {prog_mean[i]:>6.2f}±{prog_std[i]:.2f}")
    
    print("-" * 50)
    print(f"\nKey Findings:")
    
    # 找出贡献最大的步骤
    max_contrib_step = np.argmax(norm_mean)
    print(f"  - Largest contribution: Step {max_contrib_step} (t={times[max_contrib_step]:.2f}) "
          f"with {norm_mean[max_contrib_step]*100:.1f}% of total change")
    
    # 检查是否存在明显的不均匀性
    uniform_contrib = 1.0 / num_steps
    max_deviation = np.max(np.abs(norm_mean - uniform_contrib))
    print(f"  - Max deviation from uniform: {max_deviation*100:.1f}%")
    
    # 检查速度是否随时间变化
    vel_trend = vel_mean[-1] / vel_mean[0] if vel_mean[0] > 0 else float('inf')
    print(f"  - Velocity ratio (end/start): {vel_trend:.2f}")
    
    if vel_trend > 1.5:
        print("    → Velocity INCREASES toward the end (more aggressive near t=0)")
    elif vel_trend < 0.67:
        print("    → Velocity DECREASES toward the end (more aggressive at start)")
    else:
        print("    → Velocity remains relatively STABLE")


def plot_forward_backward_comparison(
    forward_record: DenoiseRecord,
    backward_record: DenoiseRecord,
    comparison: Dict[str, np.ndarray],
    sample_idx: int,
    output_dir: str
):
    """
    可视化正向和反向 ODE 的对比分析。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Forward vs Backward ODE Comparison - Sample {sample_idx}', fontsize=14)
    
    # 1. 轨迹对比（选取一个维度）
    ax1 = axes[0, 0]
    fwd_traj = np.array([t.squeeze(0)[:, 0].cpu().numpy().mean() for t in forward_record.trajectories])
    bwd_traj = np.array([t.squeeze(0)[:, 0].cpu().numpy().mean() for t in backward_record.trajectories])
    ax1.plot(fwd_traj, 'b-o', label='Forward (noise→action)', markersize=5)
    ax1.plot(bwd_traj, 'r-s', label='Backward (action→noise)', markersize=5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Value (Dim 0)')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 速度场误差
    ax2 = axes[0, 1]
    steps = np.arange(len(comparison['velocity_errors']))
    ax2.bar(steps, comparison['velocity_errors'], color='coral', alpha=0.7)
    ax2.set_xlabel('Step (matched by time)')
    ax2.set_ylabel('||v_fwd - v_bwd||')
    ax2.set_title('Velocity Field Error (at similar times)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 相对误差
    ax3 = axes[0, 2]
    ax3.bar(steps, comparison['velocity_rel_errors'] * 100, color='purple', alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Velocity Relative Error')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 正向速度范数 vs 时间
    ax4 = axes[1, 0]
    fwd_v_norms = [v.norm().item() for v in forward_record.velocities]
    bwd_v_norms = [v.norm().item() for v in backward_record.velocities]
    fwd_times = forward_record.times
    bwd_times = backward_record.times
    ax4.plot(fwd_times, fwd_v_norms, 'b-o', label='Forward', markersize=6)
    ax4.plot(bwd_times, bwd_v_norms, 'r-s', label='Backward', markersize=6)
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Velocity Norm ||v_t||')
    ax4.set_title('Velocity Magnitude vs Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 每步变化范数
    ax5 = axes[1, 1]
    fwd_step_norms = [s.norm().item() for s in forward_record.step_changes]
    bwd_step_norms = [s.norm().item() for s in backward_record.step_changes]
    width = 0.35
    x = np.arange(len(fwd_step_norms))
    ax5.bar(x - width/2, fwd_step_norms, width, label='Forward', color='blue', alpha=0.7)
    ax5.bar(x + width/2, bwd_step_norms, width, label='Backward', color='red', alpha=0.7)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('||dt * v_t||')
    ax5.set_title('Step Change Magnitude')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 循环一致性
    ax6 = axes[1, 2]
    labels = ['Noise Match\n(fwd_start vs bwd_end)', 'Action Match\n(fwd_end vs bwd_start)']
    errors = [comparison['start_match_error'], comparison['end_match_error']]
    colors = ['green' if e < 1.0 else 'orange' if e < 5.0 else 'red' for e in errors]
    bars = ax6.bar(labels, errors, color=colors, alpha=0.7)
    ax6.set_ylabel('L2 Error')
    ax6.set_title('Cycle Consistency Error')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for bar, err in zip(bars, errors):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{err:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'forward_backward_comparison_{sample_idx}.png'), dpi=150)
    plt.close()


def plot_forward_backward_contributions(
    forward_contributions: Dict[str, np.ndarray],
    backward_contributions: Dict[str, np.ndarray],
    sample_idx: int,
    output_dir: str
):
    """
    对比正向和反向 ODE 过程中每一步速度的贡献。
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Forward vs Backward Velocity Contributions - Sample {sample_idx}', fontsize=14)
    
    fwd_times = forward_contributions['times']
    bwd_times = backward_contributions['times']
    num_steps = len(fwd_times)
    steps = np.arange(num_steps)
    width = 0.35
    
    # 1. 范数贡献对比
    ax1 = axes[0, 0]
    fwd_norm = forward_contributions['norm_contribution'] * 100
    bwd_norm = backward_contributions['norm_contribution'] * 100
    ax1.bar(steps - width/2, fwd_norm, width, label='Forward (noise→action)', color='blue', alpha=0.7)
    ax1.bar(steps + width/2, bwd_norm, width, label='Backward (action→noise)', color='red', alpha=0.7)
    ax1.axhline(y=100/num_steps, color='gray', linestyle='--', label='Uniform')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Norm Contribution (%)')
    ax1.set_title('Step Contribution by Norm')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 方向贡献对比
    ax2 = axes[0, 1]
    fwd_dir = forward_contributions['direction_contribution'] * 100
    bwd_dir = backward_contributions['direction_contribution'] * 100
    ax2.bar(steps - width/2, fwd_dir, width, label='Forward', color='blue', alpha=0.7)
    ax2.bar(steps + width/2, bwd_dir, width, label='Backward', color='red', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Direction Contribution (%)')
    ax2.set_title('Contribution Along Target Direction')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 速度范数对比 vs 时间
    ax3 = axes[0, 2]
    ax3.plot(fwd_times, forward_contributions['velocity_norms'], 'b-o', 
             label='Forward', markersize=6)
    ax3.plot(bwd_times, backward_contributions['velocity_norms'], 'r-s', 
             label='Backward', markersize=6)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Velocity Norm ||v_t||')
    ax3.set_title('Velocity Magnitude vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 每步变化量对比
    ax4 = axes[1, 0]
    ax4.bar(steps - width/2, forward_contributions['step_norms'], width, 
            label='Forward', color='blue', alpha=0.7)
    ax4.bar(steps + width/2, backward_contributions['step_norms'], width, 
            label='Backward', color='red', alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Step Change ||dt * v_t||')
    ax4.set_title('Step Change Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 累积进度对比
    ax5 = axes[1, 1]
    ax5.plot(steps, forward_contributions['cumulative_progress'], 'b-o', 
             label='Forward Progress', linewidth=2, markersize=6)
    ax5.plot(steps, backward_contributions['cumulative_progress'], 'r-s', 
             label='Backward Progress', linewidth=2, markersize=6)
    ax5.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Progress')
    ax5.set_title('Cumulative Progress')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.2)
    
    # 6. 贡献差异分析
    ax6 = axes[1, 2]
    norm_diff = np.abs(fwd_norm - bwd_norm)
    dir_diff = np.abs(fwd_dir - bwd_dir)
    ax6.bar(steps - width/2, norm_diff, width, label='Norm Contrib Diff', color='purple', alpha=0.7)
    ax6.bar(steps + width/2, dir_diff, width, label='Dir Contrib Diff', color='orange', alpha=0.7)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('|Forward - Backward| (%)')
    ax6.set_title('Contribution Difference (Forward vs Backward)')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'forward_backward_contributions_{sample_idx}.png'), dpi=150)
    plt.close()


def plot_aggregated_backward_analysis(
    all_backward_contributions: List[Dict[str, np.ndarray]],
    output_dir: str
):
    """
    聚合反向 ODE 过程的贡献分析。
    """
    if len(all_backward_contributions) == 0:
        return
    
    num_samples = len(all_backward_contributions)
    num_steps = len(all_backward_contributions[0]['times'])
    times = all_backward_contributions[0]['times']
    steps = np.arange(num_steps)
    
    # 聚合数据
    norm_all = np.array([c['norm_contribution'] for c in all_backward_contributions])
    dir_all = np.array([c['direction_contribution'] for c in all_backward_contributions])
    vel_all = np.array([c['velocity_norms'] for c in all_backward_contributions])
    step_all = np.array([c['step_norms'] for c in all_backward_contributions])
    
    norm_mean, norm_std = norm_all.mean(axis=0), norm_all.std(axis=0)
    dir_mean, dir_std = dir_all.mean(axis=0), dir_all.std(axis=0)
    vel_mean, vel_std = vel_all.mean(axis=0), vel_all.std(axis=0)
    step_mean, step_std = step_all.mean(axis=0), step_all.std(axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Aggregated Backward ODE Analysis ({num_samples} samples)', fontsize=14)
    
    # 1. 平均范数贡献
    ax1 = axes[0, 0]
    ax1.bar(steps, norm_mean * 100, yerr=norm_std * 100, capsize=4, color='red', alpha=0.7)
    ax1.axhline(y=100/num_steps, color='gray', linestyle='--', label=f'Uniform ({100/num_steps:.1f}%)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Norm Contribution (%)')
    ax1.set_title('Backward: Average Norm Contribution per Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 平均方向贡献
    ax2 = axes[0, 1]
    ax2.bar(steps, dir_mean * 100, yerr=dir_std * 100, capsize=4, color='orange', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Direction Contribution (%)')
    ax2.set_title('Backward: Average Direction Contribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 平均速度范数
    ax3 = axes[1, 0]
    ax3.errorbar(times, vel_mean, yerr=vel_std, fmt='r-o', capsize=4, markersize=6)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Velocity Norm ||v_t||')
    ax3.set_title('Backward: Average Velocity Magnitude vs Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. 平均每步变化
    ax4 = axes[1, 1]
    ax4.errorbar(steps, step_mean, yerr=step_std, fmt='r-s', capsize=4, markersize=6)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Step Change ||dt * v_t||')
    ax4.set_title('Backward: Average Step Change Magnitude')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_backward_analysis.png'), dpi=150)
    plt.close()
    
    # 打印统计
    print("\n" + "=" * 60)
    print("BACKWARD ODE CONTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"\nStep-by-step contribution breakdown (Backward: action → noise):")
    print("-" * 50)
    print(f"{'Step':<6} {'Time':<8} {'Norm%':<12} {'Direction%':<12}")
    print("-" * 50)
    
    for i, t in enumerate(times):
        print(f"{i:<6} t={t:.2f}   {norm_mean[i]*100:>6.1f}±{norm_std[i]*100:.1f}   "
              f"{dir_mean[i]*100:>6.1f}±{dir_std[i]*100:.1f}")


def plot_aggregated_forward_backward(
    all_comparisons: List[Dict[str, np.ndarray]],
    output_dir: str
):
    """
    聚合多个样本的正反向对比结果。
    """
    if len(all_comparisons) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Aggregated Forward-Backward Comparison ({len(all_comparisons)} samples)', fontsize=14)
    
    # 提取数据
    all_vel_errors = np.array([c['velocity_errors'] for c in all_comparisons])
    all_rel_errors = np.array([c['velocity_rel_errors'] for c in all_comparisons])
    all_start_match = np.array([c['start_match_error'] for c in all_comparisons])
    all_end_match = np.array([c['end_match_error'] for c in all_comparisons])
    
    # 1. 平均速度误差 per step
    ax1 = axes[0, 0]
    mean_err = all_vel_errors.mean(axis=0)
    std_err = all_vel_errors.std(axis=0)
    steps = np.arange(len(mean_err))
    ax1.errorbar(steps, mean_err, yerr=std_err, fmt='o-', capsize=4, color='coral')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Velocity Error ||v_fwd - v_bwd||')
    ax1.set_title('Average Velocity Field Error per Step')
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均相对误差 per step
    ax2 = axes[0, 1]
    mean_rel = all_rel_errors.mean(axis=0) * 100
    std_rel = all_rel_errors.std(axis=0) * 100
    ax2.errorbar(steps, mean_rel, yerr=std_rel, fmt='s-', capsize=4, color='purple')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Average Velocity Relative Error per Step')
    ax2.grid(True, alpha=0.3)
    
    # 3. 循环一致性分布
    ax3 = axes[1, 0]
    ax3.hist(all_start_match, bins=15, alpha=0.6, label='Noise Match Error', color='green')
    ax3.hist(all_end_match, bins=15, alpha=0.6, label='Action Match Error', color='blue')
    ax3.axvline(all_start_match.mean(), color='green', linestyle='--', linewidth=2)
    ax3.axvline(all_end_match.mean(), color='blue', linestyle='--', linewidth=2)
    ax3.set_xlabel('L2 Error')
    ax3.set_ylabel('Count')
    ax3.set_title('Cycle Consistency Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 总结统计
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    === Forward-Backward Comparison Summary ===
    
    Velocity Field Error:
      - Mean: {all_vel_errors.mean():.4f} ± {all_vel_errors.std():.4f}
      - Max:  {all_vel_errors.max():.4f}
    
    Relative Error:
      - Mean: {all_rel_errors.mean()*100:.2f}% ± {all_rel_errors.std()*100:.2f}%
      - Max:  {all_rel_errors.max()*100:.2f}%
    
    Cycle Consistency:
      - Noise Match (fwd_start ≈ bwd_end): {all_start_match.mean():.4f} ± {all_start_match.std():.4f}
      - Action Match (fwd_end ≈ bwd_start): {all_end_match.mean():.4f} ± {all_end_match.std():.4f}
    
    Interpretation:
      - Low error = velocity field is consistent
      - High cycle error = ODE is not perfectly reversible
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aggregated_forward_backward.png'), dpi=150)
    plt.close()
    
    # 打印统计
    print("\n" + "=" * 60)
    print("FORWARD-BACKWARD COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nVelocity Field Error (at matched times):")
    print(f"  Mean: {all_vel_errors.mean():.4f} ± {all_vel_errors.std():.4f}")
    print(f"  Max:  {all_vel_errors.max():.4f}")
    print(f"\nRelative Error:")
    print(f"  Mean: {all_rel_errors.mean()*100:.2f}% ± {all_rel_errors.std()*100:.2f}%")
    print(f"\nCycle Consistency (should be ~0 for perfect ODE):")
    print(f"  Noise Match Error:  {all_start_match.mean():.4f} ± {all_start_match.std():.4f}")
    print(f"  Action Match Error: {all_end_match.mean():.4f} ± {all_end_match.std():.4f}")


@torch.no_grad()
def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Denoising Velocity Contribution Analysis")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. Loading policy...")
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
    policy = _policy_config.create_trained_policy(
        server_config,
        args.policy_dir,
        pytorch_device=args.device
    )
    
    model: PI0Pytorch = policy._model
    model.eval()
    
    print(f"   action_dim={model.config.action_dim}, action_horizon={model.config.action_horizon}")
    
    # 2. 加载数据集
    print("\n2. Loading dataset...")
    data_config_factory = server_config.data
    data_config = data_config_factory.create(server_config.assets_dirs, server_config.model)
    repo_id = args.dataset_repo_id or data_config.repo_id
    
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    raw_dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=args.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    
    from openpi import transforms as _transforms
    if data_config.prompt_from_task:
        raw_dataset = data_loader.TransformedDataset(
            raw_dataset, 
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    
    transformed_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    print(f"   Dataset size: {len(transformed_dataset)}")
    
    # 3. 分析多个样本
    print(f"\n3. Analyzing {args.num_samples} samples...")
    print("    (Forward ODE: noise → actions, Backward ODE: actions → noise)")
    
    all_contributions = []              # 正向 ODE 的贡献
    all_backward_contributions = []     # 反向 ODE 的贡献
    all_comparisons = []                # 正反向对比结果
    
    for sample_i in range(args.num_samples):
        idx = args.test_idx + sample_i * 50  # 间隔取样
        if idx >= len(transformed_dataset):
            print(f"   Skipping sample {sample_i}: index {idx} out of range")
            continue
        
        print(f"\n   Sample {sample_i} (idx={idx}):")
        
        # 获取数据并准备观测
        item = transformed_dataset[idx]
        
        def to_tensor(v):
            if isinstance(v, torch.Tensor):
                return v.to(args.device)
            elif isinstance(v, np.ndarray):
                return torch.from_numpy(v).to(args.device)
            else:
                return torch.tensor(v, device=args.device)
        
        images_raw = item['image']
        camera_names = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
        
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
                    image_masks_dict[cam_name] = torch.ones(1, dtype=torch.bool, device=args.device)
        
        state_raw = to_tensor(item['state'])
        if state_raw.shape[-1] < 32:
            state_padded = torch.zeros(32, device=args.device)
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
        
        # 生成随机噪声
        noise_shape = (1, model.config.action_horizon, model.config.action_dim)
        noise = torch.randn(noise_shape, device=args.device)
        
        # ============================================
        # Part A: 正向 ODE 分析 (noise → actions)
        # ============================================
        print(f"      [Forward ODE: noise → actions]")
        forward_record = forward_ode_with_full_recording(
            model, noise, state, prefix_pad_masks, past_key_values, args.num_ode_steps
        )
        
        # 计算正向贡献
        forward_contributions = compute_velocity_contributions(forward_record)
        all_contributions.append(forward_contributions)
        
        print(f"        Noise range: [{noise.min():.3f}, {noise.max():.3f}]")
        print(f"        Actions range: [{forward_record.actions.min():.3f}, {forward_record.actions.max():.3f}]")
        print(f"        Total change norm: {forward_contributions['total_change_norm']:.3f}")
        
        # ============================================
        # Part B: 反向 ODE 分析 (actions → noise)
        # ============================================
        print(f"      [Backward ODE: actions → noise]")
        backward_record = backward_ode_with_full_recording(
            model, forward_record.actions.clone(), state, prefix_pad_masks, past_key_values, args.num_ode_steps
        )
        
        # 计算反向贡献
        backward_contributions = compute_velocity_contributions(backward_record)
        all_backward_contributions.append(backward_contributions)
        
        print(f"        Reconstructed noise range: [{backward_record.noise.min():.3f}, {backward_record.noise.max():.3f}]")
        reconstruction_error = (backward_record.noise - noise).norm().item()
        print(f"        Reconstruction error (||z_reconstructed - z_original||): {reconstruction_error:.4f}")
        
        # ============================================
        # Part C: 正反向速度场对比
        # ============================================
        comparison = compare_forward_backward_velocities(forward_record, backward_record)
        all_comparisons.append(comparison)
        
        print(f"      [Forward-Backward Comparison]")
        print(f"        Avg velocity error: {comparison['velocity_errors'].mean():.4f}")
        print(f"        Cycle consistency (noise): {comparison['start_match_error']:.4f}")
        print(f"        Cycle consistency (action): {comparison['end_match_error']:.4f}")
        
        # 绘制单样本分析图
        # 正向贡献分析
        plot_contribution_analysis(forward_contributions, sample_i, args.output_dir)
        plot_velocity_heatmap(forward_contributions, sample_i, args.output_dir)
        # 正反向对比
        plot_forward_backward_comparison(forward_record, backward_record, comparison, sample_i, args.output_dir)
        # 正反向贡献对比（新增）
        plot_forward_backward_contributions(forward_contributions, backward_contributions, sample_i, args.output_dir)
    
    # 4. 聚合正向 ODE 分析
    if len(all_contributions) > 0:
        print(f"\n4. Aggregated forward ODE analysis over {len(all_contributions)} samples...")
        plot_aggregated_analysis(all_contributions, args.output_dir)
    
    # 5. 聚合反向 ODE 分析（新增）
    if len(all_backward_contributions) > 0:
        print(f"\n5. Aggregated backward ODE analysis over {len(all_backward_contributions)} samples...")
        plot_aggregated_backward_analysis(all_backward_contributions, args.output_dir)
    
    # 6. 聚合正反向速度场对比
    if len(all_comparisons) > 0:
        print(f"\n6. Aggregated forward-backward comparison over {len(all_comparisons)} samples...")
        plot_aggregated_forward_backward(all_comparisons, args.output_dir)
    
    print(f"\n\nAll figures saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
