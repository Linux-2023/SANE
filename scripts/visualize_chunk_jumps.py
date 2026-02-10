"""
可视化 Chunk 间跳变

基于 check_chunk_continuity.py 中的独立随机噪声和反推噪声生成的动作轨迹，
计算并可视化 chunk 边界处的跳变情况。

主要功能：
1. 计算每个 chunk 边界处的跳变大小
2. 对比不同模式（random vs inverse）的跳变分布
3. 可视化跳变随时间的变化
4. 按维度分析跳变特性
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tyro
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import logging
import os
from tqdm import tqdm
import jax

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies import policy_config as _policy_config
from lerobot.common.datasets import lerobot_dataset
from openpi.models import model as _model
from openpi import transforms as _transforms

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    # 模型路径
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    config_name: str = "pi05_piper_pick_cube"
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    
    # HDF5 路径（用于反推噪声）
    h5_path: str = "latent_dataset_10.h5"
    
    # 模拟参数
    start_idx: int = 10000
    num_chunks: int = 50  # 要模拟的 chunk 数量
    execution_horizon: int = 15  # 每个 chunk 执行的步数
    overlap_steps: int = 5  # 用于计算跳变的重叠步数
    
    # 输出
    output_dir: str = "chunk_jump_analysis"
    device: str = "cuda"


def run_simulation(
    args: Args,
    policy,
    raw_dataset,
    h5_file: Optional[h5py.File],
    mode: str = "random",
) -> Tuple[np.ndarray, List[int], List[Tuple[int, np.ndarray]]]:
    """
    运行模拟，返回执行轨迹、chunk 边界和所有解码的 chunks
    """
    print(f"\n>>> Running Simulation Mode: {mode}")
    
    current_idx = args.start_idx
    executed_trajectory = []
    chunk_boundaries = []
    all_decoded_chunks = []
    
    sim_indices = []
    for _ in range(args.num_chunks):
        sim_indices.append(current_idx)
        current_idx += args.execution_horizon

    device = args.device
    model = policy._model
    
    for step_i, idx in enumerate(tqdm(sim_indices, desc=mode)):
        if idx >= len(raw_dataset):
            print(f"Index {idx} out of bounds. Stopping.")
            break
        
        # 生成噪声
        noise_shape = (1, model.config.action_horizon, model.config.action_dim)
        
        if mode == "random":
            noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
        elif mode == "inverse" and h5_file is not None:
            # 从 HDF5 加载反推噪声
            if idx < h5_file['latents'].shape[0]:
                latent = h5_file['latents'][idx]
                noise = torch.from_numpy(latent).float().to(device).unsqueeze(0)
            else:
                noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
        else:
            noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
        
        # 获取观测
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
        
        # 解码动作
        with torch.no_grad():
            result = policy.infer(obs, noise=noise)
        
        actions = result['actions']
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        else:
            actions = np.array(actions)
        
        if actions.ndim == 3:
            actions = actions[0]
        
        all_decoded_chunks.append((idx, actions.copy()))
        
        # 执行
        to_exec = min(args.execution_horizon, actions.shape[0])
        for t in range(to_exec):
            executed_trajectory.append(actions[t])
        
        chunk_boundaries.append(len(executed_trajectory))

    return np.array(executed_trajectory), chunk_boundaries, all_decoded_chunks


def compute_chunk_jumps(
    all_chunks: List[Tuple[int, np.ndarray]],
    execution_horizon: int = 15,
    overlap_steps: int = 5,
) -> Dict[str, np.ndarray]:
    """
    计算 chunk 边界处的跳变
    
    返回：
    - jump_magnitudes: 每个边界的总跳变大小 [num_boundaries]
    - jump_per_dim: 每个边界每个维度的跳变 [num_boundaries, action_dim]
    - overlap_jumps: 重叠区域每步的跳变 [num_boundaries, overlap_steps, action_dim]
    """
    num_chunks = len(all_chunks)
    if num_chunks < 2:
        return {}
    
    action_dim = all_chunks[0][1].shape[1]
    num_boundaries = num_chunks - 1
    
    jump_magnitudes = []
    jump_per_dim = []
    overlap_jumps = []
    first_step_jumps = []  # 第一步的跳变（最重要）
    
    for i in range(num_boundaries):
        _, prev_chunk = all_chunks[i]
        _, next_chunk = all_chunks[i + 1]
        
        # 前一个 chunk 执行的最后几步
        prev_end = prev_chunk[execution_horizon - overlap_steps:execution_horizon]
        # 下一个 chunk 的前几步
        next_start = next_chunk[:overlap_steps]
        
        # 计算重叠区域的跳变
        overlap_diff = next_start - prev_end  # [overlap_steps, action_dim]
        overlap_jumps.append(overlap_diff)
        
        # 第一步跳变（chunk 边界处）
        first_jump = next_chunk[0] - prev_chunk[execution_horizon - 1]
        first_step_jumps.append(first_jump)
        
        # 每个维度的平均跳变
        dim_jump = np.abs(overlap_diff).mean(axis=0)
        jump_per_dim.append(dim_jump)
        
        # 总跳变大小（L2 范数）
        total_jump = np.sqrt((first_jump ** 2).sum())
        jump_magnitudes.append(total_jump)
    
    return {
        'jump_magnitudes': np.array(jump_magnitudes),
        'jump_per_dim': np.array(jump_per_dim),
        'overlap_jumps': np.array(overlap_jumps),
        'first_step_jumps': np.array(first_step_jumps),
    }


def plot_jump_comparison(
    results: Dict[str, Dict],
    output_dir: str,
    action_dim: int = 14,
):
    """
    绘制跳变对比图
    """
    os.makedirs(output_dir, exist_ok=True)
    
    modes = list(results.keys())
    colors = {'random': 'red', 'inverse': 'blue', 'ground_truth': 'green'}
    
    # ============================================
    # 图1: 跳变大小随 chunk 的变化
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 总跳变大小
    ax = axes[0, 0]
    for mode in modes:
        if 'jump_magnitudes' in results[mode]:
            jumps = results[mode]['jump_magnitudes']
            ax.plot(jumps, label=mode, color=colors.get(mode, 'gray'), alpha=0.8)
    ax.set_xlabel('Chunk Boundary Index')
    ax.set_ylabel('Jump Magnitude (L2)')
    ax.set_title('Total Jump Magnitude at Chunk Boundaries')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.2 跳变分布直方图
    ax = axes[0, 1]
    for mode in modes:
        if 'jump_magnitudes' in results[mode]:
            jumps = results[mode]['jump_magnitudes']
            ax.hist(jumps, bins=30, alpha=0.5, label=f'{mode} (μ={jumps.mean():.4f})', 
                   color=colors.get(mode, 'gray'))
    ax.set_xlabel('Jump Magnitude')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Jump Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1.3 按维度的平均跳变
    ax = axes[1, 0]
    x = np.arange(action_dim)
    width = 0.35
    for i, mode in enumerate(modes):
        if 'jump_per_dim' in results[mode]:
            dim_jumps = results[mode]['jump_per_dim'][:, :action_dim].mean(axis=0)
            offset = (i - len(modes)/2 + 0.5) * width
            ax.bar(x + offset, dim_jumps, width, label=mode, color=colors.get(mode, 'gray'), alpha=0.7)
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Mean Jump')
    ax.set_title('Mean Jump per Action Dimension')
    ax.legend()
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 1.4 统计摘要
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = "Jump Statistics Summary\n" + "=" * 40 + "\n\n"
    for mode in modes:
        if 'jump_magnitudes' in results[mode]:
            jumps = results[mode]['jump_magnitudes']
            stats_text += f"{mode.upper()}:\n"
            stats_text += f"  Mean:   {jumps.mean():.4f}\n"
            stats_text += f"  Std:    {jumps.std():.4f}\n"
            stats_text += f"  Max:    {jumps.max():.4f}\n"
            stats_text += f"  Min:    {jumps.min():.4f}\n"
            stats_text += f"  Median: {np.median(jumps):.4f}\n\n"
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jump_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/jump_comparison.png")
    
    # ============================================
    # 图2: 重叠区域详细分析
    # ============================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 选择几个典型的 chunk 边界进行详细展示
    sample_boundaries = [0, len(results[modes[0]]['jump_magnitudes']) // 2, -1]
    
    for ax_idx, boundary_idx in enumerate(sample_boundaries[:2]):
        ax = axes[ax_idx // 2, ax_idx % 2]
        
        for mode in modes:
            if 'overlap_jumps' in results[mode]:
                overlap = results[mode]['overlap_jumps'][boundary_idx]  # [overlap_steps, action_dim]
                # 只看前 action_dim 维
                overlap = overlap[:, :action_dim]
                jump_norm = np.sqrt((overlap ** 2).sum(axis=1))
                ax.plot(jump_norm, 'o-', label=mode, color=colors.get(mode, 'gray'))
        
        ax.set_xlabel('Overlap Step')
        ax.set_ylabel('Jump Norm')
        ax.set_title(f'Jump in Overlap Region (Boundary {boundary_idx})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2.3 平均重叠跳变
    ax = axes[1, 0]
    for mode in modes:
        if 'overlap_jumps' in results[mode]:
            overlap_all = results[mode]['overlap_jumps']  # [num_boundaries, overlap_steps, action_dim]
            overlap_norm = np.sqrt((overlap_all[:, :, :action_dim] ** 2).sum(axis=2))  # [num_boundaries, overlap_steps]
            mean_overlap = overlap_norm.mean(axis=0)
            std_overlap = overlap_norm.std(axis=0)
            x = np.arange(len(mean_overlap))
            ax.plot(x, mean_overlap, 'o-', label=mode, color=colors.get(mode, 'gray'))
            ax.fill_between(x, mean_overlap - std_overlap, mean_overlap + std_overlap, 
                           color=colors.get(mode, 'gray'), alpha=0.2)
    ax.set_xlabel('Overlap Step (0 = boundary)')
    ax.set_ylabel('Mean Jump Norm')
    ax.set_title('Average Jump Across Overlap Region')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2.4 第一步跳变的维度分解
    ax = axes[1, 1]
    for mode in modes:
        if 'first_step_jumps' in results[mode]:
            first_jumps = results[mode]['first_step_jumps'][:, :action_dim]  # [num_boundaries, action_dim]
            mean_first = np.abs(first_jumps).mean(axis=0)
            ax.plot(mean_first, 'o-', label=mode, color=colors.get(mode, 'gray'))
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Mean |First Step Jump|')
    ax.set_title('First Step Jump by Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/overlap_analysis.png")
    
    # ============================================
    # 图3: 轨迹可视化（前几个维度）
    # ============================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    dims_to_plot = min(6, action_dim)
    
    for dim_idx in range(dims_to_plot):
        ax = axes[dim_idx // 2, dim_idx % 2]
        
        for mode, data in results.items():
            if 'trajectory' in data:
                traj = data['trajectory'][:, dim_idx]
                ax.plot(traj, label=mode, color=colors.get(mode, 'gray'), alpha=0.7)
                
                # 标记 chunk 边界
                if 'boundaries' in data:
                    for b in data['boundaries'][:-1]:
                        ax.axvline(x=b, color=colors.get(mode, 'gray'), 
                                  linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Action Dim {dim_idx}')
        ax.set_title(f'Trajectory Comparison - Dimension {dim_idx}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/trajectory_comparison.png")
    
    # ============================================
    # 图4: 跳变热图
    # ============================================
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]
    
    for ax, mode in zip(axes, modes):
        if 'first_step_jumps' in results[mode]:
            jumps = np.abs(results[mode]['first_step_jumps'][:, :action_dim])
            im = ax.imshow(jumps.T, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_xlabel('Chunk Boundary Index')
            ax.set_ylabel('Action Dimension')
            ax.set_title(f'Jump Heatmap - {mode}')
            plt.colorbar(im, ax=ax, label='|Jump|')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jump_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/jump_heatmap.png")


def main(args: Args):
    """主函数"""
    print("=" * 60)
    print("Chunk Jump Visualization")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型
    print("\n1. Loading PI0 policy...")
    server_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        server_config,
        args.policy_dir,
        pytorch_device=args.device
    )
    model = policy._model
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
            key: [t / dataset_meta.fps for t in range(model.config.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    
    if data_config.prompt_from_task:
        raw_dataset = data_loader.TransformedDataset(
            raw_dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    print(f"   Dataset size: {len(raw_dataset)}")
    
    # 3. 加载 HDF5（如果存在）
    h5_file = None
    if os.path.exists(args.h5_path):
        print(f"\n3. Loading HDF5: {args.h5_path}")
        h5_file = h5py.File(args.h5_path, 'r')
        print(f"   Latents shape: {h5_file['latents'].shape}")
    else:
        print(f"\n3. HDF5 not found: {args.h5_path}")
        print("   Will only run random mode.")
    
    # 4. 运行模拟
    print("\n4. Running simulations...")
    results = {}
    
    # Random 模式
    traj_random, bounds_random, chunks_random = run_simulation(
        args, policy, raw_dataset, h5_file, mode="random"
    )
    jumps_random = compute_chunk_jumps(chunks_random, args.execution_horizon, args.overlap_steps)
    results['random'] = {
        **jumps_random,
        'trajectory': traj_random,
        'boundaries': bounds_random,
    }
    
    # Inverse 模式（如果 HDF5 存在）
    if h5_file is not None:
        traj_inverse, bounds_inverse, chunks_inverse = run_simulation(
            args, policy, raw_dataset, h5_file, mode="inverse"
        )
        jumps_inverse = compute_chunk_jumps(chunks_inverse, args.execution_horizon, args.overlap_steps)
        results['inverse'] = {
            **jumps_inverse,
            'trajectory': traj_inverse,
            'boundaries': bounds_inverse,
        }
    
    # 5. 绘图
    print("\n5. Generating plots...")
    plot_jump_comparison(results, args.output_dir, action_dim=14)
    
    # 6. 打印摘要
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for mode, data in results.items():
        if 'jump_magnitudes' in data:
            jumps = data['jump_magnitudes']
            print(f"\n{mode.upper()}:")
            print(f"  Mean Jump:   {jumps.mean():.4f}")
            print(f"  Std Jump:    {jumps.std():.4f}")
            print(f"  Max Jump:    {jumps.max():.4f}")
    
    # 计算改善比例
    if 'inverse' in results and 'random' in results:
        random_mean = results['random']['jump_magnitudes'].mean()
        inverse_mean = results['inverse']['jump_magnitudes'].mean()
        improvement = (random_mean - inverse_mean) / random_mean * 100
        print(f"\n  Inverse vs Random improvement: {improvement:.1f}%")
    
    if h5_file is not None:
        h5_file.close()
    
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    tyro.cli(main)
