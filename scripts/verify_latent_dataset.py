"""
验证 latent_dataset.h5 中反推噪声的质量，并支持在测试集上进行在线反推。

对比：
1. Ground Truth 动作（来自原始数据集）
2. 反推噪声解码的动作（训练集从 HDF5 加载，测试集在线反推）
3. 随机噪声解码的动作

以关节维度对比图的形式展示。

支持两种模式：
- train: 使用预计算的 latent_dataset.h5 (训练集)
- test: 在线反推噪声 (测试集)
"""

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

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.policies import policy_config as _policy_config
from openpi.models import model as _model
from lerobot.common.datasets import lerobot_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


# ==========================================
# 在线反推 ODE 函数
# ==========================================

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
    
    x_t = actions.clone().float()
    dt = 1.0 / num_steps
    time = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        x_t = x_t + dt * v_t
        time = time + dt
    
    return x_t


def prepare_observation_and_kv_cache(
    model: PI0Pytorch,
    item: dict,
    device: str,
    camera_names: List[str] = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
) -> Tuple[torch.Tensor, torch.Tensor, any]:
    """
    从 transformed dataset item 准备 observation 和 KV cache。
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


@dataclass
class Args:
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    
    # 训练集设置 (使用预计算的 HDF5)
    train_h5_path: str = "/home/ztlab/lgd/openpi/latent_dataset.h5"
    train_dataset_root: str = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    
    # 测试集设置 (在线反推)
    test_dataset_root: str = "/home/ztlab/lgd/openpi/test/piper_dual_lerobot"
    
    dataset_repo_id: str = "piper_dual_lerobot"
    
    # 验证参数
    mode: str = "both"  # "train", "test", 或 "both"
    episode_idx: int = 5  # 要验证的 episode
    num_chunks: int = 100  # 模拟的 chunk 数量
    execution_horizon: int = 15  # 每个 chunk 执行的帧数
    num_ode_steps: int = 10  # ODE 反推步数
    
    # 输出
    output_dir: str = "./verify_results"
    device: str = "cuda"
    
    # 只分析有效维度
    valid_dims: int = 14


def verify_dataset(
    args: Args,
    policy,
    model,
    raw_dataset,
    transformed_dataset,
    h5_latents,
    episode_ids,
    dataset_name: str,
    use_online_inverse: bool = False,
):
    """
    验证单个数据集（训练集或测试集）
    
    Args:
        use_online_inverse: 是否使用在线反推噪声（测试集为 True）
    """
    print(f"\n{'='*60}")
    print(f"Verifying {dataset_name}")
    print(f"{'='*60}")
    print(f"Online inverse: {use_online_inverse}")
    
    # 找到指定 episode 的起始索引
    if episode_ids is not None:
        episode_mask = episode_ids == args.episode_idx
        episode_indices_all = np.where(episode_mask)[0]
        if len(episode_indices_all) == 0:
            print(f"ERROR: Episode {args.episode_idx} not found!")
            unique_eps = np.unique(episode_ids)
            print(f"Available episodes: {unique_eps[:20]}...")
            # 尝试使用第一个可用的 episode
            args.episode_idx = unique_eps[0]
            episode_mask = episode_ids == args.episode_idx
            episode_indices_all = np.where(episode_mask)[0]
            print(f"Using episode {args.episode_idx} instead")
        episode_start_idx = int(episode_indices_all[0])
        episode_end_idx = int(episode_indices_all[-1])
        print(f"Episode {args.episode_idx}: {len(episode_indices_all)} frames (indices {episode_start_idx} to {episode_end_idx})")
    else:
        # 如果没有 episode_ids，使用简单的估算
        episode_start_idx = 0
        print(f"No episode_ids, starting from index 0")
    
    # 计算模拟索引
    sim_indices = []
    current_idx = episode_start_idx
    for _ in range(args.num_chunks):
        if current_idx >= len(raw_dataset):
            break
        if not use_online_inverse and h5_latents is not None and current_idx >= len(h5_latents):
            break
        sim_indices.append(current_idx)
        current_idx += args.execution_horizon
    
    print(f"Simulation: {len(sim_indices)} chunks, covering ~{len(sim_indices) * args.execution_horizon} frames")
    
    # 确定 action key
    sample_item = raw_dataset[0]
    action_key = None
    for key in ['action', 'actions']:
        if key in sample_item:
            action_key = key
            break
    if action_key is None:
        for key in sample_item.keys():
            if key.startswith('action'):
                action_key = key
                break
    print(f"Using action key: {action_key}")
    
    # 收集数据
    gt_trajectory = []
    inverse_trajectory = []
    random_trajectory = []
    
    device = args.device
    
    for step_i, idx in enumerate(tqdm(sim_indices, desc=f"Decoding {dataset_name}")):
        idx = int(idx)
        
        # 获取噪声
        if use_online_inverse:
            # 在线反推噪声
            item = transformed_dataset[idx]
            actions_normalized = item.get('actions')
            if actions_normalized is None:
                actions_normalized = item.get('action')
            
            if isinstance(actions_normalized, np.ndarray):
                actions_normalized = torch.from_numpy(actions_normalized)
            actions_normalized = actions_normalized.to(device).unsqueeze(0).float()
            
            # 准备 KV cache
            state, prefix_pad_masks, past_key_values = prepare_observation_and_kv_cache(
                model, item, device
            )
            
            # 反推噪声
            h5_latent = inverse_ode_online(
                model, actions_normalized, state, prefix_pad_masks, past_key_values, args.num_ode_steps
            )
        else:
            # 从 HDF5 加载预计算的噪声
            h5_latent = torch.from_numpy(h5_latents[idx]).unsqueeze(0).to(device).float()
        
        # 生成随机噪声
        random_noise = torch.randn_like(h5_latent)
        
        # 从原始数据集获取观测
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
        
        # 解码
        with torch.no_grad():
            result_inverse = policy.infer(obs, noise=h5_latent)
            actions_inverse = result_inverse['actions']
            if isinstance(actions_inverse, torch.Tensor):
                actions_inverse = actions_inverse.cpu().numpy()
            if actions_inverse.ndim == 3:
                actions_inverse = actions_inverse[0]
            
            result_random = policy.infer(obs, noise=random_noise)
            actions_random = result_random['actions']
            if isinstance(actions_random, torch.Tensor):
                actions_random = actions_random.cpu().numpy()
            if actions_random.ndim == 3:
                actions_random = actions_random[0]
        
        # GT 动作
        gt_actions = raw_item[action_key]
        if isinstance(gt_actions, torch.Tensor):
            gt_actions = gt_actions.numpy()
        
        # 取执行的部分
        to_exec = min(args.execution_horizon, gt_actions.shape[0], actions_inverse.shape[0])
        for t in range(to_exec):
            gt_trajectory.append(gt_actions[t])
            inverse_trajectory.append(actions_inverse[t])
            random_trajectory.append(actions_random[t])
    
    # 转换为数组
    gt_trajectory = np.array(gt_trajectory)
    inverse_trajectory = np.array(inverse_trajectory)
    random_trajectory = np.array(random_trajectory)
    
    print(f"\nTrajectory shapes:")
    print(f"  GT: {gt_trajectory.shape}")
    print(f"  Inverse: {inverse_trajectory.shape}")
    print(f"  Random: {random_trajectory.shape}")
    
    return gt_trajectory, inverse_trajectory, random_trajectory


def compute_and_print_metrics(gt, inverse, random, valid_dims, dataset_name):
    """计算并打印误差指标"""
    gt_valid = gt[:, :valid_dims]
    inverse_valid = inverse[:, :valid_dims]
    random_valid = random[:, :valid_dims]
    
    inverse_error = np.abs(inverse_valid - gt_valid)
    random_error = np.abs(random_valid - gt_valid)
    
    print(f"\n=== {dataset_name} Error Analysis (first {valid_dims} dims) ===")
    print(f"Inverse Noise -> Actions:")
    print(f"  MAE: {inverse_error.mean():.6f}")
    print(f"  Max: {inverse_error.max():.6f}")
    
    print(f"\nRandom Noise -> Actions:")
    print(f"  MAE: {random_error.mean():.6f}")
    print(f"  Max: {random_error.max():.6f}")
    
    ratio = random_error.mean() / (inverse_error.mean() + 1e-8)
    print(f"\nImprovement ratio (random/inverse): {ratio:.2f}x")
    
    # 计算 R²
    r2_inv = np.corrcoef(gt_valid.flatten(), inverse_valid.flatten())[0,1]**2
    r2_rnd = np.corrcoef(gt_valid.flatten(), random_valid.flatten())[0,1]**2
    print(f"R² - Inverse: {r2_inv:.4f}, Random: {r2_rnd:.4f}")
    
    return inverse_error, random_error


def plot_comparison(
    gt, inverse, random, inverse_error, random_error,
    valid_dims, output_dir, dataset_name, episode_idx
):
    """绘制对比图"""
    gt_valid = gt[:, :valid_dims]
    inverse_valid = inverse[:, :valid_dims]
    random_valid = random[:, :valid_dims]
    
    prefix = f"{dataset_name}_"
    
    # 计算统一的 Y 轴范围
    all_values = np.concatenate([gt_valid.flatten(), inverse_valid.flatten(), random_valid.flatten()])
    y_min, y_max = all_values.min(), all_values.max()
    y_margin = (y_max - y_min) * 0.1
    unified_ylim = (y_min - y_margin, y_max + y_margin)
    
    dims_to_plot = list(range(min(valid_dims, 14)))
    
    # 1. 每个维度单独的图
    for dim in dims_to_plot:
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        
        ax1 = axes[0]
        ax1.plot(gt_valid[:, dim], 'k-', linewidth=2, label='Ground Truth', alpha=0.9)
        ax1.plot(inverse_valid[:, dim], 'b--', linewidth=1.5, label='Inverse Noise', alpha=0.8)
        ax1.plot(random_valid[:, dim], 'r:', linewidth=1.5, label='Random Noise', alpha=0.6)
        ax1.set_ylabel('Action Value (rad)')
        ax1.set_ylim(unified_ylim)
        ax1.set_title(f'{dataset_name} - Joint {dim} - Episode {episode_idx}')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(inverse_error[:, dim], 'b-', linewidth=1.5, 
                 label=f'Inverse Error (MAE={inverse_error[:, dim].mean():.4f})')
        ax2.plot(random_error[:, dim], 'r-', linewidth=1.5, 
                 label=f'Random Error (MAE={random_error[:, dim].mean():.4f})')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Absolute Error (rad)')
        ax2.set_title(f'Joint {dim} - Error Comparison')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, f'{prefix}joint_{dim}_comparison.png')
        plt.savefig(filename, dpi=150)
        plt.close()
    
    # 2. 综合对比图
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, dim in enumerate(dims_to_plot[:16]):
        ax = axes[i]
        ax.plot(gt_valid[:, dim], 'k-', linewidth=1.5, label='GT')
        ax.plot(inverse_valid[:, dim], 'b--', linewidth=1, label='Inverse')
        ax.plot(random_valid[:, dim], 'r:', linewidth=1, alpha=0.5, label='Random')
        ax.set_ylim(unified_ylim)
        ax.set_title(f'Joint {dim} (Inv MAE={inverse_error[:, dim].mean():.4f})')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    for i in range(len(dims_to_plot), 16):
        axes[i].set_visible(False)
    
    ratio = random_error.mean() / (inverse_error.mean() + 1e-8)
    plt.suptitle(f'{dataset_name} - Episode {episode_idx} - All Joints ({len(gt_valid)} steps)\n'
                 f'Inverse MAE: {inverse_error.mean():.4f}, Random MAE: {random_error.mean():.4f}, '
                 f'Improvement: {ratio:.1f}x',
                 fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{prefix}all_joints_comparison.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")
    
    # 3. 误差分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(inverse_error.flatten(), bins=50, alpha=0.7, label='Inverse', color='blue')
    ax1.hist(random_error.flatten(), bins=50, alpha=0.5, label='Random', color='red')
    ax1.set_xlabel('Absolute Error (rad)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{dataset_name} - Error Distribution')
    ax1.legend()
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    x = np.arange(valid_dims)
    width = 0.35
    ax2.bar(x - width/2, inverse_error.mean(axis=0), width, label='Inverse', color='blue', alpha=0.7)
    ax2.bar(x + width/2, random_error.mean(axis=0), width, label='Random', color='red', alpha=0.7)
    ax2.set_xlabel('Joint Dimension')
    ax2.set_ylabel('MAE (rad)')
    ax2.set_title(f'{dataset_name} - Per-Dimension MAE')
    ax2.legend()
    ax2.set_xticks(x)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{prefix}error_distribution.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")
    
    # 4. 散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.scatter(gt_valid.flatten(), inverse_valid.flatten(), alpha=0.1, s=1)
    lims = [min(gt_valid.min(), inverse_valid.min()), max(gt_valid.max(), inverse_valid.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='y=x')
    ax1.set_xlabel('Ground Truth (rad)')
    ax1.set_ylabel('Inverse Decoded (rad)')
    r2_inv = np.corrcoef(gt_valid.flatten(), inverse_valid.flatten())[0,1]**2
    ax1.set_title(f'{dataset_name} - Inverse (R²={r2_inv:.4f})')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.scatter(gt_valid.flatten(), random_valid.flatten(), alpha=0.1, s=1)
    lims = [min(gt_valid.min(), random_valid.min()), max(gt_valid.max(), random_valid.max())]
    ax2.plot(lims, lims, 'r--', linewidth=2, label='y=x')
    ax2.set_xlabel('Ground Truth (rad)')
    ax2.set_ylabel('Random Decoded (rad)')
    r2_rnd = np.corrcoef(gt_valid.flatten(), random_valid.flatten())[0,1]**2
    ax2.set_title(f'{dataset_name} - Random (R²={r2_rnd:.4f})')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{prefix}scatter_comparison.png')
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")


def main(args: Args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载配置和模型
    print(f"Loading policy config {args.policy_config}...")
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
    print("Loading trained policy...")
    policy = _policy_config.create_trained_policy(
        server_config, 
        args.policy_dir, 
        pytorch_device=args.device
    )
    
    model = policy._model
    model.eval()
    
    # 获取数据配置
    data_config_factory = server_config.data
    data_config = data_config_factory.create(server_config.assets_dirs, server_config.model)
    repo_id = args.dataset_repo_id or data_config.repo_id
    
    from openpi import transforms as _transforms
    
    results = {}
    
    # ==========================================
    # 训练集验证 (使用预计算的 HDF5)
    # ==========================================
    if args.mode in ["train", "both"]:
        print("\n" + "="*60)
        print("TRAIN SET VERIFICATION (using precomputed HDF5)")
        print("="*60)
        
        # 加载 HDF5
        if os.path.exists(args.train_h5_path):
            f_h5 = h5py.File(args.train_h5_path, 'r')
            h5_latents = f_h5['latents']
            episode_ids = f_h5['episode_ids'][:] if 'episode_ids' in f_h5 else None
            print(f"Loaded HDF5: {h5_latents.shape}")
        else:
            print(f"ERROR: HDF5 not found at {args.train_h5_path}")
            h5_latents = None
            episode_ids = None
        
        if h5_latents is not None:
            # 加载训练集
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=args.train_dataset_root)
            
            raw_dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                root=args.train_dataset_root,
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(model_config.action_horizon)] 
                    for key in data_config.action_sequence_keys
                },
            )
            
            if data_config.prompt_from_task:
                raw_dataset = data_loader.TransformedDataset(
                    raw_dataset, 
                    [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
                )
            
            gt, inverse, random = verify_dataset(
                args, policy, model, raw_dataset, None,
                h5_latents[:], episode_ids, "Train",
                use_online_inverse=False
            )
            
            inverse_error, random_error = compute_and_print_metrics(
                gt, inverse, random, args.valid_dims, "Train"
            )
            
            plot_comparison(
                gt, inverse, random, inverse_error, random_error,
                args.valid_dims, args.output_dir, "train", args.episode_idx
            )
            
            results['train'] = {
                'inverse_mae': inverse_error.mean(),
                'random_mae': random_error.mean(),
                'improvement': random_error.mean() / (inverse_error.mean() + 1e-8)
            }
            
            f_h5.close()
    
    # ==========================================
    # 测试集验证 (在线反推噪声)
    # ==========================================
    if args.mode in ["test", "both"]:
        print("\n" + "="*60)
        print("TEST SET VERIFICATION (using online inverse ODE)")
        print("="*60)
        
        if os.path.exists(args.test_dataset_root):
            # 加载测试集
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=args.test_dataset_root)
            
            raw_dataset = lerobot_dataset.LeRobotDataset(
                repo_id,
                root=args.test_dataset_root,
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(model_config.action_horizon)] 
                    for key in data_config.action_sequence_keys
                },
            )
            
            if data_config.prompt_from_task:
                raw_dataset = data_loader.TransformedDataset(
                    raw_dataset, 
                    [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
                )
            
            # 创建 transformed dataset (用于在线反推)
            # 使用 data_loader.transform_dataset 来创建，这与 check_chunk_continuity.py 一致
            transformed_dataset = data_loader.transform_dataset(raw_dataset, data_config)
            
            print(f"Test dataset size: {len(raw_dataset)} frames")
            
            # 获取 episode_ids（快速方式：直接从底层数据获取）
            print("Getting episode_ids...")
            try:
                # 尝试找到底层的 lerobot 数据集
                base_ds = raw_dataset
                while hasattr(base_ds, 'dataset'):
                    base_ds = base_ds.dataset
                
                if hasattr(base_ds, 'hf_dataset') and 'episode_index' in base_ds.hf_dataset.column_names:
                    episode_ids = np.array(base_ds.hf_dataset['episode_index'])
                    print(f"  Got episode_ids from hf_dataset: {len(episode_ids)} items")
                else:
                    # 使用简单方法：直接创建基于 episode 长度的索引
                    # 从 meta/episodes.jsonl 读取
                    import json
                    episodes_file = os.path.join(args.test_dataset_root, "meta", "episodes.jsonl")
                    episode_ids = []
                    if os.path.exists(episodes_file):
                        with open(episodes_file, 'r') as f:
                            for line in f:
                                ep_info = json.loads(line)
                                ep_id = ep_info.get('episode_index', len(episode_ids))
                                ep_len = ep_info.get('length', 1)
                                episode_ids.extend([ep_id] * ep_len)
                        episode_ids = np.array(episode_ids)
                        print(f"  Got episode_ids from meta/episodes.jsonl: {len(episode_ids)} items")
                    else:
                        # 最终回退：假设整个数据集是一个 episode
                        print("  Warning: Could not find episode info, assuming single episode")
                        episode_ids = np.zeros(len(raw_dataset), dtype=np.int64)
            except Exception as e:
                print(f"  Warning: Could not get episode_ids: {e}")
                # 假设整个数据集是一个 episode
                episode_ids = np.zeros(len(raw_dataset), dtype=np.int64)
            print(f"  Episode IDs range: {episode_ids.min()} to {episode_ids.max()}")
            
            gt, inverse, random = verify_dataset(
                args, policy, model, raw_dataset, transformed_dataset,
                None, episode_ids, "Test",
                use_online_inverse=True
            )
            
            inverse_error, random_error = compute_and_print_metrics(
                gt, inverse, random, args.valid_dims, "Test"
            )
            
            plot_comparison(
                gt, inverse, random, inverse_error, random_error,
                args.valid_dims, args.output_dir, "test", args.episode_idx
            )
            
            results['test'] = {
                'inverse_mae': inverse_error.mean(),
                'random_mae': random_error.mean(),
                'improvement': random_error.mean() / (inverse_error.mean() + 1e-8)
            }
        else:
            print(f"ERROR: Test dataset not found at {args.test_dataset_root}")
    
    # ==========================================
    # 汇总对比
    # ==========================================
    if len(results) > 1:
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        for name, metrics in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Inverse MAE: {metrics['inverse_mae']:.6f}")
            print(f"  Random MAE: {metrics['random_mae']:.6f}")
            print(f"  Improvement: {metrics['improvement']:.2f}x")
    
    print(f"\n=== Verification Complete ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
