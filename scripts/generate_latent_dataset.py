"""
Latent Dataset Generation Script (v2 - Improved)

Improvements over v1:
1. RK4 ODE solver instead of Euler (4th order accuracy)
2. Fixed data writing bugs (no duplicate writes, proper ID handling)
3. Added data validation (NaN/Inf filtering)
4. Added statistics computation and saving
5. Added HDF5 compression
6. Better memory management
"""

import torch
import numpy as np
import os
from tqdm import tqdm
import h5py
from typing import Dict, Any, Optional, List, Tuple
import logging
import tyro
from dataclasses import dataclass
from pathlib import Path

from openpi.training import data_loader
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from lerobot.common.datasets import lerobot_dataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Args:
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    output_path: str = "latent_dataset.h5"
    num_ode_steps: int = 10
    batch_size: int = 8
    max_episodes: Optional[int] = None
    device: str = "cuda"
    num_workers: int = 8
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    
    # New options
    use_rk4: bool = True  # Use RK4 instead of Euler
    write_interval: int = 32  # Batches between HDF5 writes
    use_compression: bool = True  # Enable gzip compression
    validate_data: bool = True  # Filter NaN/Inf samples
    max_write_batch: int = 1000  # 单次写入的最大样本数，避免内存爆炸


def format_inputs_for_model(batch: Dict[str, Any], config: Any, device: str):
    """
    Format batch dict from LeRobotDataset into (images, img_masks, tokens, masks) 
    expected by PI0Pytorch.
    
    Important: This function converts images to the correct format:
    - uint8 [B, H, W, C] -> float32 [B, C, H, W] normalized to [-1, 1]
    """
    # 1. Images
    images = []
    img_masks = []
    
    def process_image(img):
        """Convert image to correct format for model."""
        if isinstance(img, torch.Tensor):
            img = img.to(device)
        else:
            img = torch.from_numpy(img).to(device)
        
        # 转换 uint8 -> float32 并归一化到 [-1, 1]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0 * 2.0 - 1.0
        
        # Permute: [B, H, W, C] -> [B, C, H, W]
        if img.ndim == 4 and img.shape[-1] in [1, 3, 4]:  # channels-last
            img = img.permute(0, 3, 1, 2)
        elif img.ndim == 3 and img.shape[-1] in [1, 3, 4]:  # [H, W, C] -> [C, H, W]
            img = img.permute(2, 0, 1)
        
        return img
    
    if "image" in batch and isinstance(batch["image"], dict):
        # 新格式: batch["image"] 是 dict
        image_keys = sorted(batch["image"].keys())
        for key in image_keys:
            img = process_image(batch["image"][key])
            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)
    elif "images" in batch and isinstance(batch["images"], dict):
        # 旧格式
        image_keys = sorted(batch["images"].keys())
        for key in image_keys:
            img = process_image(batch["images"][key])
            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)
    else:
        image_keys = [k for k in batch.keys() if k.startswith("observation.images.")]
        image_keys.sort() 
        for key in image_keys:
            img = process_image(batch[key])
            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

    # 2. Tokens
    tokens = None
    masks = None
    
    if "tokenized_prompt" in batch:
        tokens = batch["tokenized_prompt"].to(device)
        masks = batch["tokenized_prompt_mask"].to(device)
    elif "prompt_token_ids" in batch:
        tokens = batch["prompt_token_ids"].to(device)
        masks = batch["prompt_token_mask"].to(device)
    elif "observation.tokenized_prompt" in batch:
        tokens = batch["observation.tokenized_prompt"].to(device)
        masks = batch["observation.tokenized_prompt_mask"].to(device)

    # 3. State
    state = batch.get("state")
    if state is None:
        state = batch.get("observation.state")
    
    if state is not None:
        state = state.to(device)

    return images, img_masks, tokens, masks, state


class InverseFlowWrapper(torch.nn.Module):
    """
    Wraps PI0Pytorch model to add inverse ODE solving and feature extraction.
    Supports both Euler and RK4 solvers.
    """
    def __init__(self, model: PI0Pytorch):
        super().__init__()
        self.model = model
        self.config = model.config
        
    @torch.no_grad()
    def extract_features(self, images, img_masks, tokens, masks):
        """Extract conditioning features (Prefix Embeddings)."""
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _prepare_kv_cache(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        """Prepare KV cache for efficient ODE stepping."""
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values

    def _velocity_fn(self, x_t, time, state, prefix_pad_masks, past_key_values):
        """Compute velocity at (x_t, time)."""
        bsize = x_t.shape[0]
        expanded_time = time.expand(bsize) if time.dim() == 0 else time
        
        v_t = self.model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values, 
            x_t=x_t,
            timestep=expanded_time
        )
        return v_t

    @torch.no_grad()
    def invert_actions_euler(self, actions: torch.Tensor, state, prefix_pad_masks, 
                              past_key_values, num_steps: int = 100) -> torch.Tensor:
        """
        Euler method inverse ODE solver (1st order).
        经过 check_chunk_continuity.py 验证的实现。
        
        Inverse: Data (t=0) -> Noise (t=1)
        """
        device = actions.device
        bsize = actions.shape[0]
        
        # 确保 actions 是 float32
        x_t = actions.clone().float()
        
        dt = 1.0 / num_steps  # 正的步长，从 t=0 到 t=1
        time = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        for step in range(num_steps):
            expanded_time = time.expand(bsize) if time.dim() == 0 else time
            v_t = self._velocity_fn(x_t, expanded_time, state, prefix_pad_masks, past_key_values)
            x_t = x_t + dt * v_t
            time = time + dt
            
        return x_t

    @torch.no_grad()
    def invert_actions_rk4(self, actions: torch.Tensor, state, prefix_pad_masks, 
                           past_key_values, num_steps: int = 100) -> Tuple[torch.Tensor, bool]:
        """
        RK4 (Runge-Kutta 4th order) inverse ODE solver.
        经过 check_chunk_continuity.py 验证的实现。
        
        Inverse: Data (t=0) -> Noise (t=1)
        
        Returns:
            latents: The computed latents [B, H, D]
            is_valid: Whether the computation was numerically stable
        """
        device = actions.device
        bsize = actions.shape[0]
        
        # 确保 actions 是 float32
        x_t = actions.clone().float()
        
        dt = 1.0 / num_steps  # 正的步长，从 t=0 到 t=1
        time = torch.tensor(0.0, dtype=torch.float32, device=device)
        is_valid = True
        
        def velocity_fn(x, t):
            """计算速度场 v(x, t)"""
            expanded_t = t.expand(bsize) if t.dim() == 0 else t
            return self._velocity_fn(x, expanded_t, state, prefix_pad_masks, past_key_values)
        
        for step in range(num_steps):
            # RK4 四个斜率
            k1 = velocity_fn(x_t, time)
            k2 = velocity_fn(x_t + 0.5 * dt * k1, time + 0.5 * dt)
            k3 = velocity_fn(x_t + 0.5 * dt * k2, time + 0.5 * dt)
            k4 = velocity_fn(x_t + dt * k3, time + dt)
            
            # RK4 更新
            x_t = x_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            time = time + dt
            
            # Numerical stability check
            if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                logger.warning(f"NaN/Inf detected at RK4 step {step}")
                is_valid = False
                break
        
        return x_t, is_valid

    @torch.no_grad()
    def invert_actions(self, actions: torch.Tensor, images, img_masks, tokens, masks, 
                       state, num_steps=100, use_rk4=True,
                       prefix_embs=None, prefix_pad_masks=None, prefix_att_masks=None) -> Tuple[torch.Tensor, bool]:
        """
        Main inverse ODE entry point.
        
        Returns:
            latents: [B, H, D] noise latents
            is_valid: Whether computation was numerically stable
        """
        # Prepare prefix embeddings and KV cache
        if prefix_embs is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks, tokens, masks
            )
        
        past_key_values = self._prepare_kv_cache(prefix_embs, prefix_pad_masks, prefix_att_masks)
        
        # Run ODE solver
        if use_rk4:
            return self.invert_actions_rk4(actions, state, prefix_pad_masks, past_key_values, num_steps)
        else:
            latents = self.invert_actions_euler(actions, state, prefix_pad_masks, past_key_values, num_steps)
            is_valid = not (torch.isnan(latents).any() or torch.isinf(latents).any())
            return latents, is_valid


def validate_batch_data(latents_np: np.ndarray, feats_np: np.ndarray, 
                        ep_ids: Optional[np.ndarray], frame_ids: Optional[np.ndarray],
                        batch_idx: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Validate and filter invalid samples (NaN/Inf).
    
    Returns:
        Filtered arrays and count of invalid samples
    """
    # Create validity mask
    valid_mask = ~(np.isnan(latents_np).any(axis=(1, 2)) | np.isinf(latents_np).any(axis=(1, 2)))
    valid_mask &= ~(np.isnan(feats_np).any(axis=(1, 2)) | np.isinf(feats_np).any(axis=(1, 2)))
    
    # Check feature variance (features with zero variance are useless)
    feat_var = np.var(feats_np, axis=(1, 2))
    valid_mask &= (feat_var > 1e-8)
    
    invalid_count = int(np.sum(~valid_mask))
    
    if invalid_count > 0:
        logger.warning(f"Batch {batch_idx}: Filtering {invalid_count} invalid samples")
        latents_np = latents_np[valid_mask]
        feats_np = feats_np[valid_mask]
        if ep_ids is not None:
            ep_ids = ep_ids[valid_mask]
        if frame_ids is not None:
            frame_ids = frame_ids[valid_mask]
    
    return latents_np, feats_np, ep_ids, frame_ids, invalid_count


class IndexPreservingDataset(torch.utils.data.Dataset):
    """
    包装器：在 transform 后保留原始的 episode_index 和 frame_index。
    因为 repack transform 会丢弃这些键。
    """
    def __init__(self, original_dataset, transformed_dataset):
        self.original_dataset = original_dataset
        self.transformed_dataset = transformed_dataset
        
    def __len__(self):
        return len(self.transformed_dataset)
    
    def __getitem__(self, idx):
        # 获取 transformed 后的数据
        item = self.transformed_dataset[idx]
        
        # 从原始数据集获取 index 信息
        try:
            original_item = self.original_dataset[idx]
            # LeRobot 数据集的标准键名
            if 'episode_index' in original_item:
                item['episode_index'] = original_item['episode_index']
            if 'frame_index' in original_item:
                item['frame_index'] = original_item['frame_index']
            if 'index' in original_item:
                item['index'] = original_item['index']
        except Exception:
            pass
        
        return item


def flush_buffers_to_hdf5(
    buffer_latents: List[np.ndarray],
    buffer_feats: List[np.ndarray], 
    buffer_ep_ids: List[np.ndarray],
    buffer_frame_ids: List[np.ndarray],
    ds_latents: h5py.Dataset,
    ds_feats: Optional[h5py.Dataset],
    ds_ep_ids: h5py.Dataset,
    ds_frame_ids: h5py.Dataset,
    f: h5py.File,
    compression: Optional[str],
    compression_opts: Optional[int],
    max_write_batch: int = 1000
) -> h5py.Dataset:
    """
    分批写入 HDF5，避免大批次时内存爆炸。
    
    Args:
        max_write_batch: 单次写入的最大样本数
        
    Returns:
        ds_feats: 可能被初始化的 features 数据集
    """
    if not buffer_latents:
        return ds_feats
    
    # 计算总样本数
    total_samples = sum(b.shape[0] for b in buffer_latents)
    
    # 分批处理
    current_idx = 0
    buffer_idx = 0
    buffer_offset = 0
    
    while current_idx < total_samples:
        # 收集当前批次的数据
        batch_latents = []
        batch_feats = []
        batch_ep_ids = []
        batch_frame_ids = []
        batch_count = 0
        
        while batch_count < max_write_batch and buffer_idx < len(buffer_latents):
            remaining_in_buffer = buffer_latents[buffer_idx].shape[0] - buffer_offset
            take = min(remaining_in_buffer, max_write_batch - batch_count)
            
            batch_latents.append(buffer_latents[buffer_idx][buffer_offset:buffer_offset + take])
            batch_feats.append(buffer_feats[buffer_idx][buffer_offset:buffer_offset + take])
            
            if buffer_ep_ids and buffer_idx < len(buffer_ep_ids):
                batch_ep_ids.append(buffer_ep_ids[buffer_idx][buffer_offset:buffer_offset + take])
            if buffer_frame_ids and buffer_idx < len(buffer_frame_ids):
                batch_frame_ids.append(buffer_frame_ids[buffer_idx][buffer_offset:buffer_offset + take])
            
            batch_count += take
            buffer_offset += take
            
            if buffer_offset >= buffer_latents[buffer_idx].shape[0]:
                buffer_idx += 1
                buffer_offset = 0
        
        # 拼接并写入
        b_latents = np.concatenate(batch_latents, axis=0)
        b_feats = np.concatenate(batch_feats, axis=0)
        
        # 写入 latents
        ds_latents.resize(ds_latents.shape[0] + len(b_latents), axis=0)
        ds_latents[-len(b_latents):] = b_latents
        
        # 初始化并写入 features
        if ds_feats is None:
            feat_shape = (0, *b_feats.shape[1:])
            max_feat_shape = (None, *b_feats.shape[1:])
            chunk_size_feats = (1, *b_feats.shape[1:])
            ds_feats = f.create_dataset(
                'features', 
                shape=feat_shape, 
                maxshape=max_feat_shape, 
                dtype='f4', 
                chunks=chunk_size_feats,
                compression=compression,
                compression_opts=compression_opts
            )
        
        ds_feats.resize(ds_feats.shape[0] + len(b_feats), axis=0)
        ds_feats[-len(b_feats):] = b_feats
        
        # 写入 IDs
        if batch_ep_ids:
            b_ep = np.concatenate(batch_ep_ids, axis=0)
            ds_ep_ids.resize(ds_ep_ids.shape[0] + len(b_ep), axis=0)
            ds_ep_ids[-len(b_ep):] = b_ep
        
        if batch_frame_ids:
            b_frame = np.concatenate(batch_frame_ids, axis=0)
            ds_frame_ids.resize(ds_frame_ids.shape[0] + len(b_frame), axis=0)
            ds_frame_ids[-len(b_frame):] = b_frame
        
        current_idx += batch_count
    
    return ds_feats


def compute_and_save_statistics(f: h5py.File) -> Dict[str, np.ndarray]:
    """
    分块计算并保存统计量，避免大数据集时内存溢出。
    优化：避免花式索引，使用连续读取。
    """
    import gc
    
    logger.info("Computing statistics for normalization (memory-optimized)...")
    
    stats = {}
    chunk_size = 5000  # 减小 chunk size 避免 OOM
    
    # ==========================================
    # 1. Latents 统计量 (分块计算)
    # ==========================================
    latents_ds = f['latents']
    total_samples = latents_ds.shape[0]
    latent_dim = latents_ds.shape[-1]
    horizon = latents_ds.shape[1]
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    logger.info(f"Dataset: {total_samples} samples, shape {latents_ds.shape}")
    
    # 初始化累加变量 (float64 避免精度丢失)
    latents_sum = np.zeros(latents_ds.shape[1:], dtype=np.float64)
    latents_sq_sum = np.zeros(latents_ds.shape[1:], dtype=np.float64)
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_samples)
        chunk = latents_ds[start:end].astype(np.float64)
        
        latents_sum += np.sum(chunk, axis=0)
        latents_sq_sum += np.sum(chunk ** 2, axis=0)
        
        del chunk
        if i % 10 == 0:
            gc.collect()
    
    stats['latents_mean'] = (latents_sum / total_samples).astype(np.float32)
    latents_var = (latents_sq_sum / total_samples) - (stats['latents_mean'].astype(np.float64) ** 2)
    stats['latents_std'] = np.sqrt(np.maximum(latents_var, 0)).astype(np.float32)
    stats['latents_std'] = np.maximum(stats['latents_std'], 1e-6)
    
    del latents_sum, latents_sq_sum, latents_var
    gc.collect()
    
    logger.info(f"Latents stats computed from {total_samples} samples")
    
    # ==========================================
    # 2. Features 统计量 (分块计算，减小 chunk size)
    # ==========================================
    if 'features' in f:
        features_ds = f['features']
        feat_dim = features_ds.shape[-1]
        seq_len = features_ds.shape[1]
        
        # Features 维度很大，使用更小的 chunk size
        # 估算: chunk * seq_len * feat_dim * 4 bytes < 1GB
        # chunk < 1GB / (seq_len * feat_dim * 4)
        max_memory_bytes = 1 * 1024 * 1024 * 1024  # 1GB
        features_chunk_size = min(
            chunk_size,
            max(1, max_memory_bytes // (seq_len * feat_dim * 4))
        )
        logger.info(f"Features shape: {features_ds.shape}, using chunk_size={features_chunk_size}")
        
        features_sum = np.zeros(feat_dim, dtype=np.float64)
        features_sq_sum = np.zeros(feat_dim, dtype=np.float64)
        num_feat_chunks = (total_samples + features_chunk_size - 1) // features_chunk_size
        
        for i in range(num_feat_chunks):
            start = i * features_chunk_size
            end = min((i + 1) * features_chunk_size, total_samples)
            # 使用 float32 减少内存，然后转为 float64 累加
            chunk = features_ds[start:end].astype(np.float32)
            # Pool over sequence dimension
            pooled = np.mean(chunk, axis=1).astype(np.float64)  # [chunk, D]
            features_sum += np.sum(pooled, axis=0)
            features_sq_sum += np.sum(pooled ** 2, axis=0)
            
            del chunk, pooled
            if i % 10 == 0:
                gc.collect()
        
        stats['features_mean'] = (features_sum / total_samples).astype(np.float32)
        features_var = (features_sq_sum / total_samples) - (stats['features_mean'].astype(np.float64) ** 2)
        stats['features_std'] = np.sqrt(np.maximum(features_var, 0)).astype(np.float32)
        stats['features_std'] = np.maximum(stats['features_std'], 1e-6)
        
        del features_sum, features_sq_sum, features_var
        gc.collect()
    
    # ==========================================
    # 3. Delta 统计量 (最重要! 使用连续读取避免花式索引)
    # ==========================================
    logger.info("Computing delta statistics...")
    
    # 分块读取 episode_ids，避免一次性加载
    has_episode_ids = 'episode_ids' in f and f['episode_ids'].shape[0] > 0
    
    delta_sum = np.zeros(latent_dim, dtype=np.float64)
    delta_sq_sum = np.zeros(latent_dim, dtype=np.float64)
    total_delta_elements = 0
    
    # 使用连续读取：逐块读取 latents，在块内计算相邻 delta
    for i in range(num_chunks):
        start = i * chunk_size
        # 多读一个样本，用于计算跨块边界的 delta
        end = min((i + 1) * chunk_size + 1, total_samples)
        
        # 连续读取（高效）
        chunk_latents = latents_ds[start:end].astype(np.float64)
        
        # 读取对应的 episode_ids（如果有）
        if has_episode_ids:
            chunk_ep_ids = f['episode_ids'][start:end]
        
        # 计算块内相邻 delta
        if chunk_latents.shape[0] > 1:
            deltas = chunk_latents[1:] - chunk_latents[:-1]  # [chunk-1, H, D]
            
            # 过滤跨 episode 的无效转换
            if has_episode_ids:
                valid_mask = (chunk_ep_ids[:-1] == chunk_ep_ids[1:])
                deltas = deltas[valid_mask]
            
            if len(deltas) > 0:
                # 展平计算 per-dim 统计
                deltas_flat = deltas.reshape(-1, latent_dim)
                delta_sum += np.sum(deltas_flat, axis=0)
                delta_sq_sum += np.sum(deltas_flat ** 2, axis=0)
                total_delta_elements += deltas_flat.shape[0]
                
                del deltas_flat
            
            del deltas
        
        del chunk_latents
        if has_episode_ids:
            del chunk_ep_ids
        
        if i % 10 == 0:
            gc.collect()
    
    if total_delta_elements > 0:
        stats['delta_mean'] = (delta_sum / total_delta_elements).astype(np.float32)
        delta_var = (delta_sq_sum / total_delta_elements) - (stats['delta_mean'].astype(np.float64) ** 2)
        stats['delta_std'] = np.sqrt(np.maximum(delta_var, 0)).astype(np.float32)
        stats['delta_std'] = np.maximum(stats['delta_std'], 1e-6)
        
        logger.info(f"Delta stats from {total_delta_elements} elements ({total_delta_elements // horizon} transitions):")
        logger.info(f"  Mean range: [{stats['delta_mean'].min():.4f}, {stats['delta_mean'].max():.4f}]")
        logger.info(f"  Std range:  [{stats['delta_std'].min():.4f}, {stats['delta_std'].max():.4f}]")
    else:
        logger.warning("No valid delta transitions found, using defaults")
        stats['delta_mean'] = np.zeros(latent_dim, dtype=np.float32)
        stats['delta_std'] = np.ones(latent_dim, dtype=np.float32)
    
    del delta_sum, delta_sq_sum
    gc.collect()
    
    # ==========================================
    # 4. 保存到 HDF5
    # ==========================================
    for key, value in stats.items():
        if key in f:
            del f[key]
        f.create_dataset(key, data=value)
    
    logger.info("Statistics saved to HDF5")
    return stats


def main(args: Args):
    logging.basicConfig(level=logging.INFO, force=True)
    logger.info(f"Loading config from {args.policy_config}...")
    server_config = _config.get_config(args.policy_config)
    
    # Initialize Model
    logger.info("Initializing Model from Policy Checkpoint...")
    policy = _policy_config.create_trained_policy(
        server_config, 
        args.policy_dir, 
        pytorch_device=args.device
    )

    # Verify model loading
    logger.info("--- POLICY VERIFICATION ---")
    if hasattr(policy._model, "paligemma_with_expert"):
        w = next(policy._model.paligemma_with_expert.parameters())
        logger.info(f"Model Weight Sample (Mean): {w.float().mean().item():.6f}")

    has_norm = False
    for i, t in enumerate(policy._input_transform.transforms):
        if "Normalize" in str(type(t)) and "Unnormalize" not in str(type(t)):
            has_norm = True
            logger.info(f"Found Normalize transform at index {i}")
            if hasattr(t, "stats") and t.stats:
                logger.info(f"Norm Stats keys: {list(t.stats.keys())}")
    
    if not has_norm:
        logger.warning("No Normalize transform found!")
    logger.info("-------------------------")
    
    # Extract model
    if not hasattr(policy, "_model") or not isinstance(policy._model, PI0Pytorch):
        raise ValueError("Policy does not contain a PI0Pytorch model!")
        
    model = policy._model
    model.eval()
    model.to(args.device)
    model_config = server_config.model
    
    wrapper = InverseFlowWrapper(model)
    
    # Load Dataset
    logger.info("Loading dataset...")
    try:
        data_config_factory = server_config.data
        data_config = data_config_factory.create(server_config.assets_dirs, server_config.model)
        repo_id = args.dataset_repo_id or data_config.repo_id
        
        if args.dataset_root:
            logger.info(f"Using local dataset from {args.dataset_root}")
            if not os.path.exists(args.dataset_root):
                raise ValueError(f"Dataset root {args.dataset_root} does not exist")
                
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
        
        # 保存原始数据集（包含 episode_index 和 frame_index）
        raw_dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=args.dataset_root,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(model_config.action_horizon)] 
                for key in data_config.action_sequence_keys
            },
        )
        
        # 应用 transforms（会丢失 episode_index）
        dataset = raw_dataset
        if data_config.prompt_from_task:
            from openpi import transforms as _transforms
            dataset = data_loader.TransformedDataset(
                dataset, 
                [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
            )
        
        transformed_dataset = data_loader.transform_dataset(dataset, data_config)
        
        # 使用包装器保留原始索引
        dataset = IndexPreservingDataset(raw_dataset, transformed_dataset)
        logger.info("Dataset transforms applied successfully (with index preservation).")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info(f"Initializing DataLoader (num_workers={args.num_workers})...")
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    # Prepare HDF5
    logger.info(f"Creating output file at {args.output_path}...")
    logger.info(f"Using {'RK4' if args.use_rk4 else 'Euler'} ODE solver with {args.num_ode_steps} steps")

    with h5py.File(args.output_path, 'w', libver='latest') as f:
        # Dataset shapes
        latents_shape = (0, model_config.action_horizon, model_config.action_dim)
        max_latents_shape = (None, model_config.action_horizon, model_config.action_dim)
        chunk_size_latents = (1, model_config.action_horizon, model_config.action_dim)
        
        compression = "gzip" if args.use_compression else None
        compression_opts = 4 if args.use_compression else None
        
        ds_latents = f.create_dataset(
            'latents', 
            shape=latents_shape, 
            maxshape=max_latents_shape, 
            dtype='f4', 
            chunks=chunk_size_latents,
            compression=compression,
            compression_opts=compression_opts
        )
        ds_feats = None
        ds_ep_ids = f.create_dataset('episode_ids', shape=(0,), maxshape=(None,), dtype='i4')
        ds_frame_ids = f.create_dataset('frame_ids', shape=(0,), maxshape=(None,), dtype='i4')
        
        # Buffers
        buffer_latents = []
        buffer_feats = []
        buffer_ep_ids = []
        buffer_frame_ids = []
        
        # Counters
        sample_count = 0
        invalid_count = 0
        unstable_count = 0
        
        logger.info("Starting generation...")
        pbar = tqdm(dataloader, desc="Generating latents")
        
        for batch_idx, batch in enumerate(pbar):
            # Format inputs
            images, img_masks, tokens, masks, state = format_inputs_for_model(
                batch, model_config, args.device
            )
            
            # Get actions
            actions = batch.get('actions', batch.get('action'))
            if actions is None:
                raise KeyError(f"No 'actions' or 'action' in batch. Keys: {list(batch.keys())}")
            
            actions = actions.to(device=args.device, dtype=torch.float32)
            
            # Extract features
            prefix_embs, prefix_pad_masks, prefix_att_masks = wrapper.extract_features(
                images, img_masks, tokens, masks
            )

            # Run inverse ODE
            latents, is_valid = wrapper.invert_actions(
                actions, images, img_masks, tokens, masks, state, 
                num_steps=args.num_ode_steps,
                use_rk4=args.use_rk4,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks
            )
            
            if not is_valid:
                unstable_count += 1
            
            # Convert to numpy
            latents_np = latents.detach().cpu().to(dtype=torch.float32).numpy()
            feats_np = prefix_embs.detach().cpu().to(dtype=torch.float32).numpy()
            
            # Extract IDs (尝试多个可能的键名)
            ep_ids = batch.get('episode_index')
            if ep_ids is None:
                ep_ids = batch.get('episode_id')
            
            frame_ids = batch.get('frame_index')
            if frame_ids is None:
                frame_ids = batch.get('frame_id')
            
            if ep_ids is not None and isinstance(ep_ids, torch.Tensor):
                ep_ids = ep_ids.cpu().numpy()
            if frame_ids is not None and isinstance(frame_ids, torch.Tensor):
                frame_ids = frame_ids.cpu().numpy()
            
            # Validate data
            if args.validate_data:
                latents_np, feats_np, ep_ids, frame_ids, batch_invalid = validate_batch_data(
                    latents_np, feats_np, ep_ids, frame_ids, batch_idx
                )
                invalid_count += batch_invalid
            
            # Skip empty batches (all filtered)
            if len(latents_np) == 0:
                continue
            
            # Append to buffers
            buffer_latents.append(latents_np)
            buffer_feats.append(feats_np)
            if ep_ids is not None:
                buffer_ep_ids.append(ep_ids)
            if frame_ids is not None:
                buffer_frame_ids.append(frame_ids)
            
            sample_count += len(latents_np)
            
            # Periodic flush (使用分批写入函数)
            if (batch_idx + 1) % args.write_interval == 0:
                ds_feats = flush_buffers_to_hdf5(
                    buffer_latents, buffer_feats, buffer_ep_ids, buffer_frame_ids,
                    ds_latents, ds_feats, ds_ep_ids, ds_frame_ids,
                    f, compression, compression_opts, args.max_write_batch
                )
                
                # Clear all buffers
                buffer_latents = []
                buffer_feats = []
                buffer_ep_ids = []
                buffer_frame_ids = []
                
                pbar.set_postfix({
                    "samples": sample_count,
                    "invalid": invalid_count
                })

            if args.max_episodes and sample_count > args.max_episodes * 1000:
                break
        
        # Final flush (使用分批写入函数)
        if buffer_latents:
            ds_feats = flush_buffers_to_hdf5(
                buffer_latents, buffer_feats, buffer_ep_ids, buffer_frame_ids,
                ds_latents, ds_feats, ds_ep_ids, ds_frame_ids,
                f, compression, compression_opts, args.max_write_batch
            )
        
        # Compute and save statistics
        compute_and_save_statistics(f)
        
        # Save metadata
        f.attrs['num_ode_steps'] = args.num_ode_steps
        f.attrs['ode_solver'] = 'rk4' if args.use_rk4 else 'euler'
        f.attrs['policy_config'] = args.policy_config
        f.attrs['dataset_repo_id'] = args.dataset_repo_id
        f.attrs['total_samples'] = sample_count
        f.attrs['invalid_samples'] = invalid_count
        f.attrs['unstable_batches'] = unstable_count

    logger.info("=" * 50)
    logger.info(f"Generation complete!")
    logger.info(f"  Total samples: {sample_count}")
    logger.info(f"  Invalid samples filtered: {invalid_count}")
    logger.info(f"  Numerically unstable batches: {unstable_count}")
    logger.info(f"  Output file: {args.output_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    tyro.cli(main)
