"""
Noise Refiner 训练脚本

新训练目标：
- 输入：随机噪声 + features
- 输出：修正后的噪声（接近反推噪声）

网络学习：f(random_noise, features) → inverse_noise
这样推理时，给定随机噪声和当前观测，网络输出"更好"的噪声
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import logging
import tyro
import os
from tqdm import tqdm
import wandb
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. 网络定义 - Noise Refiner
# ==========================================

class NoiseRefiner(nn.Module):
    """
    噪声修正网络
    
    输入：随机噪声 [B, T, D] + features [B, cond_dim]
    输出：修正后的噪声 [B, T, D]（接近反推噪声）
    
    架构：基于 Transformer 的序列到序列映射
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        cond_dim: int = 2048,
        cond_proj_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # 条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_proj_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_proj_dim * 2, cond_proj_dim),
            nn.LayerNorm(cond_proj_dim),
        )
        
        # 输入投影：noise + cond → hidden
        self.input_proj = nn.Linear(latent_dim + cond_proj_dim, hidden_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # 残差连接的可学习权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        # 输出层小初始化（残差友好）
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self, 
        noise: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            noise: 随机噪声 [B, T, D]
            features: 条件特征 [B, cond_dim] 或 [B, T, cond_dim]
            
        Returns:
            refined_noise: 修正后的噪声 [B, T, D]
        """
        B, T, D = noise.shape
        
        # 条件处理
        if features.ndim == 3:
            # [B, T, cond_dim] -> [B, cond_dim] (mean pooling)
            cond = features.mean(dim=1)
        else:
            cond = features  # [B, cond_dim]
        
        cond = self.cond_proj(cond)  # [B, cond_proj_dim]
        cond_expanded = cond.unsqueeze(1).expand(-1, T, -1)  # [B, T, cond_proj_dim]
        
        # 拼接输入
        x = torch.cat([noise, cond_expanded], dim=-1)  # [B, T, D + cond_proj_dim]
        x = self.input_proj(x)  # [B, T, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embedding[:, :T, :]
        
        # Transformer
        x = self.transformer(x)  # [B, T, hidden_dim]
        
        # 输出
        x = self.output_norm(x)
        delta = self.output_proj(x)  # [B, T, D]
        
        # 残差连接：refined = noise + weight * delta
        refined_noise = noise + self.residual_weight * delta
        
        return refined_noise


# ==========================================
# 2. 损失函数
# ==========================================

class RefinerLoss(nn.Module):
    """
    噪声修正损失函数 - 简化版
    
    只使用 MSE Loss：让输出接近反推噪声
    """
    
    def __init__(self, action_dim: int = 14):
        super().__init__()
        self.action_dim = action_dim
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predicted: 预测的噪声 [B, T, D]
            target: 目标噪声（反推噪声）[B, T, D]
        """
        # MSE Loss（只计算前 action_dim 维度）
        mse_loss = F.mse_loss(
            predicted[:, :, :self.action_dim],
            target[:, :, :self.action_dim]
        )
        
        metrics = {
            'mse_loss': mse_loss.item(),
        }
        
        return mse_loss, metrics


# ==========================================
# 3. 数据集
# ==========================================

class RefinerDataset(torch.utils.data.Dataset):
    """
    噪声修正数据集
    
    返回：
    - random_noise: 随机噪声（输入）
    - inverse_noise: 反推噪声（目标）
    - features: 条件特征
    """
    
    def __init__(
        self,
        h5_path: str,
        indices: List[int],
        seq_len: int = 50,
        normalize: bool = True,
        latent_mean: np.ndarray = None,
        latent_std: np.ndarray = None,
        feature_mean: np.ndarray = None,
        feature_std: np.ndarray = None,
    ):
        self.h5_path = h5_path
        self.indices = indices
        self.seq_len = seq_len
        self.normalize = normalize
        
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        
        self._h5_file = None
    
    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        f = self._get_h5_file()
        real_idx = self.indices[idx]
        
        # 加载反推噪声（目标）
        inverse_noise = f['latents'][real_idx]  # [T, D]
        features = f['features'][real_idx]
        
        inverse_noise = torch.from_numpy(inverse_noise.astype(np.float32))
        features = torch.from_numpy(features.astype(np.float32))
        
        # 截断
        if inverse_noise.shape[0] > self.seq_len:
            inverse_noise = inverse_noise[:self.seq_len]
        
        # 生成随机噪声（输入）- 与反推噪声相同形状
        random_noise = torch.randn_like(inverse_noise)
        
        # 归一化反推噪声
        if self.normalize and self.latent_mean is not None:
            mean = torch.from_numpy(self.latent_mean).float()
            std = torch.from_numpy(self.latent_std).float()
            std = torch.clamp(std, min=1e-6)
            inverse_noise = (inverse_noise - mean) / std
        
        # 归一化 features
        if self.normalize and self.feature_mean is not None:
            f_mean = torch.from_numpy(self.feature_mean).float()
            f_std = torch.from_numpy(self.feature_std).float()
            f_std = torch.clamp(f_std, min=1e-6)
            if features.ndim == 1:
                features = (features - f_mean) / f_std
            else:
                features = (features - f_mean) / f_std
        
        return {
            'random_noise': random_noise,      # 输入
            'inverse_noise': inverse_noise,    # 目标
            'features': features,
        }


# ==========================================
# 4. 训练参数
# ==========================================

@dataclass
class TrainArgs:
    # 数据
    h5_path: str = "latent_dataset.h5"
    seq_len: int = 50
    normalize_data: bool = True
    episode_split: bool = True
    
    # 模型
    latent_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    cond_dim: int = 2048
    cond_proj_dim: int = 128
    dropout: float = 0.1
    
    # 损失
    action_dim: int = 14
    
    # 训练
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    
    # 验证
    patience: int = 20
    
    # 其他
    device: str = "cuda"
    save_dir: str = "checkpoints/noise_refiner"
    use_wandb: bool = True
    wandb_project: str = "noise-refiner"
    num_workers: int = 4


# ==========================================
# 5. 学习率调度器
# ==========================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


# ==========================================
# 6. 训练函数
# ==========================================

def train_step(
    model: NoiseRefiner,
    batch: Dict[str, torch.Tensor],
    criterion: RefinerLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    
    random_noise = batch['random_noise'].to(device)
    inverse_noise = batch['inverse_noise'].to(device)
    features = batch['features'].to(device)
    
    # 前向传播
    predicted = model(random_noise, features)
    
    # 计算损失
    loss, metrics = criterion(predicted, inverse_noise)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    metrics['grad_norm'] = grad_norm.item()
    
    optimizer.step()
    scheduler.step()
    
    metrics['lr'] = scheduler.get_last_lr()[0]
    
    return metrics


@torch.no_grad()
def validate(
    model: NoiseRefiner,
    dataloader: torch.utils.data.DataLoader,
    criterion: RefinerLoss,
    device: str,
) -> Dict[str, float]:
    model.eval()
    
    total_metrics = {}
    count = 0
    
    for batch in dataloader:
        random_noise = batch['random_noise'].to(device)
        inverse_noise = batch['inverse_noise'].to(device)
        features = batch['features'].to(device)
        
        predicted = model(random_noise, features)
        _, metrics = criterion(predicted, inverse_noise)
        
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        count += 1
    
    return {k: v / count for k, v in total_metrics.items()}


def create_episode_split(h5_path: str, val_ratio: float = 0.15):
    """按 episode 划分数据集"""
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['latents'].shape[0]
        
        if 'episode_ids' in f:
            episode_ids = f['episode_ids'][:]
            unique_eps = np.unique(episode_ids)
            np.random.shuffle(unique_eps)
            
            split_idx = int(len(unique_eps) * (1 - val_ratio))
            train_eps = set(unique_eps[:split_idx])
            
            train_indices = [i for i in range(total_samples) if episode_ids[i] in train_eps]
            val_indices = [i for i in range(total_samples) if episode_ids[i] not in train_eps]
        else:
            indices = list(range(total_samples))
            np.random.shuffle(indices)
            split = int(total_samples * (1 - val_ratio))
            train_indices, val_indices = indices[:split], indices[split:]
    
    return train_indices, val_indices


# ==========================================
# 7. 主训练函数
# ==========================================

def train(args: TrainArgs):
    logger.info("=" * 60)
    logger.info("Noise Refiner Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # WandB
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # 1. 加载数据集
    logger.info("\n1. Loading dataset...")
    
    latent_mean, latent_std, feature_mean, feature_std = None, None, None, None
    if args.normalize_data:
        with h5py.File(args.h5_path, 'r') as f:
            if 'latents_mean' in f and 'latents_std' in f:
                latent_mean = f['latents_mean'][:]
                latent_std = f['latents_std'][:]
                if latent_mean.ndim == 2:
                    latent_mean = latent_mean.mean(axis=0)
                    latent_std = latent_std.mean(axis=0)
                logger.info(f"  Loaded latent stats: mean_range=[{latent_mean.min():.4f}, {latent_mean.max():.4f}]")
            
            if 'features_mean' in f and 'features_std' in f:
                feature_mean = f['features_mean'][:]
                feature_std = f['features_std'][:]
                if feature_mean.ndim == 2:
                    feature_mean = feature_mean.mean(axis=0)
                    feature_std = feature_std.mean(axis=0)
                logger.info(f"  Loaded feature stats")
    
    # 数据划分
    if args.episode_split:
        train_indices, val_indices = create_episode_split(args.h5_path)
    else:
        with h5py.File(args.h5_path, 'r') as f:
            total = f['latents'].shape[0]
        indices = list(range(total))
        np.random.shuffle(indices)
        split = int(0.85 * total)
        train_indices, val_indices = indices[:split], indices[split:]
    
    logger.info(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    # 创建数据集
    train_dataset = RefinerDataset(
        args.h5_path, train_indices, args.seq_len, args.normalize_data,
        latent_mean, latent_std, feature_mean, feature_std,
    )
    val_dataset = RefinerDataset(
        args.h5_path, val_indices, args.seq_len, args.normalize_data,
        latent_mean, latent_std, feature_mean, feature_std,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # 2. 创建模型
    logger.info("\n2. Creating model...")
    model = NoiseRefiner(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        cond_dim=args.cond_dim,
        cond_proj_dim=args.cond_proj_dim,
        dropout=args.dropout,
        max_seq_len=args.seq_len + 10,
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Model params: {num_params:.2f}M")
    
    # 3. 损失函数和优化器
    criterion = RefinerLoss(action_dim=args.action_dim)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=args.min_lr,
    )
    
    # 4. 训练循环
    logger.info("\n3. Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # 训练
        model.train()
        epoch_metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            metrics = train_step(
                model, batch, criterion, optimizer, scheduler,
                args.device, args.grad_clip,
            )
            
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            
            pbar.set_postfix({
                'mse': f"{metrics['mse_loss']:.4f}",
                'lr': f"{metrics['lr']:.2e}",
            })
        
        # 平均
        num_batches = len(train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, args.device)
        
        # 日志
        logger.info(
            f"Epoch {epoch+1}: "
            f"Train MSE={epoch_metrics['mse_loss']:.4f}, "
            f"Val MSE={val_metrics['mse_loss']:.4f}"
        )
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/mse': epoch_metrics['mse_loss'],
                'val/mse': val_metrics['mse_loss'],
                'lr': epoch_metrics['lr'],
                'grad_norm': epoch_metrics['grad_norm'],
            })
        
        # 保存最佳模型
        if val_metrics['mse_loss'] < best_val_loss:
            best_val_loss = val_metrics['mse_loss']
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_metrics['mse_loss'],
                'config': vars(args),
                'latent_mean': latent_mean,
                'latent_std': latent_std,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            logger.info(f"  → New best model saved! (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'latent_mean': latent_mean,
        'latent_std': latent_std,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    logger.info("\nTraining complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    train(args)
