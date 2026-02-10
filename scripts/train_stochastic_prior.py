"""
Stochastic Motion Prior 训练脚本 (v3 - 时序学习增强版)

核心改进：
1. Scheduled Sampling：解决暴露偏差，让模型学习真正的时序依赖
2. 时序一致性损失：强制模型学习相邻帧的分布连续性
3. KL 退火策略优化：先学预测，再约束分布
4. 模型容量增大：增强时序捕捉能力
5. 自回归生成测试：验证模型是否真正学到时序规律
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
# 1. 网络定义 (增强版)
# ==========================================

class StochasticMotionPrior(nn.Module):
    """
    随机运动先验网络 (Stochastic Motion Prior) - v3
    
    改进：
    - 增大隐层维度
    - 支持单步前向（用于 Scheduled Sampling）
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,      # 从 128 增大到 256
        num_layers: int = 3,         # 从 2 增大到 3
        cond_dim: int = 2048,
        cond_proj_dim: int = 128,    # 从 64 增大到 128
        dropout: float = 0.15,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        # 条件投影（增强）
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_proj_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_proj_dim * 2, cond_proj_dim),
            nn.LayerNorm(cond_proj_dim),
        )
        
        # 缓存条件投影结果
        self._cached_cond = None
        
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
        
        # 输出头
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, latent_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.1)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight, gain=0.1)
        nn.init.constant_(self.log_std_head.bias, -1.0)
    
    def forward(
        self, 
        z_seq: torch.Tensor,
        prefix_embs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播（支持序列或单步）
        
        Args:
            z_seq: 潜变量序列 [B, T, D] 或单步 [B, 1, D]
            prefix_embs: 条件特征 [B, cond_dim] 或 [B, seq_len, cond_dim]
            hidden: GRU 隐状态 [num_layers, B, hidden_dim]
            return_hidden: 是否返回最终隐状态
        """
        B = z_seq.shape[0]
        T = z_seq.shape[1]
        
        # 条件处理
        if prefix_embs.ndim == 3:
            cond = prefix_embs.mean(dim=1)
        else:
            cond = prefix_embs
        
        cond = self.cond_proj(cond)  # [B, cond_proj_dim]
        cond_expanded = cond.unsqueeze(1).expand(-1, T, -1)  # [B, T, cond_proj_dim]
        
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


# ==========================================
# 2. 损失函数 (增强版 v2 - 时序一致性 + 分布平滑)
# ==========================================

class GaussianNLLWithKL(nn.Module):
    """
    高斯负对数似然损失 + KL 散度正则化 + 时序一致性损失 + 分布平滑损失
    
    优化：
    - 时序一致性权重从 0.1 增加到 0.5
    - 新增分布平滑损失：让预测的 μ/σ 分布更接近真实序列的统计特性
    """
    
    def __init__(
        self,
        action_dim: int = 14,
        total_dim: int = 32,
        label_smoothing: float = 0.0,
        temporal_weight: float = 0.5,     # 从0.1增加到0.5
        dist_smooth_weight: float = 0.2,  # 新增：分布平滑损失权重
    ):
        super().__init__()
        self.action_dim = action_dim
        self.total_dim = total_dim
        self.label_smoothing = label_smoothing
        self.temporal_weight = temporal_weight
        self.dist_smooth_weight = dist_smooth_weight
    
    def forward(
        self,
        mu: torch.Tensor,          # [B, T, D]
        log_std: torch.Tensor,     # [B, T, D]
        z_target: torch.Tensor,    # [B, T, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        计算 NLL、KL、时序一致性损失和分布平滑损失
        
        Returns:
            nll_loss: 负对数似然损失
            kl_loss: KL 散度损失
            aux_loss: 辅助损失（时序一致性 + 分布平滑）
            stats: 统计信息字典
        """
        std = torch.exp(log_std)
        var = std ** 2
        
        # Label smoothing
        if self.label_smoothing > 0 and self.training:
            noise = torch.randn_like(z_target) * self.label_smoothing
            z_target = z_target + noise
        
        # NLL: log σ + (z - μ)² / (2σ²)
        nll = log_std + 0.5 * ((z_target - mu) ** 2) / var
        nll_masked = nll[:, :, :self.action_dim]
        nll_loss = nll_masked.mean()
        
        # KL: 0.5 * (μ² + σ² - log σ² - 1)
        kl = 0.5 * (mu ** 2 + var - 2 * log_std - 1)
        kl_masked = kl[:, :, :self.action_dim]
        kl_loss = kl_masked.mean()
        
        # 时序一致性损失：惩罚相邻帧的 μ/σ 突变（权重增大到0.5）
        temporal_loss = torch.tensor(0.0, device=mu.device)
        if mu.shape[1] > 1:
            mu_diff = torch.abs(mu[:, 1:, :self.action_dim] - mu[:, :-1, :self.action_dim])
            std_diff = torch.abs(std[:, 1:, :self.action_dim] - std[:, :-1, :self.action_dim])
            temporal_loss = (mu_diff.mean() + std_diff.mean()) * self.temporal_weight
        
        # 分布平滑损失：让预测的分布和真实序列的分布更接近
        dist_smooth_loss = torch.tensor(0.0, device=mu.device)
        if self.dist_smooth_weight > 0:
            with torch.no_grad():
                # 计算真实序列按 batch 的均值和标准差
                true_mu = z_target[:, :, :self.action_dim].mean(dim=1, keepdim=True).expand_as(mu[:, :, :self.action_dim])
                true_std = z_target[:, :, :self.action_dim].std(dim=1, keepdim=True).expand_as(std[:, :, :self.action_dim])
                true_std = torch.clamp(true_std, min=0.1)  # 防止除零
            
            dist_smooth_loss = (
                F.mse_loss(mu[:, :, :self.action_dim], true_mu) + 
                F.mse_loss(std[:, :, :self.action_dim], true_std)
            ) * self.dist_smooth_weight
        
        # 辅助损失 = 时序一致性 + 分布平滑
        aux_loss = temporal_loss + dist_smooth_loss
        
        # 统计信息
        with torch.no_grad():
            stats = {
                'mu_mean': mu[:, :, :self.action_dim].mean().item(),
                'mu_std': mu[:, :, :self.action_dim].std().item(),
                'sigma_mean': std[:, :, :self.action_dim].mean().item(),
                'sigma_std': std[:, :, :self.action_dim].std().item(),
                'log_std_mean': log_std[:, :, :self.action_dim].mean().item(),
                'temporal_loss': temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss,
                'dist_smooth_loss': dist_smooth_loss.item() if isinstance(dist_smooth_loss, torch.Tensor) else dist_smooth_loss,
            }
        
        return nll_loss, kl_loss, aux_loss, stats


# ==========================================
# 3. Scheduled Sampling (解决暴露偏差) - 优化版
# ==========================================

class ScheduledSampler:
    """
    调度采样：训练时按概率使用模型预测的 z_t，而非真实 z_t
    
    解决 Teacher Forcing 的暴露偏差问题，让模型学习真正的时序依赖。
    
    优化：
    - 添加预热期（前 warmup_epochs 轮强制 100% Teacher Forcing）
    - 更平缓的衰减，让模型先"稳扎稳打"学好基础
    
    Args:
        start_prob: 初始使用真实 z 的概率（1.0 = 纯 Teacher Forcing）
        end_prob: 最终使用真实 z 的概率（0.8 = 保留80%真实帧）
        decay_epochs: 从 start_prob 衰减到 end_prob 的轮次
        warmup_epochs: 预热期轮次（强制 100% Teacher Forcing）
    """
    def __init__(
        self, 
        start_prob: float = 1.0, 
        end_prob: float = 0.8,      # 从0.5改为0.8，保留更多真实帧
        decay_epochs: int = 80,     # 从50改为80，衰减更慢
        warmup_epochs: int = 20,    # 新增：预热期
    ):
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs
    
    def get_true_prob(self, epoch: int) -> float:
        # 预热期：强制 100% Teacher Forcing
        if epoch < self.warmup_epochs:
            return 1.0
        
        # 衰减完成后
        effective_epoch = epoch - self.warmup_epochs
        if effective_epoch >= self.decay_epochs:
            return self.end_prob
        
        # 线性衰减
        decay_rate = (self.start_prob - self.end_prob) / self.decay_epochs
        return self.start_prob - decay_rate * effective_epoch


# ==========================================
# 4. 学习率调度器（带 Warmup）
# ==========================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """带 Warmup 的 Cosine 学习率调度器"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = self.last_epoch + 1
        
        if step < self.warmup_steps:
            lr = self.peak_lr * step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        return [lr for _ in self.base_lrs]


# ==========================================
# 5. KL Annealing (优化版 v2)
# ==========================================

class KLAnnealer:
    """
    KL 散度权重退火策略 (优化版 v2)
    
    核心改进：大幅降低 KL 权重，让模型先"学好预测"再"约束分布"
    - 前 30 轮 β=0，专注学 NLL
    - 30-90 轮 β 缓慢增长到 0.02
    - 最大 β 从 0.05 降到 0.02
    """
    
    def __init__(
        self,
        warmup_epochs: int = 30,     # 从10改为30：更长的预热期
        anneal_epochs: int = 60,     # 保持60：30-90轮缓慢增长
        beta_max: float = 0.02,      # 从0.05降到0.02：大幅降低KL约束
        cyclical: bool = False,
        cycle_length: int = 20,
    ):
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.beta_max = beta_max
        self.cyclical = cyclical
        self.cycle_length = cycle_length
    
    def get_beta(self, epoch: int) -> float:
        if self.cyclical:
            cycle_pos = epoch % self.cycle_length
            return self.beta_max * min(1.0, cycle_pos / (self.cycle_length / 2))
        
        if epoch < self.warmup_epochs:
            return 0.0
        
        if epoch < self.warmup_epochs + self.anneal_epochs:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return self.beta_max * progress
        
        return self.beta_max


# ==========================================
# 6. 数据集 (优化版 - 相对运动表示)
# ==========================================

class LatentDataset(torch.utils.data.Dataset):
    """
    潜变量数据集（支持多进程加载）- 优化版
    
    核心优化：
    - 支持「相对运动」表示，增强运动信号
    - 可选择保留部分绝对信息
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
        add_noise: bool = False,       # 训练时添加噪声
        noise_scale: float = 0.01,
        use_relative: bool = False,    # 是否使用相对运动表示
        keep_absolute_dims: int = 0,   # 保留前N维的绝对信息
    ):
        self.h5_path = h5_path
        self.indices = indices
        self.seq_len = seq_len
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.use_relative = use_relative
        self.keep_absolute_dims = keep_absolute_dims
        
        # 归一化统计量
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        
        # 每个 worker 独立的 HDF5 文件句柄
        self._h5_file = None
    
    def _get_h5_file(self):
        """延迟打开 HDF5 文件（支持多进程）"""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        f = self._get_h5_file()
        real_idx = self.indices[idx]
        
        latents = f['latents'][real_idx]  # [T, D]
        features = f['features'][real_idx]  # [seq_len, cond_dim] 或 [cond_dim]
        
        latents = torch.from_numpy(latents.astype(np.float32))
        features = torch.from_numpy(features.astype(np.float32))
        
        # 截断到 seq_len
        if latents.shape[0] > self.seq_len:
            latents = latents[:self.seq_len]
        
        # 相对运动表示：计算相对于第一帧的偏移（增强运动信号）
        if self.use_relative:
            latents_base = latents[0:1, :]  # [1, D]
            latents_relative = latents - latents_base  # [T, D]
            
            # 归一化相对运动
            if self.normalize and self.latent_mean is not None:
                mean = torch.from_numpy(self.latent_mean).float()
                std = torch.from_numpy(self.latent_std).float()
                std = torch.clamp(std, min=1e-6)
                latents_relative = (latents_relative - mean) / std
            
            # 可选：保留部分绝对信息
            if self.keep_absolute_dims > 0:
                # 绝对部分也需要归一化
                latents_abs = latents[:, :self.keep_absolute_dims]
                if self.normalize and self.latent_mean is not None:
                    latents_abs = (latents_abs - mean[:self.keep_absolute_dims]) / std[:self.keep_absolute_dims]
                latents = torch.cat([latents_abs, latents_relative], dim=-1)
            else:
                latents = latents_relative
        else:
            # 传统方式：直接归一化绝对值
            if self.normalize and self.latent_mean is not None:
                mean = torch.from_numpy(self.latent_mean).float()
                std = torch.from_numpy(self.latent_std).float()
                std = torch.clamp(std, min=1e-6)
                latents = (latents - mean) / std
        
        # Features 归一化
        if self.normalize and self.feature_mean is not None:
            f_mean = torch.from_numpy(self.feature_mean).float()
            f_std = torch.from_numpy(self.feature_std).float()
            f_std = torch.clamp(f_std, min=1e-6)
            if features.ndim == 1:
                features = (features - f_mean) / f_std
            else:
                features = (features - f_mean) / f_std
        
        # 构建输入和目标（Teacher Forcing 格式）
        z_input = latents[:-1]   # [T-1, D]
        z_target = latents[1:]   # [T-1, D]
        
        # 训练时添加噪声（增强鲁棒性）
        if self.add_noise:
            noise = torch.randn_like(z_input) * self.noise_scale
            z_input = z_input + noise
        
        return {
            'z_input': z_input,
            'z_target': z_target,
            'features': features,
            'full_latents': latents,  # 完整序列（用于验证）
        }


def create_episode_split(h5_path: str, val_ratio: float = 0.15):
    """按 episode 划分数据集，避免数据泄露"""
    with h5py.File(h5_path, 'r') as f:
        total_samples = f['latents'].shape[0]
        
        if 'episode_ids' in f:
            episode_ids = f['episode_ids'][:]
            unique_episodes = np.unique(episode_ids)
            num_episodes = len(unique_episodes)
            print(f"Using episode_ids from HDF5: {num_episodes} unique episodes")
        else:
            print("WARNING: No episode_ids in HDF5, using estimated episode size")
            episode_size = 900
            episode_ids = np.arange(total_samples) // episode_size
            unique_episodes = np.unique(episode_ids)
            num_episodes = len(unique_episodes)
    
    # 随机划分 episodes
    np.random.seed(42)
    np.random.shuffle(unique_episodes)
    
    train_episodes = int(num_episodes * (1 - val_ratio))
    train_episode_set = set(unique_episodes[:train_episodes])
    
    train_indices, val_indices = [], []
    for i in range(total_samples):
        if episode_ids[i] in train_episode_set:
            train_indices.append(i)
        else:
            val_indices.append(i)
    
    print(f"Episode split: {num_episodes} episodes -> {train_episodes} train, {num_episodes - train_episodes} val")
    print(f"  Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    return train_indices, val_indices


# ==========================================
# 7. 自回归生成测试
# ==========================================

@torch.no_grad()
def generate_sequence(
    model: StochasticMotionPrior,
    z_init: torch.Tensor,        # [B, D]
    features: torch.Tensor,      # [B, cond_dim]
    seq_len: int = 50,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自回归生成序列，用于验证模型是否学到时序规律
    
    Returns:
        generated_seq: [B, seq_len, D]
        hidden: 最终隐状态
    """
    model.eval()
    B, D = z_init.shape
    device = z_init.device
    
    generated = [z_init]
    z_t = z_init
    hidden = None
    
    for t in range(seq_len - 1):
        # 单步前向
        mu, log_std, hidden = model(
            z_t.unsqueeze(1), features, hidden, return_hidden=True
        )
        
        # 采样下一步
        z_next = model.sample(mu.squeeze(1), log_std.squeeze(1), deterministic=deterministic)
        generated.append(z_next)
        z_t = z_next
    
    return torch.stack(generated, dim=1), hidden


def evaluate_temporal_quality(
    model: StochasticMotionPrior,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    action_dim: int = 14,
    num_samples: int = 10,
) -> Dict[str, float]:
    """
    评估模型的时序预测质量
    
    计算：
    1. 真实序列的时序差分均值
    2. 生成序列的时序差分均值
    3. 两者的差距（越小说明学到了时序规律）
    """
    model.eval()
    
    true_diffs, gen_diffs = [], []
    mse_errors = []
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        z_input = batch['z_input'].to(device)
        z_target = batch['z_target'].to(device)
        features = batch['features'].to(device)
        full_latents = batch['full_latents'].to(device)
        
        B, T, D = z_input.shape
        
        # 真实序列的时序差分
        true_seq = full_latents[:, :, :action_dim]
        true_diff = torch.abs(true_seq[:, 1:] - true_seq[:, :-1]).mean().item()
        true_diffs.append(true_diff)
        
        # 自回归生成序列
        z_init = full_latents[:, 0, :]
        gen_seq, _ = generate_sequence(model, z_init, features, seq_len=T+1, deterministic=False)
        gen_seq = gen_seq[:, :, :action_dim]
        
        # 生成序列的时序差分
        gen_diff = torch.abs(gen_seq[:, 1:] - gen_seq[:, :-1]).mean().item()
        gen_diffs.append(gen_diff)
        
        # MSE（生成 vs 真实）
        min_len = min(gen_seq.shape[1], true_seq.shape[1])
        mse = F.mse_loss(gen_seq[:, :min_len], true_seq[:, :min_len]).item()
        mse_errors.append(mse)
    
    return {
        'true_diff_mean': np.mean(true_diffs),
        'gen_diff_mean': np.mean(gen_diffs),
        'diff_gap': abs(np.mean(gen_diffs) - np.mean(true_diffs)),
        'gen_mse': np.mean(mse_errors),
    }


# ==========================================
# 8. 训练参数 (优化版 v2)
# ==========================================

@dataclass
class TrainArgs:
    # 数据
    h5_path: str = "latent_dataset.h5"
    seq_len: int = 50
    normalize_data: bool = True
    episode_split: bool = True
    add_input_noise: bool = True     # 训练时添加噪声
    input_noise_scale: float = 0.01  # 噪声尺度
    use_relative: bool = False       # 是否使用相对运动表示（可选）
    keep_absolute_dims: int = 0      # 保留绝对信息的维度数（use_relative=True时生效）
    
    # 模型 (增强版)
    latent_dim: int = 32
    hidden_dim: int = 256            # 从 128 增大到 256
    num_layers: int = 3              # 从 2 增大到 3
    cond_dim: int = 2048
    cond_proj_dim: int = 128         # 从 64 增大到 128
    dropout: float = 0.2             # 从 0.15 增加到 0.2（增强正则化）
    
    # 损失
    action_dim: int = 14
    label_smoothing: float = 0.01
    temporal_weight: float = 0.5     # 从 0.1 增加到 0.5（增强时序约束）
    dist_smooth_weight: float = 0.2  # 新增：分布平滑损失权重
    
    # KL Annealing (优化版 v2)
    warmup_epochs: int = 30          # 从 10 增大到 30（更长预热期）
    anneal_epochs: int = 60          # 保持 60
    beta_max: float = 0.02           # 从 0.05 降到 0.02（大幅降低KL约束）
    cyclical_kl: bool = False
    
    # Scheduled Sampling (优化版)
    use_scheduled_sampling: bool = True
    ss_start_prob: float = 1.0       # 初始使用真实 z 的概率
    ss_end_prob: float = 0.8         # 从 0.5 改为 0.8（保留更多真实帧）
    ss_decay_epochs: int = 80        # 从 50 改为 80（衰减更慢）
    ss_warmup_epochs: int = 20       # 新增：预热期轮次
    
    # 训练 (优化版)
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 2e-4                 # 从 3e-4 降到 2e-4（降低拟合速度）
    min_lr: float = 1e-5
    warmup_steps: int = 2500
    weight_decay: float = 2e-3       # 从 1e-3 增加到 2e-3（增强正则化）
    grad_clip: float = 1.0
    dynamic_grad_clip: bool = True   # 新增：动态梯度裁剪
    
    # 验证
    eval_temporal_every: int = 5     # 从 10 改为 5（更频繁评估）
    patience: int = 25               # 从 20 增加到 25（更有耐心）
    
    # 其他
    device: str = "cuda"
    save_dir: str = "checkpoints/stochastic_prior"
    use_wandb: bool = True
    wandb_project: str = "stochastic-prior"
    num_workers: int = 4


# ==========================================
# 9. 训练函数 (Scheduled Sampling 版)
# ==========================================

def train_step_scheduled_sampling(
    model: StochasticMotionPrior,
    batch: Dict[str, torch.Tensor],
    criterion: GaussianNLLWithKL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    beta: float,
    true_prob: float,
    device: str,
    grad_clip: float,
    dynamic_grad_clip: bool = True,
) -> Dict[str, float]:
    """
    单步训练（使用 Scheduled Sampling）
    
    Args:
        true_prob: 使用真实 z 的概率（1.0 = 纯 Teacher Forcing，0.5 = 一半预测）
        dynamic_grad_clip: 是否启用动态梯度裁剪（梯度过大时降低学习率）
    """
    model.train()
    
    z_input = batch['z_input'].to(device)   # [B, T-1, D]
    z_target = batch['z_target'].to(device)
    features = batch['features'].to(device)
    
    B, T, D = z_input.shape
    
    # 初始化
    hidden = None
    mu_seq, log_std_seq = [], []
    
    # 逐步自回归训练
    z_t = z_input[:, 0, :]  # 初始帧用真实值 [B, D]
    
    for t in range(T):
        # 单步前向
        mu_t, log_std_t, hidden = model(
            z_t.unsqueeze(1), features, hidden, return_hidden=True
        )
        mu_seq.append(mu_t)
        log_std_seq.append(log_std_t)
        
        # Scheduled Sampling：决定下一步用真实 z 还是预测 z
        if t + 1 < T:
            if np.random.rand() < true_prob:
                # 用真实 z
                z_t = z_input[:, t + 1, :]
            else:
                # 用预测 z（采样）
                z_t = model.sample(mu_t.squeeze(1), log_std_t.squeeze(1))
    
    # 拼接结果
    mu = torch.cat(mu_seq, dim=1)       # [B, T, D]
    log_std = torch.cat(log_std_seq, dim=1)
    
    # 计算损失（aux_loss 包含时序一致性 + 分布平滑）
    nll_loss, kl_loss, aux_loss, stats = criterion(mu, log_std, z_target)
    loss = nll_loss + beta * kl_loss + aux_loss
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 动态梯度裁剪：梯度范数过大时降低学习率
    lr_adjusted = False
    if dynamic_grad_clip and grad_norm > 2.0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.95
        lr_adjusted = True
    
    optimizer.step()
    scheduler.step()
    
    return {
        'loss': loss.item(),
        'nll': nll_loss.item(),
        'kl': kl_loss.item(),
        'aux': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
        'grad_norm': grad_norm.item(),
        'lr_adjusted': 1.0 if lr_adjusted else 0.0,
        **stats,
    }


def train_step_teacher_forcing(
    model: StochasticMotionPrior,
    batch: Dict[str, torch.Tensor],
    criterion: GaussianNLLWithKL,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    beta: float,
    device: str,
    grad_clip: float,
    dynamic_grad_clip: bool = True,
) -> Dict[str, float]:
    """单步训练（纯 Teacher Forcing，用于对比）"""
    model.train()
    
    z_input = batch['z_input'].to(device)
    z_target = batch['z_target'].to(device)
    features = batch['features'].to(device)
    
    # 前向
    mu, log_std, _ = model(z_input, features)
    
    # 损失（aux_loss 包含时序一致性 + 分布平滑）
    nll_loss, kl_loss, aux_loss, stats = criterion(mu, log_std, z_target)
    loss = nll_loss + beta * kl_loss + aux_loss
    
    # 反向
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 动态梯度裁剪
    lr_adjusted = False
    if dynamic_grad_clip and grad_norm > 2.0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.95
        lr_adjusted = True
    
    optimizer.step()
    scheduler.step()
    
    return {
        'loss': loss.item(),
        'nll': nll_loss.item(),
        'kl': kl_loss.item(),
        'aux': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
        'grad_norm': grad_norm.item(),
        'lr_adjusted': 1.0 if lr_adjusted else 0.0,
        **stats,
    }


def train(args: TrainArgs):
    """主训练循环"""
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stochastic Motion Prior Training (v3 - Enhanced)")
    print("=" * 60)
    print(f"Scheduled Sampling: {args.use_scheduled_sampling}")
    print(f"Model: hidden={args.hidden_dim}, layers={args.num_layers}, cond_proj={args.cond_proj_dim}")
    print(f"KL Annealing: warmup={args.warmup_epochs}, anneal={args.anneal_epochs}, β_max={args.beta_max}")
    
    # 初始化 wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # 1. 加载数据集
    print("\n1. Loading dataset...")
    
    latent_mean, latent_std, feature_mean, feature_std = None, None, None, None
    if args.normalize_data:
        with h5py.File(args.h5_path, 'r') as f:
            if 'latents_mean' in f and 'latents_std' in f:
                latent_mean = f['latents_mean'][:]
                latent_std = f['latents_std'][:]
                if latent_mean.ndim == 2:
                    latent_mean = latent_mean.mean(axis=0)
                    latent_std = latent_std.mean(axis=0)
                print(f"  Loaded latent stats: mean_range=[{latent_mean.min():.4f}, {latent_mean.max():.4f}]")
            
            if 'features_mean' in f and 'features_std' in f:
                feature_mean = f['features_mean'][:]
                feature_std = f['features_std'][:]
                print(f"  Loaded feature stats: mean_range=[{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
    
    # 划分数据集
    if args.episode_split:
        train_indices, val_indices = create_episode_split(args.h5_path)
    else:
        with h5py.File(args.h5_path, 'r') as f:
            total = f['latents'].shape[0]
        indices = list(range(total))
        np.random.shuffle(indices)
        split = int(0.85 * total)
        train_indices, val_indices = indices[:split], indices[split:]
    
    # 创建数据集
    train_dataset = LatentDataset(
        args.h5_path, train_indices, args.seq_len, args.normalize_data,
        latent_mean, latent_std, feature_mean, feature_std,
        add_noise=args.add_input_noise, noise_scale=args.input_noise_scale,
        use_relative=args.use_relative, keep_absolute_dims=args.keep_absolute_dims,
    )
    val_dataset = LatentDataset(
        args.h5_path, val_indices, args.seq_len, args.normalize_data,
        latent_mean, latent_std, feature_mean, feature_std,
        add_noise=False,
        use_relative=args.use_relative, keep_absolute_dims=args.keep_absolute_dims,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    print(f"  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # 2. 创建模型
    print("\n2. Creating model...")
    model = StochasticMotionPrior(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cond_dim=args.cond_dim,
        cond_proj_dim=args.cond_proj_dim,
        dropout=args.dropout,
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params/1e6:.2f}M")
    
    # 3. 创建优化器和调度器
    criterion = GaussianNLLWithKL(
        action_dim=args.action_dim,
        label_smoothing=args.label_smoothing,
        temporal_weight=args.temporal_weight,
        dist_smooth_weight=args.dist_smooth_weight,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    total_steps = args.num_epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr,
    )
    
    kl_annealer = KLAnnealer(
        warmup_epochs=args.warmup_epochs,
        anneal_epochs=args.anneal_epochs,
        beta_max=args.beta_max,
        cyclical=args.cyclical_kl,
    )
    
    scheduled_sampler = ScheduledSampler(
        start_prob=args.ss_start_prob,
        end_prob=args.ss_end_prob,
        decay_epochs=args.ss_decay_epochs,
        warmup_epochs=args.ss_warmup_epochs,
    ) if args.use_scheduled_sampling else None
    
    # 4. 训练循环
    print("\n3. Starting training...")
    print(f"  Scheduled Sampling: warmup={args.ss_warmup_epochs}, decay={args.ss_decay_epochs}, end_prob={args.ss_end_prob}")
    print(f"  KL Annealing: warmup={args.warmup_epochs}, anneal={args.anneal_epochs}, β_max={args.beta_max}")
    print(f"  Loss weights: temporal={args.temporal_weight}, dist_smooth={args.dist_smooth_weight}")
    
    best_val_loss = float('inf')
    best_diff_gap = float('inf')
    global_step = 0
    patience = args.patience
    epochs_without_improvement = 0
    current_lr_factor = 1.0  # 用于动态梯度裁剪
    
    for epoch in range(args.num_epochs):
        beta = kl_annealer.get_beta(epoch)
        true_prob = scheduled_sampler.get_true_prob(epoch) if scheduled_sampler else 1.0
        
        # 训练
        train_metrics = {
            'nll': [], 'kl': [], 'aux': [], 'loss': [],
            'mu_mean': [], 'mu_std': [], 'sigma_mean': [], 'grad_norm': [],
            'temporal_loss': [], 'dist_smooth_loss': [], 'lr_adjusted': []
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            if args.use_scheduled_sampling:
                metrics = train_step_scheduled_sampling(
                    model, batch, criterion, optimizer, scheduler,
                    beta, true_prob, args.device, args.grad_clip,
                    args.dynamic_grad_clip,
                )
            else:
                metrics = train_step_teacher_forcing(
                    model, batch, criterion, optimizer, scheduler,
                    beta, args.device, args.grad_clip,
                    args.dynamic_grad_clip,
                )
            
            global_step += 1
            
            for k, v in metrics.items():
                if k in train_metrics:
                    train_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                nll=f"{metrics['nll']:.4f}",
                kl=f"{metrics['kl']:.4f}",
                aux=f"{metrics['aux']:.4f}",
                β=f"{beta:.2e}",
                p=f"{true_prob:.2f}",
            )
        
        # 验证
        model.eval()
        val_metrics = {'nll': [], 'kl': [], 'aux': [], 'mse': [], 'sigma_mean': []}
        
        with torch.no_grad():
            for batch in val_loader:
                z_input = batch['z_input'].to(args.device)
                z_target = batch['z_target'].to(args.device)
                features = batch['features'].to(args.device)
                
                mu, log_std, _ = model(z_input, features)
                nll_loss, kl_loss, aux_loss, stats = criterion(mu, log_std, z_target)
                mse = F.mse_loss(mu[:, :, :args.action_dim], z_target[:, :, :args.action_dim])
                
                val_metrics['nll'].append(nll_loss.item())
                val_metrics['kl'].append(kl_loss.item())
                val_metrics['aux'].append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
                val_metrics['mse'].append(mse.item())
                val_metrics['sigma_mean'].append(stats['sigma_mean'])
        
        # 时序质量评估
        temporal_eval = None
        if (epoch + 1) % args.eval_temporal_every == 0:
            temporal_eval = evaluate_temporal_quality(
                model, val_loader, args.device, args.action_dim, num_samples=10
            )
            print(f"\n  [Temporal Eval] True diff: {temporal_eval['true_diff_mean']:.4f}, "
                  f"Gen diff: {temporal_eval['gen_diff_mean']:.4f}, "
                  f"Gap: {temporal_eval['diff_gap']:.4f}, "
                  f"Gen MSE: {temporal_eval['gen_mse']:.4f}")
        
        # 统计
        train_avg = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in train_metrics.items()}
        val_avg = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in val_metrics.items()}
        val_loss = val_avg['nll'] + beta * val_avg['kl'] + val_avg['aux']
        
        # 日志
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train: NLL={train_avg['nll']:.4f}, KL={train_avg['kl']:.4f}, Aux={train_avg['aux']:.4f}")
        print(f"  Val:   NLL={val_avg['nll']:.4f}, KL={val_avg['kl']:.4f}, MSE={val_avg['mse']:.4f}")
        print(f"  Stats: μ_mean={train_avg['mu_mean']:.4f}, σ_mean={train_avg['sigma_mean']:.4f}, val_σ={val_avg['sigma_mean']:.4f}")
        print(f"  β={beta:.2e}, true_prob={true_prob:.2f}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        if args.use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/nll': train_avg['nll'],
                'train/kl': train_avg['kl'],
                'train/aux': train_avg['aux'],
                'train/loss': train_avg['loss'],
                'train/mu_mean': train_avg['mu_mean'],
                'train/sigma_mean': train_avg['sigma_mean'],
                'train/grad_norm': train_avg['grad_norm'],
                'train/temporal_loss': train_avg.get('temporal_loss', 0.0),
                'train/dist_smooth_loss': train_avg.get('dist_smooth_loss', 0.0),
                'train/lr_adjusted': train_avg.get('lr_adjusted', 0.0),
                'val/nll': val_avg['nll'],
                'val/kl': val_avg['kl'],
                'val/aux': val_avg['aux'],
                'val/mse': val_avg['mse'],
                'val/sigma_mean': val_avg['sigma_mean'],
                'val/loss': val_loss,
                'beta': beta,
                'true_prob': true_prob,
                'lr': scheduler.get_last_lr()[0],
            }
            if temporal_eval:
                log_dict.update({
                    'temporal/true_diff': temporal_eval['true_diff_mean'],
                    'temporal/gen_diff': temporal_eval['gen_diff_mean'],
                    'temporal/diff_gap': temporal_eval['diff_gap'],
                    'temporal/gen_mse': temporal_eval['gen_mse'],
                })
            wandb.log(log_dict)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'config': vars(args),
                'latent_mean': latent_mean,
                'latent_std': latent_std,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  → New best model saved! (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement}/{patience} epochs")
            if epochs_without_improvement >= patience:
                print(f"\n>>> Early stopping triggered! Best val_loss: {best_val_loss:.4f}")
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
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    tyro.cli(train)
