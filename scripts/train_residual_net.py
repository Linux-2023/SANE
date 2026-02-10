import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import tyro
from dataclasses import dataclass
import math

# ==========================================
# 1. 模型定义: 1D-ResNet + FiLM
# ==========================================

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
    gamma * x + beta
    
    改进：FiLM 位置调整到 conv → act → film 结构
    """
    def __init__(self, cond_dim, num_features):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, num_features * 2)
        # 初始化: gamma=1, beta=0 (Identity modulation)
        # 这比 gamma=0 更稳定，因为 gamma=0 会直接把特征清零
        self.cond_proj.weight.data.zero_()
        # gamma 初始化为 1，beta 初始化为 0
        self.cond_proj.bias.data[:num_features] = 1.0  # gamma = 1
        self.cond_proj.bias.data[num_features:] = 0.0  # beta = 0

    def forward(self, x, cond):
        # x: [Batch, Channels, Length]
        # cond: [Batch, Cond_Dim]
        
        style = self.cond_proj(cond) # [B, 2*C]
        style = style.unsqueeze(-1)  # [B, 2*C, 1]
        gamma, beta = style.chunk(2, dim=1)
        
        return gamma * x + beta

class ResBlock1D(nn.Module):
    """
    改进结构: conv → act → film → conv → act → film + residual
    这样 FiLM 在激活后调制，更符合 StyleGAN 的最佳实践
    """
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
        cond_dim=2048,  # 修复：默认值与实际数据一致
        hidden_dim=256,
        num_blocks=4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        
        # 1. 输入归一化
        # 注意：nn.LayerNorm(D) 会对输入的最后一维归一化
        # 对于 [B, H, D] 输入，会对每个 (B, H) 位置的 D 维度归一化（正确）
        # nn.LayerNorm([H, D]) 会错误地将 H*D 整体归一化
        self.z_norm = nn.LayerNorm(latent_dim)
        
        # 2. 条件归一化
        self.cond_dim = cond_dim
        self.cond_norm = nn.LayerNorm(cond_dim)
        
        # 3. 输入投影
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        
        # 4. 残差主干
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_dim, cond_dim) for _ in range(num_blocks)
        ])
        
        # 5. 输出头 (新增：LayerNorm before output)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        
        # Zero-Init Output Projection
        self.output_proj.weight.data.zero_()
        self.output_proj.bias.data.zero_()
        
        # Delta 归一化统计量 (训练时注册)
        self.register_buffer('delta_mean', torch.zeros(latent_dim))
        self.register_buffer('delta_std', torch.ones(latent_dim))

    def set_delta_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """设置 delta 归一化统计量"""
        self.delta_mean = mean.to(self.delta_mean.device)
        self.delta_std = std.to(self.delta_std.device)

    def forward(self, z_t, prefix_embs, normalize_output=True):
        """
        Args:
            z_t: [Batch, Horizon, Latent_Dim]
            prefix_embs: [Batch, Seq_Len, Embed_Dim] or [Batch, Embed_Dim]
            normalize_output: 如果 True，输出归一化后的 delta；推理时设为 False
        Returns:
            delta_z: [Batch, Horizon, Latent_Dim]
        """
        # 输入归一化
        z_t = self.z_norm(z_t)
        
        # [B, H, D] -> [B, D, H]
        x = z_t.transpose(1, 2)
        
        # Global Average Pooling: [B, Seq, Dim] -> [B, Dim]
        if prefix_embs.ndim == 3:
            cond = prefix_embs.mean(dim=1) 
        else:
            cond = prefix_embs
        
        # 条件归一化
        cond = self.cond_norm(cond)
        
        # Encoding
        x = self.input_proj(x)
        
        # Residual Blocks
        for block in self.blocks:
            x = block(x, cond)
        
        # Output Head with LayerNorm
        # [B, C, H] -> [B, H, C] for LayerNorm
        x = x.transpose(1, 2)
        x = self.output_norm(x)
        x = x.transpose(1, 2)
        
        # Decoding
        delta = self.output_proj(x)
        
        # [B, D, H] -> [B, H, D]
        delta = delta.transpose(1, 2)
        
        # 反归一化（推理时）
        if not normalize_output and self.delta_std.sum() > 0:
            delta = delta * self.delta_std + self.delta_mean
        
        return delta

# ==========================================
# 2. 数据集定义
# ==========================================

def load_dataset_stats(h5_path: str) -> dict:
    """
    从 HDF5 文件加载预计算的统计量。
    新版 generate_latent_dataset.py 会自动保存这些统计量。
    """
    stats = {}
    with h5py.File(h5_path, 'r') as f:
        # Delta 统计量（最重要）
        if 'delta_mean' in f:
            stats['delta_mean'] = torch.from_numpy(f['delta_mean'][:]).float()
            stats['delta_std'] = torch.from_numpy(f['delta_std'][:]).float()
            print(f"Loaded delta stats from HDF5:")
            print(f"  Mean range: [{stats['delta_mean'].min():.4f}, {stats['delta_mean'].max():.4f}]")
            print(f"  Std range:  [{stats['delta_std'].min():.4f}, {stats['delta_std'].max():.4f}]")
        else:
            print("Warning: No precomputed delta stats in HDF5, will compute on-the-fly")
            stats['delta_mean'] = None
            stats['delta_std'] = None
        
        # Latents 统计量
        if 'latents_mean' in f:
            stats['latents_mean'] = torch.from_numpy(f['latents_mean'][:]).float()
            stats['latents_std'] = torch.from_numpy(f['latents_std'][:]).float()
        
        # Features 统计量
        if 'features_mean' in f:
            stats['features_mean'] = torch.from_numpy(f['features_mean'][:]).float()
            stats['features_std'] = torch.from_numpy(f['features_std'][:]).float()
        
        # 元数据
        if 'total_samples' in f.attrs:
            stats['total_samples'] = f.attrs['total_samples']
            stats['ode_solver'] = f.attrs.get('ode_solver', 'unknown')
            stats['num_ode_steps'] = f.attrs.get('num_ode_steps', 0)
            print(f"Dataset metadata:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  ODE solver: {stats['ode_solver']}")
            print(f"  ODE steps: {stats['num_ode_steps']}")
    
    return stats


class LatentResidualDataset(Dataset):
    def __init__(self, h5_path, delta_mean=None, delta_std=None):
        super().__init__()
        self.h5_path = h5_path
        self.pairs = self._build_indices()
        
        # Delta 归一化统计量
        self.delta_mean = delta_mean
        self.delta_std = delta_std
        
    def _build_indices(self):
        """
        构建训练对索引。
        返回 list[int] 以兼容 PyTorch DataLoader（避免 np.int64 类型问题）
        """
        print(f"Indexing dataset from {self.h5_path}...")
        
        with h5py.File(self.h5_path, 'r') as f:
            # Check if episode_ids exists and is not empty
            if 'episode_ids' not in f or f['episode_ids'].shape[0] == 0:
                # Fallback: Assume continuous if no IDs
                total_len = f['latents'].shape[0]
                print("Warning: No episode_ids found or empty. Assuming continuous single episode.")
                # 返回 list[int] 兼容 PyTorch
                pairs = list(range(total_len - 1))
            else:
                ep_ids = f['episode_ids'][:]
                # Vectorized search for valid transitions
                # valid if id[i] == id[i+1]
                valid_mask = (ep_ids[:-1] == ep_ids[1:])
                # np.int64 → list[int] 兼容 PyTorch DataLoader
                pairs = np.where(valid_mask)[0].tolist()
                    
        print(f"Found {len(pairs)} valid transitions.")
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # 延迟加载 HDF5
        idx_curr = int(self.pairs[idx])
        idx_next = idx_curr + 1
        
        with h5py.File(self.h5_path, 'r') as f:
            # 1. 输入: z_t, feat_t
            z_curr = torch.from_numpy(f['latents'][idx_curr]).float()
            
            feat_key = 'features' if 'features' in f else 'image_feats'
            feat_curr = torch.from_numpy(f[feat_key][idx_curr]).float()
            
            # 2. 目标: z_{t+1}
            z_next = torch.from_numpy(f['latents'][idx_next]).float()
            
            # 计算残差目标
            target_delta = z_next - z_curr
            
            # 归一化 delta
            if self.delta_mean is not None and self.delta_std is not None:
                target_delta = (target_delta - self.delta_mean) / (self.delta_std + 1e-8)
            
            # [Safety] Clamp extremums (归一化后 5 sigma 已经很极端了)
            target_delta = torch.clamp(target_delta, -5.0, 5.0)
            
        return z_curr, feat_curr, target_delta

# ==========================================
# 3. 训练逻辑
# ==========================================

@dataclass
class TrainArgs:
    dataset_path: str = "/home/ztlab/lgd/openpi/latent_dataset.h5"
    save_dir: str = "checkpoints/residual_net"
    batch_size: int = 64
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    device: str = "cuda"
    num_workers: int = 4  # DataLoader workers, 设为 0 禁用多进程
    
    # Model Params
    latent_dim: int = 32
    horizon: int = 50
    cond_dim: int = 2048  # Matches SigLIP/PaliGemma embedding dim
    hidden_dim: int = 256
    num_blocks: int = 4
    
    # Delta 归一化（现在从 HDF5 加载，无需重新计算）
    normalize_delta: bool = True
    
    # WandB
    wandb_project: str = "openpi-residual-policy"
    wandb_entity: str = ""
    name: str = "residual_net_v2"

def train(args: TrainArgs):
    # Init WandB
    wandb.init(
        project=args.wandb_project,
        name=args.name,
        config=vars(args),
        entity=args.wandb_entity if args.wandb_entity else None
    )

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # 1. 从 HDF5 加载预计算的统计量
    print(f"Loading dataset stats from {args.dataset_path}...")
    dataset_stats = load_dataset_stats(args.dataset_path)
    
    delta_mean, delta_std = None, None
    if args.normalize_delta:
        delta_mean = dataset_stats.get('delta_mean')
        delta_std = dataset_stats.get('delta_std')
        
        if delta_mean is not None:
            # 保存统计量（兼容旧的推理代码）
            stats_path = os.path.join(args.save_dir, "delta_stats.pt")
            torch.save({'mean': delta_mean, 'std': delta_std}, stats_path)
            print(f"Delta stats saved to {stats_path}")
        else:
            print("Warning: No delta stats found, training without normalization")
            args.normalize_delta = False
    
    # 2. 准备数据
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = LatentResidualDataset(
        args.dataset_path,
        delta_mean=delta_mean,
        delta_std=delta_std
    )
    
    # 划分训练/验证集 (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoader 配置：兼容 num_workers=0
    loader_kwargs = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    # persistent_workers 只在 num_workers > 0 时有效
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    
    # 3. 初始化模型
    model = ResidualTransitionNet(
        latent_dim=args.latent_dim, 
        horizon=args.horizon, 
        cond_dim=args.cond_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks
    ).to(device)
    
    # 设置 delta 统计量到模型（用于推理时反归一化）
    if delta_mean is not None:
        model.set_delta_stats(delta_mean.to(device), delta_std.to(device))
    
    print(f"Model initialized on {device}. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Warmup + Cosine Decay Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.MSELoss()
    
    # 4. 训练循环
    best_loss = float('inf')
    global_step = 0
    
    print(f"Start training for {args.epochs} epochs...")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (z_t, feat, target_delta) in enumerate(pbar):
            z_t = z_t.to(device)
            feat = feat.to(device)
            target_delta = target_delta.to(device)
            
            # Forward
            pred_delta = model(z_t, feat, normalize_output=True)
            
            # Loss
            loss = criterion(pred_delta, target_delta)

            # Nan Check
            if torch.isnan(loss) or loss.item() > 1e6:
                print(f"Abnormal loss {loss.item()} at step {i}, skipping...")
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            global_step += 1
            
            # Log step metrics
            current_lr = optimizer.param_groups[0]['lr']
            if i % 10 == 0:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": current_lr,
                    "train/pred_mean": pred_delta.mean().item(),
                    "train/pred_std": pred_delta.std().item(),
                    "train/target_mean": target_delta.mean().item(),
                    "train/target_std": target_delta.std().item(),
                    "step": global_step
                })
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}"
            })
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for z_t, feat, target_delta in val_loader:
                z_t = z_t.to(device)
                feat = feat.to(device)
                target_delta = target_delta.to(device)
                
                pred_delta = model(z_t, feat, normalize_output=True)
                loss = criterion(pred_delta, target_delta)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Log Epoch Metrics
        wandb.log({
            "epoch/train_loss": avg_train_loss,
            "epoch/val_loss": avg_val_loss,
            "epoch/best_val_loss": min(best_loss, avg_val_loss),
            "epoch": epoch + 1
        })
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Save Best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  → New best model saved! (val_loss: {best_loss:.6f})")
            
    # Save Final
    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    
    # Save model artifact to WandB
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file(os.path.join(args.save_dir, "best_model.pth"))
    stats_file = os.path.join(args.save_dir, "delta_stats.pt")
    if os.path.exists(stats_file):
        artifact.add_file(stats_file)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print("Training finished.")

if __name__ == "__main__":
    tyro.cli(train)
