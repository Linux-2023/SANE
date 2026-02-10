"""
Action Refiner 网络训练

目标：训练一个后处理网络，放在 PI0 模型后面，用于：
1. 对齐新 chunk 和上一个 chunk 的边界（消除跳变）
2. 消除 chunk 内的抖动
3. 保持原始动作的意图

输入：
- prev_chunk: 上一个执行的 chunk [B, exec_horizon, D]（如 15 步）
- new_chunk: PI0 输出的新 chunk [B, action_horizon, D]（如 50 步）

输出：
- refined_chunk: 修改后的新 chunk [B, action_horizon, D]

损失函数：
L_total = λ1 * L_align + λ2 * L_smooth + λ3 * L_fidelity

训练策略：
- 前期：重点 L_fidelity，让模型学会"不改错"
- 中期：加大 L_align 权重，解决跳变
- 后期：精调 L_smooth，消除震荡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import logging
import tyro
import os
from tqdm import tqdm
import wandb
import h5py

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.policies import policy_config as _policy_config
from openpi.models import model as _model
from openpi import transforms as _transforms
from lerobot.common.datasets import lerobot_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. Action Refiner 网络
# ==========================================

class ActionRefinerResNet(nn.Module):
    """
    基于残差的 Action Refiner 网络
    
    设计理念：
    - 只做小的修改（残差连接）
    - 使用 1D CNN 处理时序信息
    - 输出残差 delta，refined = original + delta
    """
    
    def __init__(
        self,
        action_dim: int = 32,
        action_horizon: int = 50,
        exec_horizon: int = 15,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        
        # 输入编码
        # prev_chunk: [B, exec_horizon, D]
        # new_chunk: [B, action_horizon, D]
        # 拼接后: [B, exec_horizon + action_horizon, D]
        total_len = exec_horizon + action_horizon
        
        self.input_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # 1D 卷积残差块
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(
                ResidualBlock1D(hidden_dim, kernel_size, dropout)
            )
        
        # 输出投影（只输出 new_chunk 部分的残差）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 残差缩放因子（初始为小值，让初始输出接近原始）
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self,
        prev_chunk: torch.Tensor,  # [B, exec_horizon, D]
        new_chunk: torch.Tensor,   # [B, action_horizon, D]
    ) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            refined_chunk: [B, action_horizon, D]
        """
        B = new_chunk.shape[0]
        
        # 1. 拼接输入
        combined = torch.cat([prev_chunk, new_chunk], dim=1)  # [B, total_len, D]
        
        # 2. 输入投影
        x = self.input_proj(combined)  # [B, total_len, hidden_dim]
        
        # 3. 转换为卷积格式 [B, C, T]
        x = x.permute(0, 2, 1)  # [B, hidden_dim, total_len]
        
        # 4. 残差块
        for block in self.res_blocks:
            x = block(x)
        
        # 5. 转换回 [B, T, C]
        x = x.permute(0, 2, 1)  # [B, total_len, hidden_dim]
        
        # 6. 只取 new_chunk 部分
        x = x[:, self.exec_horizon:, :]  # [B, action_horizon, hidden_dim]
        
        # 7. 输出残差
        delta = self.output_proj(x)  # [B, action_horizon, D]
        
        # 8. 残差连接
        refined_chunk = new_chunk + self.residual_scale * delta
        
        return refined_chunk


class ResidualBlock1D(nn.Module):
    """1D 卷积残差块"""
    
    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]"""
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        
        return self.act(x + residual)


class ActionRefinerTransformer(nn.Module):
    """
    基于 Transformer 的 Action Refiner 网络
    
    使用自注意力机制处理时序依赖
    """
    
    def __init__(
        self,
        action_dim: int = 32,
        action_horizon: int = 50,
        exec_horizon: int = 15,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.exec_horizon = exec_horizon
        total_len = exec_horizon + action_horizon
        
        # 输入投影
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, total_len, hidden_dim) * 0.02)
        
        # 类型编码（区分 prev_chunk 和 new_chunk）
        self.type_encoding = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 残差缩放
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self,
        prev_chunk: torch.Tensor,
        new_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = new_chunk.shape[0]
        
        # 1. 拼接输入
        combined = torch.cat([prev_chunk, new_chunk], dim=1)  # [B, total_len, D]
        
        # 2. 输入投影
        x = self.input_proj(combined)  # [B, total_len, hidden_dim]
        
        # 3. 添加位置编码
        x = x + self.pos_encoding
        
        # 4. 添加类型编码
        type_emb = torch.cat([
            self.type_encoding[:, 0:1, :].expand(B, self.exec_horizon, -1),
            self.type_encoding[:, 1:2, :].expand(B, self.action_horizon, -1),
        ], dim=1)
        x = x + type_emb
        
        # 5. Transformer
        x = self.transformer(x)
        
        # 6. 只取 new_chunk 部分
        x = x[:, self.exec_horizon:, :]
        
        # 7. 输出残差
        delta = self.output_proj(x)
        
        # 8. 残差连接
        refined_chunk = new_chunk + self.residual_scale * delta
        
        return refined_chunk


# ==========================================
# 2. 损失函数
# ==========================================

class RefinerLoss(nn.Module):
    """
    Action Refiner 的组合损失函数
    
    L_total = λ1 * L_align + λ2 * L_smooth + λ3 * L_fidelity
    """
    
    def __init__(
        self,
        lambda_align: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_fidelity: float = 1.0,
        align_steps: int = 3,  # 对齐的步数
        smooth_order: int = 2,  # 平滑的阶数（1=速度，2=加速度）
        action_dim: int = 14,  # 有效动作维度
    ):
        super().__init__()
        
        self.lambda_align = lambda_align
        self.lambda_smooth = lambda_smooth
        self.lambda_fidelity = lambda_fidelity
        self.align_steps = align_steps
        self.smooth_order = smooth_order
        self.action_dim = action_dim
        
    def forward(
        self,
        refined_chunk: torch.Tensor,   # [B, H, D]
        original_chunk: torch.Tensor,  # [B, H, D]
        prev_chunk: torch.Tensor,      # [B, exec_H, D]
        gt_chunk: Optional[torch.Tensor] = None,  # [B, H, D] Ground truth（可选）
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 只考虑有效动作维度
        refined = refined_chunk[..., :self.action_dim]
        original = original_chunk[..., :self.action_dim]
        prev = prev_chunk[..., :self.action_dim]
        
        # 1. 边界对齐损失 L_align
        # 让 refined 的前几帧与 prev 的最后一帧对齐
        prev_last = prev[:, -1:, :]  # [B, 1, D]
        refined_start = refined[:, :self.align_steps, :]  # [B, align_steps, D]
        
        # 位置对齐
        L_align_pos = F.mse_loss(refined_start, prev_last.expand_as(refined_start))
        
        # 速度对齐（可选）：让边界处的速度也连续
        if prev.shape[1] >= 2:
            prev_vel = prev[:, -1, :] - prev[:, -2, :]  # [B, D]
            refined_vel = refined[:, 1, :] - refined[:, 0, :]  # [B, D]
            L_align_vel = F.mse_loss(refined_vel, prev_vel)
        else:
            L_align_vel = torch.tensor(0.0, device=refined.device)
        
        L_align = L_align_pos + 0.5 * L_align_vel
        
        # 2. 平滑损失 L_smooth
        # 一阶导数（速度）
        velocity = torch.diff(refined, dim=1)  # [B, H-1, D]
        L_smooth_1 = (velocity ** 2).mean()
        
        # 二阶导数（加速度）
        if self.smooth_order >= 2:
            acceleration = torch.diff(velocity, dim=1)  # [B, H-2, D]
            L_smooth_2 = (acceleration ** 2).mean()
        else:
            L_smooth_2 = torch.tensor(0.0, device=refined.device)
        
        # 三阶导数（jerk，可选）
        if self.smooth_order >= 3:
            jerk = torch.diff(acceleration, dim=1)
            L_smooth_3 = (jerk ** 2).mean()
        else:
            L_smooth_3 = torch.tensor(0.0, device=refined.device)
        
        L_smooth = L_smooth_1 + 0.5 * L_smooth_2 + 0.1 * L_smooth_3
        
        # 3. 保真度损失 L_fidelity
        # 惩罚与原始输出的偏差
        L_fidelity = F.mse_loss(refined, original[..., :self.action_dim])
        
        # 4. 可选：Ground Truth 损失
        if gt_chunk is not None:
            gt = gt_chunk[..., :self.action_dim]
            L_gt = F.mse_loss(refined, gt)
        else:
            L_gt = torch.tensor(0.0, device=refined.device)
        
        # 总损失
        total_loss = (
            self.lambda_align * L_align +
            self.lambda_smooth * L_smooth +
            self.lambda_fidelity * L_fidelity
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'align': L_align.item(),
            'align_pos': L_align_pos.item(),
            'align_vel': L_align_vel.item() if isinstance(L_align_vel, torch.Tensor) else L_align_vel,
            'smooth': L_smooth.item(),
            'smooth_vel': L_smooth_1.item(),
            'smooth_acc': L_smooth_2.item() if isinstance(L_smooth_2, torch.Tensor) else L_smooth_2,
            'fidelity': L_fidelity.item(),
            'gt': L_gt.item() if isinstance(L_gt, torch.Tensor) else L_gt,
        }
        
        return total_loss, loss_dict


# ==========================================
# 3. 数据集
# ==========================================

class PrecomputedChunkDataset(torch.utils.data.Dataset):
    """
    从预生成的 HDF5 文件读取 (prev_chunk, new_chunk, gt_chunk) 三元组
    
    HDF5 结构：
    - prev_chunks: [N, exec_horizon, action_dim]
    - new_chunks: [N, action_horizon, action_dim]  (PI0 随机噪声输出)
    - gt_chunks: [N, action_horizon, action_dim]
    """
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        
        self.prev_chunks = self.h5_file['prev_chunks']
        self.new_chunks = self.h5_file['new_chunks']
        self.gt_chunks = self.h5_file['gt_chunks']
        
        self.num_samples = self.prev_chunks.shape[0]
        print(f"PrecomputedChunkDataset: {self.num_samples} samples from {h5_path}")
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'prev_chunk': torch.from_numpy(self.prev_chunks[idx]).float(),
            'new_chunk': torch.from_numpy(self.new_chunks[idx]).float(),
            'gt_chunk': torch.from_numpy(self.gt_chunks[idx]).float(),
        }
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def generate_chunk_dataset(
    policy,
    raw_dataset,
    output_path: str,
    exec_horizon: int = 15,
    action_horizon: int = 50,
    action_dim: int = 14,
    latent_dim: int = 32,
    num_samples: int = 5000,
    device: str = "cuda",
):
    """
    预生成 chunk 数据集并保存到 HDF5 文件
    
    Args:
        policy: PI0 策略模型
        raw_dataset: 原始 LeRobot 数据集
        output_path: 输出 HDF5 文件路径
        exec_horizon: 执行时域（prev_chunk 长度）
        action_horizon: 动作时域（new_chunk/gt_chunk 长度）
        action_dim: 实际动作维度
        latent_dim: PI0 噪声潜空间维度
        num_samples: 生成样本数
        device: 设备
    """
    print(f"\n{'='*60}")
    print("Generating Chunk Dataset")
    print(f"{'='*60}")
    
    # 计算有效索引
    max_idx = len(raw_dataset) - action_horizon - exec_horizon
    indices = list(range(exec_horizon, max_idx, exec_horizon))
    
    if len(indices) > num_samples:
        indices = indices[:num_samples]
    
    actual_samples = len(indices)
    print(f"Total samples to generate: {actual_samples}")
    
    # 创建 HDF5 文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # 预分配数据集
        prev_chunks = f.create_dataset(
            'prev_chunks', 
            shape=(actual_samples, exec_horizon, action_dim),
            dtype=np.float32
        )
        new_chunks = f.create_dataset(
            'new_chunks',
            shape=(actual_samples, action_horizon, action_dim),
            dtype=np.float32
        )
        gt_chunks = f.create_dataset(
            'gt_chunks',
            shape=(actual_samples, action_horizon, action_dim),
            dtype=np.float32
        )
        
        # 保存元数据
        f.attrs['exec_horizon'] = exec_horizon
        f.attrs['action_horizon'] = action_horizon
        f.attrs['action_dim'] = action_dim
        f.attrs['latent_dim'] = latent_dim
        f.attrs['num_samples'] = actual_samples
        
        # 生成数据
        for i, start_idx in enumerate(tqdm(indices, desc="Generating samples")):
            # 1. 获取 prev_chunk（GT 动作）
            prev_chunk = _get_gt_actions_static(raw_dataset, start_idx - exec_horizon, exec_horizon)
            
            # 2. 获取 new_chunk（PI0 随机噪声输出）
            new_chunk = _get_pi0_output_static(
                policy, raw_dataset, start_idx, 
                action_horizon, latent_dim, device
            )
            
            # 3. 获取 gt_chunk（GT 动作）
            gt_chunk = _get_gt_actions_static(raw_dataset, start_idx, action_horizon)
            
            # 只保留有效动作维度
            prev_chunks[i] = prev_chunk[:, :action_dim].numpy()
            new_chunks[i] = new_chunk[:, :action_dim].numpy()
            gt_chunks[i] = gt_chunk[:, :action_dim].numpy()
            
            # 每 100 个样本刷新一次
            if (i + 1) % 100 == 0:
                f.flush()
    
    print(f"\nDataset saved to: {output_path}")
    print(f"  - prev_chunks: {(actual_samples, exec_horizon, action_dim)}")
    print(f"  - new_chunks: {(actual_samples, action_horizon, action_dim)}")
    print(f"  - gt_chunks: {(actual_samples, action_horizon, action_dim)}")


def _get_gt_actions_static(raw_dataset, start_idx: int, length: int) -> torch.Tensor:
    """获取 Ground Truth 动作序列（静态函数版本）"""
    actions = []
    for i in range(length):
        idx = start_idx + i
        if idx >= len(raw_dataset):
            break
        
        item = raw_dataset[idx]
        action = item.get('action', item.get('actions'))
        if isinstance(action, torch.Tensor):
            action = action[0] if action.ndim > 1 else action
        else:
            action = torch.tensor(action[0] if len(action.shape) > 1 else action)
        actions.append(action)
    
    # 补齐
    while len(actions) < length:
        actions.append(actions[-1])
    
    return torch.stack(actions)  # [length, D]


def _get_pi0_output_static(
    policy, raw_dataset, idx: int, 
    action_horizon: int, latent_dim: int, device: str
) -> torch.Tensor:
    """获取 PI0 模型输出（静态函数版本）"""
    raw_item = raw_dataset[idx]
    
    # 构建观测
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
    
    # 使用独立随机高斯噪声
    noise_shape = (1, action_horizon, latent_dim)
    noise = torch.randn(noise_shape, device=device, dtype=torch.float32)
    
    # 调用 PI0
    with torch.no_grad():
        result = policy.infer(obs, noise=noise)
    
    actions = result['actions']
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    
    return actions.float()  # [action_horizon, D]


# ==========================================
# 4. 训练参数和主函数
# ==========================================

@dataclass
class TrainArgs:
    # 运行模式
    mode: str = "train"  # "generate" 生成数据集, "train" 训练模型
    
    # 预生成数据集路径
    h5_path: str = "chunk_dataset.h5"  # 预生成数据集路径
    
    # 模型路径
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    
    # 动作维度（重要：区分实际动作维度和噪声潜空间维度）
    action_dim: int = 14       # 实际动作维度（piper机器人是14维）
    latent_dim: int = 32       # PI0 噪声潜空间维度（固定为32）
    
    # 网络参数
    model_type: str = "resnet"  # "resnet" 或 "transformer"
    hidden_dim: int = 128
    num_blocks: int = 4  # ResNet 块数
    num_heads: int = 4   # Transformer 头数
    
    # 损失权重
    lambda_align: float = 1.0
    lambda_smooth: float = 0.1
    lambda_fidelity: float = 1.0
    align_steps: int = 3
    
    # 训练参数
    exec_horizon: int = 15
    num_samples: int = 5000
    batch_size: int = 32
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    
    # 权重调度（训练阶段）
    # 前 30%: fidelity 为主
    # 中 40%: align 增大
    # 后 30%: smooth 增大
    schedule_fidelity: str = "1.0->0.5"  # 起始->结束
    schedule_align: str = "0.5->2.0"
    schedule_smooth: str = "0.05->0.5"
    
    # 其他
    device: str = "cuda"
    save_dir: str = "checkpoints/action_refiner"
    save_interval: int = 10
    use_wandb: bool = True
    wandb_project: str = "action-refiner"


def parse_schedule(schedule_str: str, epoch: int, total_epochs: int) -> float:
    """解析权重调度字符串"""
    parts = schedule_str.split("->")
    start = float(parts[0])
    end = float(parts[1])
    progress = epoch / total_epochs
    return start + (end - start) * progress


def generate_mode(args: TrainArgs):
    """数据集生成模式"""
    print("=" * 60)
    print("Action Refiner - Dataset Generation Mode")
    print("=" * 60)
    
    # 1. 加载 PI0 模型
    print("\n1. Loading PI0 policy...")
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
    policy = _policy_config.create_trained_policy(
        server_config,
        args.policy_dir,
        pytorch_device=args.device
    )
    print(f"   action_dim={args.action_dim}, latent_dim={args.latent_dim}, action_horizon={model_config.action_horizon}")
    
    # 2. 加载原始数据集
    print("\n2. Loading raw dataset...")
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
    
    if data_config.prompt_from_task:
        raw_dataset = data_loader.TransformedDataset(
            raw_dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    print(f"   Dataset size: {len(raw_dataset)}")
    
    # 3. 生成预计算数据集
    generate_chunk_dataset(
        policy=policy,
        raw_dataset=raw_dataset,
        output_path=args.h5_path,
        exec_horizon=args.exec_horizon,
        action_horizon=model_config.action_horizon,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        num_samples=args.num_samples,
        device=args.device,
    )


def train_mode(args: TrainArgs):
    """训练模式（使用预生成数据集）"""
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Action Refiner - Training Mode")
    print("=" * 60)
    
    # 检查数据集是否存在
    if not os.path.exists(args.h5_path):
        raise FileNotFoundError(
            f"Dataset not found: {args.h5_path}\n"
            f"Please run with --args.mode generate first to create the dataset."
        )
    
    # 初始化 wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # 1. 加载预生成数据集
    print("\n1. Loading precomputed dataset...")
    chunk_dataset = PrecomputedChunkDataset(args.h5_path)
    
    # 读取元数据
    with h5py.File(args.h5_path, 'r') as f:
        action_horizon = f.attrs['action_horizon']
        exec_horizon = f.attrs['exec_horizon']
        action_dim = f.attrs['action_dim']
    
    print(f"   exec_horizon={exec_horizon}, action_horizon={action_horizon}, action_dim={action_dim}")
    
    # 划分训练/验证集
    train_size = int(0.9 * len(chunk_dataset))
    val_size = len(chunk_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        chunk_dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # 2. 创建模型
    print("\n2. Creating model...")
    if args.model_type == "resnet":
        model = ActionRefinerResNet(
            action_dim=action_dim,
            action_horizon=action_horizon,
            exec_horizon=exec_horizon,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
        ).to(args.device)
    else:
        model = ActionRefinerTransformer(
            action_dim=action_dim,
            action_horizon=action_horizon,
            exec_horizon=exec_horizon,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_blocks,
        ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model type: {args.model_type}")
    print(f"   Parameters: {num_params/1e6:.2f}M")
    
    # 3. 创建损失函数和优化器
    criterion = RefinerLoss(
        lambda_align=args.lambda_align,
        lambda_smooth=args.lambda_smooth,
        lambda_fidelity=args.lambda_fidelity,
        align_steps=args.align_steps,
        action_dim=action_dim,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.1
    )
    
    # 4. 训练循环
    print("\n3. Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # 更新损失权重
        criterion.lambda_align = parse_schedule(args.schedule_align, epoch, args.num_epochs)
        criterion.lambda_smooth = parse_schedule(args.schedule_smooth, epoch, args.num_epochs)
        criterion.lambda_fidelity = parse_schedule(args.schedule_fidelity, epoch, args.num_epochs)
        
        # 训练
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            prev_chunk = batch['prev_chunk'].to(args.device)
            new_chunk = batch['new_chunk'].to(args.device)
            gt_chunk = batch['gt_chunk'].to(args.device)
            
            # 前向
            refined_chunk = model(prev_chunk, new_chunk)
            
            # 计算损失
            loss, loss_dict = criterion(
                refined_chunk, new_chunk, prev_chunk, gt_chunk
            )
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss_dict)
            pbar.set_postfix(loss=f"{loss_dict['total']:.4f}")
        
        # 验证
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                prev_chunk = batch['prev_chunk'].to(args.device)
                new_chunk = batch['new_chunk'].to(args.device)
                gt_chunk = batch['gt_chunk'].to(args.device)
                
                refined_chunk = model(prev_chunk, new_chunk)
                loss, loss_dict = criterion(refined_chunk, new_chunk, prev_chunk, gt_chunk)
                val_losses.append(loss_dict)
        
        # 统计
        train_avg = {k: np.mean([d[k] for d in train_losses]) for k in train_losses[0]}
        val_avg = {k: np.mean([d[k] for d in val_losses]) for k in val_losses[0]}
        
        # 日志
        print(f"\nEpoch {epoch+1}: "
              f"Train Loss={train_avg['total']:.4f}, Val Loss={val_avg['total']:.4f}")
        print(f"  Weights: align={criterion.lambda_align:.2f}, "
              f"smooth={criterion.lambda_smooth:.2f}, fidelity={criterion.lambda_fidelity:.2f}")
        print(f"  Train: align={train_avg['align']:.4f}, smooth={train_avg['smooth']:.4f}, "
              f"fidelity={train_avg['fidelity']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/total': train_avg['total'],
                'train/align': train_avg['align'],
                'train/smooth': train_avg['smooth'],
                'train/fidelity': train_avg['fidelity'],
                'val/total': val_avg['total'],
                'val/align': val_avg['align'],
                'val/smooth': val_avg['smooth'],
                'val/fidelity': val_avg['fidelity'],
                'weight/align': criterion.lambda_align,
                'weight/smooth': criterion.lambda_smooth,
                'weight/fidelity': criterion.lambda_fidelity,
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # 保存最佳模型
        if val_avg['total'] < best_val_loss:
            best_val_loss = val_avg['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_avg['total'],
                'config': vars(args),
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  -> New best model saved! (val_loss: {val_avg['total']:.4f})")
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
            }, os.path.join(args.save_dir, f'checkpoint_{epoch+1}.pth'))
        
        scheduler.step()
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    if args.use_wandb:
        wandb.finish()


def main(args: TrainArgs):
    """主函数：根据 mode 选择运行模式"""
    if args.mode == "generate":
        generate_mode(args)
    elif args.mode == "train":
        train_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use 'generate' or 'train'.")


if __name__ == "__main__":
    tyro.cli(main)
