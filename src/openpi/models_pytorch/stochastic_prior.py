"""
StochasticMotionPrior 网络定义

随机运动先验网络，用于生成时序相关的噪声序列。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


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
        hidden_dim: int = 256,       # 增大到 256
        num_layers: int = 3,          # 增大到 3
        cond_dim: int = 2048,
        cond_proj_dim: int = 128,     # 增大到 128
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
