"""
Noise Refiner 网络

功能：将随机噪声修正为更优的噪声
输入：随机噪声 [B, T, D] + features [B, cond_dim]
输出：修正后的噪声 [B, T, D]
"""

import torch
import torch.nn as nn


class NoiseRefiner(nn.Module):
    """
    噪声修正网络
    
    基于 Transformer 的序列到序列映射
    学习 f(random_noise, features) → refined_noise
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
            norm_first=True,
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
            cond = features.mean(dim=1)  # [B, cond_dim]
        else:
            cond = features
        
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
        
        # 残差连接
        refined_noise = noise + self.residual_weight * delta
        
        return refined_noise
