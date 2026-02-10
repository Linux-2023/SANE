import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: gamma * x + beta
    
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
        
        style = self.cond_proj(cond)  # [B, 2*C]
        style = style.unsqueeze(-1)   # [B, 2*C, 1]
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
    """
    残差过渡网络：预测隐变量的变化量 delta_z = z_{t+1} - z_t
    
    架构：1D-ResNet + FiLM 条件调制
    输入：z_t [B, H, D] 和条件特征 prefix_embs [B, C]
    输出：delta_z [B, H, D]
    """
    def __init__(
        self, 
        latent_dim=32, 
        horizon=50, 
        cond_dim=2048,  # prefix_embs 的维度
        hidden_dim=256,
        num_blocks=4
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.cond_dim = cond_dim
        
        # 1. 输入归一化
        # 注意：nn.LayerNorm(D) 会对输入的最后一维归一化
        # 对于 [B, H, D] 输入，会对每个 (B, H) 位置的 D 维度归一化（正确）
        self.z_norm = nn.LayerNorm(latent_dim)
        
        # 2. 条件归一化
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