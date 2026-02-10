"""
强化学习训练噪声生成策略

目标：学习一个网络，根据 VLM 特征和上一个噪声，生成下一个噪声，
      使得相邻 chunk 之间的动作跳变最小。

核心组件：
1. NoiseEnv: Gym 环境，封装 PI0 模型和 chunk 执行逻辑
2. NoisePolicy: Actor 网络，输出下一个噪声
3. PPO 训练循环

奖励设计：
- 主奖励: -||a_{new}[0:overlap] - a_{old}[exec_h-overlap:exec_h]||
- 正则项: -β * KL(z || N(0,1))，保持噪声在标准高斯范围内
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import jax
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import logging
import tyro
import os
from tqdm import tqdm
import wandb
from collections import deque

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.policies import policy_config as _policy_config
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi import transforms as _transforms
from lerobot.common.datasets import lerobot_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. 噪声生成策略网络 (Actor) - 基于 GRU 架构
# ==========================================

class StochasticMotionPriorActor(nn.Module):
    """
    基于 GRU 的随机运动先验网络（与预训练的 StochasticMotionPrior 架构兼容）
    
    结构：
    - 输入: z_t (32 dim) + pooled prefix_embs (2048 -> cond_proj_dim)
    - 核心: GRU (hidden_dim, num_layers)
    - 输出: μ (32 dim) + log σ (32 dim)
    
    可以加载预训练的 stochastic_prior 权重作为初始化。
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        cond_dim: int = 2048,
        cond_proj_dim: int = 64,
        dropout: float = 0.1,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
        horizon: int = 50,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.horizon = horizon
        self.cond_proj_dim = cond_proj_dim
        
        # 条件投影（与 StochasticMotionPrior 一致）
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_proj_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_proj_dim * 2, cond_proj_dim),
            nn.LayerNorm(cond_proj_dim),
        )
        
        # 输入投影（与 StochasticMotionPrior 一致）
        gru_input_dim = latent_dim + cond_proj_dim
        self.input_proj = nn.Sequential(
            nn.Linear(gru_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # GRU 核心（与 StochasticMotionPrior 一致）
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # 输出头：均值（与 StochasticMotionPrior 一致）
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # 输出头：对数标准差（与 StochasticMotionPrior 一致）
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # GRU 隐状态（用于自回归生成）
        self._hidden = None
        
    def reset_hidden(self):
        """重置 GRU 隐状态"""
        self._hidden = None
        
    def forward(
        self,
        prefix_embs: torch.Tensor,  # [B, cond_dim] 或 [B, seq_len, cond_dim]
        prev_noise: torch.Tensor,   # [B, H, D]
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            prefix_embs: VLM 特征 [B, cond_dim] 或 [B, seq_len, cond_dim]
            prev_noise: 上一个噪声 [B, H, D]
            hidden: GRU 隐状态 [num_layers, B, hidden_dim]
            
        Returns:
            mean: [B, H, D]
            log_std: [B, H, D]
            new_hidden: [num_layers, B, hidden_dim]
        """
        B, H, D = prev_noise.shape
        
        # 1. 处理条件特征
        if prefix_embs.ndim == 3:
            cond = prefix_embs.mean(dim=1)  # [B, cond_dim]
        else:
            cond = prefix_embs
        cond = self.cond_proj(cond)  # [B, cond_proj_dim]
        
        # 2. 扩展条件到每个时间步
        cond_expanded = cond.unsqueeze(1).expand(-1, H, -1)  # [B, H, cond_proj_dim]
        
        # 3. 拼接噪声和条件
        combined = torch.cat([prev_noise, cond_expanded], dim=-1)  # [B, H, D + cond_proj_dim]
        
        # 4. 输入投影
        projected = self.input_proj(combined)  # [B, H, hidden_dim]
        
        # 5. GRU 前向
        if hidden is None:
            hidden = torch.zeros(self.num_layers, B, self.hidden_dim, device=prev_noise.device)
        
        gru_out, new_hidden = self.gru(projected, hidden)  # gru_out: [B, H, hidden_dim]
        
        # 6. 输出均值和对数标准差
        mean = self.mu_head(gru_out)  # [B, H, D]
        log_std = self.log_std_head(gru_out)  # [B, H, D]
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        return mean, log_std, new_hidden
    
    def sample(
        self,
        prefix_embs: torch.Tensor,
        prev_noise: torch.Tensor,
        deterministic: bool = False,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样噪声
        
        Returns:
            noise: 采样的噪声 [B, H, D]
            log_prob: 对数概率 [B]
        """
        mean, log_std, new_hidden = self.forward(prefix_embs, prev_noise, hidden)
        std = torch.exp(log_std)
        
        # 保存隐状态
        self._hidden = new_hidden.detach()
        
        if deterministic:
            noise = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device)
        else:
            # 重参数化采样
            dist = Normal(mean, std)
            noise = dist.rsample()
            log_prob = dist.log_prob(noise).sum(dim=[1, 2])  # [B]
        
        return noise, log_prob
    
    def evaluate_actions(
        self,
        prefix_embs: torch.Tensor,
        prev_noise: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率和熵
        
        Returns:
            log_prob: [B]
            entropy: [B]
        """
        mean, log_std, _ = self.forward(prefix_embs, prev_noise)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(noise).sum(dim=[1, 2])
        entropy = dist.entropy().sum(dim=[1, 2])
        
        return log_prob, entropy
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cuda", **kwargs):
        """
        从预训练的 StochasticMotionPrior 加载权重
        
        Args:
            checkpoint_path: 预训练模型路径
            device: 设备
            **kwargs: 额外的模型参数
            
        Returns:
            model: 加载了权重的模型
        """
        print(f"Loading pretrained weights from: {checkpoint_path}")
        
        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 获取模型配置（如果保存了的话）
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Found config: hidden_dim={config.get('hidden_dim', 128)}, "
                  f"num_layers={config.get('num_layers', 2)}")
        
        # 创建模型
        model = cls(**kwargs)
        
        # 加载权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试加载（允许部分匹配）
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)
        
        model.load_state_dict(model_state)
        
        print(f"  Loaded {len(loaded_keys)} keys, skipped {len(skipped_keys)} keys")
        if skipped_keys:
            print(f"  Skipped keys: {skipped_keys[:5]}...")
        
        return model.to(device)


# 为了向后兼容，保留 NoisePolicy 作为别名
NoisePolicy = StochasticMotionPriorActor


class ValueNetwork(nn.Module):
    """
    价值网络 (Critic)，用于 PPO 的 baseline
    """
    
    def __init__(
        self,
        noise_dim: int = 32,
        horizon: int = 50,
        cond_dim: int = 2048,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim * horizon, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        prefix_embs: torch.Tensor,
        prev_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            value: [B]
        """
        B = prev_noise.shape[0]
        
        if prefix_embs.ndim == 3:
            cond = prefix_embs.mean(dim=1)
        else:
            cond = prefix_embs
        cond = self.cond_proj(cond)
        
        noise_flat = prev_noise.view(B, -1)
        noise_feat = self.noise_encoder(noise_flat)
        
        combined = torch.cat([cond, noise_feat], dim=-1)
        value = self.value_head(combined).squeeze(-1)
        
        return value


# ==========================================
# 2. 噪声生成环境
# ==========================================

class NoiseGenerationEnv:
    """
    强化学习环境：噪声生成
    
    状态: (prefix_embs, prev_noise, prev_actions)
    动作: next_noise
    奖励: -||chunk_boundary_jump|| - β * KL_penalty
    
    关键设计：
    1. 通过 PI0 的 get_features() 获取真实的 prefix_embs
    2. 噪声约束使用 KL 散度，鼓励噪声保持在标准高斯分布
    3. 主奖励是 chunk 边界跳变的负值
    """
    
    def __init__(
        self,
        policy,           # PI0 policy
        dataset,          # LeRobot dataset
        input_transform,  # Policy 输入变换
        execution_horizon: int = 15,
        overlap_steps: int = 5,
        kl_coef: float = 0.01,
        device: str = "cuda",
    ):
        self.policy = policy
        self.model: PI0Pytorch = policy._model
        self.dataset = dataset
        self.input_transform = input_transform
        self.execution_horizon = execution_horizon
        self.overlap_steps = overlap_steps
        self.kl_coef = kl_coef
        self.device = device
        
        self.action_horizon = self.model.config.action_horizon
        self.action_dim = self.model.config.action_dim
        
        # 当前状态
        self.current_idx = 0
        self.prev_noise = None
        self.prev_actions = None
        self.prefix_embs = None
        self.episode_rewards = []
        
    def reset(self, start_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """重置环境"""
        if start_idx is None:
            start_idx = np.random.randint(0, len(self.dataset) - 500)
        
        self.current_idx = start_idx
        self.episode_rewards = []
        
        # 获取初始观测
        obs_dict = self._get_observation(self.current_idx)
        self.prefix_embs = obs_dict['prefix_embs']
        
        # 初始噪声从标准高斯采样
        self.prev_noise = torch.randn(
            1, self.action_horizon, self.action_dim,
            device=self.device
        )
        
        # 执行第一个 chunk 获取初始动作
        with torch.no_grad():
            result = self.policy.infer(obs_dict['raw_obs'], noise=self.prev_noise)
        self.prev_actions = torch.tensor(result['actions'], device=self.device)
        if self.prev_actions.ndim == 2:
            self.prev_actions = self.prev_actions.unsqueeze(0)
        
        return {
            'prefix_embs': self.prefix_embs,
            'prev_noise': self.prev_noise,
        }
    
    def step(self, noise: torch.Tensor) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步
        
        Args:
            noise: 策略网络生成的噪声 [1, H, D]
            
        Returns:
            next_state, reward, done, info
        """
        # 1. 更新索引
        self.current_idx += self.execution_horizon
        
        # 检查是否结束
        if self.current_idx >= len(self.dataset) - self.action_horizon:
            return {}, 0.0, True, {}
        
        # 2. 获取新的观测
        obs_dict = self._get_observation(self.current_idx)
        self.prefix_embs = obs_dict['prefix_embs']
        
        # 3. 使用给定噪声解码动作
        with torch.no_grad():
            result = self.policy.infer(obs_dict['raw_obs'], noise=noise)
        new_actions = torch.tensor(result['actions'], device=self.device)
        if new_actions.ndim == 2:
            new_actions = new_actions.unsqueeze(0)
        
        # 4. 计算奖励
        reward, info = self._compute_reward(noise, new_actions)
        self.episode_rewards.append(reward)
        
        # 5. 更新状态
        self.prev_noise = noise.clone()
        self.prev_actions = new_actions
        
        # 6. 构建下一个状态
        next_state = {
            'prefix_embs': self.prefix_embs,
            'prev_noise': self.prev_noise,
        }
        
        done = False
        
        # 保存最新的 info 用于日志
        self.last_info = info
        
        return next_state, reward, done, info
    
    def _get_observation(self, idx: int) -> Dict:
        """获取观测，包括从 PI0 提取的 prefix_embs"""
        raw_item = self.dataset[idx]
        
        # 构建 policy.infer 需要的观测格式
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
        
        # 通过 PI0 获取 prefix_embs
        with torch.no_grad():
            # 应用输入变换
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self.input_transform(inputs)
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self.device)[None, ...],
                inputs
            )
            
            # 构建 Observation 并提取特征
            from openpi.models import model as _model
            observation = _model.Observation.from_dict(inputs)
            prefix_embs, _, _ = self.model.get_features(observation)
            
            # prefix_embs: [1, seq_len, hidden_dim] -> 池化为 [1, hidden_dim]
            # 转换为 float32，因为 PI0 使用 bfloat16
            prefix_embs_pooled = prefix_embs.mean(dim=1).float()  # [1, 2048]
        
        return {
            'raw_obs': obs,
            'prefix_embs': prefix_embs_pooled,
            'prefix_embs_full': prefix_embs.float(),  # 保留完整特征以备使用
        }
    
    def _compute_reward(
        self,
        noise: torch.Tensor,
        new_actions: torch.Tensor,
    ) -> Tuple[float, Dict]:
        """
        计算奖励
        
        主奖励: 负的 chunk 边界跳变
        正则项: 
          1. KL 散度惩罚（保持噪声接近标准高斯）
          2. 方差惩罚（强制方差接近 1.0）
          3. 范围惩罚（惩罚超出 [-3,3] 的噪声）
        """
        # 1. 计算 chunk 边界跳变
        overlap = self.overlap_steps
        exec_h = self.execution_horizon
        
        prev_end = self.prev_actions[0, exec_h - overlap:exec_h, :14]  # 只看前 14 维动作
        new_start = new_actions[0, :overlap, :14]
        
        # 计算均方跳变
        jump = (prev_end - new_start).pow(2).mean().sqrt().item()
        
        # 2. 计算 KL 散度惩罚
        noise_mean = noise.mean()
        noise_var = noise.var()
        noise_std = noise.std().item()
        
        # KL(N(μ,σ²) || N(0,1)) = 0.5 * (μ² + σ² - 1 - log(σ²))
        kl_penalty = 0.5 * (noise_mean.pow(2) + noise_var - 1 - torch.log(noise_var + 1e-8)).item()
        kl_penalty = max(0, kl_penalty)  # KL 不能为负
        
        # 3. 方差惩罚（强制方差接近 1.0）
        # 使用平方惩罚，偏离 1.0 越多惩罚越大
        var_penalty = (noise_var - 1.0).pow(2).item()
        
        # 4. 均值惩罚（强制均值接近 0）
        mean_penalty = noise_mean.pow(2).item()
        
        # 5. 范围惩罚
        out_of_range = (noise.abs() > 3.0).float().mean().item()
        range_penalty = out_of_range * 5.0
        
        # 6. 总奖励
        # 主要目标：减小 jump
        # 约束：保持噪声在标准高斯分布
        reward = (
            -jump  # 主奖励：减小跳变
            - self.kl_coef * kl_penalty  # KL 惩罚
            - self.kl_coef * 2.0 * var_penalty  # 方差惩罚（权重更高）
            - self.kl_coef * mean_penalty  # 均值惩罚
            - range_penalty  # 范围惩罚
        )
        
        info = {
            'jump': jump,
            'kl_penalty': kl_penalty,
            'var_penalty': var_penalty,
            'mean_penalty': mean_penalty,
            'range_penalty': range_penalty,
            'noise_mean': noise_mean.item(),
            'noise_std': noise_std,
            'noise_var': noise_var.item(),
            'out_of_range_ratio': out_of_range,
        }
        
        return reward, info


# ==========================================
# 3. PPO 训练器
# ==========================================

class PPOTrainer:
    """PPO 训练器"""
    
    def __init__(
        self,
        actor: NoisePolicy,
        critic: ValueNetwork,
        env: NoiseGenerationEnv,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.actor = actor
        self.critic = critic
        self.env = env
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)
        
    def collect_rollout(self, num_steps: int) -> Dict[str, List]:
        """收集 rollout 数据"""
        rollout = {
            'prefix_embs': [],
            'prev_noise': [],
            'noise': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            # 详细指标
            'jumps': [],
            'kl_penalties': [],
            'noise_means': [],
            'noise_stds': [],
            'var_penalties': [],
        }
        
        state = self.env.reset()
        
        for _ in range(num_steps):
            prefix_embs = state['prefix_embs']
            prev_noise = state['prev_noise']
            
            # 采样动作
            with torch.no_grad():
                noise, log_prob = self.actor.sample(prefix_embs, prev_noise)
                value = self.critic(prefix_embs, prev_noise)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(noise)
            
            # 存储数据
            rollout['prefix_embs'].append(prefix_embs)
            rollout['prev_noise'].append(prev_noise)
            rollout['noise'].append(noise)
            rollout['log_probs'].append(log_prob)
            rollout['rewards'].append(reward)
            rollout['values'].append(value)
            rollout['dones'].append(done)
            
            # 存储详细指标
            if info:
                rollout['jumps'].append(info.get('jump', 0))
                rollout['kl_penalties'].append(info.get('kl_penalty', 0))
                rollout['noise_means'].append(info.get('noise_mean', 0))
                rollout['noise_stds'].append(info.get('noise_std', 1))
                rollout['var_penalties'].append(info.get('var_penalty', 0))
            
            if done:
                state = self.env.reset()
            else:
                state = next_state
        
        return rollout
    
    def compute_gae(self, rollout: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE 优势估计"""
        rewards = torch.tensor(rollout['rewards'], device=self.device, dtype=torch.float32)
        values = torch.stack(rollout['values']).squeeze(-1)  # [num_steps]
        dones = rollout['dones']
        
        num_steps = len(rewards)
        advantages = torch.zeros(num_steps, device=self.device)
        
        last_gae = 0
        last_value = values[-1].item() if not dones[-1] else 0.0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1].item()
            
            next_non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t].item() + self.gamma * next_value * next_non_terminal - values[t].item()
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, rollout: Dict) -> Dict[str, float]:
        """PPO 更新"""
        # 准备数据
        prefix_embs = torch.cat(rollout['prefix_embs'], dim=0)
        prev_noise = torch.cat(rollout['prev_noise'], dim=0)
        noise = torch.cat(rollout['noise'], dim=0)
        old_log_probs = torch.cat(rollout['log_probs'], dim=0)
        
        advantages, returns = self.compute_gae(rollout)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        num_samples = len(rollout['rewards'])
        
        for _ in range(self.ppo_epochs):
            # 随机打乱
            indices = torch.randperm(num_samples)
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = indices[start:end]
                
                batch_prefix_embs = prefix_embs[batch_indices]
                batch_prev_noise = prev_noise[batch_indices]
                batch_noise = noise[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor 更新
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_prefix_embs, batch_prev_noise, batch_noise
                )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Critic 更新
                values = self.critic(batch_prefix_embs, batch_prev_noise)
                critic_loss = F.mse_loss(values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        num_updates = self.ppo_epochs * (num_samples // self.batch_size + 1)
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }


# ==========================================
# 4. 训练参数和主函数
# ==========================================

@dataclass
class TrainArgs:
    # 模型路径
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    
    # 预训练模型（StochasticMotionPrior）
    pretrained_actor: Optional[str] = "/home/ztlab/lgd/openpi/checkpoints/stochastic_prior/best_model.pth"
    
    # HDF5 路径（用于反推噪声对比）
    h5_path: str = "latent_dataset_10.h5"
    
    # 环境参数
    execution_horizon: int = 15
    overlap_steps: int = 5
    kl_coef: float = 0.1  # KL 系数
    
    # 网络参数（与 StochasticMotionPrior 保持一致）
    hidden_dim: int = 128  # 与预训练模型一致
    num_layers: int = 2    # 与预训练模型一致
    cond_proj_dim: int = 64  # 与预训练模型一致
    
    # PPO 参数
    lr_actor: float = 5e-5  # 使用较小的学习率进行微调
    lr_critic: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.001  # 低 entropy 系数，减少随机探索
    ppo_epochs: int = 4
    
    # 训练参数
    num_iterations: int = 1000
    rollout_steps: int = 64
    batch_size: int = 32
    save_interval: int = 50
    log_interval: int = 10
    
    # 其他
    device: str = "cuda"
    save_dir: str = "checkpoints/rl_noise_policy"
    use_wandb: bool = True
    wandb_project: str = "rl-noise-policy"


def main(args: TrainArgs):
    """主训练循环"""
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("RL Noise Policy Training")
    print("=" * 60)
    
    # 初始化 wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    # 1. 加载 PI0 模型
    print("\n1. Loading PI0 policy...")
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
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
    
    # 3. 创建网络
    print("\n3. Creating networks...")
    
    # Actor: 使用 GRU 架构（与 StochasticMotionPrior 兼容）
    if args.pretrained_actor and os.path.exists(args.pretrained_actor):
        print(f"   Loading pretrained actor from: {args.pretrained_actor}")
        actor = StochasticMotionPriorActor.from_pretrained(
            args.pretrained_actor,
            device=args.device,
            latent_dim=model.config.action_dim,
            horizon=model.config.action_horizon,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cond_dim=2048,
            cond_proj_dim=args.cond_proj_dim,
        )
    else:
        print("   Creating new actor (no pretrained weights)")
        actor = StochasticMotionPriorActor(
            latent_dim=model.config.action_dim,
            horizon=model.config.action_horizon,
            cond_dim=2048,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            cond_proj_dim=args.cond_proj_dim,
        ).to(args.device)
    
    # Critic
    critic = ValueNetwork(
        noise_dim=model.config.action_dim,
        horizon=model.config.action_horizon,
        cond_dim=2048,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"   Actor params: {actor_params/1e6:.2f}M")
    print(f"   Critic params: {critic_params/1e6:.2f}M")
    
    # 4. 创建环境
    print("\n4. Creating environment...")
    # 获取 policy 的输入变换
    input_transform = policy._input_transform
    
    env = NoiseGenerationEnv(
        policy=policy,
        dataset=raw_dataset,
        input_transform=input_transform,
        execution_horizon=args.execution_horizon,
        overlap_steps=args.overlap_steps,
        kl_coef=args.kl_coef,
        device=args.device,
    )
    
    # 5. 创建训练器
    trainer = PPOTrainer(
        actor=actor,
        critic=critic,
        env=env,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # 6. 训练循环
    print("\n5. Starting training...")
    reward_history = deque(maxlen=100)
    jump_history = deque(maxlen=100)
    noise_std_history = deque(maxlen=100)
    
    best_reward = float('-inf')
    
    for iteration in tqdm(range(args.num_iterations), desc="Training"):
        # 收集 rollout
        rollout = trainer.collect_rollout(args.rollout_steps)
        
        # 更新网络
        losses = trainer.update(rollout)
        
        # 记录指标
        avg_reward = np.mean(rollout['rewards'])
        reward_history.append(avg_reward)
        
        # 从 env 的 info 中获取详细指标
        # 注：这里简化处理，实际可以从 rollout 中收集 info
        
        if iteration % args.log_interval == 0:
            # 计算详细指标
            avg_jump = np.mean(rollout['jumps']) if rollout['jumps'] else 0
            avg_kl = np.mean(rollout['kl_penalties']) if rollout['kl_penalties'] else 0
            avg_var_penalty = np.mean(rollout['var_penalties']) if rollout['var_penalties'] else 0
            avg_noise_mean = np.mean(rollout['noise_means']) if rollout['noise_means'] else 0
            avg_noise_std = np.mean(rollout['noise_stds']) if rollout['noise_stds'] else 1
            
            metrics = {
                'iteration': iteration,
                # 奖励指标
                'reward/avg': avg_reward,
                'reward/avg_100': np.mean(reward_history) if len(reward_history) > 0 else avg_reward,
                # 核心指标：chunk 跳变
                'continuity/jump': avg_jump,
                'penalty/kl': avg_kl,
                'penalty/var': avg_var_penalty,
                # 噪声分布指标（关键监控）
                'noise/mean': avg_noise_mean,
                'noise/std': avg_noise_std,  # 目标：接近 1.0
                'noise/var': avg_noise_std ** 2,  # 目标：接近 1.0
                # 训练损失
                'loss/actor': losses['actor_loss'],
                'loss/critic': losses['critic_loss'],
                'loss/entropy': losses['entropy'],
            }
            
            print(f"\nIter {iteration}: reward={avg_reward:.4f}, jump={avg_jump:.4f}, "
                  f"noise_std={avg_noise_std:.3f} (target=1.0), entropy={losses['entropy']:.2f}")
            
            if args.use_wandb:
                wandb.log(metrics)
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'iteration': iteration,
                'reward': avg_reward,
            }, best_path)
        
        # 定期保存模型
        if (iteration + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_{iteration+1}.pth')
            torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'iteration': iteration,
            }, save_path)
            print(f"   Saved checkpoint to {save_path}")
    
    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'config': vars(args),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best reward achieved: {best_reward:.4f}")
    
    if args.use_wandb:
        wandb.finish()


# ==========================================
# 5. 评估函数
# ==========================================

def evaluate_policy(
    actor: NoisePolicy,
    env: NoiseGenerationEnv,
    num_episodes: int = 10,
    num_steps_per_episode: int = 50,
    deterministic: bool = True,
    compare_with_random: bool = True,
) -> Dict[str, List[float]]:
    """
    评估训练好的策略
    
    Args:
        actor: 训练好的 Actor 网络
        env: 环境
        num_episodes: 评估的 episode 数量
        num_steps_per_episode: 每个 episode 的步数
        deterministic: 是否使用确定性策略
        compare_with_random: 是否与随机噪声对比
        
    Returns:
        评估结果字典
    """
    actor.eval()
    
    results = {
        'policy_rewards': [],
        'policy_jumps': [],
        'policy_noise_stds': [],
        'random_rewards': [],
        'random_jumps': [],
    }
    
    for ep in range(num_episodes):
        # 评估学习到的策略
        state = env.reset()
        ep_rewards = []
        ep_jumps = []
        ep_noise_stds = []
        
        for _ in range(num_steps_per_episode):
            prefix_embs = state['prefix_embs']
            prev_noise = state['prev_noise']
            
            with torch.no_grad():
                noise, _ = actor.sample(prefix_embs, prev_noise, deterministic=deterministic)
            
            next_state, reward, done, info = env.step(noise)
            
            ep_rewards.append(reward)
            if 'jump' in info:
                ep_jumps.append(info['jump'])
                ep_noise_stds.append(info['noise_std'])
            
            if done:
                break
            state = next_state
        
        results['policy_rewards'].append(np.mean(ep_rewards))
        results['policy_jumps'].append(np.mean(ep_jumps) if ep_jumps else 0)
        results['policy_noise_stds'].append(np.mean(ep_noise_stds) if ep_noise_stds else 0)
        
        # 评估随机噪声基线
        if compare_with_random:
            state = env.reset()
            ep_rewards = []
            ep_jumps = []
            
            for _ in range(num_steps_per_episode):
                # 生成随机高斯噪声
                noise = torch.randn_like(state['prev_noise'])
                next_state, reward, done, info = env.step(noise)
                
                ep_rewards.append(reward)
                if 'jump' in info:
                    ep_jumps.append(info['jump'])
                
                if done:
                    break
                state = next_state
            
            results['random_rewards'].append(np.mean(ep_rewards))
            results['random_jumps'].append(np.mean(ep_jumps) if ep_jumps else 0)
    
    return results


def print_evaluation_results(results: Dict[str, List[float]]):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\n[Learned Policy]")
    print(f"  Avg Reward:    {np.mean(results['policy_rewards']):.4f} ± {np.std(results['policy_rewards']):.4f}")
    print(f"  Avg Jump:      {np.mean(results['policy_jumps']):.4f} ± {np.std(results['policy_jumps']):.4f}")
    print(f"  Avg Noise Std: {np.mean(results['policy_noise_stds']):.4f}")
    
    if results['random_rewards']:
        print(f"\n[Random Noise Baseline]")
        print(f"  Avg Reward:    {np.mean(results['random_rewards']):.4f} ± {np.std(results['random_rewards']):.4f}")
        print(f"  Avg Jump:      {np.mean(results['random_jumps']):.4f} ± {np.std(results['random_jumps']):.4f}")
        
        improvement = (np.mean(results['policy_rewards']) - np.mean(results['random_rewards'])) / abs(np.mean(results['random_rewards']) + 1e-8) * 100
        jump_reduction = (np.mean(results['random_jumps']) - np.mean(results['policy_jumps'])) / (np.mean(results['random_jumps']) + 1e-8) * 100
        print(f"\n[Improvement over Random]")
        print(f"  Reward Improvement: {improvement:.1f}%")
        print(f"  Jump Reduction:     {jump_reduction:.1f}%")


if __name__ == "__main__":
    tyro.cli(main)
