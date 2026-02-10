"""
验证脚本：确保反推 ODE 与正向 ODE 完全一致。

测试流程：
1. 使用标准噪声 z 调用 policy.infer() 得到动作 a
2. 使用反推 ODE 从动作 a 得到隐变量 z'
3. 验证 z' ≈ z（应该非常接近）
4. 使用 z' 再次调用 policy.infer() 得到动作 a'
5. 验证 a' ≈ a（应该非常接近）
"""

import torch
import numpy as np
import h5py
from dataclasses import dataclass
from typing import Optional
import logging

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models import model as _model
from lerobot.common.datasets import lerobot_dataset

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"
    device: str = "cuda"
    num_ode_steps: int = 10
    test_idx: int = 100  # 测试样本索引


def invert_actions_with_model(
    model: PI0Pytorch,
    actions: torch.Tensor,  # [B, H, D] 归一化后的动作
    observation: _model.Observation,
    num_steps: int = 10,
    device: str = "cuda"
) -> torch.Tensor:
    """
    使用与 sample_actions 完全相同的组件进行反推。
    
    关键：确保 embed_prefix, _prepare_kv_cache, denoise_step 的调用方式一致。
    """
    from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
    
    # 1. 预处理观测（与 sample_actions 一致）
    observation = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    images = list(observation.images.values())
    img_masks = list(observation.image_masks.values())
    lang_tokens = observation.tokenized_prompt
    lang_masks = observation.tokenized_prompt_mask
    state = observation.state
    
    # 2. 嵌入前缀（与 sample_actions 一致）
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    
    # 3. 准备 KV Cache（与 sample_actions 一致）
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
    
    # 4. 反向 ODE：从 t=0 积分到 t=1
    bsize = actions.shape[0]
    dt = 1.0 / num_steps
    x_t = actions.clone()
    time = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        # 使用与 sample_actions 完全相同的 denoise_step
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        # Euler 步进
        x_t = x_t + dt * v_t
        time = time + dt
    
    return x_t


def forward_ode_with_model(
    model: PI0Pytorch,
    noise: torch.Tensor,  # [B, H, D] 噪声/隐变量
    observation: _model.Observation,
    num_steps: int = 10,
    device: str = "cuda"
) -> torch.Tensor:
    """
    正向 ODE：与 sample_actions 完全一致的实现。
    """
    from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
    
    # 1. 预处理观测
    observation = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    images = list(observation.images.values())
    img_masks = list(observation.image_masks.values())
    lang_tokens = observation.tokenized_prompt
    lang_masks = observation.tokenized_prompt_mask
    state = observation.state
    
    # 2. 嵌入前缀
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    
    # 3. 准备 KV Cache
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
    
    # 4. 正向 ODE：从 t=1 积分到 t=0
    bsize = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        # Euler 步进
        x_t = x_t + dt * v_t
        time = time + dt
    
    return x_t


def main(args: Args):
    print("=" * 60)
    print("Inverse ODE Verification Script")
    print("=" * 60)
    
    # 1. 加载 policy
    print(f"\n1. Loading policy config {args.policy_config}...")
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
    policy = _policy_config.create_trained_policy(
        server_config,
        args.policy_dir,
        pytorch_device=args.device
    )
    
    model = policy._model
    model.eval()
    
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
    
    # 3. 获取一个测试样本
    print(f"\n3. Getting test sample at index {args.test_idx}...")
    item = raw_dataset[args.test_idx]
    
    # 转换为 policy 期望的格式
    obs = {}
    obs["state"] = item.get("observation.state")
    if isinstance(obs["state"], torch.Tensor):
        obs["state"] = obs["state"].numpy()
    
    images_dict = {}
    for key in item.keys():
        if key.startswith("observation.images."):
            cam_name = key.replace("observation.images.", "")
            img_val = item[key]
            if isinstance(img_val, torch.Tensor):
                img_val = img_val.numpy()
            images_dict[cam_name] = img_val
    obs["images"] = images_dict
    
    # Prompt 处理
    from openpi import transforms as _transforms
    prompt_transform = _transforms.PromptFromLeRobotTask(dataset_meta.tasks)
    obs_with_prompt = prompt_transform({"task_index": item.get("task_index", 0)})
    obs["prompt"] = obs_with_prompt.get("prompt", "")
    
    print(f"   State shape: {obs['state'].shape}")
    print(f"   Image keys: {list(obs['images'].keys())}")
    print(f"   Prompt: '{obs['prompt'][:50]}...'")
    
    # 4. 使用随机噪声进行正向推理
    print("\n4. Forward inference with random noise...")
    noise_shape = (1, model_config.action_horizon, model_config.action_dim)
    random_noise = torch.randn(noise_shape, device=args.device)
    
    with torch.no_grad():
        result = policy.infer(obs, noise=random_noise)
    
    actions_from_random = result['actions']
    if isinstance(actions_from_random, torch.Tensor):
        actions_from_random = actions_from_random.cpu().numpy()
    print(f"   Actions shape: {actions_from_random.shape}")
    print(f"   Actions range: [{actions_from_random.min():.4f}, {actions_from_random.max():.4f}]")
    
    # 5. 从输出动作反推隐变量
    print("\n5. Inverse ODE: actions -> latents...")
    
    # 准备模型输入 - 不能直接用 policy._input_transform，因为它包含嵌套结构
    # 我们需要直接使用 model.infer_step 的方式
    
    # 首先，使用 policy 的内部方法获取处理后的输入
    inputs_for_model = obs.copy()
    inputs_for_model = policy._input_transform(inputs_for_model)
    
    # 手动转换为 tensor（处理嵌套结构）
    def to_tensor(v, device, key=None):
        if isinstance(v, dict):
            return {k: to_tensor(vv, device, key=k) for k, vv in v.items()}
        elif isinstance(v, bool) or isinstance(v, np.bool_):
            # image_masks 需要是 tensor，不能是 Python bool
            return torch.tensor([v], dtype=torch.bool, device=device)
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v).to(device)
            # 所有非 4D 的 tensor 都需要添加 batch 维度
            if t.dim() == 0:
                t = t.unsqueeze(0)
            elif t.dim() == 1:
                t = t.unsqueeze(0)  # [L] -> [1, L]
            elif t.dim() == 2:
                t = t.unsqueeze(0)  # [H, W] or [S, D] -> [1, H, W] or [1, S, D]
            elif t.dim() == 3:
                t = t.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            return t
        elif isinstance(v, torch.Tensor):
            t = v.to(device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            elif t.dim() == 1:
                t = t.unsqueeze(0)
            elif t.dim() == 2:
                t = t.unsqueeze(0)
            elif t.dim() == 3:
                t = t.unsqueeze(0)
            return t
        elif isinstance(v, (list, tuple)):
            return [to_tensor(vv, device) for vv in v]
        elif isinstance(v, str):
            return v  # 保持字符串不变
        else:
            return v
    
    inputs_tensor = to_tensor(inputs_for_model, args.device)
    observation = _model.Observation.from_dict(inputs_tensor)
    
    # 反归一化动作（因为模型输出的是未归一化的动作）
    # 但实际上 policy.infer 输出的已经是反归一化的
    # 我们需要先把动作归一化回去，才能做反推
    
    # 获取归一化统计量
    norm_stats = data_config.norm_stats
    if norm_stats and 'actions' in norm_stats:
        action_mean = norm_stats['actions'].mean
        action_std = norm_stats['actions'].std
        print(f"   Action norm mean: {action_mean[:3]}...")
        print(f"   Action norm std: {action_std[:3]}...")
        
        # 归一化动作
        actions_normalized = (actions_from_random - action_mean) / (action_std + 1e-6)
        actions_normalized_tensor = torch.from_numpy(actions_normalized).float().to(args.device)
        if actions_normalized_tensor.ndim == 2:
            actions_normalized_tensor = actions_normalized_tensor.unsqueeze(0)
        
        print(f"   Normalized actions range: [{actions_normalized_tensor.min():.4f}, {actions_normalized_tensor.max():.4f}]")
    else:
        print("   WARNING: No action normalization stats found!")
        actions_normalized_tensor = torch.from_numpy(actions_from_random).float().to(args.device)
        if actions_normalized_tensor.ndim == 2:
            actions_normalized_tensor = actions_normalized_tensor.unsqueeze(0)
    
    # 执行反向 ODE
    with torch.no_grad():
        inverted_latents = invert_actions_with_model(
            model,
            actions_normalized_tensor,
            observation,
            num_steps=args.num_ode_steps,
            device=args.device
        )
    
    print(f"   Inverted latents shape: {inverted_latents.shape}")
    print(f"   Inverted latents range: [{inverted_latents.min():.4f}, {inverted_latents.max():.4f}]")
    
    # 6. 比较原始噪声和反推噪声
    print("\n6. Comparing original noise and inverted latents...")
    
    diff = (random_noise - inverted_latents).abs()
    print(f"   Original noise range:  [{random_noise.min():.4f}, {random_noise.max():.4f}]")
    print(f"   Inverted latent range: [{inverted_latents.min():.4f}, {inverted_latents.max():.4f}]")
    print(f"   Absolute difference:   Mean={diff.mean():.6f}, Max={diff.max():.6f}")
    
    # 7. 使用反推的隐变量再次进行正向推理
    print("\n7. Forward inference with inverted latents...")
    
    with torch.no_grad():
        result_from_inverted = policy.infer(obs, noise=inverted_latents)
    
    actions_from_inverted = result_from_inverted['actions']
    if isinstance(actions_from_inverted, torch.Tensor):
        actions_from_inverted = actions_from_inverted.cpu().numpy()
    
    print(f"   Actions shape: {actions_from_inverted.shape}")
    print(f"   Actions range: [{actions_from_inverted.min():.4f}, {actions_from_inverted.max():.4f}]")
    
    # 8. 比较两次推理得到的动作
    print("\n8. Comparing actions...")
    action_diff = np.abs(actions_from_random - actions_from_inverted)
    print(f"   Original actions range:        [{actions_from_random.min():.4f}, {actions_from_random.max():.4f}]")
    print(f"   Actions from inverted latents: [{actions_from_inverted.min():.4f}, {actions_from_inverted.max():.4f}]")
    print(f"   Absolute difference:           Mean={action_diff.mean():.6f}, Max={action_diff.max():.6f}")
    
    # 9. 判断结论
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if diff.mean() < 0.1 and action_diff.mean() < 0.01:
        print("✓ Inverse ODE is CORRECT! Noise reconstruction error is small.")
    elif action_diff.mean() < 0.1:
        print("△ Inverse ODE has some error, but action reconstruction is acceptable.")
    else:
        print("✗ Inverse ODE is INCORRECT! There is a significant mismatch.")
        print("  Possible issues:")
        print("  - Normalization/unnormalization mismatch")
        print("  - Different preprocessing (images, state)")
        print("  - ODE solver configuration mismatch")
    print("=" * 60)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
