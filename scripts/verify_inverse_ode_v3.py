"""
验证脚本 v3：保存正向 ODE 的速度场，实现完美可逆

关键洞察（用户建议）：
- 既然正向 ODE 使用速度场 v_t 从 noise 到 actions
- 如果保存这些速度场，反向时直接使用 -v_t（取反）
- 这样可以确保完美的可逆性

新测试流程：
1. 正向 ODE：noise -> actions，同时保存每一步的 v_t
2. 反向 ODE 方案A：使用保存的 v_t（取反），验证能否回到原始 noise
3. 反向 ODE 方案B：重新计算 v_t（原方法），对比差异
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging
import tyro

from openpi.training import config as _config
from openpi.training import data_loader
from openpi.policies import policy_config as _policy_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models import model as _model
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
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
    test_idx: int = 100


@torch.no_grad()
def forward_ode_with_velocity_recording(
    model: PI0Pytorch,
    noise: torch.Tensor,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    num_steps: int
) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
    """
    正向 ODE 同时记录每一步的速度场。
    
    Returns:
        actions: 最终的 actions
        velocities: 每一步的速度场 v_t
        times: 每一步的时间 t
    """
    device = noise.device
    bsize = noise.shape[0]
    
    # 与 sample_actions 一致的设置
    dt = -1.0 / num_steps  # 负的步长，从 t=1 到 t=0
    x_t = noise.clone()
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    velocities = []
    times = []
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        # 计算速度场
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        # 保存速度和时间
        velocities.append(v_t.clone())
        times.append(time.item())
        
        # Euler 更新：x_t = x_t + dt * v_t
        x_t = x_t + dt * v_t
        time = time + dt
    
    return x_t, velocities, times


@torch.no_grad()
def inverse_ode_with_saved_velocities(
    actions: torch.Tensor,
    velocities: List[torch.Tensor],
    num_steps: int
) -> torch.Tensor:
    """
    使用保存的速度场进行反向 ODE（完美可逆）。
    
    关键：正向 ODE 使用 dt=-1/N，所以更新是 x_t = x_t - v_t/N
    反向时应该使用 x_t = x_t + v_t/N（按相反顺序）
    """
    dt = 1.0 / num_steps  # 正的步长，从 t=0 到 t=1
    x_t = actions.clone()
    
    # 反向遍历速度场（从最后一步到第一步）
    for v_t in reversed(velocities):
        # 反向更新：正向是 x -= v*|dt|，反向就是 x += v*|dt|
        x_t = x_t + dt * v_t
    
    return x_t


@torch.no_grad()
def inverse_ode_recompute_velocities(
    actions: torch.Tensor,
    model: PI0Pytorch,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    num_steps: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    重新计算速度场进行反向 ODE（原方法）。
    
    Returns:
        noise: 反推得到的 noise
        velocities: 反向过程中计算的速度场
    """
    device = actions.device
    bsize = actions.shape[0]
    
    dt = 1.0 / num_steps  # 正的步长
    x_t = actions.clone()
    time = torch.tensor(0.0, dtype=torch.float32, device=device)
    
    velocities = []
    
    for step in range(num_steps):
        expanded_time = time.expand(bsize)
        
        # 在当前位置计算速度场
        v_t = model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=expanded_time,
        )
        
        velocities.append(v_t.clone())
        
        # Euler 更新
        x_t = x_t + dt * v_t
        time = time + dt
    
    return x_t, velocities


@torch.no_grad()
def main(args: Args):
    print("=" * 60)
    print("Inverse ODE Verification Script v3")
    print("关键思想：保存正向速度场，确保完美可逆")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. Loading policy config...")
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
    policy = _policy_config.create_trained_policy(
        server_config,
        args.policy_dir,
        pytorch_device=args.device
    )
    
    model: PI0Pytorch = policy._model
    model.eval()
    
    print(f"   action_dim={model.config.action_dim}, action_horizon={model.config.action_horizon}")
    
    # 2. 加载数据集并获取测试样本
    print("\n2. Loading dataset and getting test sample...")
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
    
    from openpi import transforms as _transforms
    if data_config.prompt_from_task:
        raw_dataset = data_loader.TransformedDataset(
            raw_dataset, 
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    
    transformed_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    item = transformed_dataset[args.test_idx]
    
    # 3. 构建 Observation
    print("\n3. Building observation...")
    
    def to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.to(args.device)
        elif isinstance(v, np.ndarray):
            return torch.from_numpy(v).to(args.device)
        else:
            return torch.tensor(v, device=args.device)
    
    images_raw = item['image']
    camera_names = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
    
    images_dict = {}
    image_masks_dict = {}
    
    if isinstance(images_raw, dict):
        for cam_name in camera_names:
            if cam_name in images_raw:
                img = to_tensor(images_raw[cam_name])
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                images_dict[cam_name] = img[None, ...]
                image_masks_dict[cam_name] = torch.ones(1, dtype=torch.bool, device=args.device)
    else:
        for i, cam_name in enumerate(camera_names):
            if i < images_raw.shape[0]:
                img = to_tensor(images_raw[i])
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                images_dict[cam_name] = img[None, ...]
                image_masks_dict[cam_name] = torch.ones(1, dtype=torch.bool, device=args.device)
    
    state_raw = to_tensor(item['state'])
    if state_raw.shape[-1] < 32:
        state_padded = torch.zeros(32, device=args.device)
        state_padded[:state_raw.shape[-1]] = state_raw
        state_tensor = state_padded[None, ...]
    else:
        state_tensor = state_raw[None, ...]
    
    tokenized_prompt = to_tensor(item['tokenized_prompt'])[None, ...]
    tokenized_prompt_mask = to_tensor(item['tokenized_prompt_mask'])[None, ...]
    
    observation = _model.Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        state=state_tensor,
        tokenized_prompt=tokenized_prompt.long(),
        tokenized_prompt_mask=tokenized_prompt_mask.bool(),
    )
    
    # 预处理观测
    obs_preprocessed = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    
    images = list(obs_preprocessed.images.values())
    img_masks = list(obs_preprocessed.image_masks.values())
    lang_tokens = obs_preprocessed.tokenized_prompt
    lang_masks = obs_preprocessed.tokenized_prompt_mask
    state = obs_preprocessed.state
    
    # 嵌入前缀
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    
    # 准备 KV Cache
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
    
    print(f"   State shape: {state.shape}")
    print(f"   prefix_embs shape: {prefix_embs.shape}")
    
    # 4. 生成随机 noise
    print("\n4. Generating random noise...")
    noise_shape = (1, model.config.action_horizon, model.config.action_dim)
    original_noise = torch.randn(noise_shape, device=args.device)
    print(f"   Noise shape: {original_noise.shape}")
    print(f"   Noise range: [{original_noise.min():.4f}, {original_noise.max():.4f}]")
    
    # 5. 正向 ODE + 记录速度场
    print(f"\n5. Forward ODE (noise -> actions) with velocity recording...")
    print(f"   Using {args.num_ode_steps} steps")
    
    actions, forward_velocities, forward_times = forward_ode_with_velocity_recording(
        model, original_noise, state, prefix_pad_masks, past_key_values, args.num_ode_steps
    )
    
    print(f"   Actions shape: {actions.shape}")
    print(f"   Actions range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"   Recorded {len(forward_velocities)} velocity fields")
    
    # 计算速度场统计
    v_norms = [v.norm().item() for v in forward_velocities]
    print(f"   Velocity norms: min={min(v_norms):.4f}, max={max(v_norms):.4f}, mean={np.mean(v_norms):.4f}")
    
    # 6. 方案A：使用保存的速度场反向（完美可逆）
    print("\n6. Inverse ODE Method A: Using saved velocities (perfect reversibility)...")
    noise_from_saved_v = inverse_ode_with_saved_velocities(
        actions, forward_velocities, args.num_ode_steps
    )
    
    error_saved = (noise_from_saved_v - original_noise).abs()
    rmse_saved = torch.sqrt((error_saved ** 2).mean()).item()
    print(f"   Reconstructed noise range: [{noise_from_saved_v.min():.4f}, {noise_from_saved_v.max():.4f}]")
    print(f"   RMSE (should be ~0): {rmse_saved:.8f}")
    print(f"   Max error: {error_saved.max():.8f}")
    
    # 7. 方案B：重新计算速度场反向（原方法）
    print("\n7. Inverse ODE Method B: Recomputing velocities...")
    noise_from_recompute, inverse_velocities = inverse_ode_recompute_velocities(
        actions, model, state, prefix_pad_masks, past_key_values, args.num_ode_steps
    )
    
    error_recompute = (noise_from_recompute - original_noise).abs()
    rmse_recompute = torch.sqrt((error_recompute ** 2).mean()).item()
    print(f"   Reconstructed noise range: [{noise_from_recompute.min():.4f}, {noise_from_recompute.max():.4f}]")
    print(f"   RMSE: {rmse_recompute:.6f}")
    print(f"   Max error: {error_recompute.max():.6f}")
    
    # 8. 分析速度场差异
    print("\n8. Analyzing velocity field differences...")
    print("   注意：正向和反向在不同的位置 x 计算速度，所以速度自然不同！")
    print("\n   当前比较（位置不同，速度自然不同）:")
    
    # 正向速度是按 t=1, t=0.9, ... 顺序
    # 反向速度是按 t=0, t=0.1, ... 顺序
    
    for i in range(min(3, len(forward_velocities))):
        fwd_v = forward_velocities[i]
        inv_v = inverse_velocities[-(i+1)] if i < len(inverse_velocities) else None
        
        if inv_v is not None:
            v_diff = (fwd_v - inv_v).abs()
            print(f"   Step {i}: forward time={forward_times[i]:.2f}")
            print(f"     Forward v norm: {fwd_v.norm():.4f} (at original_noise trajectory)")
            print(f"     Inverse v norm: {inv_v.norm():.4f} (at recomputed trajectory)")
            print(f"     Difference norm: {v_diff.norm():.4f}")
    
    # 8.5 关键验证：在相同位置计算速度，应该相同
    print("\n   关键验证：在完全相同的 (x, t) 处计算速度...")
    test_time = torch.tensor(0.5, device=args.device)
    test_x = original_noise.clone()  # 使用相同的 x
    
    # 计算两次速度
    v_check1 = model.denoise_step(
        state=state,
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        x_t=test_x,
        timestep=test_time.expand(1),
    )
    
    v_check2 = model.denoise_step(
        state=state,
        prefix_pad_masks=prefix_pad_masks,
        past_key_values=past_key_values,
        x_t=test_x,
        timestep=test_time.expand(1),
    )
    
    v_same_check = (v_check1 - v_check2).abs()
    print(f"     v(x, t=0.5) 第一次: norm={v_check1.norm():.6f}")
    print(f"     v(x, t=0.5) 第二次: norm={v_check2.norm():.6f}")
    print(f"     差异: {v_same_check.max():.10f} (应该是 0)")
    
    if v_same_check.max() < 1e-6:
        print("     ✓ 确认：相同 (x, t) 给出相同速度！")
        print("\n     结论：速度差异完全是因为 x 位置不同（离散化误差累积），")
        print("            而不是模型本身的问题。")
    
    # 8.6 展示 x 位置差异如何影响速度
    print("\n   位置差异如何影响速度:")
    x_diff = (noise_from_recompute - original_noise).abs()
    print(f"     原始 noise vs 反推 noise 的 RMSE: {rmse_recompute:.4f}")
    print(f"     这个位置差异导致速度场计算结果不同。")
    
    # 9. 验证使用重新计算的 noise 解码
    print("\n9. Verification: Forward ODE with recomputed noise...")
    actions_from_recompute = model.sample_actions(
        device=args.device,
        observation=observation,
        noise=noise_from_recompute.clone(),
        num_steps=args.num_ode_steps
    )
    
    action_error = (actions_from_recompute - actions).abs()
    action_rmse = torch.sqrt((action_error ** 2).mean()).item()
    print(f"   Action RMSE: {action_rmse:.6f}")
    
    # 10. 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n方案A (保存速度场):")
    print(f"  Noise RMSE: {rmse_saved:.10f}")
    if rmse_saved < 1e-5:
        print("  ✓ 完美可逆！保存速度场方案有效。")
    else:
        print("  ✗ 仍有误差，可能是数值精度问题")
    
    print(f"\n方案B (重新计算速度场):")
    print(f"  Noise RMSE: {rmse_recompute:.6f}")
    print(f"  Action RMSE: {action_rmse:.6f}")
    
    if rmse_recompute > 0.1:
        print("  ✗ 噪声重建误差大，但...")
        if action_rmse < 0.01:
            print("  ✓ Action 重建误差小！这说明不同噪声可以映射到相近的 actions")
            print("\n  结论：Flow-Matching 不是双射，多个噪声可以生成相同的 actions。")
            print("  因此，即使反推的噪声不同，只要解码后的 actions 一致，反推就是有效的！")
    
    print("\n" + "=" * 60)
    print("关键结论：如果需要完美可逆，必须保存速度场。")
    print("对于 latent dataset 生成：只要解码后 actions 一致，反推的 noise 就是有效的。")
    print("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)
