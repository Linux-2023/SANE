"""
验证脚本 v2：确保反推 ODE 与正向 ODE 完全一致。

关键发现：
- model.sample_actions 在 32 维 latent space 操作
- policy 输出只取前 14 维作为真实动作
- 反推应该在 32 维 latent space 进行

测试流程：
1. 生成 32 维随机 noise
2. 调用 model.sample_actions() 得到 32 维 latent actions
3. 从 32 维 latent 反推得到 noise'
4. 验证 noise' ≈ noise
5. 使用 noise' 再次解码得到 latent actions'
6. 验证 actions' ≈ actions
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
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
def main(args: Args):
    print("=" * 60)
    print("Inverse ODE Verification Script v2")
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
    
    # 应用 repack 变换得到正确格式
    from openpi import transforms as _transforms
    if data_config.prompt_from_task:
        raw_dataset = data_loader.TransformedDataset(
            raw_dataset, 
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    
    # 应用完整的数据变换（包括 repack）
    transformed_dataset = data_loader.transform_dataset(raw_dataset, data_config)
    
    # 3. 获取测试样本
    print(f"\n3. Getting test sample at index {args.test_idx}...")
    item = transformed_dataset[args.test_idx]
    
    # 打印可用的键和形状
    print(f"   Available keys: {list(item.keys())}")
    for k, v in item.items():
        if hasattr(v, 'shape'):
            print(f"   {k}: shape={v.shape}, dtype={v.dtype}")
    
    # transform_dataset 后的数据已经是模型可直接使用的格式
    # 键是: image, image_mask, state, tokenized_prompt, tokenized_prompt_mask, actions
    # 直接转换为 tensor 并添加 batch 维度
    
    def to_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.to(args.device)
        elif isinstance(v, np.ndarray):
            return torch.from_numpy(v).to(args.device)
        else:
            return torch.tensor(v, device=args.device)
    
    # 构建 observation tensor dict
    # image 可能是 dict 或 tensor，取决于 transform 阶段
    images_raw = item['image']
    image_masks_raw = item['image_mask']
    
    # 获取相机名称 - 根据 piper 配置
    camera_names = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
    
    images_dict = {}
    image_masks_dict = {}
    
    if isinstance(images_raw, dict):
        # 如果已经是 dict 格式
        for cam_name in camera_names:
            if cam_name in images_raw:
                img = to_tensor(images_raw[cam_name])
                # 转换为 float 并归一化到 [0, 1]
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                # 如果是 [H, W, C] 格式，转换为 [C, H, W]
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                images_dict[cam_name] = img[None, ...]  # [1, C, H, W]
                image_masks_dict[cam_name] = torch.ones(1, dtype=torch.bool, device=args.device)
    else:
        # 如果是 tensor 格式 [N_cameras, H, W, C] 或 [N_cameras, C, H, W]
        images_tensor = images_raw
        image_masks_tensor = image_masks_raw
        for i, cam_name in enumerate(camera_names):
            if i < images_tensor.shape[0]:
                img = to_tensor(images_tensor[i])
                # 转换为 float 并归一化到 [0, 1]
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                # 如果是 [H, W, C] 格式，转换为 [C, H, W]
                if img.ndim == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
                images_dict[cam_name] = img[None, ...]  # [1, C, H, W]
                if image_masks_tensor is not None:
                    image_masks_dict[cam_name] = to_tensor(image_masks_tensor[i:i+1]).bool()  # [1]
                else:
                    image_masks_dict[cam_name] = torch.ones(1, dtype=torch.bool, device=args.device)
    
    # State: 需要 padding 到 32 维（如果是 14 维）
    state_raw = to_tensor(item['state'])
    if state_raw.shape[-1] < 32:
        state_padded = torch.zeros(32, device=args.device)
        state_padded[:state_raw.shape[-1]] = state_raw
        state_tensor = state_padded[None, ...]  # [1, 32]
    else:
        state_tensor = state_raw[None, ...]
    
    # Tokenized prompt
    tokenized_prompt = to_tensor(item['tokenized_prompt'])[None, ...]  # [1, L]
    tokenized_prompt_mask = to_tensor(item['tokenized_prompt_mask'])[None, ...]  # [1, L]
    
    # 构建 Observation
    observation = _model.Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        state=state_tensor,
        tokenized_prompt=tokenized_prompt.long(),
        tokenized_prompt_mask=tokenized_prompt_mask.bool(),
    )
    
    print(f"   State shape: {observation.state.shape}")
    print(f"   Images keys: {list(observation.images.keys())}")
    for k, v in observation.images.items():
        print(f"   {k}: {v.shape}")
    
    # 4. 生成随机 noise（32 维）
    print("\n4. Generating random noise (32-dim latent space)...")
    noise_shape = (1, model.config.action_horizon, model.config.action_dim)  # (1, 50, 32)
    original_noise = torch.randn(noise_shape, device=args.device)
    print(f"   Noise shape: {original_noise.shape}")
    print(f"   Noise range: [{original_noise.min():.4f}, {original_noise.max():.4f}]")
    
    # 5. 正向 ODE：noise -> latent_actions（32 维）
    print("\n5. Forward ODE: noise -> latent_actions...")
    latent_actions = model.sample_actions(
        device=args.device,
        observation=observation,
        noise=original_noise.clone(),
        num_steps=args.num_ode_steps
    )
    print(f"   Latent actions shape: {latent_actions.shape}")  # 应该是 (1, 50, 32)
    print(f"   Latent actions range: [{latent_actions.min():.4f}, {latent_actions.max():.4f}]")
    
    # 6. 反向 ODE：latent_actions -> reconstructed_noise
    print("\n6. Inverse ODE: latent_actions -> reconstructed_noise...")
    
    # 预处理观测（与 sample_actions 内部一致）
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
    
    # 反向 ODE：从 t=0 积分到 t=1
    bsize = latent_actions.shape[0]
    dt = 1.0 / args.num_ode_steps
    x_t = latent_actions.clone()
    time = torch.tensor(0.0, dtype=torch.float32, device=args.device)
    
    print(f"   Running inverse ODE with {args.num_ode_steps} steps...")
    for step in range(args.num_ode_steps):
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
    
    reconstructed_noise = x_t
    print(f"   Reconstructed noise shape: {reconstructed_noise.shape}")
    print(f"   Reconstructed noise range: [{reconstructed_noise.min():.4f}, {reconstructed_noise.max():.4f}]")
    
    # 7. 验证 noise 重建误差
    print("\n7. Verification: noise reconstruction error...")
    noise_diff = (reconstructed_noise - original_noise).abs()
    print(f"   Max absolute error: {noise_diff.max():.6f}")
    print(f"   Mean absolute error: {noise_diff.mean():.6f}")
    print(f"   RMSE: {torch.sqrt((noise_diff ** 2).mean()):.6f}")
    
    # 8. 使用重建的 noise 再次解码
    print("\n8. Forward ODE with reconstructed noise...")
    reconstructed_actions = model.sample_actions(
        device=args.device,
        observation=observation,
        noise=reconstructed_noise.clone(),
        num_steps=args.num_ode_steps
    )
    print(f"   Reconstructed actions shape: {reconstructed_actions.shape}")
    
    # 9. 验证 actions 重建误差
    print("\n9. Verification: action reconstruction error...")
    action_diff = (reconstructed_actions - latent_actions).abs()
    print(f"   Max absolute error: {action_diff.max():.6f}")
    print(f"   Mean absolute error: {action_diff.mean():.6f}")
    print(f"   RMSE: {torch.sqrt((action_diff ** 2).mean()):.6f}")
    
    # 10. 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    noise_rmse = torch.sqrt((noise_diff ** 2).mean()).item()
    action_rmse = torch.sqrt((action_diff ** 2).mean()).item()
    
    if noise_rmse < 0.01 and action_rmse < 0.01:
        print("✓ SUCCESS: Inverse ODE is consistent with forward ODE!")
        print(f"  Noise RMSE: {noise_rmse:.6f}")
        print(f"  Action RMSE: {action_rmse:.6f}")
    else:
        print("✗ FAILURE: Inverse ODE is NOT consistent!")
        print(f"  Noise RMSE: {noise_rmse:.6f} (should be < 0.01)")
        print(f"  Action RMSE: {action_rmse:.6f} (should be < 0.01)")
        
        # 调试：显示前几个元素的对比
        print("\n  Sample comparison (first 5 elements of first timestep):")
        print(f"  Original noise[0,0,:5]: {original_noise[0,0,:5].cpu().numpy()}")
        print(f"  Reconstructed noise[0,0,:5]: {reconstructed_noise[0,0,:5].cpu().numpy()}")


if __name__ == "__main__":
    tyro.cli(main)
