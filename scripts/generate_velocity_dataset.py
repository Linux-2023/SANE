import torch
import numpy as np
import os
from tqdm import tqdm
import h5py
from typing import Dict, Any, Optional, List
import logging
import tyro
from dataclasses import dataclass, field

from openpi.training import data_loader
from openpi.training import config as _config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from lerobot.common.datasets import lerobot_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Args:
    policy_dir: str = "/home/ztlab/lgd/openpi/checkpoints/pi05_piper_pytorch"
    policy_config: str = "pi05_piper_pick_cube"
    output_path: str = "velocity_dataset.h5"
    num_ode_steps: int = 10
    # 想要保存速度场的步数索引。例如 [0] 表示只存第一步(t=0, Data端)，[9] 表示最后一步(t=1, Noise端)
    # 默认为 [0, 5, 9] (起点、中点、终点)
    save_velocity_indices: List[int] = field(default_factory=lambda: [0, 5, 9]) 
    batch_size: int = 8
    max_episodes: Optional[int] = None
    device: str = "cuda"
    num_workers: int = 8
    dataset_repo_id: str = "piper_dual_lerobot"
    dataset_root: Optional[str] = "/home/ztlab/lgd/openpi/pick_anything_100/piper_dual_lerobot"

def format_inputs_for_model(batch: Dict[str, Any], config: Any, device: str):
    """
    Format batch dict from LeRobotDataset into (images, img_masks, tokens, masks) 
    """
    # 1. Images
    images = []
    img_masks = []
    
    if "images" in batch and isinstance(batch["images"], dict):
        image_keys = sorted(batch["images"].keys())
        for key in image_keys:
            img = batch["images"][key]
            if isinstance(img, torch.Tensor): img = img.to(device)
            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))
    else:
        image_keys = [k for k in batch.keys() if k.startswith("observation.images.")]
        image_keys.sort() 
        for key in image_keys:
            img = batch[key]
            if isinstance(img, torch.Tensor): img = img.to(device)
            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))

    # 2. Tokens
    tokens = None
    masks = None
    if "tokenized_prompt" in batch:
        tokens = batch["tokenized_prompt"].to(device)
        masks = batch["tokenized_prompt_mask"].to(device)
    elif "observation.tokenized_prompt" in batch:
        tokens = batch["observation.tokenized_prompt"].to(device)
        masks = batch["observation.tokenized_prompt_mask"].to(device)

    # 3. State
    state = batch.get("state")
    if state is None: state = batch.get("observation.state")
    if state is not None: state = state.to(device)

    return images, img_masks, tokens, masks, state

class InverseFlowWrapper(torch.nn.Module):
    def __init__(self, model: PI0Pytorch):
        super().__init__()
        self.model = model
        self.config = model.config
        
    @torch.no_grad()
    def extract_features(self, images, img_masks, tokens, masks):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    @torch.no_grad()
    def invert_actions_with_velocity(
        self, actions: torch.Tensor, images, img_masks, tokens, masks, state, 
        num_steps=10, save_indices=[0],
        prefix_embs=None, prefix_pad_masks=None, prefix_att_masks=None
    ):
        """
        Inverse ODE Solver that also captures Velocity Fields.
        Returns:
            latents: Noise z at t=1 [B, H, D]
            velocities: Stacked velocities [B, Num_Save, H, D]
        """
        bsize = actions.shape[0]
        device = actions.device
        
        # 1. Precompute Prefix
        if prefix_embs is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks, tokens, masks
            )
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        # Caching
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 2. ODE Integration
        dt = 1.0 / num_steps
        x_t = actions.clone()
        time = torch.tensor(0.0, dtype=torch.float32, device=device)
        
        captured_velocities = []
        
        for step_i in range(num_steps):
            expanded_time = time.expand(bsize)
            
            # Compute v_t
            v_t = self.model.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values, 
                x_t=x_t,
                timestep=expanded_time
            )
            
            # Save velocity if index matches
            if step_i in save_indices:
                captured_velocities.append(v_t.clone())
            
            # Euler update
            x_t = x_t + dt * v_t
            time = time + dt
            
        # Stack velocities: [B, N_Save, H, D]
        if captured_velocities:
            velocities_stack = torch.stack(captured_velocities, dim=1)
        else:
            velocities_stack = torch.zeros(bsize, 0, *x_t.shape[1:], device=device)
            
        return x_t, velocities_stack

def main(args: Args):
    server_config = _config.get_config(args.policy_config)
    model_config = server_config.model
    
    print("Initializing Model...")
    model = PI0Pytorch(model_config)
    model.eval()
    model.to(args.device)
    
    wrapper = InverseFlowWrapper(model)
    
    # Dataset Load Logic (Same as before)
    print("Loading dataset...")
    data_config_factory = server_config.data
    data_config = data_config_factory.create(server_config.assets_dirs, server_config.model)
    repo_id = args.dataset_repo_id or data_config.repo_id
    
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=args.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)] 
            for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
         from openpi import transforms as _transforms
         dataset = data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    dataset = data_loader.transform_dataset(dataset, data_config)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    print(f"Creating output file at {args.output_path}...")
    # Validate save indices
    valid_indices = [i for i in args.save_velocity_indices if 0 <= i < args.num_ode_steps]
    if len(valid_indices) != len(args.save_velocity_indices):
        logger.warning(f"Filtered invalid indices. Using: {valid_indices}")
    num_save_steps = len(valid_indices)

    with h5py.File(args.output_path, 'w') as f:
        # Datasets
        latents_shape = (0, model_config.action_horizon, model_config.action_dim)
        max_latents_shape = (None, model_config.action_horizon, model_config.action_dim)
        
        # Velocity Shape: [N, Num_Steps, Horizon, Dim]
        vel_shape = (0, num_save_steps, model_config.action_horizon, model_config.action_dim)
        max_vel_shape = (None, num_save_steps, model_config.action_horizon, model_config.action_dim)

        ds_latents = f.create_dataset('latents', shape=latents_shape, maxshape=max_latents_shape, dtype='f4')
        ds_vels = f.create_dataset('velocities', shape=vel_shape, maxshape=max_vel_shape, dtype='f4')
        ds_feats = None 
        ds_ep_ids = f.create_dataset('episode_ids', shape=(0,), maxshape=(None,), dtype='i4')
        
        # Save indices metadata for later reference
        f.attrs['saved_velocity_step_indices'] = valid_indices
        f.attrs['total_ode_steps'] = args.num_ode_steps

        buffer_latents, buffer_vels, buffer_feats, buffer_ep_ids = [], [], [], []
        
        current_episode_idx = 0
        last_episode_index_in_batch = -1 # Tracker for episode changes if available
        
        print(f"Starting generation... Saving velocities at steps {valid_indices}")
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            
            # Move to device & format
            images, img_masks, tokens, masks, state = format_inputs_for_model(batch, model_config, args.device)
            actions = batch.get('actions', batch.get('action')).to(device=args.device, dtype=torch.float32)
            
            # 1. Extract Features
            prefix_embs, prefix_pad_masks, prefix_att_masks = wrapper.extract_features(images, img_masks, tokens, masks)

            # 2. Run Inverse Flow with Velocity Capture
            latents, velocities = wrapper.invert_actions_with_velocity(
                actions, images, img_masks, tokens, masks, state, 
                num_steps=args.num_ode_steps,
                save_indices=valid_indices,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks
            )
            
            # 3. Buffer Data
            buffer_latents.append(latents.detach().cpu().float().numpy())
            buffer_vels.append(velocities.detach().cpu().float().numpy())
            buffer_feats.append(prefix_embs.detach().cpu().float().numpy())
            
            # Simple Episode ID logic (assuming sequential batching)
            # If LeRobot provides 'episode_index', use it. Otherwise, increment on 'index' jump or just linear.
            # Here we just use batch index as a proxy or extract real ID if available.
            if 'episode_index' in batch:
                ep_ids = batch['episode_index'].numpy()
            else:
                # Fallback: assume sequential
                bs = latents.shape[0]
                ep_ids = np.full(bs, -1) # Placeholder
            
            buffer_ep_ids.append(ep_ids)

            # 4. Flush Buffer
            if (batch_idx + 1) % 16 == 0:
                b_lat = np.concatenate(buffer_latents, axis=0)
                b_vel = np.concatenate(buffer_vels, axis=0)
                b_feat = np.concatenate(buffer_feats, axis=0)
                b_ep = np.concatenate(buffer_ep_ids, axis=0)
                
                # Resize
                curr_len = ds_latents.shape[0]
                add_len = len(b_lat)
                
                ds_latents.resize(curr_len + add_len, axis=0)
                ds_latents[-add_len:] = b_lat
                
                ds_vels.resize(curr_len + add_len, axis=0)
                ds_vels[-add_len:] = b_vel
                
                if ds_feats is None:
                     ds_feats = f.create_dataset('features', shape=(0, *b_feat.shape[1:]), maxshape=(None, *b_feat.shape[1:]), dtype='f4')
                ds_feats.resize(curr_len + add_len, axis=0)
                ds_feats[-add_len:] = b_feat
                
                ds_ep_ids.resize(curr_len + add_len, axis=0)
                ds_ep_ids[-add_len:] = b_ep
                
                buffer_latents, buffer_vels, buffer_feats, buffer_ep_ids = [], [], [], []
                
        # Final Flush
        if buffer_latents:
            b_lat = np.concatenate(buffer_latents, axis=0)
            b_vel = np.concatenate(buffer_vels, axis=0)
            b_feat = np.concatenate(buffer_feats, axis=0)
            b_ep = np.concatenate(buffer_ep_ids, axis=0)
            
            ds_latents.resize(ds_latents.shape[0] + len(b_lat), axis=0)
            ds_latents[-len(b_lat):] = b_lat
            
            ds_vels.resize(ds_vels.shape[0] + len(b_vel), axis=0)
            ds_vels[-len(b_vel):] = b_vel
            
            if ds_feats is None:
                 ds_feats = f.create_dataset('features', shape=(0, *b_feat.shape[1:]), maxshape=(None, *b_feat.shape[1:]), dtype='f4')
            ds_feats.resize(ds_feats.shape[0] + len(b_feat), axis=0)
            ds_feats[-len(b_feat):] = b_feat
            
            ds_ep_ids.resize(ds_ep_ids.shape[0] + len(b_ep), axis=0)
            ds_ep_ids[-len(b_ep):] = b_ep

    print(f"Done. Dataset saved to {args.output_path}")

if __name__ == "__main__":
    tyro.cli(main)