"""
检查 latent_dataset.h5 的结构和内容
"""

import h5py
import numpy as np
import argparse


def inspect_dataset(path: str):
    print(f"{'='*60}")
    print(f"Inspecting: {path}")
    print(f"{'='*60}\n")
    
    with h5py.File(path, 'r') as f:
        # 1. 元数据
        print("📋 METADATA (Attributes)")
        print("-" * 40)
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        print()
        
        # 2. 数据集结构
        print("📁 DATASETS")
        print("-" * 40)
        for name in f.keys():
            ds = f[name]
            if isinstance(ds, h5py.Dataset):
                print(f"  {name}:")
                print(f"    Shape: {ds.shape}")
                print(f"    Dtype: {ds.dtype}")
                if ds.compression:
                    print(f"    Compression: {ds.compression}")
                print()
        
        # 3. 统计量详情
        print("📊 STATISTICS")
        print("-" * 40)
        
        stat_keys = ['delta_mean', 'delta_std', 'latents_mean', 'latents_std', 
                     'features_mean', 'features_std']
        for key in stat_keys:
            if key in f:
                data = f[key][:]
                print(f"  {key}:")
                print(f"    Shape: {data.shape}")
                print(f"    Range: [{data.min():.6f}, {data.max():.6f}]")
                print(f"    Mean:  {data.mean():.6f}")
                print()
        
        # 4. 主数据详情
        print("📈 DATA SUMMARY")
        print("-" * 40)
        
        if 'latents' in f:
            latents = f['latents']
            print(f"  Latents: {latents.shape}")
            # 采样一小部分来查看，只分析前14维（有效维度）
            valid_dims = 14
            sample = latents[:min(1000, latents.shape[0]), :, :valid_dims]
            print(f"    (Analyzing first {valid_dims} dims only)")
            print(f"    Sample range: [{sample.min():.4f}, {sample.max():.4f}]")
            print(f"    Sample mean:  {sample.mean():.4f}")
            print(f"    Sample std:   {sample.std():.4f}")
            print()
        
        if 'features' in f:
            features = f['features']
            print(f"  Features: {features.shape}")
            sample = features[:min(100, features.shape[0])]
            print(f"    Sample range: [{sample.min():.4f}, {sample.max():.4f}]")
            print(f"    Sample mean:  {sample.mean():.4f}")
            print(f"    Sample std:   {sample.std():.4f}")
            print()
        
        # 5. Episode 分析
        print("🎬 EPISODE ANALYSIS")
        print("-" * 40)
        
        if 'episode_ids' in f and f['episode_ids'].shape[0] > 0:
            ep_ids = f['episode_ids'][:]
            unique_eps = np.unique(ep_ids)
            print(f"  Total samples: {len(ep_ids)}")
            print(f"  Unique episodes: {len(unique_eps)}")
            print(f"  Episode IDs: {unique_eps[:10]}{'...' if len(unique_eps) > 10 else ''}")
            
            # 每个 episode 的长度
            ep_counts = np.bincount(ep_ids)
            ep_counts = ep_counts[ep_counts > 0]
            print(f"  Episode lengths: min={ep_counts.min()}, max={ep_counts.max()}, mean={ep_counts.mean():.1f}")
            
            # 有效转换数
            valid_transitions = np.sum(ep_ids[:-1] == ep_ids[1:])
            print(f"  Valid transitions: {valid_transitions}")
        else:
            print("  No episode_ids found!")
        print()
        
        # 6. Delta 分布（计算实际值）
        print("🔄 DELTA DISTRIBUTION (Actual)")
        print("-" * 40)
        
        if 'latents' in f and 'episode_ids' in f and f['episode_ids'].shape[0] > 0:
            latents = f['latents']
            ep_ids = f['episode_ids'][:]
            
            # 找有效转换
            valid_mask = ep_ids[:-1] == ep_ids[1:]
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # 采样计算，只分析前14维
                valid_dims = 14
                sample_size = min(5000, len(valid_indices))
                sample_idx = np.random.choice(valid_indices, sample_size, replace=False)
                sample_idx = np.sort(sample_idx)  # HDF5 需要索引递增
                
                z_curr = latents[sample_idx][:, :, :valid_dims]
                z_next = latents[sample_idx + 1][:, :, :valid_dims]
                deltas = z_next - z_curr
                
                print(f"  Computed from {sample_size} transitions (first {valid_dims} dims):")
                print(f"    Delta shape: {deltas.shape}")
                print(f"    Delta range: [{deltas.min():.4f}, {deltas.max():.4f}]")
                print(f"    Delta mean:  {deltas.mean():.6f}")
                print(f"    Delta std:   {deltas.std():.4f}")
                
                # 每维度统计
                delta_per_dim = deltas.reshape(-1, deltas.shape[-1])
                dim_std = delta_per_dim.std(axis=0)
                print(f"    Per-dim std range: [{dim_std.min():.4f}, {dim_std.max():.4f}]")
        print()
        
        print(f"{'='*60}")
        print("Inspection complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/ztlab/lgd/openpi/latent_dataset.h5")
    args = parser.parse_args()
    
    inspect_dataset(args.path)
