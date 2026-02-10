import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def analyze_residual_distribution(dataset_path, episode_idx=9, output_filename="latent_residual_analysis.png"):
    """
    分析一个 episode 下噪声的残差（前后两个噪声的差）分布
    残差 = z_{t+1} - z_t
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    print(f"Loading episode {episode_idx} from {dataset_path}...")
    
    with h5py.File(dataset_path, 'r') as f:
        if 'latents' not in f:
            print(f"Error: 'latents' dataset not found in {dataset_path}. Keys: {list(f.keys())}")
            return
            
        total_len = f['latents'].shape[0]
        if episode_idx >= total_len:
            print(f"Error: Episode index {episode_idx} is out of bounds (Total: {total_len})")
            return

        # 读取数据
        # shape: [Horizon, Action_Dim] (e.g., [50, 14])
        latents = f['latents'][episode_idx]
        
    print(f"Latent shape: {latents.shape}")
    
    # 只分析前14维（有效维度）
    valid_dims = 14
    latents = latents[:, :valid_dims]
    print(f"Analyzing first {valid_dims} dimensions only")
    
    # 计算残差：z_{t+1} - z_t
    residuals = latents[1:] - latents[:-1]  # shape: [T-1, D]
    
    print(f"\n=== Residual Statistics ===")
    print(f"Residual shape: {residuals.shape}")
    print(f"Overall: Mean={np.mean(residuals):.6f}, Std={np.std(residuals):.6f}")
    print(f"         Min={np.min(residuals):.6f}, Max={np.max(residuals):.6f}")
    
    # 每个维度的残差统计
    print(f"\nPer-dimension residual statistics:")
    for dim in range(valid_dims):
        dim_res = residuals[:, dim]
        print(f"  Dim {dim:2d}: Mean={np.mean(dim_res):+.6f}, Std={np.std(dim_res):.6f}, "
              f"Range=[{np.min(dim_res):.4f}, {np.max(dim_res):.4f}]")

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # --- 图 1: 残差热力图 ---
    # 目的：直观展示残差在时间和维度上的分布
    # 如果大部分区域接近 0（白色/浅色），说明噪声变化平缓
    # 如果有明显的正/负条纹，说明存在持续的趋势
    sns.heatmap(residuals.T, ax=axes[0, 0], cmap="RdBu_r", center=0, 
                cbar_kws={'label': 'Residual ($z_{t+1} - z_t$)'})
    axes[0, 0].set_title(f"Residual Heatmap (Episode {episode_idx})")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Action Dimension")
    
    # --- 图 2: 残差直方图 ---
    # 目的：查看残差的整体分布是否接近正态分布
    # 如果是正态分布且均值接近 0，说明噪声变化是随机的
    residuals_flat = residuals.flatten()
    axes[0, 1].hist(residuals_flat, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 拟合正态分布
    mu, sigma = np.mean(residuals_flat), np.std(residuals_flat)
    x = np.linspace(residuals_flat.min(), residuals_flat.max(), 100)
    gaussian = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))
    axes[0, 1].plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian fit\n$\\mu$={mu:.4f}, $\\sigma$={sigma:.4f}')
    
    axes[0, 1].set_title(f"Residual Distribution (All Dims)")
    axes[0, 1].set_xlabel("Residual Value ($z_{t+1} - z_t$)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # --- 图 3: 各维度残差的箱线图 ---
    # 目的：比较不同维度的残差分布
    # 如果某个维度的箱子明显大，说明该维度变化更剧烈
    axes[1, 0].boxplot([residuals[:, d] for d in range(valid_dims)], 
                       labels=[f'D{d}' for d in range(valid_dims)])
    axes[1, 0].set_title("Residual Distribution by Dimension")
    axes[1, 0].set_xlabel("Dimension")
    axes[1, 0].set_ylabel("Residual Value")
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)

    # --- 图 4: 残差的时序演化（选定维度）---
    # 目的：查看残差随时间的变化
    # 如果残差在 0 附近波动，说明噪声变化是稳定的
    # 如果有持续的正/负偏移，说明噪声有漂移趋势
    dims_to_plot = [0, 1, 2, 3, 6, 7]  # 选择几个关键维度
    dims_to_plot = [d for d in dims_to_plot if d < valid_dims]
    
    for dim in dims_to_plot:
        axes[1, 1].plot(residuals[:, dim], label=f"Dim {dim}", marker='o', markersize=2, alpha=0.7)
    
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_title("Residual Time Series (Selected Dims)")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Residual ($z_{t+1} - z_t$)")
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis saved to {output_filename}")


if __name__ == "__main__":
    # 使用方法：
    # python -m scripts.analyze_latents [dataset_path] [episode_idx]
    
    default_path = "latent_dataset.h5"
    episode_idx = 2
    
    if len(sys.argv) > 1:
        default_path = sys.argv[1]
    if len(sys.argv) > 2:
        episode_idx = int(sys.argv[2])
        
    analyze_residual_distribution(default_path, episode_idx=episode_idx)
