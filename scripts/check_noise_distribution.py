import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import os

def check_noise_statistics(dataset_path):
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    print(f"Loading latents from {dataset_path}...")
    with h5py.File(dataset_path, 'r') as f:
        # shape: [N_episodes, Horizon, Action_Dim]
        # 例如: [1000, 50, 14]
        latents = f['latents'][:] 
        
    print(f"Data shape: {latents.shape}")
    
    # ---------------------------------------------
    # 1. 全局统计 (Global Statistics)
    # ---------------------------------------------
    global_mean = np.mean(latents)
    global_std = np.std(latents)
    global_min = np.min(latents)
    global_max = np.max(latents)
    
    print("\n>>> Global Statistics <<<")
    print(f"Mean: {global_mean:.6f} (Ideal: 0.0)")
    print(f"Std : {global_std:.6f}  (Ideal: 1.0)")
    print(f"Range: [{global_min:.4f}, {global_max:.4f}]")
    
    # ---------------------------------------------
    # 2. 详细分析 & 可视化
    # ---------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3)
    
    # [图1] 全局直方图 vs 标准正态分布
    # 如果偏移很大，推理时必须修正初始采样
    ax = axes[0, 0]
    data_flat = latents.flatten()
    # 随机采样一部分画图，避免太慢
    if len(data_flat) > 100000:
        data_flat = np.random.choice(data_flat, 100000, replace=False)
        
    sns.histplot(data_flat, bins=100, stat="density", color="skyblue", alpha=0.6, ax=ax, label="Dataset Latents")
    
    # 画标准正态分布曲线
    x = np.linspace(-4, 4, 100)
    ax.plot(x, norm.pdf(x, 0, 1), 'r--', linewidth=2, label="Standard N(0,1)")
    ax.set_title(f"Global Distribution\nMean={global_mean:.4f}, Std={global_std:.4f}")
    ax.legend()
    
    # [图2] 随时间步的分布 (Temporal Structure)
    # 检查 Horizon (0~49) 上，每一步的均值和方差是否一致
    # 理论上 Flow Matching 中整个 chunk 共享同一个 time t，所以应该一致
    ax = axes[0, 1]
    # mean over (Batch, Dim), preserve Horizon
    mean_per_step = np.mean(latents, axis=(0, 2))
    std_per_step = np.std(latents, axis=(0, 2))
    
    steps = np.arange(len(mean_per_step))
    ax.plot(steps, mean_per_step, 'b-', label="Mean")
    ax.fill_between(steps, mean_per_step - std_per_step, mean_per_step + std_per_step, color='b', alpha=0.2, label="Std Range")
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title("Distribution per Horizon Step")
    ax.set_xlabel("Horizon Step (0-49)")
    ax.legend()
    
    # [图3] 随维度的分布 (Dimensional Structure)
    # 检查是不是某些关节（比如底盘 vs 手臂）的噪声分布不同
    ax = axes[1, 0]
    # mean over (Batch, Horizon), preserve Dim
    mean_per_dim = np.mean(latents, axis=(0, 1))
    std_per_dim = np.std(latents, axis=(0, 1))
    
    dims = np.arange(len(mean_per_dim))
    ax.errorbar(dims, mean_per_dim, yerr=std_per_dim, fmt='o', capsize=5, ecolor='gray')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_title("Mean & Std per Action Dimension")
    ax.set_xlabel("Dimension Index")
    ax.set_xticks(dims)
    
    # [图4] 样本相关性 (Covariance Heatmap)
    # 看看维度之间是不是独立的
    # 我们把 (Horizon * Dim) 展平来看看结构
    ax = axes[1, 1]
    # 取前1000个样本计算协方差
    sample_flat = latents[:1000].reshape(1000, -1) 
    # 计算相关系数矩阵
    corr_matrix = np.corrcoef(sample_flat.T)
    # 只画前 50x50 (即前几步的维度) 避免图太大
    subset_size = min(corr_matrix.shape[0], 64) 
    sns.heatmap(corr_matrix[:subset_size, :subset_size], cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Correlation Matrix (First {subset_size} Features)")
    
    plt.tight_layout()
    plt.savefig("noise_distribution_check.png")
    print("\nAnalysis plot saved to 'noise_distribution_check.png'")
    
    # ---------------------------------------------
    # 3. 结论建议
    # ---------------------------------------------
    print("\n>>> Analysis Report <<<")
    if abs(global_mean) > 0.1 or abs(global_std - 1.0) > 0.1:
        print("[WARNING] The dataset distribution deviates from Standard Normal!")
        print("建议方案 1 (推荐): 在推理脚本中，初始化噪声时使用统计出的 Mean 和 Std。")
        print(f"    --> noise = torch.randn(...) * {global_std:.4f} + {global_mean:.4f}")
        print("建议方案 2: 重新检查 generate_dataset 脚本，是否 ODE 步数太少导致没推回到纯噪声？")
    else:
        print("[OK] Distribution looks close to Standard Normal. Random initialization is safe.")

if __name__ == "__main__":
    # 请修改为你的数据集路径
    DATASET_PATH = "/home/ztlab/lgd/openpi/velocity_dataset.h5"
    check_noise_statistics(DATASET_PATH)