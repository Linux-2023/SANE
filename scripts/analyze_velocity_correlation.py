import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os

def analyze_velocity_drift(dataset_path, select_idx=-1, max_samples=5000):
    print(f"Loading dataset from {dataset_path}...")
    
    with h5py.File(dataset_path, 'r') as f:
        latents = f['latents'][:] # [N, H, D]
        velocities = f['velocities'][:] # [N, Steps, H, D]
        ep_ids = f['episode_ids'][:]
        
        # 读取保存的步数索引信息
        saved_indices = f.attrs.get('saved_velocity_step_indices', [0, 5, 9])
        
        print(f"Velocity dataset shape: {velocities.shape}")
        print(f"Saved ODE steps: {saved_indices}")
        
        # 根据 select_idx 选择特定步数的速度场
        # select_idx = 0  -> Saved Step 0 (Data Space, t=0)
        # select_idx = 1  -> Saved Step 5
        # select_idx = -1 -> Saved Step 9 (Noise Space, t=1)
        
        real_step = saved_indices[select_idx]
        print(f"--> Selecting velocity at Index {select_idx} (ODE Step {real_step})")
        
        target_velocity = velocities[:, select_idx, :, :] # [N, H, D]
        
    print(f"Total samples: {len(latents)}")
    
    # 1. 构建相邻帧对 (t, t+1)
    drift_vectors = []
    velocity_vectors = []
    
    # 为了避免计算量过大，我们可以只取部分样本
    indices = np.arange(len(latents) - 1)
    if max_samples and len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False)
    
    valid_count = 0
    
    for i in indices:
        # 必须是同一个 Episode
        if ep_ids[i] != ep_ids[i+1]:
            continue
            
        z_t = latents[i]     # [H, D]
        z_next = latents[i+1] # [H, D]
        
        # 计算漂移: Delta z
        # 注意：这是整条轨迹的漂移，我们将其展平处理
        drift = (z_next - z_t).flatten()
        
        # 提取速度场: v
        # 使用上面选择的 t 时刻的速度
        v = target_velocity[i].flatten()
        
        drift_vectors.append(drift)
        velocity_vectors.append(v)
        valid_count += 1
        
    drift_vectors = np.array(drift_vectors) # [M, H*D]
    velocity_vectors = np.array(velocity_vectors) # [M, H*D]
    
    print(f"Analyzed {valid_count} transitions.")
    
    # 2. 计算余弦相似度
    # CosSim = (A . B) / (|A| * |B|)
    
    # 先归一化
    norm_drift = np.linalg.norm(drift_vectors, axis=1, keepdims=True)
    norm_vel = np.linalg.norm(velocity_vectors, axis=1, keepdims=True)
    
    # 避免除零
    norm_drift[norm_drift < 1e-6] = 1e-6
    norm_vel[norm_vel < 1e-6] = 1e-6
    
    drift_unit = drift_vectors / norm_drift
    vel_unit = velocity_vectors / norm_vel
    
    # 点积计算相似度
    cos_sim = np.sum(drift_unit * vel_unit, axis=1)
    
    mean_sim = np.mean(cos_sim)
    print(f"\n>>> Mean Cosine Similarity: {mean_sim:.4f}")
    
    # 3. 可视化
    plot_results(cos_sim, drift_vectors, velocity_vectors)

def plot_results(cos_sim, drift_vectors, velocity_vectors):
    plt.figure(figsize=(18, 6))
    
    # --- 图 1: 余弦相似度直方图 ---
    plt.subplot(1, 3, 1)
    sns.histplot(cos_sim, bins=50, kde=True, color='purple')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=np.mean(cos_sim), color='r', linestyle='-', label=f'Mean: {np.mean(cos_sim):.2f}')
    plt.title("Cosine Similarity Distribution\n(Drift vs. Noise Velocity)")
    plt.xlabel("Cosine Similarity (-1 to 1)")
    plt.ylabel("Count")
    plt.legend()
    
    # --- 图 2: 向量模长对比 ---
    # 看看速度场的大小和漂移的大小是否成比例
    plt.subplot(1, 3, 2)
    drift_mags = np.linalg.norm(drift_vectors, axis=1)
    vel_mags = np.linalg.norm(velocity_vectors, axis=1)
    
    plt.scatter(drift_mags, vel_mags, alpha=0.1, s=5)
    plt.xlabel("|Drift| Magnitude")
    plt.ylabel("|Velocity| Magnitude")
    plt.title("Magnitude Correlation")
    
    # 拟合一条线看看比例关系
    m, b = np.polyfit(drift_mags, vel_mags, 1)
    plt.plot(drift_mags, m*drift_mags + b, color='r', alpha=0.8, label=f"Fit: y={m:.2f}x+{b:.2f}")
    plt.legend()
    
    # --- 图 3: PCA 2D 向量场可视化 ---
    # 这是一个很酷的图，把高维的两个向量投影到平面上，看方向是否一致
    plt.subplot(1, 3, 3)
    
    # 选取前 100 个样本进行可视化，否则太乱
    N_vis = 100
    subset_drift = drift_vectors[:N_vis]
    subset_vel = velocity_vectors[:N_vis]
    
    # 用 Drift 数据训练 PCA，把大家投影到 Drift 变化最大的平面
    pca = PCA(n_components=2)
    pca.fit(subset_drift)
    
    drift_2d = pca.transform(subset_drift)
    vel_2d = pca.transform(subset_vel)
    
    # 归一化以便画箭头
    drift_2d /= np.linalg.norm(drift_2d, axis=1, keepdims=True) + 1e-6
    vel_2d /= np.linalg.norm(vel_2d, axis=1, keepdims=True) + 1e-6
    
    origin = np.zeros_like(drift_2d)
    
    # 画 Drift 箭头 (蓝色)
    plt.quiver(origin[:,0], origin[:,1], drift_2d[:,0], drift_2d[:,1], 
               color='blue', alpha=0.5, angles='xy', scale_units='xy', scale=1, label='Drift Direction')
    
    # 画 Velocity 箭头 (红色)
    plt.quiver(origin[:,0], origin[:,1], vel_2d[:,0], vel_2d[:,1], 
               color='red', alpha=0.5, angles='xy', scale_units='xy', scale=1, label='Velocity Direction')
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title("Vector Direction in 2D PCA Space\n(Blue=Drift, Red=Velocity)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("velocity_correlation_analysis.png")
    plt.show()
    print("Analysis plot saved to 'velocity_correlation_analysis.png'")

if __name__ == "__main__":
    # 请修改为你的 dataset 路径
    DATASET_PATH = "/home/ztlab/lgd/openpi/velocity_dataset.h5"
    if os.path.exists(DATASET_PATH):
        # 参数 select_idx 控制选择哪一步的速度场进行分析
        # 0: Data Space (t=0)
        # 1: Middle (t=0.5)
        # -1: Noise Space (t=1)
        analyze_velocity_drift(DATASET_PATH, select_idx=1)
    else:
        print(f"File not found: {DATASET_PATH}")