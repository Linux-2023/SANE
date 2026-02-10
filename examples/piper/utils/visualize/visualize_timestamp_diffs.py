#!/usr/bin/env python3
"""
HDF5文件时间戳差异可视化工具

该脚本用于读取HDF5文件，提取每个obs中的timestamps中的start时间戳，
计算相邻obs之间的时间差，并可视化时间差图像。
"""

import h5py
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from typing import List, Optional


def find_timestamps_in_hdf5(file_path: str) -> Optional[np.ndarray]:
    """
    在HDF5文件中查找timestamps数据
    
    Args:
        file_path: HDF5文件路径
    
    Returns:
        包含start时间戳的数组，如果未找到则返回None
    """
    with h5py.File(file_path, 'r') as f:
        # 尝试多种可能的位置
        possible_paths = [
            'observations/timestamps/start',
            'timestamps/start',
            'observations/timestamps',
            'timestamps',
        ]
        
        for path in possible_paths:
            if path in f:
                data = f[path]
                if isinstance(data, h5py.Dataset):
                    # 如果是数据集，直接读取
                    timestamps = data[:]
                    # 如果是一维数组，直接返回
                    if timestamps.ndim == 1:
                        return timestamps
                    # 如果是二维数组，可能是 (num_steps, num_timestamps)
                    # 尝试提取第一列（假设是start）
                    elif timestamps.ndim == 2:
                        return timestamps[:, 0]
                elif isinstance(data, h5py.Group):
                    # 如果是组，查找start字段
                    if 'start' in data:
                        return data['start'][:]
        
        # 如果以上路径都没找到，尝试遍历整个文件结构
        def search_recursive(group, path=""):
            """递归搜索timestamps"""
            for key in group.keys():
                current_path = f"{path}/{key}" if path else key
                item = group[key]
                
                if isinstance(item, h5py.Group):
                    # 如果是组，检查是否包含timestamps相关的键
                    if 'start' in item or 'timestamps' in key.lower():
                        if 'start' in item:
                            return item['start'][:]
                    # 递归搜索
                    result = search_recursive(item, current_path)
                    if result is not None:
                        return result
                elif isinstance(item, h5py.Dataset):
                    # 检查数据集名称是否包含timestamp
                    if 'timestamp' in key.lower() and 'start' in key.lower():
                        return item[:]
            
            return None
        
        result = search_recursive(f)
        if result is not None:
            return result
    
    return None


def extract_timestamps_from_obs_structure(file_path: str) -> Optional[np.ndarray]:
    """
    从obs结构中提取timestamps（如果数据是按obs结构存储的）
    
    Args:
        file_path: HDF5文件路径
    
    Returns:
        包含start时间戳的数组
    """
    with h5py.File(file_path, 'r') as f:
        # 检查是否有observations组
        if 'observations' not in f:
            return None
        
        obs_group = f['observations']
        
        # 检查是否有timestamps组
        if 'timestamps' in obs_group:
            ts_group = obs_group['timestamps']
            if 'start' in ts_group:
                return ts_group['start'][:]
        
        # 如果没有timestamps组，检查是否有其他结构
        # 可能timestamps是作为属性存储的，或者在其他位置
        return None


def extract_all_timestamps(file_path: str) -> Optional[dict]:
    """
    从HDF5文件中提取所有timestamps字段
    
    Args:
        file_path: HDF5文件路径
    
    Returns:
        包含所有时间戳字段的字典，如果未找到则返回None
    """
    with h5py.File(file_path, 'r') as f:
        # 检查是否有observations/timestamps组
        if 'observations' in f and 'timestamps' in f['observations']:
            ts_group = f['observations']['timestamps']
            timestamps_dict = {}
            
            # 提取所有可用的时间戳字段
            required_fields = ['start', 'cam_high', 'cam_left_wrist', 'robot']
            for field in required_fields:
                if field in ts_group:
                    timestamps_dict[field] = ts_group[field][:]
                else:
                    print(f"警告：未找到timestamps字段 '{field}'")
                    return None
            
            return timestamps_dict
        
        return None


def extract_actions(file_path: str) -> Optional[np.ndarray]:
    """
    从HDF5文件中提取action数据
    
    Args:
        file_path: HDF5文件路径
    
    Returns:
        动作数组，形状为 (num_steps, action_dim)，如果未找到则返回None
    """
    with h5py.File(file_path, 'r') as f:
        group = '/observations/qpos'
        if group in f:
            actions = f[group][:]
            return actions
        
        return None


def calculate_timestamp_diffs(timestamps: np.ndarray) -> np.ndarray:
    """
    计算相邻时间戳之间的时间差
    
    Args:
        timestamps: 时间戳数组
    
    Returns:
        时间差数组（第一个元素为NaN或0，因为第一个obs没有前一个obs）
    """
    if len(timestamps) < 2:
        return np.array([])
    
    # 计算相邻时间戳的差值
    diffs = np.diff(timestamps)
    
    # 在开头插入0，使得diffs[i]表示step i和step i-1之间的时间差
    # 这样diffs[0] = 0（第一个step没有前一个step）
    diffs = np.insert(diffs, 0, 0.0)
    
    return diffs


def visualize_timestamp_diffs(
    timestamps: np.ndarray,
    diffs: np.ndarray,
    output_file: Optional[str] = None,
    show_plot: bool = True,
    stage_latencies: Optional[dict] = None,
    actions: Optional[np.ndarray] = None
) -> None:
    """
    可视化时间差图像
    
    Args:
        timestamps: 原始时间戳数组
        diffs: 时间差数组
        output_file: 输出图像文件路径（可选）
        show_plot: 是否显示图像
        stage_latencies: 各阶段延迟字典，包含 'start_to_cam_high', 'cam_high_to_cam_left_wrist', 'cam_left_wrist_to_robot'
        actions: 动作数组，形状为 (num_steps, action_dim)
    """
    steps = np.arange(len(diffs))
    
    # 计算需要的子图数量
    num_plots = 2  # 基础的两个图（时间差和累积时间戳）
    if stage_latencies is not None:
        num_plots += 2  # 阶段延迟的两个图
    if actions is not None:
        num_plots += 1  # 动作轨迹图
    
    # 根据子图数量决定布局
    if num_plots <= 2:
        fig, axes = plt.subplots(2, 1, figsize=(18, 14))
        ax_list = [axes[0], axes[1]]
    elif num_plots == 3:
        fig, axes = plt.subplots(3, 1, figsize=(18, 18))
        ax_list = list(axes)
    elif num_plots == 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        ax_list = axes.flatten()
    else:  # num_plots == 5
        fig, axes = plt.subplots(3, 2, figsize=(20, 20))
        ax_list = axes.flatten()
    
    ax1 = ax_list[0]
    ax2 = ax_list[1]
    plot_idx = 2  # 下一个可用的子图索引
    
    # 第一个子图：时间差
    ax1.plot(steps, diffs, 'b-', linewidth=2.5, marker='o', markersize=5)
    ax1.set_xlabel('Step', fontsize=16)
    ax1.set_ylabel('Time Difference (s)', fontsize=16)
    ax1.set_title('Time Difference Between Adjacent Observations', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, len(steps) - 0.5)
    ax1.tick_params(labelsize=14)
    
    # 添加统计信息
    if len(diffs) > 1:
        # 跳过第一个元素（0），因为它没有实际意义
        valid_diffs = diffs[1:]
        mean_diff = np.mean(valid_diffs)
        std_diff = np.std(valid_diffs)
        min_diff = np.min(valid_diffs)
        max_diff = np.max(valid_diffs)
        
        stats_text = f'Mean: {mean_diff:.4f}s\nStd: {std_diff:.4f}s\nMin: {min_diff:.4f}s\nMax: {max_diff:.4f}s'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=14)
    
    # 第二个子图：原始时间戳（可选，用于参考）
    if len(timestamps) > 0:
        # 将时间戳转换为相对于第一个时间戳的时间
        relative_timestamps = timestamps - timestamps[0]
        ax2.plot(steps, relative_timestamps, 'g-', linewidth=2.5, marker='s', markersize=5)
        ax2.set_xlabel('Step', fontsize=16)
        ax2.set_ylabel('Relative Time (s)', fontsize=16)
        ax2.set_title('Cumulative Timestamps (Relative to First Observation)', fontsize=18, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.5, len(steps) - 0.5)
        ax2.tick_params(labelsize=14)
    
    # 如果有阶段延迟数据，添加额外的子图
    if stage_latencies is not None:
        # 第三个子图：start到cam_high的延迟
        if 'start_to_cam_high' in stage_latencies:
            ax3 = ax_list[plot_idx]
            plot_idx += 1
            latencies = stage_latencies['start_to_cam_high']
            ax3.plot(steps, latencies * 1000, 'r-', linewidth=2.5, marker='o', markersize=5, label='Start to Cam High')
            ax3.set_xlabel('Step', fontsize=16)
            ax3.set_ylabel('Latency (ms)', fontsize=16)
            ax3.set_title('Latency: Start to Cam High', fontsize=18, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(-0.5, len(steps) - 0.5)
            ax3.tick_params(labelsize=14)
            
            # 添加统计信息
            if len(latencies) > 0:
                mean_lat = np.mean(latencies) * 1000
                std_lat = np.std(latencies) * 1000
                min_lat = np.min(latencies) * 1000
                max_lat = np.max(latencies) * 1000
                stats_text = f'Mean: {mean_lat:.2f}ms\nStd: {std_lat:.2f}ms\nMin: {min_lat:.2f}ms\nMax: {max_lat:.2f}ms'
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                        fontsize=14)
        
        # 第四个子图：cam_high到cam_left_wrist和cam_left_wrist到robot的延迟
        if 'cam_high_to_cam_left_wrist' in stage_latencies and 'cam_left_wrist_to_robot' in stage_latencies:
            ax4 = ax_list[plot_idx]
            plot_idx += 1
            lat1 = stage_latencies['cam_high_to_cam_left_wrist'] * 1000
            lat2 = stage_latencies['cam_left_wrist_to_robot'] * 1000
            ax4.plot(steps, lat1, 'm-', linewidth=2.5, marker='s', markersize=5, label='Cam High to Cam Left Wrist')
            ax4.plot(steps, lat2, 'c-', linewidth=2.5, marker='^', markersize=5, label='Cam Left Wrist to Robot')
            ax4.set_xlabel('Step', fontsize=16)
            ax4.set_ylabel('Latency (ms)', fontsize=16)
            ax4.set_title('Latency: Camera Pipeline', fontsize=18, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(-0.5, len(steps) - 0.5)
            ax4.tick_params(labelsize=14)
            ax4.legend(fontsize=14)
            
            # 添加统计信息
            if len(lat1) > 0 and len(lat2) > 0:
                mean_lat1 = np.mean(lat1)
                mean_lat2 = np.mean(lat2)
                stats_text = f'Cam High→Wrist:\n  Mean: {mean_lat1:.2f}ms\n\nWrist→Robot:\n  Mean: {mean_lat2:.2f}ms'
                ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                        fontsize=12)
    
    # 如果有动作数据，添加动作维度可视化
    if actions is not None:
        ax_actions = ax_list[plot_idx]
        action_steps = np.arange(len(actions))
        action_dim = actions.shape[1]
        
        # 为每个动作维度绘制轨迹
        colors = plt.cm.tab10(np.linspace(0, 1, min(action_dim, 10)))
        for dim in range(action_dim):
            color = colors[dim % len(colors)]
            label = f'Dim {dim}'
            # 如果是常见的7维动作（6关节+1gripper），使用更友好的标签
            if action_dim == 7:
                labels = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Gripper']
                label = labels[dim]
            ax_actions.plot(action_steps, actions[:, dim], '-', linewidth=2, 
                          color=color, label=label, alpha=0.8)
        
        ax_actions.set_xlabel('Step', fontsize=16)
        ax_actions.set_ylabel('Action Value', fontsize=16)
        ax_actions.set_title(f'Action Trajectory ({action_dim} dimensions)', fontsize=18, fontweight='bold')
        ax_actions.grid(True, alpha=0.3)
        ax_actions.set_xlim(-0.5, len(action_steps) - 0.5)
        ax_actions.tick_params(labelsize=14)
        ax_actions.legend(fontsize=12, ncol=2 if action_dim <= 7 else 3, loc='upper right')
        plot_idx += 1  # 动作图已使用
    
    # 隐藏多余的子图（从当前plot_idx开始到列表末尾）
    for i in range(plot_idx, len(ax_list)):
        ax_list[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图像
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {output_file}")
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_hdf5_timestamps(file_path: str, output_image: Optional[str] = None, show_plot: bool = True) -> bool:
    """
    分析HDF5文件中的时间戳并可视化
    
    Args:
        file_path: HDF5文件路径
        output_image: 输出图像路径（可选）
        show_plot: 是否显示图像
    
    Returns:
        是否成功分析
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return False
    
    print(f"正在读取文件: {file_path}")
    
    # 尝试提取action数据
    actions = extract_actions(file_path)
    if actions is not None:
        print(f"找到动作数据: {actions.shape[0]} 步, {actions.shape[1]} 维")
    else:
        print("未找到动作数据")
    
    # 尝试提取所有timestamps字段
    all_timestamps = extract_all_timestamps(file_path)
    
    # 如果成功提取所有timestamps，使用它们
    if all_timestamps is not None:
        timestamps = all_timestamps['start']
        
        # 计算各阶段延迟
        stage_latencies = {}
        stage_latencies['start_to_cam_high'] = all_timestamps['cam_high'] - all_timestamps['start']
        stage_latencies['cam_high_to_cam_left_wrist'] = all_timestamps['cam_left_wrist'] - all_timestamps['cam_high']
        stage_latencies['cam_left_wrist_to_robot'] = all_timestamps['robot'] - all_timestamps['cam_left_wrist']
        
        print(f"找到 {len(timestamps)} 个时间戳")
        print(f"时间戳范围: {timestamps[0]:.6f} 到 {timestamps[-1]:.6f}")
        print(f"总时长: {timestamps[-1] - timestamps[0]:.6f} 秒")
        
        # 打印阶段延迟统计
        print(f"\n阶段延迟统计:")
        print(f"  Start → Cam High:")
        print(f"    平均值: {np.mean(stage_latencies['start_to_cam_high']) * 1000:.2f} ms")
        print(f"    标准差: {np.std(stage_latencies['start_to_cam_high']) * 1000:.2f} ms")
        print(f"    最小值: {np.min(stage_latencies['start_to_cam_high']) * 1000:.2f} ms")
        print(f"    最大值: {np.max(stage_latencies['start_to_cam_high']) * 1000:.2f} ms")
        
        print(f"  Cam High → Cam Left Wrist:")
        print(f"    平均值: {np.mean(stage_latencies['cam_high_to_cam_left_wrist']) * 1000:.2f} ms")
        print(f"    标准差: {np.std(stage_latencies['cam_high_to_cam_left_wrist']) * 1000:.2f} ms")
        print(f"    最小值: {np.min(stage_latencies['cam_high_to_cam_left_wrist']) * 1000:.2f} ms")
        print(f"    最大值: {np.max(stage_latencies['cam_high_to_cam_left_wrist']) * 1000:.2f} ms")
        
        print(f"  Cam Left Wrist → Robot:")
        print(f"    平均值: {np.mean(stage_latencies['cam_left_wrist_to_robot']) * 1000:.2f} ms")
        print(f"    标准差: {np.std(stage_latencies['cam_left_wrist_to_robot']) * 1000:.2f} ms")
        print(f"    最小值: {np.min(stage_latencies['cam_left_wrist_to_robot']) * 1000:.2f} ms")
        print(f"    最大值: {np.max(stage_latencies['cam_left_wrist_to_robot']) * 1000:.2f} ms")
    else:
        # 回退到只提取start时间戳
        timestamps = find_timestamps_in_hdf5(file_path)
        
        if timestamps is None:
            # 如果没找到，尝试从obs结构提取
            timestamps = extract_timestamps_from_obs_structure(file_path)
        
        if timestamps is None:
            print("错误：无法在HDF5文件中找到timestamps数据")
            print("\n尝试查找文件结构...")
            with h5py.File(file_path, 'r') as f:
                def print_structure(name, obj):
                    print(f"  {name}: {type(obj).__name__}")
                f.visititems(print_structure)
            return False
        
        stage_latencies = None
        print(f"找到 {len(timestamps)} 个时间戳")
        print(f"时间戳范围: {timestamps[0]:.6f} 到 {timestamps[-1]:.6f}")
        print(f"总时长: {timestamps[-1] - timestamps[0]:.6f} 秒")
    
    # 计算时间差
    diffs = calculate_timestamp_diffs(timestamps)
    
    if len(diffs) == 0:
        print("错误：时间戳数据不足，无法计算时间差")
        return False
    
    # 打印一些统计信息
    if len(diffs) > 1:
        valid_diffs = diffs[1:]  # 跳过第一个0
        print(f"\n相邻Obs时间差统计:")
        print(f"  平均值: {np.mean(valid_diffs):.6f} 秒")
        print(f"  标准差: {np.std(valid_diffs):.6f} 秒")
        print(f"  最小值: {np.min(valid_diffs):.6f} 秒")
        print(f"  最大值: {np.max(valid_diffs):.6f} 秒")
        print(f"  中位数: {np.median(valid_diffs):.6f} 秒")
    
    # 可视化
    visualize_timestamp_diffs(timestamps, diffs, output_image, show_plot, stage_latencies, actions)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="可视化HDF5文件中相邻obs之间的时间戳差异",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 可视化单个文件
  python visualize_timestamp_diffs.py -f episode_001.hdf5
  
  # 保存图像到文件
  python visualize_timestamp_diffs.py -f episode_001.hdf5 -o timestamp_diffs.png
  
  # 不显示图像，只保存
  python visualize_timestamp_diffs.py -f episode_001.hdf5 -o timestamp_diffs.png --no-show
        """
    )
    
    parser.add_argument('-f', '--file', 
                       required=True,
                       help='要分析的HDF5文件路径')
    parser.add_argument('-o', '--output',
                       help='输出图像文件路径（可选）')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图像，仅保存（如果指定了输出文件）')
    
    args = parser.parse_args()
    
    show_plot = not args.no_show
    
    success = analyze_hdf5_timestamps(
        args.file,
        args.output,
        show_plot
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()

