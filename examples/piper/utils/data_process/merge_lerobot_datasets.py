#!/usr/bin/env python3
"""
脚本用于合并两个LeRobot数据集为一个新的数据集

Example usage:
    python merge_lerobot_datasets.py \
        --dataset1 /path/to/dataset1 \
        --dataset2 /path/to/dataset2 \
        --output /path/to/merged_dataset \
        --output-repo-id merged_dataset_name

功能说明：
- 将两个LeRobot数据集的所有episodes合并到一个新的数据集中
- 自动保留所有原始的observations、actions和metadata
- 支持视频和图像两种模式
- 保持原始的task标签和其他属性
"""

import argparse
import shutil
from pathlib import Path
from typing import Literal
import os

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tqdm


# 设置LeRobot默认路径
if os.getenv("HF_LEROBOT_HOME") is None:
    os.environ["HF_LEROBOT_HOME"] = str(Path.home() / ".cache" / "huggingface" / "lerobot")
LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME"))


def validate_datasets_compatibility(dataset1: LeRobotDataset, dataset2: LeRobotDataset) -> tuple[bool, str]:
    """
    验证两个数据集是否兼容，可以合并
    
    Returns:
        (is_compatible, message): 兼容性检查结果和消息
    """
    # 检查FPS是否一致
    if dataset1.fps != dataset2.fps:
        return False, f"FPS不匹配: dataset1={dataset1.fps}, dataset2={dataset2.fps}"
    
    # 检查robot_type是否一致
    if dataset1.meta.robot_type != dataset2.meta.robot_type:
        return False, f"机器人类型不匹配: dataset1={dataset1.meta.robot_type}, dataset2={dataset2.meta.robot_type}"
    
    # 检查特征是否一致
    features1 = set(dataset1.hf_dataset.column_names)
    features2 = set(dataset2.hf_dataset.column_names)
    
    if features1 != features2:
        missing_in_1 = features2 - features1
        missing_in_2 = features1 - features2
        msg = "特征不匹配:\n"
        if missing_in_1:
            msg += f"  dataset1缺少: {missing_in_1}\n"
        if missing_in_2:
            msg += f"  dataset2缺少: {missing_in_2}\n"
        return False, msg
    
    # 检查每个特征的shape是否一致（排除batch维度）
    for feature in features1:
        if feature in ["index", "episode_index", "frame_index", "timestamp", "task_index"]:
            continue  # 这些是metadata，不需要检查shape
        
        try:
            sample1 = dataset1.hf_dataset[0][feature]
            sample2 = dataset2.hf_dataset[0][feature]
            
            if hasattr(sample1, 'shape') and hasattr(sample2, 'shape'):
                if sample1.shape != sample2.shape:
                    return False, f"特征'{feature}'的shape不匹配: {sample1.shape} vs {sample2.shape}"
        except Exception as e:
            print(f"警告: 检查特征'{feature}'时出错: {e}")
    
    return True, "数据集兼容"


def create_merged_dataset(
    dataset1: LeRobotDataset,
    dataset2: LeRobotDataset,
    output_repo_id: str,
    mode: Literal["video", "image"] = "video",
) -> LeRobotDataset:
    """
    创建一个新的空数据集，配置与源数据集一致
    """
    # 获取features配置（从dataset1）
    features = {}
    
    # 获取所有列名
    column_names = dataset1.hf_dataset.column_names
    
    # 排除元数据列
    metadata_cols = ["index", "episode_index", "frame_index", "timestamp", "task_index"]
    
    for col in column_names:
        if col in metadata_cols:
            continue
        
        # 获取第一帧的样本
        sample = dataset1.hf_dataset[0][col]
        
        if hasattr(sample, 'shape') and hasattr(sample, 'dtype'):
            # 处理图像特征
            if "images" in col:
                features[col] = {
                    "dtype": mode,
                    "shape": tuple(sample.shape),
                    "names": ["channels", "height", "width"],
                }
            else:
                # 处理其他张量特征（state, action, velocity, effort等）
                features[col] = {
                    "dtype": str(sample.dtype).replace("torch.", ""),
                    "shape": tuple(sample.shape),
                    "names": None,
                }
    
    # 删除已存在的输出目录
    output_path = LEROBOT_HOME / output_repo_id
    if output_path.exists():
        print(f"⚠️  删除已存在的输出目录: {output_path}")
        shutil.rmtree(output_path)
    
    # 创建新数据集
    print(f"🔨 创建新数据集: {output_repo_id}")
    merged_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=dataset1.fps,
        robot_type=dataset1.meta.robot_type,
        features=features,
        use_videos=(mode == "video"),
    )
    
    return merged_dataset


def copy_episode_to_dataset(
    source_dataset: LeRobotDataset,
    target_dataset: LeRobotDataset,
    episode_idx: int,
) -> int:
    """
    将源数据集的指定episode复制到目标数据集
    
    Returns:
        复制的帧数
    """
    # 获取episode的帧索引范围
    episode_data_index = source_dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_idx].item()
    to_idx = episode_data_index["to"][episode_idx].item()
    
    # 复制每一帧
    for frame_idx in range(from_idx, to_idx):
        frame_data = {}
        
        # 获取所有特征
        for col in source_dataset.hf_dataset.column_names:
            if col in ["index", "episode_index", "frame_index", "timestamp"]:
                continue  # 这些由target_dataset自动生成
            
            # task_index需要转换为task文本
            if col == "task_index":
                task_index = source_dataset.hf_dataset[frame_idx][col]
                # 从meta.tasks中获取实际的task文本
                task_text = source_dataset.meta.tasks.get(int(task_index))
                if task_text is not None:
                    # 添加task字段，save_episode会自动转换为task_index
                    frame_data["task"] = task_text
                continue
            
            value = source_dataset.hf_dataset[frame_idx][col]
            frame_data[col] = value
        
        target_dataset.add_frame(frame_data)
    
    # 保存episode
    target_dataset.save_episode()
    
    return to_idx - from_idx


def merge_datasets(
    dataset1_path: str,
    dataset2_path: str,
    output_repo_id: str,
    mode: Literal["video", "image"] = "video",
) -> LeRobotDataset:
    """
    合并两个LeRobot数据集
    
    Args:
        dataset1_path: 第一个数据集路径
        dataset2_path: 第二个数据集路径
        output_repo_id: 输出数据集的repo_id
        mode: 'video' 或 'image' 模式
    
    Returns:
        合并后的数据集
    """
    print("="*80)
    print("📦 加载数据集...")
    print("="*80)
    
    # 加载两个数据集
    print(f"加载数据集1: {dataset1_path}")
    dataset1 = LeRobotDataset(dataset1_path)
    print(f"  ✅ 已加载 {dataset1.num_episodes} episodes, {len(dataset1)} 帧")
    
    print(f"\n加载数据集2: {dataset2_path}")
    dataset2 = LeRobotDataset(dataset2_path)
    print(f"  ✅ 已加载 {dataset2.num_episodes} episodes, {len(dataset2)} 帧")
    
    # 验证兼容性
    print("\n" + "="*80)
    print("🔍 验证数据集兼容性...")
    print("="*80)
    
    is_compatible, message = validate_datasets_compatibility(dataset1, dataset2)
    if not is_compatible:
        raise ValueError(f"数据集不兼容: {message}")
    print(f"  ✅ {message}")
    
    # 创建合并后的数据集
    print("\n" + "="*80)
    print("🔨 创建合并数据集...")
    print("="*80)
    
    merged_dataset = create_merged_dataset(dataset1, dataset2, output_repo_id, mode)
    
    # 复制dataset1的所有episodes
    print("\n" + "="*80)
    print(f"📋 复制数据集1的episodes...")
    print("="*80)
    
    total_frames = 0
    for ep_idx in tqdm.tqdm(range(dataset1.num_episodes), desc="Dataset1"):
        num_frames = copy_episode_to_dataset(dataset1, merged_dataset, ep_idx)
        total_frames += num_frames
    
    print(f"  ✅ 已复制 {dataset1.num_episodes} episodes, {total_frames} 帧")
    
    # 复制dataset2的所有episodes
    print("\n" + "="*80)
    print(f"📋 复制数据集2的episodes...")
    print("="*80)
    
    total_frames = 0
    for ep_idx in tqdm.tqdm(range(dataset2.num_episodes), desc="Dataset2"):
        num_frames = copy_episode_to_dataset(dataset2, merged_dataset, ep_idx)
        total_frames += num_frames
    
    print(f"  ✅ 已复制 {dataset2.num_episodes} episodes, {total_frames} 帧")
    
    # 打印合并结果
    print("\n" + "="*80)
    print("✨ 合并完成！")
    print("="*80)
    print(f"总Episodes数: {merged_dataset.num_episodes}")
    print(f"  - 来自数据集1: {dataset1.num_episodes}")
    print(f"  - 来自数据集2: {dataset2.num_episodes}")
    print(f"总帧数: {len(merged_dataset)}")
    print(f"  - 来自数据集1: {len(dataset1)}")
    print(f"  - 来自数据集2: {len(dataset2)}")
    print(f"保存位置: {LEROBOT_HOME / output_repo_id}")
    print("="*80)
    
    return merged_dataset


def main():
    parser = argparse.ArgumentParser(
        description="合并两个LeRobot数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 合并两个数据集（video模式）
  python merge_lerobot_datasets.py \\
      --dataset1 /path/to/dataset1 \\
      --dataset2 /path/to/dataset2 \\
      --output-repo-id merged_dataset \\
      --mode video
  
  # 合并两个数据集（image模式）
  python merge_lerobot_datasets.py \\
      --dataset1 /path/to/dataset1 \\
      --dataset2 /path/to/dataset2 \\
      --output-repo-id merged_dataset \\
      --mode image
        """
    )
    
    parser.add_argument(
        "--dataset1",
        type=str,
        required=True,
        help="第一个数据集的路径"
    )
    
    parser.add_argument(
        "--dataset2",
        type=str,
        required=True,
        help="第二个数据集的路径"
    )
    
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="输出数据集的repo_id（名称）"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="video",
        choices=["video", "image"],
        help="数据集模式: video（使用视频压缩）或 image（使用独立图像）"
    )
    
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="合并后是否consolidate数据集（可能需要较长时间）"
    )
    
    args = parser.parse_args()
    
    # 执行合并
    try:
        merged_dataset = merge_datasets(
            args.dataset1,
            args.dataset2,
            args.output_repo_id,
            args.mode,
        )
        
        # 可选的consolidate步骤
        if args.consolidate:
            print("\n" + "="*80)
            print("🔄 Consolidating数据集...")
            print("="*80)
            merged_dataset.consolidate()
            print("  ✅ Consolidate完成")
        
        print("\n✅ 所有操作完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
