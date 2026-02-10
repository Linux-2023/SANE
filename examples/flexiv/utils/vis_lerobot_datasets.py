#!/usr/bin/env python3
"""
脚本用于可视化LeRobot数据集并查看task信息
"""
import argparse
from lerobot.scripts.visualize_dataset import visualize_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def print_dataset_info(dataset: LeRobotDataset):
    """打印数据集的基本信息"""
    print(f"\n{'='*80}")
    print(f"📊 数据集基本信息")
    print(f"{'='*80}")
    print(f"总帧数: {len(dataset)}")
    print(f"总Episode数: {dataset.num_episodes}")
    print(f"帧率: {dataset.fps} fps")
    
    # 打印所有特征
    print(f"\n可用特征:")
    for key in dataset.hf_dataset.column_names:
        print(f"  - {key}")
    print()


def print_episode_info(dataset: LeRobotDataset, episode_index: int):
    """打印指定episode的详细信息"""
    if episode_index >= dataset.num_episodes:
        print(f"❌ 错误: episode_index {episode_index} 超出范围（总共 {dataset.num_episodes} 个episodes）")
        return False
    
    print(f"\n{'='*80}")
    print(f"📝 Episode {episode_index} 详细信息")
    print(f"{'='*80}")
    
    # 获取该episode的帧索引范围
    episode_data_index = dataset.episode_data_index
    from_idx = episode_data_index["from"][episode_index].item()
    to_idx = episode_data_index["to"][episode_index].item()
    num_frames = to_idx - from_idx
    
    print(f"帧索引范围: [{from_idx}, {to_idx})")
    print(f"包含帧数: {num_frames}")
    
    # 读取并打印task信息
    print(f"\n📋 Task信息:")
    if "task" in dataset.hf_dataset.column_names:
        try:
            # 获取第一帧的task
            first_frame_task = dataset.hf_dataset[from_idx]["task"]
            
            # 处理不同类型的task数据
            if isinstance(first_frame_task, bytes):
                first_frame_task = first_frame_task.decode('utf-8')
            
            print(f"  Task (第一帧): {first_frame_task}")
            
            # 检查episode中的task是否一致
            tasks_in_episode = set()
            sample_size = min(10, num_frames)  # 采样检查
            check_indices = [from_idx + i * (num_frames // sample_size) for i in range(sample_size)]
            
            for idx in check_indices:
                if idx < to_idx:
                    task = dataset.hf_dataset[idx]["task"]
                    if isinstance(task, bytes):
                        task = task.decode('utf-8')
                    tasks_in_episode.add(task)
            
            if len(tasks_in_episode) == 1:
                print(f"  ✅ Episode中所有帧的task一致")
            else:
                print(f"  ⚠️  Episode中发现 {len(tasks_in_episode)} 个不同的task:")
                for i, task in enumerate(tasks_in_episode):
                    print(f"    [{i+1}] {task}")
        
        except Exception as e:
            print(f"  ❌ 读取task时出错: {e}")
    else:
        print(f"  ⚠️  数据集不包含task字段")
    
    # 打印 advantage 信息
    print(f"\n💡 Advantage信息:")
    if "advantage" in dataset.hf_dataset.column_names:
        try:
            first_adv = dataset.hf_dataset[from_idx]["advantage"]
            if isinstance(first_adv, bytes):
                first_adv = first_adv.decode("utf-8")
            print(f"  Advantage (第一帧): {first_adv}")

            adv_in_episode = set()
            sample_size = min(10, num_frames)
            step = max(1, num_frames // sample_size) if sample_size else 1
            check_indices = [from_idx + i * step for i in range(sample_size)]

            for idx in check_indices:
                if idx < to_idx:
                    adv_value = dataset.hf_dataset[idx]["advantage"]
                    if isinstance(adv_value, bytes):
                        adv_value = adv_value.decode("utf-8")
                    adv_in_episode.add(adv_value)

            if len(adv_in_episode) == 1:
                print(f"  ✅ Episode中所有帧的advantage一致")
            else:
                print(f"  ⚠️  Episode中发现 {len(adv_in_episode)} 个不同的advantage:")
                for i, adv_value in enumerate(adv_in_episode):
                    print(f"    [{i+1}] {adv_value}")
        except Exception as e:
            print(f"  ❌ 读取advantage时出错: {e}")
    else:
        print(f"  ⚠️  数据集不包含advantage字段")
    
    # 打印其他观测信息
    print(f"\n🔍 观测数据形状:")
    try:
        first_frame = dataset[from_idx]
        for key, value in first_frame.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")
    except Exception as e:
        print(f"  ❌ 读取观测数据时出错: {e}")
    
    print(f"\n{'='*80}\n")
    return True


def print_all_episodes_tasks(dataset: LeRobotDataset):
    """打印所有episodes的task列表"""
    print(f"\n{'='*80}")
    print(f"📋 所有Episodes的Task列表")
    print(f"{'='*80}\n")
    
    if "task" not in dataset.hf_dataset.column_names:
        print("⚠️  数据集不包含task字段")
        return
    
    episode_data_index = dataset.episode_data_index
    task_counts = {}
    
    for ep_idx in range(dataset.num_episodes):
        from_idx = episode_data_index["from"][ep_idx].item()
        
        try:
            task = dataset.hf_dataset[from_idx]["task"]
            if isinstance(task, bytes):
                task = task.decode('utf-8')
            
            print(f"Episode {ep_idx:3d}: {task}")
            
            # 统计task出现次数
            task_counts[task] = task_counts.get(task, 0) + 1
        
        except Exception as e:
            print(f"Episode {ep_idx:3d}: ❌ 读取失败 - {e}")
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"📊 Task统计")
    print(f"{'='*80}")
    print(f"不同task数量: {len(task_counts)}")
    print(f"\nTask分布:")
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  [{count:3d}次] {task}")
    print(f"\n{'='*80}\n")


def interactive_session(
    dataset: LeRobotDataset,
    start_episode: int = 0,
    mode: str = "local",
    auto_visualize: bool = True,
) -> None:
    """交互式浏览所有 episodes。"""
    if dataset.num_episodes == 0:
        print("⚠️ 数据集中没有可用的 episodes。")
        return

    current_episode = max(0, min(start_episode, dataset.num_episodes - 1))

    instructions = """
============================ 交互模式说明 ============================
  - Enter/回车 或 n / next : 查看下一个 episode
  - p / prev               : 查看上一个 episode
  - <数字>                 : 跳转到指定 episode
  - v / view               : 仅可视化当前 episode
  - l / list               : 打印所有 episode 的 task
  - info                   : 重新打印数据集概览
  - q / quit / exit        : 退出交互模式
=====================================================================
"""
    print(instructions)

    while True:
        success = print_episode_info(dataset, current_episode)
        if success and auto_visualize:
            print(f"🎬 自动可视化 Episode {current_episode}...")
            visualize_dataset(dataset, episode_index=current_episode, mode=mode)

        try:
            cmd = input("请输入指令（回车=下一条，q退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️ 检测到中断，退出交互模式。")
            break

        if not cmd or cmd.lower() in {"n", "next"}:
            current_episode = min(current_episode + 1, dataset.num_episodes - 1)
            continue

        lower = cmd.lower()
        if lower in {"q", "quit", "exit"}:
            print("👋 退出交互模式。")
            break
        if lower in {"p", "prev", "previous"}:
            current_episode = max(current_episode - 1, 0)
            continue
        if lower in {"l", "list"}:
            print_all_episodes_tasks(dataset)
            continue
        if lower in {"info"}:
            print_dataset_info(dataset)
            continue
        if lower in {"v", "view"}:
            print(f"🎬 可视化 Episode {current_episode}...")
            visualize_dataset(dataset, episode_index=current_episode, mode=mode)
            continue

        # 尝试解析为 episode 编号
        try:
            idx = int(cmd)
            if 0 <= idx < dataset.num_episodes:
                current_episode = idx
            else:
                print(f"⚠️ Episode {idx} 超出范围（0 ~ {dataset.num_episodes - 1}）")
        except ValueError:
            print("⚠️ 无法识别的指令，请重新输入。")


def main():
    parser = argparse.ArgumentParser(description="可视化LeRobot数据集并查看task信息")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/workspace/pjk/ELM/openpi/datasets/piper_lerobot_data_pack",
        help="LeRobot数据集路径"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1,
        help="要可视化的episode索引（默认: 0）"
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="列出所有episodes的task"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="不启动可视化，只打印信息"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "distant"],
        help="可视化模式（默认: local)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互模式，一次加载后多次查看不同 episodes"
    )
    parser.add_argument(
        "--auto-visualize",
        action="store_true",
        help="交互模式下，每次切换 episode 自动调用 visualize_dataset"
    )
    
    args = parser.parse_args()
    
    # 加载数据集
    print(f"正在加载数据集: {args.dataset_path}")
    try:
        dataset = LeRobotDataset(args.dataset_path)
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return
    
    # 打印数据集基本信息
    print_dataset_info(dataset)
    
    # 如果请求列出所有episodes
    if args.list_all:
        print_all_episodes_tasks(dataset)
    
    # 交互模式
    if args.interactive:
        interactive_session(
            dataset,
            start_episode=args.episode,
            mode=args.mode,
            auto_visualize=args.auto_visualize and not args.no_visualize,
        )
        return

    # 打印指定episode的信息
    if print_episode_info(dataset, args.episode):
        # 启动可视化
        if not args.no_visualize:
            print(f"🎬 启动可视化 Episode {args.episode}...")
            visualize_dataset(dataset, episode_index=args.episode, mode=args.mode)


if __name__ == "__main__":
    main()