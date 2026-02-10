#!/usr/bin/env python3
"""
A script to collect robot interaction data using the PiperEnvironment.

Usage:
- Run the script. A window will appear showing the camera feeds.
- Press 's' to start recording an episode.
- Move the robot arm or interact with the environment.
- Press 'q' to stop recording. The episode will be saved to a unique HDF5 file.
- Press Ctrl+C in the terminal or 'ESC' in the window to exit the program.
"""
import time
import numpy as np
import cv2
import h5py
from datetime import datetime
import os
import argparse

from env import PiperEnvironment

PROMPT_SET = {

    # pick 
    'pick': 
    [
        "pick up the red cube",
        "pick up the yellow cube",
        "pick up the blue cube",
        "pick up the orange cube",
        "pick up the green cube",
    ],
    # pick and place
    'pick_and_place': 
    [
        "pick up red cubes and put it in the black box",
        "pick up blue cubes and put it in the black box",
        "pick up yellow cubes and put it in the black box",
        "pick up blue cubes and put it in the brown box",
        "pick up yellow cubes and put it in the brown box",
        "pick up orange cubes and put it in the brown box",
        "pick up yellow cubes and put it in the white box",
        "pick up orange cubes and put it in the white box",
        "pick up green cubes and put it in the white box",
    ],
    # push
    'push': ["push the red cube away from yourself",
    "push the yellow cube away from yourself",
    "push the blue cube away from yourself",
    "pull the blue cube close to yourself",
    "pull the orange cube close to yourself",
    "pull the green cube close to yourself",
    ],
    # sort
    'sort': ["sort the cubes by color",
    "sort the red cubes",
    "sort the yellow cubes",
    "sort the blue cubes",
    "sort the green cubes",
    "sort the orange cubes",
    "sort the orange and blue cubes",
    "sort the red and yellow cubes",
    ],
    # clean
    'clean': ["remove the cubes to clean the table",],
    # count
    'count': [
    "What is 1 plus 1?",
    "What is 2 plus 1?",
    "What is 3 plus 2?",
    "What is 3 minus 1?",
    "What is 4 minus 3?",
    "What is 5 minus 2?",
    ],
    # build tower
    'build': [
    "build a tower with the cubes, the color sequence is yellow, blue",
    "build a tower with the cubes, the color sequence is red, green, orange",
    "put the yellow cube on the red cube",
    "put the blue cube on the green cube",
    "put the orange cube on the blue cube",
    ],
}

PACKAGE_PROMPT_SET = {
    'pack': [
        "Repeatedly pack the camera kits into the box and push them to the front",
        "For each empty packaging box, bring the box to the center of the table, then place one card reader in the right side of the box, one battery in the right side of the box and left side of the card reader, one cable in the bottom-left corner of the box, one camera in the top-left corner of the box, and finally push the box to the front"
    ],
    'pick': [
        "pick up the card reader",
        "pick up the battery",
        "pick up the cable",
        "pick up the camera",
    ],
    'pick_and_place': [
        "pick up card readers and put them in the box",
        "pick up batteries and put them in the box",
        "pick up cables and put them in the box",
        "pick up cameras and put them in the box",
        "place one card reader in the right side of the box",
        "place one battery in the right side of the box and left side of the card reader",
        "place one cable in the bottom-left corner of the box",
        "place one camera in the top-left corner of the box",
    ],
    'push': [
        "push the box away from yourself",
        "push the box to the front",
    ],
    'bring':[
        "bring the box to the center of the table",
        "Process the boxes one by one: bring each box to the center of the table, then push it to the front",
    ]
}

PICK_AND_PLACE_PROMPT_SET = {
    'pick_and_place': [
        "pick up the camera and put them in the box",
        "pick up the battery and put them in the box",
        "pick up the tape and put them in the box",
        "pick up anything and put them in the box",
    ],
}

def display_observations(obs: dict, window_name: str = "Observations", recording: bool = False):
    """Display all images from observation dict in a single window."""
    images_dict = obs.get('images', {})
    if not images_dict:
        return

    display_images = []
    for cam_name, img in images_dict.items():
        img_hwc = np.transpose(img, (1, 2, 0))
        img_bgr = img_hwc[:, :, ::-1].copy()
        
        label_color = (0, 0, 255) if recording else (0, 255, 0)
        label_text = f"{cam_name} (RECORDING)" if recording else cam_name
        cv2.putText(img_bgr, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
        
        display_images.append(img_bgr)
    
    combined = np.hstack(display_images) if len(display_images) > 1 else display_images[0]
    
    # 在底部添加prompt信息
    prompt = obs.get('prompt', '')
    if prompt:
        # 创建一个底部横幅来显示prompt
        banner_height = 60
        banner = np.zeros((banner_height, combined.shape[1], 3), dtype=np.uint8)
        # 使用较小的字体显示prompt
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)
        cv2.putText(banner, f"Task: {prompt}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, text_color, thickness)
        # 将banner添加到combined图像的底部
        combined = np.vstack([combined, banner])
    
    cv2.imshow(window_name, combined)
    return cv2.waitKey(1) & 0xFF

def save_episode_to_hdf5(episode_data: list, directory: str = "recorded_data", prompt: str = ""):
    """
    Saves a list of observation dictionaries to an HDF5 file with a structure
    compatible with the convert_piper_data_to_lerobot.py script.
    """
    if not episode_data:
        print("No data to save.")
        return

    # Ensure the save directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Generate a unique filename with prompt
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 将prompt转换为文件名安全的格式
    if prompt:
        # 将空格替换为下划线，移除或替换特殊字符
        safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_prompt = safe_prompt.replace(",", "").replace(".", "")
        # 限制长度（避免文件名过长）
        max_prompt_len = 80
        if len(safe_prompt) > max_prompt_len:
            safe_prompt = safe_prompt[:max_prompt_len]
        filename = os.path.join(directory, f"episode_{timestamp_str}_{safe_prompt}.hdf5")
    else:
        filename = os.path.join(directory, f"episode_{timestamp_str}.hdf5")
    
    print(f"Saving episode to {filename}...")

    with h5py.File(filename, "w") as f:
        action_chunk_size = 50
        
        # Pre-allocate numpy arrays by inspecting the first observation
        first_obs = episode_data[0]
        num_steps = len(episode_data)
        
        # Create datasets based on the expected structure
        obs_group = f.create_group("observations")
        img_group = obs_group.create_group("images")
        
        state_shape = first_obs['state'].shape
        obs_group.create_dataset("qpos", (num_steps,) + state_shape, dtype=first_obs['state'].dtype)
        
        # # Action is a chunk of future states
        # action_shape = (num_steps, action_chunk_size) + state_shape
        # f.create_dataset("action", action_shape, dtype=first_obs['state'].dtype)
        f.create_dataset("action", (num_steps,) + state_shape, dtype=first_obs['state'].dtype)

        for cam_name, img in first_obs['images'].items():
            img_shape = img.shape
            img_group.create_dataset(cam_name, (num_steps,) + img_shape, dtype=img.dtype)
            
        # Save the instruction prompt
        if 'prompt' in first_obs:
            prompt_len = len(first_obs['prompt'])
            fix_prompt_type = np.dtype(f'S{prompt_len}')
            f.create_dataset("task", (num_steps,), dtype=fix_prompt_type)

        # Save timestamps if available
        timestamps_group = None
        if 'timestamps' in first_obs and first_obs['timestamps']:
            timestamps_group = obs_group.create_group("timestamps")
            # Create datasets for each timestamp field
            for ts_key in first_obs['timestamps'].keys():
                timestamps_group.create_dataset(ts_key, (num_steps,), dtype=np.float64)

        # Populate the datasets
        for i in range(num_steps):
            # Populate observations
            obs_group['qpos'][i] = episode_data[i]['state']

            action_index = min(i+1, num_steps - 1)
            f['action'][i] = episode_data[action_index]['state']

            for cam_name, img in episode_data[i]['images'].items():
                img_group[cam_name][i] = img
            if 'prompt' in first_obs:
                f['task'][i] = episode_data[i]['prompt']
            
            # Save timestamps
            if timestamps_group is not None and 'timestamps' in episode_data[i]:
                for ts_key, ts_value in episode_data[i]['timestamps'].items():
                    if ts_key in timestamps_group:
                        timestamps_group[ts_key][i] = ts_value

            # # Populate actions (chunk of future states)
            # action_chunk = []
            # for j in range(action_chunk_size):
            #     # Find the index of the future state
            #     future_index = min(i + j, num_steps - 1)
            #     action_chunk.append(episode_data[future_index]['state'])
            
            # f['action'][i] = np.array(action_chunk)
                
    print(f"Successfully saved {num_steps} steps.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Piper数据采集脚本")
    
    parser.add_argument(
        "--task_type",
        type=str,
        required=False,
        choices=list(PROMPT_SET.keys()),
        help=f"任务类型，可选: {', '.join(PROMPT_SET.keys())}（如果提供了--prompt参数则此参数可选）"
    )
    
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=0,
        help="该任务类型中prompt的索引（从0开始）"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="直接指定prompt文本，如果提供此参数则忽略task_type和prompt_index"
    )
    
    parser.add_argument(
        "--can_port",
        type=str,
        default="can0",
        help="CAN总线端口名称（默认: can0）"
    )
    
    
    parser.add_argument(
        "--camera_fps",
        type=int,
        default=30,
        help="相机帧率（默认: 30）"
    )

    parser.add_argument(
        "--high_camera_id",
        type=int,
        default=6,
        help="全局相机ID（默认: 0）"
    )

    parser.add_argument(
        "--left_wrist_camera_id",
        type=int,
        default=4,
        help="手腕相机ID（默认: 6）"
    )
    
    parser.add_argument(
        "--tele_mode",
        action="store_true",
        default=True,
        help="记录模式（禁用机器人运动，默认: True）"
    )
    
    return parser.parse_args()

def main():
    """Main data collection loop."""
    args = parse_args()
    
    # 决策逻辑：优先使用--prompt参数，否则从PROMPT_SET中选取
    if args.prompt:
        # 如果提供了prompt参数，直接使用
        prompt = args.prompt
        task_type = "custom"
        print("=== Piper Data Collector ===\n")
        print(f"使用自定义Prompt: {prompt}\n")
    elif args.task_type:
        # 如果提供了task_type，从PROMPT_SET中获取
        prompts_list = PROMPT_SET[args.task_type]
        if args.prompt_index >= len(prompts_list):
            print(f"错误: prompt_index {args.prompt_index} 超出范围，任务 '{args.task_type}' 只有 {len(prompts_list)} 个prompt")
            print(f"可用的prompts:")
            for i, p in enumerate(prompts_list):
                print(f"  [{i}] {p}")
            return
        
        prompt = prompts_list[args.prompt_index]
        task_type = args.task_type
        print("=== Piper Data Collector ===\n")
        print(f"任务类型: {task_type}")
        print(f"Prompt: {prompt}\n")
    else:
        # 两个参数都没提供，报错
        print("错误: 必须提供 --prompt 参数或 --task_type 参数")
        print("\n使用方式:")
        print("  方式1: 直接指定prompt文本")
        print("    python collect_data.py --prompt \"你的任务描述\"")
        print("\n  方式2: 从预设任务中选择")
        print(f"    python collect_data.py --task_type <任务类型> [--prompt_index <索引>]")
        print(f"\n可用的任务类型: {', '.join(PROMPT_SET.keys())}")
        return

    env = PiperEnvironment(
        can_port=args.can_port,
        camera_fps=args.camera_fps,
        max_episode_steps=10000000,
        tele_mode=args.tele_mode,
        high_camera_id=args.high_camera_id,
        left_wrist_camera_id=args.left_wrist_camera_id,
        prompt=prompt  # 将prompt传递给环境
    )
    
    cv2.namedWindow("Observations", cv2.WINDOW_AUTOSIZE)
    
    episode_data = []
    recording = False
    
    try:
        print("Initializing environment and cameras...")
        env.reset()
        print("\nReady to record. Press 's' to start, 'q' to stop, ESC to quit.")
        
        while True:
            start_time = time.time()
            obs = env.get_observation()
            key = display_observations(obs, recording=recording)

            if key == 27:  # ESC key
                break
            elif key == ord('s'):
                if not recording:
                    print("Starting recording...")
                    recording = True
                    episode_data = []
                else:
                    print("Already recording.")
            elif key == ord('q'):
                if recording:
                    print("Stopping recording...")
                    recording = False
                    save_episode_to_hdf5(episode_data, prompt=prompt)
                else:
                    print("Not currently recording.")

            if recording:
                episode_data.append(obs)
                print(f"Recording... Steps: {len(episode_data)}", end='\r')

            # The loop naturally runs based on the environment's update rate
            cost_time = time.time() - start_time
            if cost_time < 1.0 / env._update_rate:
                time.sleep(1.0 / env._update_rate - cost_time)


    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        if recording and episode_data:
            print("Saving partially recorded episode...")
            save_episode_to_hdf5(episode_data, prompt=prompt)
            
    finally:
        print("\nCleaning up...")
        env.close()
        cv2.destroyAllWindows()
        print("Program finished.")

if __name__ == "__main__":
    main()
