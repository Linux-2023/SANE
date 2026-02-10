"""
Script to convert Flexiv hdf5 data to the LeRobot dataset v2.0 format.

Example usage: 
  # 使用 qpos (关节角) 作为 state 和 action
  # repo_id 会自动添加 _qpos 后缀，最终为 <org>/<dataset-name>_qpos
  uv run examples/flexiv/utils/convert_flexiv_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name> --state-type qpos
  
  # 使用 pose (位姿) 作为 state 和 action
  # repo_id 会自动添加 _pose 后缀，最终为 <org>/<dataset-name>_pose
  uv run examples/flexiv/utils/convert_flexiv_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name> --state-type pose
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import gc

import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import os
import cv2

# 导入 quat2rotvec 转换函数
from data_process.quat2rotvec import dual_arm_quat_to_rotvec, fix_quaternion_discontinuity

if os.getenv("HF_LEROBOT_HOME") is None:
    os.environ["HF_LEROBOT_HOME"] = str(Path.home() / ".cache" / "huggingface" / "lerobot")
LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME"))


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    has_advantage: bool = False,
    has_prev_state: bool = False,
    state_type: Literal["qpos", "pose"] = "qpos",
    state_shape: tuple[int, ...] | None = None,
    cameras: list[str] | None = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        # Left arm (7 joints) + left gripper (1)
        "left_joint_0",
        "left_joint_1",
        "left_joint_2",
        "left_joint_3",
        "left_joint_4",
        "left_joint_5",
        "left_joint_6",
        "left_gripper",
        # Right arm (7 joints) + right gripper (1)
        "right_joint_0",
        "right_joint_1",
        "right_joint_2",
        "right_joint_3",
        "right_joint_4",
        "right_joint_5",
        "right_joint_6",
        "right_gripper",
    ]
    
    # 如果指定了 state_shape，使用它；否则使用 motors 的长度
    if state_shape is None:
        state_shape = (len(motors),)
    
    # 根据 state_type 决定 names
    if state_type == "qpos":
        state_names = [motors]
    else:  # pose
        # 对于 pose，使用通用的名称
        state_names = None  # 或者可以创建 pose 相关的名称
    
    # 如果没有指定 cameras，使用默认值
    if cameras is None:
        cameras = [
            "cam_left_wrist",
            "cam_right_wrist",
        ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }
    
    if has_advantage:
        features["advantage"] = {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        }
    
    if has_prev_state:
        features["observation.prev_state"] = {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 224, 224),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep

def has_advantage(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/advantage" in ep

def has_prev_pose(hdf5_files: list[Path]) -> bool:
    """检查 hdf5 文件中是否存在 /observations/prev_pose"""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/prev_pose" in ep

def get_state_shape(hdf5_files: list[Path], state_type: Literal["qpos", "pose"]) -> tuple[int, ...]:
    """
    获取 state 的 shape（硬编码维度）
    
    Args:
        hdf5_files: HDF5 文件列表（保留参数以保持接口兼容）
        state_type: 状态类型 "qpos" 或 "pose"
    
    Returns:
        state 的 shape: qpos 为 (16,), pose 为 (14,)
    """
    if state_type == "qpos":
        return (16,)
    else:  # pose
        return (16,)


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    state_type: Literal["qpos", "pose"] = "qpos",
    cameras: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, str, np.ndarray | None]:
    with h5py.File(ep_path, "r") as ep:
        # 根据 state_type 读取相应的 state 和 action
        if state_type == "qpos":
            if "/observations/qpos" not in ep:
                raise ValueError(f"qpos not found in {ep_path}")
            state = torch.from_numpy(ep["/observations/qpos"][:])
            if "/action" in ep:
                action = torch.from_numpy(ep["/action"][:])
            else:
                action = None
        else:  # pose
            if "/observations/pose" not in ep:
                raise ValueError(f"pose not found in {ep_path}")
            state_np = ep["/observations/pose"][:]
            
            # 检查维度，如果是16维（四元数格式），先修复跳变，再转换为14维（旋转向量格式）
            if state_np.shape[-1] == 16:
                # 修复四元数跳变
                num_frames = state_np.shape[0]
                for i in range(1, num_frames):
                    state_np[i] = fix_quaternion_discontinuity(state_np[i-1], state_np[i])
                # 转换为14维旋转向量格式
            #     state_np = dual_arm_quat_to_rotvec(state_np)
            # elif state_np.shape[-1] != 14:
            #     raise ValueError(
            #         f"pose维度应为14或16，实际为 {state_np.shape[-1]}。"
            #         f"14维为旋转向量格式，16维为四元数格式。"
            #     )
            state = torch.from_numpy(state_np)
            
            if "/action_pose" in ep:
                action_np = ep["/action_pose"][:]
                
                # 检查维度，如果是16维（四元数格式），先修复跳变，再转换为14维（旋转向量格式）
                if action_np.shape[-1] == 16:
                    # 修复四元数跳变
                    num_frames = action_np.shape[0]
                    for i in range(1, num_frames):
                        action_np[i] = fix_quaternion_discontinuity(action_np[i-1], action_np[i])
                    # 转换为14维旋转向量格式
                    # action_np = dual_arm_quat_to_rotvec(action_np)
                # elif action_np.shape[-1] != 14:
                #     raise ValueError(
                #         f"action_pose维度应为14或16，实际为 {action_np.shape[-1]}。"
                #         f"14维为旋转向量格式，16维为四元数格式。"
                #     )
                action = torch.from_numpy(action_np)
            else:
                action = None
        
        # 读取 prev_pose（只在 pose 模式下）
        prev_pose = None
        if state_type == "pose" and "/observations/prev_pose" in ep:
            prev_pose_np = ep["/observations/prev_pose"][:]
            # 检查维度，如果是16维（四元数格式），修复四元数跳变
            if prev_pose_np.shape[-1] == 16:
                # 修复四元数跳变
                num_frames = prev_pose_np.shape[0]
                for i in range(1, num_frames):
                    prev_pose_np[i] = fix_quaternion_discontinuity(prev_pose_np[i-1], prev_pose_np[i])
            prev_pose = torch.from_numpy(prev_pose_np)
        
        task_prompt = ep["/task"][:]
        advantage = None
        if "/advantage" in ep:
            advantage = ep["/advantage"][:]

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        # 如果没有指定 cameras，从 hdf5 文件中获取
        if cameras is None:
            cameras = [key for key in ep["/observations/images"].keys() if "depth" not in key]

        imgs_per_cam = load_raw_images_per_camera(ep, cameras)

    return imgs_per_cam, state, action, velocity, effort, prev_pose, task_prompt, advantage


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    state_type: Literal["qpos", "pose"] = "qpos",
    cameras: list[str] | None = None,
) -> LeRobotDataset:
    def resize_and_reformat(img: np.ndarray) -> np.ndarray:
        """Ensure image is CHW 3x224x224; resize if needed."""
        if img.ndim != 3:
            return img

        # If channel-first, temporarily move to HWC for resize
        channel_first = img.shape[0] == 3 and img.shape[1] != 3 and img.shape[2] != 3
        if channel_first:
            img_hwc = np.transpose(img, (1, 2, 0))
        else:
            img_hwc = img

        h, w = img_hwc.shape[:2]
        if (h, w) != (224, 224):
            img_hwc = cv2.resize(img_hwc, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Ensure channel-first output 3x224x224
        if img_hwc.shape[2] == 3:
            img_chw = np.transpose(img_hwc, (2, 0, 1))
        else:
            img_chw = img_hwc

        return img_chw

    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        # 一次加载整个episode的数据
        imgs_per_cam, state, action, velocity, effort, prev_pose, task_prompt, advantage = load_raw_episode_data(
            ep_path, state_type=state_type, cameras=cameras
        )
        num_frames = state.shape[0]

        for i in range(num_frames):
            # 如果action不存在，使用下一帧的state作为action（最后一帧使用当前帧）
            action_value = action[i] if action is not None else (state[i + 1] if i + 1 < num_frames else state[i])
            
            frame = {
                "observation.state": state[i],
                "action": action_value,
                "task": str(task_prompt[i], 'utf-8'),
            }

            # 添加 prev_state（如果存在，只在 pose 模式下）
            if prev_pose is not None:
                frame["observation.prev_state"] = prev_pose[i]

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = resize_and_reformat(img_array[i])

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            
            if advantage is not None:
                frame["advantage"] = str(advantage[i], 'utf-8')

            dataset.add_frame(frame)

        # 保存episode
        dataset.save_episode()
        
        # 显式删除episode数据，释放内存
        del imgs_per_cam, state, action, velocity, effort, prev_pose, task_prompt, advantage
        
        # 每处理一定数量的episode后，清理 hf_dataset 以释放内存
        # 注意：hf_dataset 在 save_episode 中会持续增长，即使数据已写入磁盘
        if (ep_idx + 1) % 10 == 0:
            # 尝试清理 hf_dataset 的内部缓存
            # 由于数据已经写入 parquet 文件，可以安全地清理内存中的 dataset
            if hasattr(dataset, 'hf_dataset') and dataset.hf_dataset is not None:
                # 创建一个空的 dataset 来替换，释放内存
                # 注意：这不会影响已保存的 parquet 文件
                try:
                    # 获取 dataset 的 features 和 split
                    features = dataset.hf_dataset.features
                    split = dataset.hf_dataset.split
                    # 创建空 dataset 来释放内存
                    import datasets
                    empty_dict = {key: [] for key in features.keys()}
                    dataset.hf_dataset = datasets.Dataset.from_dict(empty_dict, features=features, split=split)
                except Exception as e:
                    # 如果清理失败，至少尝试删除引用
                    print(f"警告: 清理 hf_dataset 时出错: {e}")
                    pass
            
            # 强制垃圾回收，释放内存
            gc.collect()
            
            # 如果是PyTorch，清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return dataset


def port_flexiv(
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    raw_dir: Path = Path("datasets/recorded_data"),
    repo_id: str = "flexiv_lerobot_data",
    state_type: Literal["qpos", "pose"] = "qpos",
):
    """
    转换 Flexiv HDF5 数据到 LeRobot 格式
    
    Args:
        state_type: 使用 "qpos" (关节角) 或 "pose" (位姿) 作为 state 和 action
    """
    # 根据 state_type 在 repo_id 后添加后缀
    suffix = f"_{state_type}"
    if not repo_id.endswith(suffix):
        repo_id = f"{repo_id}{suffix}"
    
    print(f"Using repo_id: {repo_id} (state_type: {state_type})")
    
    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        else:
            raise ValueError("Downloading raw data is not supported in this script.")

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No hdf5 files found in {raw_dir}")

    # 获取 state 的 shape（硬编码：qpos=16, pose=14）
    state_shape = get_state_shape(hdf5_files, state_type)
    print(f"Using {state_type} with shape: {state_shape}")
    
    # 如果使用pose，检查第一个文件的实际维度，判断是否需要转换
    if state_type == "pose":
        with h5py.File(hdf5_files[0], "r") as ep:
            if "/observations/pose" in ep:
                actual_dim = ep["/observations/pose"].shape[-1]
                if actual_dim == 16:
                    print(f"  检测到16维pose数据（四元数格式）")
                elif actual_dim == 14:
                    print(f"  检测到14维pose数据（旋转向量格式）")
                else:
                    print(f"  警告: pose维度为 {actual_dim}，预期为14或16")

    # 获取可用的相机列表（包括 cam_high）
    cameras = get_cameras(hdf5_files)
    print(f"Found cameras: {cameras}")

    # 只在 pose 模式下检查 prev_pose
    has_prev_state_val = False
    if state_type == "pose":
        has_prev_state_val = has_prev_pose(hdf5_files)
        if has_prev_state_val:
            print(f"  检测到 prev_pose 数据，将添加到数据集中")
        else:
            print(f"  未检测到 prev_pose 数据，将不添加到数据集中")
    
    dataset = create_empty_dataset(
        repo_id,
        robot_type="flexiv",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        has_advantage=has_advantage(hdf5_files),
        has_prev_state=has_prev_state_val,
        state_type=state_type,
        state_shape=state_shape,
        cameras=cameras,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
        state_type=state_type,
        cameras=cameras,
    )

    # if push_to_hub:
    #     dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_flexiv)
