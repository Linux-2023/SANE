"""
将包含位姿的 hdf5 数据转换为 LeRobot 数据集格式（绝对位姿）

直接保存原始绝对位姿数据：
- observation.state: 当前帧的绝对位姿
- action: 当前帧的绝对位姿（从 action 数据读取）

用法:
    python convert_flexiv_data_to_lerobot_relative_pose.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import os
import cv2

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
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # 位姿格式：左臂位置(3) + 左臂四元数(4) + 左臂夹爪(1) + 右臂位置(3) + 右臂四元数(4) + 右臂夹爪(1) = 16维
    pose_names = [
        "left_pos_x", "left_pos_y", "left_pos_z",
        "left_quat_w", "left_quat_x", "left_quat_y", "left_quat_z",
        "left_gripper",
        "right_pos_x", "right_pos_y", "right_pos_z",
        "right_quat_w", "right_quat_x", "right_quat_y", "right_quat_z",
        "right_gripper",
    ]
    
    cameras = ["cam_left_wrist", "cam_right_wrist"]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [pose_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": [pose_names],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": [pose_names],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": [pose_names],
        }
    
    if has_prev_state:
        features["observation.prev_state"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": [pose_names],
        }
    
    if has_advantage:
        features["advantage"] = {
            "dtype": "string",
            "shape": (1,),
            "names": None,
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
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def has_advantage(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/advantage" in ep


def has_prev_qpos(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/prev_qpos" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, str, np.ndarray | None]:
    with h5py.File(ep_path, "r") as ep:
        qpos = torch.from_numpy(ep["/observations/qpos"][:])  # shape: (num_steps, 16)
        action = torch.from_numpy(ep["/action"][:])
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
        
        prev_qpos = None
        if "/observations/prev_qpos" in ep:
            prev_qpos = torch.from_numpy(ep["/observations/prev_qpos"][:])  # shape: (num_steps, 16)

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            ["cam_left_wrist", "cam_right_wrist"],
        )

    return imgs_per_cam, qpos, action, velocity, effort, prev_qpos, task_prompt, advantage


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    def resize_and_reformat(img: np.ndarray) -> np.ndarray:
        if img.ndim != 3:
            return img

        channel_first = img.shape[0] == 3 and img.shape[1] != 3 and img.shape[2] != 3
        if channel_first:
            img_hwc = np.transpose(img, (1, 2, 0))
        else:
            img_hwc = img

        h, w = img_hwc.shape[:2]
        if (h, w) != (224, 224):
            img_hwc = cv2.resize(img_hwc, (224, 224), interpolation=cv2.INTER_LINEAR)

        if img_hwc.shape[2] == 3:
            img_chw = np.transpose(img_hwc, (2, 0, 1))
        else:
            img_chw = img_hwc

        return img_chw

    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, qpos, action, velocity, effort, prev_qpos, task_prompt, advantage = load_raw_episode_data(ep_path)
        num_frames = qpos.shape[0]
        qpos_np = qpos.numpy()  # shape: (num_frames, 16)
        action_np = action.numpy()  # shape: (num_frames, 16)
        prev_qpos_np = prev_qpos.numpy() if prev_qpos is not None else None

        for i in range(num_frames):
            # 直接使用原始绝对位姿
            # observation.state: 当前帧的绝对位姿
            obs_state = qpos_np[i].astype(np.float32)
            
            # action: 当前帧的绝对位姿（从 action 数据读取）
            action_state = action_np[i].astype(np.float32)

            frame = {
                "observation.state": obs_state,
                "action": action_state,
                "task": str(task_prompt[i], 'utf-8'),
            }

            # 添加 prev_state（如果存在）
            if prev_qpos_np is not None:
                frame["observation.prev_state"] = prev_qpos_np[i].astype(np.float32)

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = resize_and_reformat(img_array[i])

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            
            if advantage is not None:
                frame["advantage"] = str(advantage[i], 'utf-8')

            dataset.add_frame(frame)

        dataset.save_episode()

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
    repo_id: str = "flexiv_lerobot_data/",
):
    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        else:
            raise ValueError("Downloading raw data is not supported in this script.")

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="flexiv",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        has_advantage=has_advantage(hdf5_files),
        has_prev_state=has_prev_qpos(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )


if __name__ == "__main__":
    tyro.cli(port_flexiv)

