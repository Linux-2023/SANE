import logging
import pathlib
from typing import List, Optional

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """保存 episode 数据为视频文件。
    
    支持保存多个相机的视频，优先使用全局相机，如果没有则使用左腕相机。
    """

    def __init__(
        self, 
        out_dir: pathlib.Path, 
        subsample: int = 1,
        camera_name: Optional[str] = None
    ) -> None:
        """初始化 VideoSaver。
        
        Args:
            out_dir: 输出目录
            subsample: 子采样率（每 N 帧保存一帧）
            camera_name: 要保存的相机名称。如果为 None，将自动选择：
                - 优先使用 "cam_high"（全局相机）
                - 如果没有，使用 "cam_left_wrist"（左腕相机）
                - 如果还没有，使用 "cam_right_wrist"（右腕相机）
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: List[np.ndarray] = []
        self._subsample = subsample
        self._camera_name = camera_name

    @override
    def on_episode_start(self) -> None:
        self._images = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        images = observation.get("images", {})
        
        # 确定要使用的相机
        camera_name = self._camera_name
        if camera_name is None:
            # 自动选择相机：优先全局相机，然后左腕，最后右腕
            if "cam_high" in images:
                camera_name = "cam_high"
            elif "cam_left_wrist" in images:
                camera_name = "cam_left_wrist"
            elif "cam_right_wrist" in images:
                camera_name = "cam_right_wrist"
            else:
                # 如果没有找到任何相机，使用第一个可用的
                if images:
                    camera_name = list(images.keys())[0]
                else:
                    logging.warning("No camera images found in observation")
                    return
        
        if camera_name not in images:
            logging.warning(f"Camera {camera_name} not found in observation")
            return
        
        im = images[camera_name]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im)

    @override
    def on_episode_end(self) -> None:
        if not self._images:
            logging.warning("No images to save")
            return
        
        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}.mp4"

        logging.info(f"Saving video to {out_path}")
        imageio.mimwrite(
            out_path,
            [np.asarray(x) for x in self._images[:: self._subsample]],
            fps=50 // max(1, self._subsample),
            codec='libx264',
        )


