"""
Script to resize images in HDF5 files to 224x224 resolution.

Example usage: uv run examples/flexiv/utils/resize_images_in_hdf5.py --input-dir /path/to/hdf5/files --output-dir /path/to/output
"""

import dataclasses
from pathlib import Path
from typing import Literal

import h5py
import cv2
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class ResizeConfig:
    input_size: tuple[int, int] = (224, 224)
    interpolation: str = "linear"


DEFAULT_RESIZE_CONFIG = ResizeConfig()


def get_interpolation_method(method: str) -> int:
    """Convert string method name to OpenCV interpolation constant."""
    methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    return methods.get(method.lower(), cv2.INTER_LINEAR)


def load_and_resize_images_per_camera(
    ep: h5py.File, cameras: list[str], target_size: tuple[int, int], interpolation: int
) -> dict[str, np.ndarray]:
    """Load and resize images from HDF5 file for all cameras."""
    imgs_per_cam = {}
    for camera in cameras:
        if f"/observations/images/{camera}" not in ep:
            print(f"Warning: Camera {camera} not found in episode")
            continue

        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                img = cv2.imdecode(data, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs_array.append(img)
            imgs_array = np.array(imgs_array)

        # Resize all images to target size
        num_frames = imgs_array.shape[0]
        resized_imgs = []
        for i in range(num_frames):
            resized_img = cv2.resize(imgs_array[i], target_size, interpolation=interpolation)
            resized_imgs.append(resized_img)
        
        imgs_per_cam[camera] = np.array(resized_imgs)
    
    return imgs_per_cam


def resize_hdf5_images(
    input_dir: Path,
    output_dir: Path,
    target_size: tuple[int, int] = (224, 224),
    interpolation: str = "linear",
    overwrite: bool = False,
):
    """Resize all images in HDF5 files from input_dir and save to output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hdf5_files = sorted(input_dir.glob("episode_*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")
    
    interp_method = get_interpolation_method(interpolation)
    
    # Get all cameras from first episode
    with h5py.File(hdf5_files[0], "r") as ep:
        if "/observations/images" not in ep:
            raise ValueError("No images found in HDF5 file")
        cameras = [key for key in ep["/observations/images"].keys() if "depth" not in key]
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    print(f"Cameras: {cameras}")
    print(f"Target size: {target_size}")
    print(f"Interpolation method: {interpolation}")
    
    for ep_idx in tqdm.tqdm(hdf5_files, desc="Processing episodes"):
        output_path = output_dir / ep_idx.name
        
        if output_path.exists() and not overwrite:
            print(f"Skipping {ep_idx.name} (already exists, use --overwrite to replace)")
            continue
        
        with h5py.File(ep_idx, "r") as input_ep:
            # Load and resize images
            imgs_per_cam = load_and_resize_images_per_camera(
                input_ep, cameras, target_size, interp_method
            )
            
            # Create output file
            with h5py.File(output_path, "w") as output_ep:
                # Copy all data except images, recursively
                def copy_data_recursive(src_group, dst_group, skip_images=False):
                    """Recursively copy groups and datasets, optionally skipping images."""
                    for key in src_group.keys():
                        src_item = src_group[key]
                        
                        # Skip images group
                        if skip_images and key == "images" and isinstance(src_item, h5py.Group):
                            continue
                        
                        if isinstance(src_item, h5py.Dataset):
                            dst_group.create_dataset(key, data=src_item[...])
                        elif isinstance(src_item, h5py.Group):
                            new_group = dst_group.create_group(key)
                            copy_data_recursive(src_item, new_group, skip_images=(key == "observations"))
                
                # Copy all data except images
                copy_data_recursive(input_ep, output_ep)
                
                # Create and write resized images
                images_group = output_ep["observations"].create_group("images")
                for camera, resized_imgs in imgs_per_cam.items():
                    images_group.create_dataset(camera, data=resized_imgs, compression="gzip")


if __name__ == "__main__":
    tyro.cli(resize_hdf5_images)
