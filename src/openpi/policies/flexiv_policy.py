import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_flexiv_example() -> dict:
    """Creates a random input example for the Flexiv policy."""
    return {
        "state": np.ones((16,)),
        "images": {
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class FlexivInputs(transforms.DataTransformFn):
    """Inputs for the Flexiv policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [16] (left arm 7 + gripper, right arm 7 + gripper)
    - actions: [action_horizon, 16]
    """

    adapt_to_pi: bool = False

    # Flexiv only uses two wrist cameras.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_flexiv(data)

        in_images = data["images"]
        # if set(in_images) - set(self.EXPECTED_CAMERAS):
        #     raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Use cam_high or cam_low as base; if neither exists, create empty image with mask=False.
        base_image = in_images.get("cam_high")
        if base_image is None:
            base_image = in_images.get("cam_low")
        
        if base_image is not None:
            base_image_mask = np.True_
        else:
            base_image = np.zeros((224, 224, 3), dtype=np.uint8)
            base_image_mask = np.False_

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": base_image_mask,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = actions.copy()

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        if "rtc_obs" in data:
            inputs["rtc_obs"] = data["rtc_obs"].copy()
        
        if "advantage" in data:
            inputs["advantage"] = data["advantage"]

        if "prev_state" in data:
            inputs["prev_state"] = data["prev_state"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FlexivOutputs(transforms.DataTransformFn):
    """Outputs for the Flexiv policy."""

    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        # Return 16-dim actions (left arm 8 + right arm 8).
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": actions}


def _decode_flexiv(data: dict) -> dict:
    """Pass-through for state and converts images from CHW to HWC uint8."""
    state = np.asarray(data["state"])
    prev_state = data.get("prev_state")
    if prev_state is not None:
        prev_state = np.asarray(prev_state)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        if img.ndim == 3 and img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")
        return img

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    if prev_state is not None:
        data["prev_state"] = prev_state
    return data
