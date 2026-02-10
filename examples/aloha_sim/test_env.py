import gym_aloha  # noqa: F401
import gymnasium
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import time



env = gymnasium.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", observation_width = 224, observation_height = 224, render_mode=None)
env.reset()
for _ in range(100):
    start = time.time()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    img = obs["pixels"]["top"]
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    # Convert axis order from [H, W, C] --> [C, H, W]
    img = np.transpose(img, (2, 0, 1))
    end = time.time()
    print("step time:", end - start)
    if terminated or truncated:
        env.reset()