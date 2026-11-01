# Normalization statistics

Following common practice, our models normalize the proprioceptive state inputs and action targets during policy training and inference. The statistics used for normalization are computed over the training data and stored alongside the model checkpoint.

## Reloading normalization statistics

When you fine-tune one of our models on a new dataset, you need to decide whether to (A) reuse existing normalization statistics or (B) compute new statistics over your new training data. Which option is better for you depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. Below, we list all the available pre-training normalization statistics for each model.

**If your target robot matches one of these pre-training statistics, consider reloading the same normalization statistics.** By reloading the normalization statistics, the actions in your dataset will be more "familiar" to the model, which can lead to better performance. You can reload the normalization statistics by adding an `AssetsConfig` to your training config that points to the corresponding checkpoint directory and normalization statistics ID, like below for the `Trossen` (aka ALOHA) robot statistics of the `pi0_base` checkpoint:

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

For an example of a full training config that reloads normalization statistics, see the `pi0_aloha_pen_uncap` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

**Note:** To successfully reload normalization statistics, it's important that your robot + dataset are following the action space definitions used in pre-training. We provide a detailed description of our action space definitions below.

**Note #2:** Whether reloading normalization statistics is beneficial depends on the similarity of your robot and task to the robot and task distribution in the pre-training dataset. We recommend to always try both, reloading and training with a fresh set of statistics computed on your new dataset (see [main README](../README.md) for instructions on how to compute new statistics), and pick the one that works better for your task.


## Provided Pre-training Normalization Statistics

Below is a list of all the pre-training normalization statistics we provide. We provide them for both, the `pi0_base` and `pi0_fast_base` models. For `pi0_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_base/assets` and for `pi0_fast_base`, set the `assets_dir` to `gs://openpi-assets/checkpoints/pi0_fast_base/assets`.
| Robot | Description | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF dual arm robot with parallel grippers | trossen |
| Mobile ALOHA | Mobile version of ALOHA mounted on a Slate base | trossen_mobile |
| Franka Emika (DROID) | 7-DoF arm with parallel gripper based on the DROID setup | droid |
| Franka Emika (non-DROID) | Franka FR3 arm with Robotiq 2F-85 gripper | franka |
| UR5e | 6-DoF UR5e arm with Robotiq 2F-85 gripper | ur5e |
| UR5e bi-manual | Bi-manual UR5e setup with Robotiq 2F-85 grippers | ur5e_dual |
| ARX | Bi-manual ARX-5 robot arm setup with parallel gripper | arx |
| ARX mobile | Mobile version of bi-manual ARX-5 robot arm setup mounted on a Slate base | arx_mobile |
| Fibocom mobile | Fibocom mobile robot with 2x ARX-5 arms | fibocom_mobile |


## Pi0 Model Action Space Definitions

Out of the box, both the `pi0_base` and `pi0_fast_base` use the following action space definitions (left and right are defined looking from behind the robot towards the workspace):
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

The proprioceptive state uses the same definitions as the action space, except for the base x-y position (the last two dimensions) for mobile robots, which we don't include in the proprioceptive state.

For 7-DoF robots (e.g. Franka), we use the first 7 dimensions of the action space for the joint actions, and the 8th dimension for the gripper action.

General info for Pi robots:
- Joint angles are expressed in radians, with position zero corresponding to the zero position reported by each robot's interface library, except for ALOHA, where the standard ALOHA code uses a slightly different convention (see the [ALOHA example code](../examples/aloha_real/README.md) for details).
- Gripper positions are in [0.0, 1.0], with 0.0 corresponding to fully open and 1.0 corresponding to fully closed.
- Control frequencies are either 20 Hz for UR5e and Franka, and 50 Hz for ARX and Trossen (ALOHA) arms.

For DROID, we use the original DROID action configuration, with joint velocity actions in the first 7 dimensions and gripper actions in the 8th dimension + a control frequency of 15 Hz.

## Piper dual tidy_desk Training-RTC (pi0.5)

The `pi05_piper_dual_tidy_desk_ros2_shifted_training_rtc` preset and its JAX
training implementation are included in `main`; no separate patch is required.
Other presets keep Training-RTC disabled by default.

### Data and objective

- Dataset: [`HITdongdong/piper_dual_tidy_desk_ros2_shifted`](https://huggingface.co/datasets/HITdongdong/piper_dual_tidy_desk_ros2_shifted),
  revision `1a127a8d018c7cbf1afacc92dc7dcaaf510ee7272`: 50 episodes,
  86,764 frames at 30 Hz, with `cam_high`, `cam_left_wrist`, and `cam_right_wrist` images.
- State and action each have 14 values: six left joint positions, left gripper,
  six right joint positions, and right gripper. `adapt_to_pi=False` preserves
  the dataset's units. The 12 joint targets are converted to deltas from the
  current state; grippers remain absolute. Compute normalization statistics
  through this preset, not from the raw absolute actions or another robot's assets.
- The model has action dimension 32 and horizon 50. Training samples a clean
  prefix uniformly from 0 through 15 positions, with an additional probability
  of 0.25 of forcing the prefix length to zero. Prefix positions use per-token
  diffusion time `t=0`; suffix positions retain the sampled diffusion time.
- Flow-matching loss supervises only the 14 real action dimensions and valid
  suffix positions. Clean prefixes and LeRobot `action_is_pad` positions beyond
  the episode end are excluded; the reduction is normalized by the valid count.
- `discrete_state_input=False` deliberately matches the completed Piper run.
  This pi0.5 preset does not supply state tokens or a continuous state projection;
  measured state is still required for joint-delta encoding and decoding.

### Reproducing training

Install the repository dependencies as described in the [README](../README.md).
Download the pinned dataset into a dedicated LeRobot cache:

```bash
export HF_LEROBOT_HOME=/absolute/path/to/lerobot
uv run python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo = "HITdongdong/piper_dual_tidy_desk_ros2_shifted"
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    revision="1a127a8d018c7cbf1afacc92dc7dcaaf510ee7272",
    local_dir=Path(os.environ["HF_LEROBOT_HOME"]) / repo,
)
PY
export TRAIN_CONFIG=pi05_piper_dual_tidy_desk_ros2_shifted_training_rtc
uv run scripts/compute_norm_stats.py --config-name "$TRAIN_CONFIG"
uv run scripts/train.py "$TRAIN_CONFIG" --exp-name tidy_desk_training_rtc \
    --batch-size 16 --fsdp-devices 1 --num-workers 4
```

The preset starts from `gs://openpi-assets/checkpoints/pi05_base/params`, trains
for 30,000 steps, and saves every 5,000 steps. Its defaults are batch 64 and
FSDP 2; the command above uses the completed run's batch 16 / FSDP 1 overrides.
The completed run's final checkpoint is `29999` because steps are zero-indexed.
Normalization statistics live under
`assets/pi05_piper_dual_tidy_desk_ros2_shifted_training_rtc/HITdongdong/piper_dual_tidy_desk_ros2_shifted/`
and are copied into checkpoints. Preserve these assets when exporting a model.

### Verification and inference boundary

`src/openpi/models/rtc_training_test.py` covers clean-prefix inputs, token-wise
time conditioning, valid-position loss masking, and scalar-time compatibility.
The existing pi0 configuration regressions also pass. A CPU smoke run of the
integrated code restored the full `gemma_2b` / `gemma_300m` checkpoint and used
real first-frame and episode-end samples through the training transforms and
`compute_loss(train=True)`. It verified three-camera inputs, normalization,
episode padding, finite RTC loss, excluded-prefix/padding loss, and the RTC-off path.

This is a training implementation, not a clean-prefix inference server. The
standard PyTorch serving path uses zero-prefix sampling; legacy inference-time
RTC guidance is not the same algorithm. A finite loss or offline smoke run does
not establish real-robot success or safety. This publication does not restart
training or change the exported checkpoint.

