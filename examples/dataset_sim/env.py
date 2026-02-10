from typing import Optional, Dict, Any
import os

import numpy as np
from typing_extensions import override

try:
    # OpenPI environment base (for API compatibility)
    from openpi_client.runtime import environment as _environment
except Exception:
    # Fallback: minimal base class if openpi_client not available
    class _BaseEnv:
        def reset(self) -> None: ...
        def is_episode_complete(self) -> bool: ...
        def get_observation(self) -> Dict[str, Any]: ...
        def apply_action(self, action: Dict[str, Any]) -> None: ...
        def close(self) -> None: ...
    _environment = type("environment", (), {"Environment": _BaseEnv})

# LeRobot dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def fix_quaternion_discontinuity(last: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    修复两个16维向量之间四元数的数值跳变
    
    四元数在表示同一个旋转时有两种等价表示：q 和 -q。这可能导致相邻帧之间的
    四元数数值发生跳变（例如从 [1, 0, 0, 0] 跳到 [-1, 0, 0, 0]）。
    此函数检测并修复这种跳变，确保 current 中的四元数与 last 保持一致的方向。
    
    输入格式（16维）：
    - 左臂位姿（3维: x, y, z）
    - 左臂四元数（4维: qw, qx, qy, qz）- 索引 3-6
    - 左臂夹爪（1维）- 索引 7
    - 右臂位姿（3维: x, y, z）- 索引 8-10
    - 右臂四元数（4维: qw, qx, qy, qz）- 索引 11-14
    - 右臂夹爪（1维）- 索引 15
    
    Args:
        last: 上一帧的双臂位姿向量 (16,)
        current: 当前帧的双臂位姿向量 (16,)，可能会被修改
    
    Returns:
        fixed_current: 修复后的当前帧向量 (16,)
    """
    last = np.asarray(last, dtype=np.float32).reshape(16)
    current = np.asarray(current, dtype=np.float32).reshape(16).copy()
    
    # 提取左臂四元数（索引 3-6: qw, qx, qy, qz）
    last_left_quat = last[3:7]
    current_left_quat = current[3:7]
    
    # 计算四元数点积，如果为负说明需要取反
    left_quat_dot = np.dot(last_left_quat, current_left_quat)
    if left_quat_dot < 0:
        # 取反以保持一致的方向
        current[3:7] = -current_left_quat
    
    # 提取右臂四元数（索引 11-14: qw, qx, qy, qz）
    last_right_quat = last[11:15]
    current_right_quat = current[11:15]
    
    # 计算四元数点积，如果为负说明需要取反
    right_quat_dot = np.dot(last_right_quat, current_right_quat)
    if right_quat_dot < 0:
        # 取反以保持一致的方向
        current[11:15] = -current_right_quat
    
    return current

class DatasetSimEnvironment(_environment.Environment):
    """基于LeRobot数据集的简洁仿真环境。

    - reset(): 重置到指定episode的第一帧
    - get_observation(): 返回数据集中的observation字段
    - apply_action(action): 与数据集中的action对比并记录差异
    - is_episode_complete(): 当前episode是否结束
    """

    def __init__(
        self,
        dataset_path: str,
        episode_index: int = 0,
        step_stride: int = 1,
        save_dir: str = "data/action_diffs",
    ) -> None:
        """初始化环境。

        Args:
            dataset_path: LeRobot数据集路径。
            episode_index: 启动时选择的episode索引。
            step_stride: 每次动作后的帧步进（默认1）。
            save_dir: 保存图表的目录。
        """
        self.dataset = LeRobotDataset(dataset_path)
        self.episode_index = max(0, min(episode_index, self.dataset.num_episodes - 1))
        self.step_stride = max(1, int(step_stride))
        self.save_dir = save_dir

        self._done = True
        self._cur_idx = 0
        self._from_idx = 0
        self._to_idx = 0

        # 记录最近一次动作差异
        self.last_action_diff: Optional[Dict[str, Any]] = None
        
        # 记录历史l2和mae
        self.l2_history: list = []
        self.mae_history: list = []
        self.step_history: list = []

        # 直接准备episode边界，reset时激活
        self._set_episode_bounds(self.episode_index)

    def _set_episode_bounds(self, ep: int) -> None:
        epi = self.dataset.episode_data_index
        self._from_idx = int(epi["from"][ep].item())
        self._to_idx = int(epi["to"][ep].item())

    @override
    def reset(self) -> None:
        """重置到当前episode的第一帧。"""
        self._set_episode_bounds(self.episode_index)
        self._cur_idx = self._from_idx
        self._done = False
        self.last_action_diff = None
        # 清空历史记录
        self.l2_history = []
        self.mae_history = []
        self.step_history = []

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    def _current_item(self) -> Dict[str, Any]:
        return self.dataset[self._cur_idx]

    def _get_by_path(self, obj: Any, path: str) -> Any:
        cur = obj
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    def _to_numpy(self, obj: Any) -> Any:
        # 轻量转换：支持 tensor、numpy、dict、list/tuple
        if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
            return obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, dict):
            return {k: self._to_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_numpy(v) for v in obj)
        return obj

    @override
    def get_observation(self) -> Dict[str, Any]:
        """返回按 LeRobotFlexivDataConfig repack_transforms 组织的 observation。

        结构：
        {
          "images": {
             "cam_left_wrist": observation.images.left_wrist,
             "cam_right_wrist": observation.images.right_wrist,
          },
                    "state": observation.state,
                    "prev_state": observation.prev_state (如果存在)
        }
        所有张量将被转换为 numpy。
        """
        item = self._current_item()

        # 提取图像
        cam_left = item['observation.images.cam_left_wrist']#self._get_by_path(item, "observation.images.cam_left_wrist")
        cam_right = item['observation.images.cam_right_wrist'] #self._get_by_path(item, "observation.images.cam_right_wrist")

        # 提取状态
        state = item['observation.state'] #self._get_by_path(item, "observation.state")
        if state is None and isinstance(item, dict):
            state = item.get("state")

        # 提取 prev_state（用于 pose 模式或需要前一状态的情况）
        try:
            prev_state = item['observation.prev_state'] #self._get_by_path(item, "observation.prev_state")
            if prev_state is None and isinstance(item, dict):
                prev_state = item.get("prev_state")
        except:
            prev_state = None
        
        
        prompt = item['task']
        
        obs: Dict[str, Any] = {"images": {}}
        if cam_left is not None:
            obs["images"]["cam_left_wrist"] = cam_left
        if cam_right is not None:
            obs["images"]["cam_right_wrist"] = cam_right
        if state is not None:
            obs["state"] = state
        if prev_state is not None:
            obs["prev_state"] = prev_state
        if prompt is not None:
            obs["prompt"] = prompt

        return self._to_numpy(obs)

    def _get_expert_action(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        # 兼容不同字段命名
        if "actions" in item:
            return np.asarray(item["actions"])  # 已是向量或数组
        if "action" in item:
            return np.asarray(item["action"])  # 单步动作
        return None

    @override
    def apply_action(self, action: Dict[str, Any]) -> None:
        """对比传入action与数据集中的action并记录差异。

        期望输入格式：{"actions": np.ndarray | list}
        """
        if self._done:
            return

        item = self._current_item()
        expert = self._get_expert_action(item)

        user_act = action.get("actions")
        if user_act is None:
            raise ValueError("缺少'actions'键，期望格式: {'actions': np.ndarray | list}")
        user_act = np.asarray(user_act, dtype=np.float32)
        self._last_user_action = user_act

        if expert is None:
            print("Warning: No expert action available for comparison in dataset.")
            # 无专家动作时，只记录用户动作信息
            self.last_action_diff = {
                "has_expert": False,
                "user_action_shape": user_act.shape,
            }
        else:
            expert = np.asarray(expert, dtype=np.float32)
            user_act = fix_quaternion_discontinuity(expert, user_act)
            # 对齐形状（取最小维度）
            dim = min(user_act.shape[-1], expert.shape[-1])
            ua = user_act[..., :dim]
            ea = expert[..., :dim]
            diff = ua - ea
            l2 = float(np.linalg.norm(diff))
            mae = float(np.mean(np.abs(diff)))

            self.last_action_diff = {
                "has_expert": True,
                "dim": dim,
                "l2": l2,
                "mae": mae,
                "diff": diff.astype(np.float32),
            }
            
            # 记录l2和mae到历史
            step = len(self.step_history)
            self.l2_history.append(l2)
            self.mae_history.append(mae)
            self.step_history.append(step)

            if l2 > 0.5:
                print("predict:", ua)
                print("expert :", ea)
                print("diff   :", diff)
                print("=============================")

        # 前进到下一帧
        self._cur_idx += self.step_stride
        if self._cur_idx >= self._to_idx:
            self._cur_idx = self._to_idx - 1
            self._done = True
    
    def plot_action_diff(self) -> None:
        """保存action差异的图表到文件。"""
        if len(self.l2_history) == 0:
            print("No action history to plot")
            return
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 导入matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制L2距离
        ax1.plot(self.step_history, self.l2_history, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('L2 Distance')
        ax1.set_title('Action Difference: L2 Distance over Steps')
        ax1.grid(True, alpha=0.3)
        
        # 绘制MAE
        ax2.plot(self.step_history, self.mae_history, 'r-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Mean Absolute Error (MAE)')
        ax2.set_title('Action Difference: MAE over Steps')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(self.save_dir, f"action_diff_ep{self.episode_index}.png")
        plt.savefig(save_path, dpi=100)
        print(f"✓ Action difference plot saved to: {save_path}")
        
        # 保存数据为CSV
        csv_path = os.path.join(self.save_dir, f"action_diff_ep{self.episode_index}.csv")
        with open(csv_path, 'w') as f:
            f.write("step,l2,mae\n")
            for step, l2, mae in zip(self.step_history, self.l2_history, self.mae_history):
                f.write(f"{step},{l2:.6f},{mae:.6f}\n")
        print(f"✓ Action difference data saved to: {csv_path}")
        
        plt.close()
    
    def close(self) -> None:
        pass
