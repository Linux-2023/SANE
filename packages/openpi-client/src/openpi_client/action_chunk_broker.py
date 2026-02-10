from typing import Dict, Tuple, List
import threading
from collections import deque
import atexit
import math

import numpy as np
import tree
from typing_extensions import override
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt

from openpi_client import base_policy as _base_policy
import time


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, fps: int = 30):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._fps = fps

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class ActionChunkBroker_RTC(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time with async inference.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is made asynchronously every
    action_horizon steps, allowing the current chunks to be returned without
    blocking on model inference.
    """
    
    _instances = []  # Class-level list to track all instances
    _atexit_registered = False  # Flag to ensure atexit is only registered once

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, fps: int = 30, actions_during_latency: int = 1, use_rtc: bool = True):
        self._policy = policy
        self._max_horizon = 50
        self._action_dim = 14
        self._action_horizon = action_horizon
        self._fps = fps
        self._infer_latency = deque(maxlen=10)  # Buffer for last 10 inference latencies
        self._avg_latency_ms: float = 0.0  # Average latency in milliseconds
        self._actions_during_latency = actions_during_latency  # Number of actions executed during avg latency
        self._actions_during_latency_realtime: int = 0  # Number of actions executed during avg latency
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_obs: Dict | None = None
        self._prev_results: Dict[str, np.ndarray] | None = None
        self._prev_obs: Dict | None = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None
        self._should_stop = False

        self._use_rtc = use_rtc
        
        # Action variation tracking
        self._prev_returned_action: Dict | None = None
        self._total_error: float = 0.0
        self._error_count: int = 0
        self._action_history: List[np.ndarray] = []  # Store executed action vectors for post-analysis
        self._chunk_boundaries: List[int] = []  # Store step indices where new chunks start
        self._chunk_action_records: List[Dict[str, np.ndarray | int]] = []  # Keep full chunk outputs and overlaps
        
    
    def _record_chunk_actions(self, results: Dict, overlap_steps: int) -> None:
        """Persist raw chunk actions along with overlap metadata."""
        if 'actions' not in results:
            return
        actions = results['actions']
        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)
        max_len = self._action_horizon + max(0, int(overlap_steps))
        max_len = min(max_len, actions.shape[0] if actions.ndim > 0 else max_len)
        if max_len > 0:
            actions = actions[:max_len].copy()
        record = {
            'actions': actions,
            'overlap': max(0, min(int(overlap_steps), actions.shape[0] if actions.ndim > 0 else 0)),
        }
        self._chunk_action_records.append(record)

    def _build_chunk_action_series(self) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]], List[Tuple[int, np.ndarray]]]:
        """Return stitched chunk series plus overlap segments for plotting."""
        if len(self._chunk_action_records) == 0:
            return np.empty((0, self._action_dim)), [], []

        stitched_chunks: List[np.ndarray] = []
        overlap_segments: List[Tuple[int, np.ndarray]] = []
        chunk_segments: List[Tuple[int, np.ndarray]] = []
        executed_steps = 0

        prev_overlap = 0
        prev_chunk_actions = None
        for idx, record in enumerate(self._chunk_action_records):
            chunk_actions = record['actions']
            if not isinstance(chunk_actions, np.ndarray):
                chunk_actions = np.asarray(chunk_actions)
            overlap = int(record['overlap'])
            chunk_len = chunk_actions.shape[0] if chunk_actions.ndim > 0 else 0

            if idx > 0 and overlap > 0 and chunk_len > 0:
                # Calculate error between overlapping parts before adjusting prev_overlap
                if prev_chunk_actions is not None:
                    # Determine actual overlap size (use the minimum of prev_overlap and current overlap)
                    actual_overlap = min(prev_overlap, overlap, prev_chunk_actions.shape[0], chunk_len)
                    if actual_overlap > 0:
                        # Previous chunk's overlap region (last actual_overlap steps)
                        prev_overlap_region = prev_chunk_actions[-actual_overlap:]
                        # Current chunk's overlap region (first actual_overlap steps)
                        curr_overlap_region = chunk_actions[:actual_overlap]
                        
                        # Calculate mean squared error (MSE) and mean absolute error (MAE)
                        error = prev_overlap_region - curr_overlap_region
                        mse = np.mean(np.square(error))
                        mae = np.mean(np.abs(error))
                        max_error = np.max(np.abs(error))
                        
                        print(f"Chunk {idx-1} -> {idx} overlap error (overlap={actual_overlap}): MSE={mse:.6f}, MAE={mae:.6f}, Max={max_error:.6f}")
                
                prev_overlap = min(prev_overlap, chunk_len)
                overlap_segments.append(
                    (max(executed_steps - prev_overlap, 0), chunk_actions[:prev_overlap+1].copy())
                )
            non_overlap_slice = (
                chunk_actions[prev_overlap:]
                if chunk_len > prev_overlap
                else np.empty((0,) + chunk_actions.shape[1:])
            )
            if idx == 0:
                stitched_chunks.append(chunk_actions.copy())
                chunk_segments.append((executed_steps, chunk_actions.copy()))
                executed_steps += chunk_len
            else:
                if non_overlap_slice.size > 0:
                    stitched_chunks.append(non_overlap_slice)
                    chunk_segments.append((executed_steps, non_overlap_slice.copy()))
                    executed_steps += non_overlap_slice.shape[0]
            prev_overlap = overlap
            prev_chunk_actions = chunk_actions.copy()

        if len(stitched_chunks) == 0:
            return np.empty((0, self._action_dim)), overlap_segments, chunk_segments

        stitched_matrix = np.concatenate(stitched_chunks, axis=0)
        return stitched_matrix, overlap_segments, chunk_segments

    def _plot_error_history(self) -> None:
        """Plot action variation over time (position change, velocity change, acceleration change)."""
        if len(self._action_history) < 2:
            return
        
        step_duration_ms = 33.0  # Each step lasts 20ms
        
        def flatten_action(action: np.ndarray) -> np.ndarray:
            return np.asarray(action).astype(np.float64).reshape(-1)
        
        action_matrix = np.stack([flatten_action(a) for a in self._action_history], axis=0)
        
        # First-order difference: position change
        position_change_vectors = action_matrix[1:] - action_matrix[:-1]
        position_change_magnitudes = np.linalg.norm(position_change_vectors, axis=1)
        
        # Second-order difference: velocity change (acceleration)
        velocity_change_vectors = np.diff(position_change_vectors, axis=0) if position_change_vectors.shape[0] >= 2 else np.empty((0, action_matrix.shape[1]))
        velocity_change_magnitudes = np.linalg.norm(velocity_change_vectors, axis=1) if velocity_change_vectors.size > 0 else np.array([])
        
        # Third-order difference: acceleration change (jerk)
        acceleration_change_vectors = np.diff(velocity_change_vectors, axis=0) if velocity_change_vectors.shape[0] >= 2 else np.empty((0, action_matrix.shape[1] if velocity_change_vectors.size > 0 else 0))
        acceleration_change_magnitudes = np.linalg.norm(acceleration_change_vectors, axis=1) if acceleration_change_vectors.size > 0 else np.array([])
        
        def build_time_series(values: np.ndarray, offset_steps: int) -> Tuple[List[float], List[float]]:
            time_points: List[float] = []
            plot_values: List[float] = []
            for i, val in enumerate(values):
                step_index = i + offset_steps
                step_start_time = step_index * step_duration_ms
                step_end_time = (step_index + 1) * step_duration_ms
                time_points.extend([step_start_time, step_end_time])
                plot_values.extend([float(val), float(val)])
            return time_points, plot_values
        
        time_points_pos, pos_values = build_time_series(position_change_magnitudes, offset_steps=0)
        time_points_vel, vel_values = build_time_series(velocity_change_magnitudes, offset_steps=1) if velocity_change_magnitudes.size > 0 else ([], [])
        time_points_acc, acc_values = build_time_series(acceleration_change_magnitudes, offset_steps=2) if acceleration_change_magnitudes.size > 0 else ([], [])
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        axes[0].plot(time_points_pos, pos_values, linewidth=2, drawstyle='steps-post', color='blue', label='Position Change')
        axes[0].set_ylabel('Position Change (L2 norm)', fontsize=11)
        axes[0].set_title('Action Variation: Position, Velocity Change, Acceleration Change', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=9)
        
        if len(vel_values) > 0:
            axes[1].plot(time_points_vel, vel_values, linewidth=2, drawstyle='steps-post', color='green', label='Velocity Change (Acceleration)')
        axes[1].set_ylabel('Velocity Change (L2 norm)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=9)
        
        if len(acc_values) > 0:
            axes[2].plot(time_points_acc, acc_values, linewidth=2, drawstyle='steps-post', color='orange', label='Acceleration Change (Jerk)')
        axes[2].set_ylabel('Acceleration Change (L2 norm)', fontsize=11)
        axes[2].set_xlabel('Time (ms)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right', fontsize=9)
        
        if len(self._chunk_boundaries) > 0:
            for boundary_step in self._chunk_boundaries:
                boundary_time = boundary_step * step_duration_ms
                label = 'Chunk Boundary' if boundary_step == self._chunk_boundaries[0] else ''
                for ax in axes:
                    ax.axvline(x=boundary_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                    if label:
                        label = ''
        
        plt.tight_layout()
        plt.savefig('data/chunk_error_history.pdf', bbox_inches='tight')
        plt.close()
        
        # Plot each action dimension over time (position, velocity, acceleration per row)
        num_steps, num_dims = action_matrix.shape
        position_time = np.arange(num_steps) * step_duration_ms
        velocity_steps = position_change_vectors.shape[0]
        acceleration_steps = velocity_change_vectors.shape[0]
        velocity_time = np.arange(velocity_steps) * step_duration_ms + step_duration_ms
        acceleration_time = np.arange(acceleration_steps) * step_duration_ms + (step_duration_ms * 1.5)
        
        chunk_matrix, overlap_segments, chunk_segments = self._build_chunk_action_series()
        fig_dims, axs = plt.subplots(num_dims, 4, figsize=(16, 2.4 * num_dims), sharex=False)
        if num_dims == 1:
            axs = np.expand_dims(axs, axis=0)
        axs = np.asarray(axs)
        
        for dim in range(num_dims):
            pos_ax = axs[dim, 0]
            vel_ax = axs[dim, 1]
            acc_ax = axs[dim, 2]
            chunk_ax = axs[dim, 3]
            
            pos_series = np.asarray(action_matrix[:, dim], dtype=np.float64).copy()
            if len(self._chunk_boundaries) > 1:
                unique_boundaries = sorted(set(self._chunk_boundaries[1:]))
                for boundary_step in unique_boundaries:
                    if 0 <= boundary_step < pos_series.shape[0]:
                        pos_series[boundary_step] = np.nan
            pos_ax.plot(position_time, pos_series, linewidth=1.3, color='tab:blue')
            pos_ax.set_ylabel(f'Dim {dim}', fontsize=9)
            pos_ax.grid(True, alpha=0.2)
            if dim == 0:
                pos_ax.set_title('Position', fontsize=10, fontweight='bold')
            
            if velocity_steps > 0 and position_change_vectors.size > 0:
                plot_len = min(velocity_steps, position_change_vectors.shape[0])
                vel_ax.plot(
                    velocity_time[:plot_len],
                    position_change_vectors[:plot_len, dim],
                    linewidth=1.3,
                    color='tab:green',
                )
            vel_ax.grid(True, alpha=0.2)
            if dim == 0:
                vel_ax.set_title('Velocity Change', fontsize=10, fontweight='bold')
            
            if acceleration_steps > 0 and velocity_change_vectors.size > 0:
                plot_len = min(acceleration_steps, velocity_change_vectors.shape[0])
                acc_ax.plot(
                    acceleration_time[:plot_len],
                    velocity_change_vectors[:plot_len, dim],
                    linewidth=1.3,
                    color='tab:orange',
                )
            acc_ax.grid(True, alpha=0.2)
            if dim == 0:
                acc_ax.set_title('Acceleration Change', fontsize=10, fontweight='bold')
            acc_ax.set_xlabel('Time (ms)', fontsize=9)

            if chunk_segments:
                for start_idx, segment in chunk_segments:
                    if segment.ndim == 1:
                        if dim == 0:
                            segment_time = (np.arange(segment.shape[0]) + start_idx) * step_duration_ms
                            chunk_ax.plot(segment_time, segment, linewidth=1.3, color='tab:purple')
                        continue
                    if dim >= segment.shape[1]:
                        continue
                    segment_time = (np.arange(segment.shape[0]) + start_idx) * step_duration_ms
                    chunk_ax.plot(segment_time, segment[:, dim], linewidth=1.3, color='tab:purple')
            if overlap_segments:
                for start_idx, segment in overlap_segments:
                    if segment.ndim == 1:
                        if dim == 0:
                            segment_time = (np.arange(segment.shape[0]) + start_idx) * step_duration_ms
                            chunk_ax.plot(segment_time, segment, linewidth=1.1, color='tab:red', alpha=0.9)
                        continue
                    if dim >= segment.shape[1]:
                        continue
                    segment_time = (np.arange(segment.shape[0]) + start_idx) * step_duration_ms
                    chunk_ax.plot(segment_time, segment[:, dim], linewidth=1.1, color='tab:red', alpha=0.9)
            chunk_ax.grid(True, alpha=0.2)
            if dim == 0:
                chunk_ax.set_title('Chunk Actions', fontsize=10, fontweight='bold')
            chunk_ax.set_xlabel('Time (ms)', fontsize=9)
            
            if len(self._chunk_boundaries) > 0:
                for boundary_step in self._chunk_boundaries:
                    boundary_time = boundary_step * step_duration_ms
                    for ax in (pos_ax, vel_ax, acc_ax, chunk_ax):
                        ax.axvline(x=boundary_time, color='red', linestyle='--', linewidth=1, alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('data/action_dimensions_history.pdf', bbox_inches='tight')
        plt.close()
        
        # # Plot each action dimension over time
        # num_steps, num_dims = action_matrix.shape
        # time_axis_actions = np.arange(num_steps) * step_duration_ms
        # cols = min(4, num_dims)
        # rows = math.ceil(num_dims / cols)
        # fig_dims, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), sharex=True)
        # axs = np.atleast_2d(axs)
        
        # for dim in range(num_dims):
        #     r, c = divmod(dim, cols)
        #     ax = axs[r, c]
        #     ax.plot(time_axis_actions, action_matrix[:, dim], linewidth=1.2, color='purple')
        #     ax.set_title(f"Dimension {dim}", fontsize=9)
        #     ax.grid(True, alpha=0.2)
        #     if r == rows - 1:
        #         ax.set_xlabel('Time (ms)', fontsize=9)
        #     ax.set_ylabel('Value', fontsize=9)
            
        #     if len(self._chunk_boundaries) > 0:
        #         for boundary_step in self._chunk_boundaries:
        #             boundary_time = boundary_step * step_duration_ms
        #             ax.axvline(x=boundary_time, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # total_plots = rows * cols
        # if total_plots > num_dims:
        #     for dim in range(num_dims, total_plots):
        #         r, c = divmod(dim, cols)
        #         fig_dims.delaxes(axs[r, c])
        
        # plt.tight_layout()
        # plt.savefig('data/action_dimensions_history.png', dpi=150, bbox_inches='tight')
        # plt.close()
        
        # # Plot each action dimension over time
        # num_steps, num_dims = action_matrix.shape
        # time_axis_actions = np.arange(num_steps) * step_duration_ms
        # cols = min(4, num_dims)
        # rows = math.ceil(num_dims / cols)
        # fig_dims, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), sharex=True)
        # axs = np.atleast_2d(axs)
        
        # for dim in range(num_dims):
        #     r, c = divmod(dim, cols)
        #     ax = axs[r, c]
        #     ax.plot(time_axis_actions, action_matrix[:, dim], linewidth=1.2, color='purple')
        #     ax.set_title(f"Dimension {dim}", fontsize=9)
        #     ax.grid(True, alpha=0.2)
        #     if r == rows - 1:
        #         ax.set_xlabel('Time (ms)', fontsize=9)
        #     ax.set_ylabel('Value', fontsize=9)
            
        #     # Draw chunk boundaries
        #     if len(self._chunk_boundaries) > 0:
        #         for boundary_step in self._chunk_boundaries:
        #             boundary_time = boundary_step * step_duration_ms
        #             ax.axvline(x=boundary_time, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # # Hide unused subplots if any
        # total_plots = rows * cols
        # if total_plots > num_dims:
        #     for dim in range(num_dims, total_plots):
        #         r, c = divmod(dim, cols)
        #         fig_dims.delaxes(axs[r, c])
        
        # plt.tight_layout()
        # plt.savefig('data/action_dimensions_history.png', dpi=150, bbox_inches='tight')
        # plt.close()

    def _update_latency_stats(self) -> None:
        """Calculate average latency and number of actions during that latency.
        
        Must be called within a lock context.
        """
        if len(self._infer_latency) > 0:
            # Calculate average latency in milliseconds
            self._avg_latency_ms = float(np.mean(self._infer_latency))
            
            # Calculate how many actions are executed during this latency
            # latency is in ms, fps is actions per second
            # actions = (latency_ms / 1000) * fps
            self._actions_during_latency_realtime = max(0, self._cur_step - self._action_horizon) #int((self._avg_latency_ms / 1000.0) * self._fps)
            if self._actions_during_latency_realtime > self._action_horizon:
                self._actions_during_latency_realtime = self._action_horizon
        else:
            self._avg_latency_ms = 0.0
            self._actions_during_latency_realtime = self._actions_during_latency  # --- IGNORE ---

    def _async_inference(self, obs: Dict, prev_action: Dict, prev_obs: Dict) -> None:
        """Run inference in background and update results when ready."""
        try:
            a_start = time.time()
            if self._use_rtc:
                print(f"prev_action: {prev_action['actions'][:self._actions_during_latency,...]}")
                rtc_obs = {
                    'prev_action': prev_action['actions'],
                    'prev_state': prev_obs['state'],
                    's': self._action_horizon,
                    'd': self._actions_during_latency,
                }
                obs['rtc_obs'] = rtc_obs
            # Measure actual inference time
            #print(f"Obtained obs times used: {obs['timestamps']['robot']-obs['timestamps']['start']}")
            #print(f"Delta time from obtained to model infer start: {time.time() - obs['timestamps']['robot']} ")
            start_time = time.time()
            print(f"transmit time: {(time.time() - prev_action['timestamps']) * 1000.0} ms")
            new_results = self._policy.infer(obs)
            end_time = time.time()
            
            # Calculate inference latency in milliseconds
            infer_ms = (end_time - start_time) * 1000.0
            
            with self._lock:
                # Update inference latency buffer with measured time
                self._infer_latency.append(infer_ms)
                # Calculate average latency and actions during latency
                self._update_latency_stats()
                
                # Calculate and print average action variation error
                if self._error_count > 0:
                    avg_error = self._total_error / self._error_count
                    print(f"action chunk updated, avg_action_error={avg_error:.6f}, infer_latency={self._avg_latency_ms}, s={self._action_horizon}, d={self._actions_during_latency_realtime}, cur_step={self._cur_step}")
                else:
                    print(f"action chunk updated, infer_latency={self._avg_latency_ms}, s={self._action_horizon}, d={self._actions_during_latency_realtime}, cur_step={self._cur_step}")
                
                # Reset error tracking for new chunk
                self._total_error = 0.0
                self._error_count = 0
                #self._prev_returned_action = None
                
                # Record chunk boundary: new chunk starts at current number of executed steps
                boundary_step = max(len(self._action_history) - 1, 0)
                self._chunk_boundaries.append(boundary_step)
                
                # def slicer_excute(x):
                #     if isinstance(x, np.ndarray):
                #         already_executed = x[0:self._actions_during_latency_realtime, ...]
                #         return x[self._actions_during_latency_realtime:, ...]#np.concatenate([already_executed, x], axis=0)
                #     else:
                #         return x
                
                # def slicer_prev(x):
                #     if isinstance(x, np.ndarray):
                #         # Get the actions from _actions_during_latency onwards
                #         sliced = x[self._action_horizon:, ...]
                #         # Create zero padding with the same shape as one action
                #         zero_shape = (self._action_horizon,) + sliced.shape[1:]
                #         zeros = np.zeros(zero_shape, dtype=sliced.dtype)
                #         # Concatenate along the first dimension
                #         return np.concatenate([sliced, zeros], axis=0)
                #     else:
                #         return x
                
                def slicer_smooth(x):
                    if isinstance(x, np.ndarray):
                        prev_actions = self._last_results['actions']
                        # 取上一段与当前段相同区间的动作
                        latency_len = self._actions_during_latency
                        prev_sliced = prev_actions[
                            self._cur_step : self._cur_step + latency_len, ...
                        ]
                        current_sliced = x[
                            self._actions_during_latency_realtime : self._actions_during_latency_realtime + latency_len, ...
                        ]

                        # 设计权重：越靠近后面的动作给予当前结果更大权重（线性增长）
                        weights = np.linspace(
                            0.2, 0.8, num=latency_len, dtype=np.float32
                        ).reshape(
                            (latency_len,) + (1,) * (current_sliced.ndim - 1)
                        )
                        smooth_sliced = prev_sliced * (1.0 - weights) + current_sliced * weights

                        fixed_sliced = x[self._actions_during_latency_realtime + latency_len :, ...]
                        return np.concatenate([smooth_sliced, fixed_sliced], axis=0)
                    else:
                        return x
                
                self._record_chunk_actions(self._last_results, self._actions_during_latency_realtime)
                # self._prev_results = tree.map_structure(slicer_prev, new_results)
                # smooth the last results
                # smooth_results = tree.map_structure(slicer_excute, new_results)
                self._last_results = new_results #smooth_results #tree.map_structure(slicer_excute, new_results)
                self._prev_obs = obs.copy() # for delta action culation
                self._cur_step = self._actions_during_latency_realtime
            a_end = time.time()
            print(f"infer time: {infer_ms} ms, async time: {(a_end - a_start) * 1000.0} ms")

        except Exception as e:
            print(f"Error in async inference: {e}")

    def _warmup_inference(self, obs: Dict) -> None:
        # if self._use_rtc:
        #     rtc_obs = {
        #         'prev_action': np.zeros((self._max_horizon, self._action_dim)),
        #         'prev_state': np.zeros((self._action_dim)),
        #         's': self._action_horizon,
        #         'd': self._actions_during_latency,
                
        #     }
        #     obs['rtc_obs'] = rtc_obs
        results = self._policy.infer(obs)
        return results

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._lock:
            # Store the latest observation
            self._last_obs = obs
            
            #print(f"Delta time from obtained to infer start: {time.time() - obs['timestamps']['robot']}")

            # Initialize on first call
            if self._last_results is None:
                # Measure actual inference time
                start_time = time.time()
                self._last_results = self._warmup_inference(obs)
                end_time = time.time()
                # Calculate inference latency in milliseconds
                infer_ms = (end_time - start_time) * 1000.0
                # Update inference latency buffer with measured time
                self._infer_latency.append(infer_ms)
                # Calculate average latency and actions during latency
                self._update_latency_stats()
                self._cur_step = 0
                
                # def slicer_prev(x):
                #     if isinstance(x, np.ndarray):
                #         # Get the actions from _actions_during_latency onwards
                #         sliced = x[self._action_horizon:, ...]
                #         # Create zero padding with the same shape as one action
                #         zero_shape = (self._action_horizon,) + sliced.shape[1:]
                #         zeros = np.zeros(zero_shape, dtype=sliced.dtype)
                #         # Concatenate along the first dimension
                #         return np.concatenate([sliced, zeros], axis=0)
                #     else:
                #         return x
                # # Update previous results with zero-padded actions
                # self._prev_results = tree.map_structure(slicer_prev, self._last_results)
                self._prev_obs = obs.copy()
                #self._record_chunk_actions(self._last_results, overlap_steps=0)
                
                # Record first chunk boundary at step 0
                if not self._chunk_boundaries:
                    self._chunk_boundaries.append(0)

            def slicer(x):
                if isinstance(x, np.ndarray):
                    return x[self._cur_step, ...]
                else:
                    return x
            results = tree.map_structure(slicer, self._last_results)
            
            # Calculate action variation for logging (position change only)
            if self._prev_returned_action is not None:
                if 'actions' in results and 'actions' in self._prev_returned_action:
                    current_actions = results['actions']
                    prev_actions = self._prev_returned_action['actions']
                    if isinstance(current_actions, np.ndarray) and isinstance(prev_actions, np.ndarray):
                        position_error = float(np.linalg.norm(current_actions - prev_actions))
                        self._total_error += position_error
                        self._error_count += 1
            
            # Update previous returned action (only save 'actions' field) and record history
            if 'actions' in results:
                action_value = results['actions']
                if isinstance(action_value, np.ndarray):
                    action_copy = action_value.copy()
                else:
                    action_copy = np.asarray(action_value).copy()
                self._action_history.append(action_copy)
                if self._prev_returned_action is None:
                    self._prev_returned_action = {}
                self._prev_returned_action['actions'] = action_copy
            

            # Trigger async inference 
            if self._cur_step >= self._action_horizon:
                # Only start new thread if previous one is done
                if self._inference_thread is None or not self._inference_thread.is_alive():

                    def slicer_current(x):
                        if isinstance(x, np.ndarray):
                            # Get the actions from _actions_during_latency onwards
                            sliced = x[self._cur_step:, ...]
                            # # Create zero padding with the same shape as one action
                            zero_shape = (self._cur_step,) + sliced.shape[1:]
                            zeros = np.zeros(zero_shape, dtype=sliced.dtype)
                            # Concatenate along the first dimension
                            sliced = np.concatenate([sliced, zeros], axis=0)
                            return sliced
                        else:
                            
                            return x
                    
                    prev_action = tree.map_structure(
                        slicer_current,
                        self._last_results.copy()
                    )
                    prev_action['timestamps'] = time.time()
                    
                    #print(f"mother fuckerrrrrrrrrrr prev_action: {prev_action['actions'].shape}, cur_step: {self._cur_step}")
                    self._inference_thread = threading.Thread(
                        target=self._async_inference,
                        args=(self._last_obs.copy(), prev_action, self._prev_obs.copy()),
                        daemon=True
                    )
                    self._inference_thread.start()

            self._cur_step += 1
            # If no action can be executed, wait
            # if self._cur_step >= self._action_horizon:
            #     self._cur_step = 0
        
        return results

    @override
    def reset(self) -> None:
        with self._lock:
            self._should_stop = True
            # Wait for inference thread to complete if running
            if self._inference_thread is not None and self._inference_thread.is_alive():
                # Don't wait too long, just let it finish naturally
                pass
            
            self._policy.reset()
            self._last_results = None
            self._last_obs = None
            self._cur_step = 0
            self._inference_thread = None
            self._should_stop = False
            self._infer_latency.clear()
            self._avg_latency_ms = 0.0
            self._prev_returned_action = None
            self._total_error = 0.0
            self._error_count = 0
            self._action_history.clear()
            self._chunk_boundaries.clear()
            self._chunk_action_records.clear()
    
class ActionChunkBroker_RTC_Fake(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time with async inference.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is made asynchronously every
    action_horizon steps, allowing the current chunks to be returned without
    blocking on model inference.
    """
    
    _instances = []  # Class-level list to track all instances
    _atexit_registered = False  # Flag to ensure atexit is only registered once

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, fps: int = 30, actions_during_latency: int = 1, use_rtc: bool = True):
        self._policy = policy
        self._max_horizon = 50
        self._action_dim = 14
        self._action_horizon = action_horizon
        self._fps = fps
        self._infer_latency = deque(maxlen=10)  # Buffer for last 10 inference latencies
        self._avg_latency_ms: float = 0.0  # Average latency in milliseconds
        self._actions_during_latency: int = 0 #actions_during_latency  # Number of actions executed during avg latency
        self._actions_during_latency_realtime: int = 0  # Number of actions executed during avg latency
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_obs: Dict | None = None
        self._prev_results: Dict[str, np.ndarray] | None = None
        self._prev_obs: Dict | None = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._inference_thread: threading.Thread | None = None
        self._should_stop = False

        self._use_rtc = use_rtc
        
        # Action variation tracking
        self._prev_returned_action: Dict | None = None
        self._total_error: float = 0.0
        self._error_count: int = 0
        self._action_history: List[np.ndarray] = []  # Store raw action vectors for post-analysis
        self._chunk_boundaries: List[int] = []  # Store step indices where new chunks start
        
    # @classmethod
    # def _plot_all_instances(cls) -> None:
    #     """Plot error history for all instances when program exits."""
    #     for instance in cls._instances:
    #         if hasattr(instance, '_step_errors') and len(instance._step_errors) > 0:
    #             instance._plot_error_history()
    
    def _update_latency_stats(self) -> None:
        """Calculate average latency and number of actions during that latency.
        
        Must be called within a lock context.
        """
        if len(self._infer_latency) > 0:
            # Calculate average latency in milliseconds
            self._avg_latency_ms = float(np.mean(self._infer_latency))
            
            # Calculate how many actions are executed during this latency
            # latency is in ms, fps is actions per second
            # actions = (latency_ms / 1000) * fps
            self._actions_during_latency_realtime = 0 #self._cur_step - self._action_horizon #int((self._avg_latency_ms / 1000.0) * self._fps)
            
        else:
            self._avg_latency_ms = 0.0
            self._actions_during_latency_realtime = self._actions_during_latency  # --- IGNORE ---
    
    def _plot_error_history(self) -> None:
        """Plot action variation over time (position change, velocity change, acceleration change)."""
        if len(self._action_history) < 2:
            return
        
        step_duration_ms = 20.0  # Each step lasts 20ms
        
        def flatten_action(action: np.ndarray) -> np.ndarray:
            return np.asarray(action).astype(np.float64).reshape(-1)
        
        action_matrix = np.stack([flatten_action(a) for a in self._action_history], axis=0)
        
        position_vectors = action_matrix[1:] - action_matrix[:-1]
        position_changes = np.linalg.norm(position_vectors, axis=1)
        
        velocity_change_vectors = np.diff(position_vectors, axis=0) if position_vectors.shape[0] >= 2 else np.empty((0, action_matrix.shape[1]))
        velocity_change_magnitudes = np.linalg.norm(velocity_change_vectors, axis=1) if velocity_change_vectors.size > 0 else np.array([])
        
        acceleration_change_vectors = np.diff(velocity_change_vectors, axis=0) if velocity_change_vectors.shape[0] >= 2 else np.empty((0, action_matrix.shape[1] if velocity_change_vectors.size > 0 else 0))
        acceleration_change_magnitudes = np.linalg.norm(acceleration_change_vectors, axis=1) if acceleration_change_vectors.size > 0 else np.array([])
        
        def build_time_series(values: np.ndarray, offset_steps: int) -> Tuple[List[float], List[float]]:
            time_points: List[float] = []
            plot_values: List[float] = []
            for i, val in enumerate(values):
                step_index = i + offset_steps
                step_start_time = step_index * step_duration_ms
                step_end_time = (step_index + 1) * step_duration_ms
                time_points.extend([step_start_time, step_end_time])
                plot_values.extend([float(val), float(val)])
            return time_points, plot_values
        
        time_points_pos, pos_values = build_time_series(position_changes, offset_steps=0)
        time_points_vel, vel_values = build_time_series(velocity_change_magnitudes, offset_steps=1) if velocity_change_magnitudes.size > 0 else ([], [])
        time_points_acc, acc_values = build_time_series(acceleration_change_magnitudes, offset_steps=2) if acceleration_change_magnitudes.size > 0 else ([], [])
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        axes[0].plot(time_points_pos, pos_values, linewidth=2, drawstyle='steps-post', color='blue', label='Position Change')
        axes[0].set_ylabel('Position Change (L2 norm)', fontsize=11)
        axes[0].set_title('Action Variation: Position, Velocity Change, Acceleration Change', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=9)
        
        if len(vel_values) > 0:
            axes[1].plot(time_points_vel, vel_values, linewidth=2, drawstyle='steps-post', color='green', label='Velocity Change (Acceleration)')
        axes[1].set_ylabel('Velocity Change (L2 norm)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=9)
        
        if len(acc_values) > 0:
            axes[2].plot(time_points_acc, acc_values, linewidth=2, drawstyle='steps-post', color='orange', label='Acceleration Change (Jerk)')
        axes[2].set_ylabel('Acceleration Change (L2 norm)', fontsize=11)
        axes[2].set_xlabel('Time (ms)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right', fontsize=9)
        
        if len(self._chunk_boundaries) > 0:
            for boundary_step in self._chunk_boundaries:
                boundary_time = boundary_step * step_duration_ms
                label = 'Chunk Boundary' if boundary_step == self._chunk_boundaries[0] else ''
                for ax in axes:
                    ax.axvline(x=boundary_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                    if label:
                        label = ''
        
        plt.tight_layout()
        plt.savefig('data/chunk_error_history.pdf', bbox_inches='tight')
        plt.close()

    def _async_inference(self, obs: Dict, prev_action: Dict, prev_obs: Dict) -> None:
        """Run inference in background and update results when ready."""
        try:
            if self._use_rtc:
                rtc_obs = {
                    'prev_action': prev_action['actions'],
                    'prev_state': prev_obs['state'],
                    's': self._action_horizon,
                    'd': self._actions_during_latency,
                
                }
                obs['rtc_obs'] = rtc_obs
            # Measure actual inference time
            start_time = time.time()
            new_results = self._policy.infer(obs)
            end_time = time.time()
            
            # Calculate inference latency in milliseconds
            infer_ms = (end_time - start_time) * 1000.0
            
            with self._lock:
                # Update inference latency buffer with measured time
                self._infer_latency.append(infer_ms)
                # Calculate average latency and actions during latency
                self._update_latency_stats()
                
                # Calculate and print average action variation error
                if self._error_count > 0:
                    avg_error = self._total_error / self._error_count
                    print(f"action chunk updated, infer_latency={self._avg_latency_ms}, s={self._action_horizon}, d={self._actions_during_latency}, cur_step={self._cur_step}, avg_action_error={avg_error:.6f}")
                else:
                    print(f"action chunk updated, infer_latency={self._avg_latency_ms}, s={self._action_horizon}, d={self._actions_during_latency}, cur_step={self._cur_step}")
                
                # Reset error tracking for new chunk
                self._total_error = 0.0
                self._error_count = 0
                
                # def slicer_excute(x):
                #     if isinstance(x, np.ndarray):
                #         return x[self._actions_during_latency_realtime:, ...]
                #     else:
                #         return x
                
                def slicer_prev(x):
                    if isinstance(x, np.ndarray):
                        # Get the actions from _actions_during_latency onwards
                        sliced = x[self._action_horizon:, ...]
                        # Create zero padding with the same shape as one action
                        zero_shape = (self._action_horizon,) + sliced.shape[1:]
                        zeros = np.zeros(zero_shape, dtype=sliced.dtype)
                        # Concatenate along the first dimension
                        return np.concatenate([sliced, zeros], axis=0)
                    else:
                        return x
                
                self._prev_results = tree.map_structure(slicer_prev, new_results)
                self._last_results = new_results #tree.map_structure(slicer_excute, new_results)
                self._prev_obs = obs.copy()
                self._cur_step = 0
                
                # Record chunk boundary: new chunk starts at current number of executed steps
                boundary_step = max(len(self._action_history) - 1, 0)
                self._chunk_boundaries.append(boundary_step)
        except Exception as e:
            print(f"Error in async inference: {e}")

    def _warmup_inference(self, obs: Dict) -> None:
        if self._use_rtc:
            rtc_obs = {
                    'prev_action': np.zeros((self._max_horizon, self._action_dim)),
                    'prev_state': np.zeros((self._action_dim)),
                    's': self._action_horizon,
                    'd': self._actions_during_latency,
                    
                }
            obs['rtc_obs'] = rtc_obs
        results = self._policy.infer(obs)
        return results

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._lock:
            # Store the latest observation
            self._last_obs = obs
            
            # Initialize on first call
            if self._last_results is None:
                # Measure actual inference time
                start_time = time.time()
                self._last_results = self._warmup_inference(obs)
                end_time = time.time()
                # Calculate inference latency in milliseconds
                infer_ms = (end_time - start_time) * 1000.0
                # Update inference latency buffer with measured time
                self._infer_latency.append(infer_ms)
                # Calculate average latency and actions during latency
                self._update_latency_stats()
                self._cur_step = 0
                
                def slicer_prev(x):
                    if isinstance(x, np.ndarray):
                        # Get the actions from _actions_during_latency onwards
                        sliced = x[self._action_horizon:, ...]
                        # Create zero padding with the same shape as one action
                        zero_shape = (self._action_horizon,) + sliced.shape[1:]
                        zeros = np.zeros(zero_shape, dtype=sliced.dtype)
                        # Concatenate along the first dimension
                        return np.concatenate([sliced, zeros], axis=0)
                    else:
                        return x
                # Update previous results with zero-padded actions
                self._prev_results = tree.map_structure(slicer_prev, self._last_results)
                self._prev_obs = obs.copy()
                
                # Record first chunk boundary at step 0
                if not self._chunk_boundaries:
                    self._chunk_boundaries.append(0)

            def slicer(x):
                if isinstance(x, np.ndarray):
                    return x[self._cur_step, ...]
                else:
                    return x
            results = tree.map_structure(slicer, self._last_results)
            
            # Calculate action variation for logging (position change only)
            if self._prev_returned_action is not None:
                if 'actions' in results and 'actions' in self._prev_returned_action:
                    current_actions = results['actions']
                    prev_actions = self._prev_returned_action['actions']
                    if isinstance(current_actions, np.ndarray) and isinstance(prev_actions, np.ndarray):
                        position_error = float(np.linalg.norm(current_actions - prev_actions))
                        self._total_error += position_error
                        self._error_count += 1
            
            # Update previous returned action (only save 'actions' field) and record history
            if 'actions' in results:
                action_value = results['actions']
                if isinstance(action_value, np.ndarray):
                    action_copy = action_value.copy()
                else:
                    action_copy = np.asarray(action_value).copy()
                self._action_history.append(action_copy)
                if self._prev_returned_action is None:
                    self._prev_returned_action = {}
                self._prev_returned_action['actions'] = action_copy
            
            if self._cur_step >= self._action_horizon:
                # Only start new thread if previous one is done
                if self._inference_thread is None or not self._inference_thread.is_alive():
                    self._inference_thread = threading.Thread(
                        target=self._async_inference,
                        args=(self._last_obs,self._prev_results,self._prev_obs),
                        daemon=True
                    )
                    self._inference_thread.start()
            else:
                self._cur_step += 1

            # Trigger async inference 
            
                
            
            # If no action can be executed, wait
            # if self._cur_step >= self._action_horizon:
            #     self._cur_step = 0
        
        return results

    @override
    def reset(self) -> None:
        with self._lock:
            self._should_stop = True
            # Wait for inference thread to complete if running
            if self._inference_thread is not None and self._inference_thread.is_alive():
                # Don't wait too long, just let it finish naturally
                pass
            
            self._policy.reset()
            self._last_results = None
            self._last_obs = None
            self._cur_step = 0
            self._inference_thread = None
            self._should_stop = False
            self._infer_latency.clear()
            self._avg_latency_ms = 0.0
            self._prev_returned_action = None
            self._total_error = 0.0
            self._error_count = 0
            self._action_history.clear()
            self._chunk_boundaries.clear()
    
