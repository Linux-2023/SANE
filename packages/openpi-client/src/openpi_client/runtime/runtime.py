import logging
import threading
import time
from typing import List, Optional

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: List[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
        reset_pormpt: Optional[str] = None,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
        self._reset_pormpt = reset_pormpt

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()
        time.sleep(1)
        try:
            self._environment.close()
        except:
            logging.warning("Failed to close environment.")

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        if self._reset_pormpt is not None:
            self._reset_with_policy()
            time.sleep(1)

        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            self._step()
            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
            print("Running step:", self._episode_steps)
            self._episode_steps += 1

        logging.info("Episode completed.")
        #for subscriber in self._subscribers:
        #    subscriber.on_episode_end()
        
        # custom function
        try:
            self._environment._save_episode_to_hdf5()
        except Exception as exc:
            logging.exception("Failed to save episode: %s", exc)
        try:
            self._agent._policy._plot_error_history()
        except Exception as exc:
            logging.exception("Failed to plot error history: %s", exc)
        

    def _step(self) -> None:
        """A single step of the runtime loop."""
        start = time.time()
        observation = self._environment.get_observation()

        obs_time = time.time()
        action = self._agent.get_action(observation)

        action_time = time.time()
        self._environment.apply_action(action)
        apply_time = time.time()
        
        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
            try:
                self._environment.plot_action_diff()
            except Exception as exc:
                logging.exception("Failed to plot action difference: %s", exc)
        end = time.time()
        
        # print(
        #     f"Timing: obs {obs_time - start:.3f}s, infer {action_time - obs_time:.3f}s, apply {apply_time - action_time:.3f}s, other {end - apply_time:.3f}s"
        # )

    def _reset_with_policy(self) -> None:
        self._environment.reset()
        self._agent.reset()

        reset_steps = int(self._environment._max_episode_steps/5)
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()
        for i in range(reset_steps):
            observation = self._environment.get_observation()
            observation["prompt"] = self._reset_pormpt
            print(self._reset_pormpt)
            action = self._agent.get_action(observation)
            self._environment.apply_action(action)
            
            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
            print("Running reset step:", i)
        print("Reset complete")
        self._environment.reset()
        self._agent.reset()

        