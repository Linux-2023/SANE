import dataclasses
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro


@dataclasses.dataclass
class Args:
    dataset_path: pathlib.Path = pathlib.Path("/workspace/pjk/ELM/openpi/datasets/flexiv/gripperpap/train/flexiv_lerobot_data_pose")
    episode_index: int = 0
    host: str = "0.0.0.0"
    port: int = 8000
    action_horizon: int = 1
    fps: int = 30
    actions_during_latency: int = 5
    num_episodes: int = 1
    use_async: bool = False
    use_rtc: bool = False


def main(args: Args) -> None:
    env = _env.DatasetSimEnvironment(
        dataset_path=str(args.dataset_path),
        episode_index=args.episode_index,
    )
    env.reset()

    if args.use_async:
        agent = _policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker_RTC(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
                fps=args.fps,
                actions_during_latency=args.actions_during_latency,
                use_rtc=args.use_rtc,
            )
        )
    else:
        agent = _policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
                fps=args.fps,
            )
        )

    runtime = _runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[],
        max_hz=args.fps,
        num_episodes=args.num_episodes,
    )
    runtime.run()


if __name__ == "__main__":
    tyro.cli(main)
