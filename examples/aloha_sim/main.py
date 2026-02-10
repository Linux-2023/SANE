import dataclasses
import logging
import pathlib

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0"
    seed: int = 42

    action_horizon: int = 15
    actions_during_latency : int = 8 # action_horizon >= actions_during_latency

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False

    fps: int = 30


def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker_RTC(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
                fps=args.fps,
                actions_during_latency=args.actions_during_latency,
                use_rtc=True,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=args.fps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
