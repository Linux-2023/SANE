import dataclasses
import logging
import pathlib
import signal
import sys
import os
import threading
import time

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/piper/videos")

    seed: int = 0

    action_horizon: int = 15
    fps: int = 30
    actions_during_latency: int = 5
    num_steps: int = 6000
    num_episodes: int = 1
    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False

    high_camera_id: int = 8
    left_wrist_camera_id: int = 4

    prompt: str = "pick up the bottle"
    reset_pormpt: str = "Pour the objects in the box onto the table"

    use_async: bool = True
    use_rtc: bool = True
    gripper_norm: bool = True
    record_mode: bool = False

def main(args: Args) -> None:
    environment = None
    runtime = None
    emergency_shutdown = threading.Event()
    
    # 设置信号处理器来捕获 Ctrl+C
    def signal_handler(sig, frame):
        print("\n🛑 检测到 Ctrl+C，正在安全关闭...")
        emergency_shutdown.set()
        try:
            if environment:
                environment.close()
        except Exception as e:
            print(f"❌ 关闭环境时出错: {e}")
        finally:
            print("✅ 程序已退出")
            os._exit(0)  # 强制退出
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🚀 正在初始化 Piper 环境...")
        environment = _env.PiperEnvironment(
            high_camera_id=args.high_camera_id,
            left_wrist_camera_id=args.left_wrist_camera_id,
            camera_fps=args.fps,
            seed=args.seed,
            tele_mode=False,
            prompt=args.prompt,
            max_episode_steps=args.num_steps,
            watchdog_timeout=5.0,  # 5秒超时
            show_usb_camera=args.display,
            gripper_norm=args.gripper_norm,
            record_mode=args.record_mode,
        )
        print("✅ Piper 环境初始化成功")
        
        print("🚀 正在初始化运行时...")
        if args.use_async:
            runtime = _runtime.Runtime(
                environment=environment,
                agent=_policy_agent.PolicyAgent(
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
                ),
                subscribers=[
                    _saver.VideoSaver(args.out_dir),
                ],
                max_hz=args.fps,
                #reset_pormpt=args.reset_pormpt,
                num_episodes=args.num_episodes,
                )
        else:
            runtime = _runtime.Runtime(
                    environment=environment,
                    agent=_policy_agent.PolicyAgent(
                        policy=action_chunk_broker.ActionChunkBroker(
                            policy=_websocket_client_policy.WebsocketClientPolicy(
                                host=args.host,
                                port=args.port,
                            ),
                            action_horizon=args.action_horizon,
                            fps=args.fps,
                        )
                    ),
                    subscribers=[
                        _saver.VideoSaver(args.out_dir),
                    ],
                    max_hz=args.fps,
                    num_episodes=args.num_episodes,
                )
        print("✅ 运行时初始化成功")
        
        print("🏃 开始运行策略...")
        runtime.run()
        
    except RuntimeError as e:
        # 捕获设备故障错误
        error_msg = str(e)
        if "看门狗超时" in error_msg or "不健康状态" in error_msg or "断开连接" in error_msg:
            print(f"\n{'='*60}")
            print("🚨 检测到硬件设备故障！")
            print(f"{'='*60}")
            print(f"错误信息: {e}")
            print("\n可能的原因:")
            print("  1. USB 相机（腕部相机或全局相机）突然断开连接")
            print("  2. CAN 总线连接中断，机械臂失去通信")
            print("  3. USB 带宽不足或供电不稳定")
            print("  4. 驱动程序卡死")
            print("\n建议的解决方案:")
            print("  1. 检查所有 USB 连接是否牢固")
            print("  2. 检查 CAN 总线连接: bash third_party/piper_sdk/piper_sdk/find_all_can_port.sh")
            print("  3. 尝试更换 USB 端口（使用主板直连端口而非 HUB）")
            print("  4. 检查电源供应是否充足")
            print("  5. 重启机械臂和相机")
            print("  6. 重新运行 can_activate.sh 脚本")
            print(f"{'='*60}\n")
        else:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n🛑 检测到 Ctrl+C，正在安全关闭...")
    
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔄 正在清理资源...")
        if environment:
            try:
                environment.close()
            except Exception as e:
                print(f"❌ 关闭环境时出错: {e}")
        
        print("✅ 程序已完成")
        # 确保程序完全退出
        time.sleep(0.5)
        os._exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
