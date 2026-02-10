import dataclasses
import logging
import pathlib
import signal
import sys
import os
import threading
import time
from typing import Optional

import env_dual_feetech as _env_dual_feetech
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


@dataclasses.dataclass
class Args:
    """Flexiv 双臂环境主程序参数（Feetech 夹爪版本）"""
    # 输出目录
    out_dir: pathlib.Path = pathlib.Path("data/flexiv/videos")
    
    # 配置文件路径（如果提供，将从配置文件加载参数）
    config_path: str = "./dual_arm_env_feetech_config.yaml"
    
    # 随机种子
    seed: int = 0
    
    # 动作相关参数
    action_horizon: int = 15
    fps: int = 30
    actions_during_latency: int = 8
    #num_steps: int = 6000
    num_episodes: int = 1
    
    # WebSocket 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 显示配置
    display: bool = False
    
    # 左臂配置（如果未使用配置文件）
    left_robot_sn: str = ""
    left_gripper_side: str = "left"
    left_gripper_config_file: Optional[str] = None
    left_wrist_camera_id: int = 2
    left_wrist_camera_type: str = "usb"  # "usb", "realsense", "gopro"
    
    # 右臂配置（如果未使用配置文件）
    right_robot_sn: str = ""
    right_gripper_side: str = "right"
    right_gripper_config_file: Optional[str] = None
    right_wrist_camera_id: int = 0
    right_wrist_camera_type: str = "usb"
    
    # 全局相机配置（如果未使用配置文件）
    high_camera_id: Optional[int] = None
    high_camera_type: str = "usb"
    
    # 相机分辨率
    wrist_camera_width: int = 480
    wrist_camera_height: int = 480
    high_camera_width: int = 1280
    high_camera_height: int = 960
    
    # 任务提示
    prompt: str = "put the object in the box"
    reset_pormpt: str = ""
    
    # 运行时配置
    use_async: bool = True
    use_rtc: bool = True
    record_mode: bool = False
    
    # 环境配置（如果未使用配置文件）
    max_episode_steps: int = 10000
    watchdog_timeout: float = 5.0
    collision_check_enabled: bool = True
    collision_safety_threshold: float = 0.03
    # 夹爪阈值（Feetech 版本）
    close_threshold: float = 0.05
    open_threshold: float = 0.95
    # 动作缩放参数（用于控制模型推理动作的应用幅度）
    action_scale: float = 1.0  # 1.0 = 完全应用，< 1.0 = 减小幅度


def main(args: Args) -> None:
    print(args)
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
        print("🚀 正在初始化 Flexiv 双臂环境（Feetech 夹爪版本）...")
        
        # 尝试从配置文件加载环境
        if os.path.exists(args.config_path):
            print(f"📄 从配置文件加载: {args.config_path}")
            environment = _env_dual_feetech.create_dual_arm_feetech_env_from_config(
                args.config_path,
                seed=args.seed,
                tele_mode=False,
                prompt=args.prompt,
                max_episode_steps=args.max_episode_steps,
                watchdog_timeout=args.watchdog_timeout,
                show_camera_preview=args.display,
                record_mode=args.record_mode,
                action_scale=args.action_scale,  # 允许命令行参数覆盖配置文件
            )
            print("environment_max_episode_steps:", environment._max_episode_steps)
        else:
            print("⚠️  配置文件不存在，使用命令行参数...")
            # 使用命令行参数创建环境
            environment = _env_dual_feetech.DualArmEnvironmentFeetech(
                # 左臂配置
                left_robot_sn=args.left_robot_sn,
                left_gripper_side=args.left_gripper_side,
                left_gripper_config_file=args.left_gripper_config_file,
                left_wrist_camera_id=args.left_wrist_camera_id,
                left_wrist_camera_type=args.left_wrist_camera_type,
                # 右臂配置
                right_robot_sn=args.right_robot_sn,
                right_gripper_side=args.right_gripper_side,
                right_gripper_config_file=args.right_gripper_config_file,
                right_wrist_camera_id=args.right_wrist_camera_id,
                right_wrist_camera_type=args.right_wrist_camera_type,
                # 全局相机配置
                high_camera_id=args.high_camera_id,
                high_camera_type=args.high_camera_type,
                # 相机分辨率
                wrist_camera_width=args.wrist_camera_width,
                wrist_camera_height=args.wrist_camera_height,
                high_camera_width=args.high_camera_width,
                high_camera_height=args.high_camera_height,
                camera_fps=args.fps,
                # 环境参数
                max_episode_steps=args.max_episode_steps,
                seed=args.seed,
                tele_mode=False,
                prompt=args.prompt,
                watchdog_timeout=args.watchdog_timeout,
                show_camera_preview=args.display,
                record_mode=args.record_mode,
                # 碰撞检测
                collision_check_enabled=args.collision_check_enabled,
                collision_safety_threshold=args.collision_safety_threshold,
                # 夹爪阈值
                close_threshold=args.close_threshold,
                open_threshold=args.open_threshold,
                # 动作缩放参数
                action_scale=args.action_scale,
            )
        
        print("✅ Flexiv 双臂环境（Feetech 夹爪版本）初始化成功")
        
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
            print("  5. Feetech 夹爪通信失败")
            print("\n建议的解决方案:")
            print("  1. 检查所有 USB 连接是否牢固")
            print("  2. 检查 CAN 总线连接")
            print("  3. 检查 Feetech 夹爪连接和配置")
            print("  4. 尝试更换 USB 端口（使用主板直连端口而非 HUB）")
            print("  5. 检查电源供应是否充足")
            print("  6. 重启机械臂和相机")
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

