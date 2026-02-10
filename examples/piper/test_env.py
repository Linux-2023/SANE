#!/usr/bin/env python3
"""Test script for PiperEnvironment with real hardware.

This demonstrates the basic reset/observe/act loop for the
PiperEnvironment that integrates Piper robot arm and RealSense camera.
"""
import time
import numpy as np
from env import PiperEnvironment
import random
import cv2


def display_observations(obs: dict, window_name: str = "Observations"):
    """Display all images from observation dict in a single window.
    
    Args:
        obs: Observation dictionary with 'images' key containing camera images
        window_name: Name of the display window
    """
    images_dict = obs.get('images', {})
    
    if not images_dict:
        return
    
    # Convert images from CHW to HWC format and RGB to BGR for OpenCV display
    display_images = []
    labels = []
    
    for cam_name, img in images_dict.items():
        # img is in CHW format (3, 224, 224), convert to HWC
        img_hwc = np.transpose(img, (1, 2, 0))  # (224, 224, 3)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = img_hwc[:, :, ::-1].copy()
        
        # Add label to image
        img_labeled = img_bgr.copy()
        cv2.putText(img_labeled, cam_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        display_images.append(img_labeled)
        labels.append(cam_name)
    
    # Concatenate images horizontally
    if len(display_images) == 1:
        combined = display_images[0]
    else:
        combined = np.hstack(display_images)
    
    # Display
    cv2.imshow(window_name, combined)
    cv2.waitKey(1)  # Small delay to update window


def test_basic_loop(tele_mode=False):
    """Test basic environment reset and action loop.
    
    Args:
        tele_mode: Whether to enable recording mode to save episode data
    """
    print("=== Testing PiperEnvironment ===\n")
    print(f"Record mode: {'ENABLED' if tele_mode else 'DISABLED'}\n")
    
    # Create environment with default settings
    env = PiperEnvironment(
        can_port="can0",
        camera_fps=30,
        high_camera_id=8,
        left_wrist_camera_id=4,
        max_episode_steps=50,
        initial_joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001],
        tele_mode=tele_mode
    )
    
    try:
        # Create display window
        cv2.namedWindow("Observations", cv2.WINDOW_AUTOSIZE)
        
        # Reset environment
        print("Resetting environment...")
        env.reset()
        
        # Get initial observation
        obs = env.get_observation()
        print(f"Initial observation:")
        print(f"  - State shape: {obs['state'].shape}")
        print(f"  - State values: {obs['state']}")
        print(f"  - Image keys: {list(obs['images'].keys())}")
        print(f"  - Image shape: {obs['images']['cam_high'].shape}")

        # Print timestamp differences
        timestamps = obs.get("timestamps", {})
        if timestamps:
            robot_ts = timestamps.get("robot", 0)
            wrist_ts = timestamps.get("cam_left_wrist", 0)
            high_ts = timestamps.get("cam_high", 0)
            print(f"  - Timestamps (s): Robot={robot_ts:.4f}, Wrist={wrist_ts:.4f}, High={high_ts:.4f}")
            print(f"  - Time diff (Robot vs Wrist): {abs(robot_ts - wrist_ts) * 1000:.2f} ms")
            print(f"  - Time diff (Robot vs High):  {abs(robot_ts - high_ts) * 1000:.2f} ms")
            print(f"  - Time diff (Wrist vs High):  {abs(wrist_ts - high_ts) * 1000:.2f} ms")
        print()
        
        # Display initial observation
        display_observations(obs)
        
        # For calculating average latency
        total_diff_robot_wrist = 0
        total_diff_robot_high = 0
        total_diff_wrist_high = 0
        total_obs_duration = 0
        num_measurements = 0
        
        action_set = [
            [0,-1.5,1.5,0,0,0,0],
            [0,0.23,-0.59,-0.04,0.71,0.028,0.05],
            [0,0.8,-0.89,-0.08,0.24,0.07,0.025]
        ]
        # Run a few steps with simple actions
        print("Running 50 test steps...")
        for step in range(10000):
            # Create a small random perturbation action
            # Start from current state and add small noise
            current_state = obs['state']
            
            action = action_set[random.randint(0, len(action_set)-1)]
            
            print(f"Step {step + 1}: Applying action {action}")
            
            # Apply action
            env.apply_action({"actions": action})
            
            # Get new observation
            obs = env.get_observation()
            print(f"  New state: {obs['state']}")
            print(f"  Episode complete: {env.is_episode_complete()}")

            # Print timestamp differences
            timestamps = obs.get("timestamps", {})
            if timestamps:
                start_ts = timestamps.get("start", 0)
                robot_ts = timestamps.get("robot", 0)
                wrist_ts = timestamps.get("cam_left_wrist", 0)
                high_ts = timestamps.get("cam_high", 0)

                # Calculate total observation time for this step
                latest_ts = max(robot_ts, wrist_ts, high_ts)
                obs_duration_ms = (latest_ts - start_ts) * 1000
                total_obs_duration += obs_duration_ms
                print(f"  - Obs. Duration: {obs_duration_ms:.2f} ms")

                # Create a list of (name, timestamp) tuples for sorting
                ts_data = [
                    ("Robot", robot_ts),
                    ("Wrist", wrist_ts),
                    ("High", high_ts)
                ]
                ts_data.sort(key=lambda x: x[1])

                # Print the acquisition order
                order_str = " -> ".join([name for name, ts in ts_data])
                print(f"  - Acquisition Order: {order_str}")

                # Print differences between consecutive devices
                for i in range(1, len(ts_data)):
                    prev_name, prev_ts = ts_data[i-1]
                    curr_name, curr_ts = ts_data[i]
                    diff_ms = (curr_ts - prev_ts) * 1000
                    print(f"    - {prev_name} to {curr_name}: {diff_ms:.2f} ms")

                # Accumulate for average calculation
                diff_rw = abs(robot_ts - wrist_ts) * 1000
                diff_rh = abs(robot_ts - high_ts) * 1000
                diff_wh = abs(wrist_ts - high_ts) * 1000
                total_diff_robot_wrist += diff_rw
                total_diff_robot_high += diff_rh
                total_diff_wrist_high += diff_wh
                num_measurements += 1
            
            # Display images in real-time
            display_observations(obs)

            time.sleep(1 / env._update_rate)  # Small delay between steps

        print("\n=== Test completed successfully ===")
        
        # Calculate and print average latencies
        if num_measurements > 0:
            avg_diff_rw = total_diff_robot_wrist / num_measurements
            avg_diff_rh = total_diff_robot_high / num_measurements
            avg_diff_wh = total_diff_wrist_high / num_measurements
            avg_obs_duration = total_obs_duration / num_measurements
            print("\n--- Average Latency & Duration ---")
            print(f"  - Avg. Obs. Duration: {avg_obs_duration:.2f} ms")
            print(f"  - Avg. Robot vs Wrist: {avg_diff_rw:.2f} ms")
            print(f"  - Avg. Robot vs High:  {avg_diff_rh:.2f} ms")
            print(f"  - Avg. Wrist vs High:  {avg_diff_wh:.2f} ms")
            print("------------------------------------")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nCleaning up...")
        cv2.destroyAllWindows()
        env.close()


# def test_observation_only():
#     """Test observation capture without moving the robot."""
#     print("=== Testing Observation Capture (no motion) ===\n")
    
#     env = PiperEnvironment(can_port="can0")
    
#     try:
#         # Create display window
#         cv2.namedWindow("Observations", cv2.WINDOW_AUTOSIZE)
        
#         env.reset()
        
#         # Capture 10 observations
#         print("Capturing 10 observations...")
#         for i in range(10):
#             obs = env.get_observation()
#             print(f"Observation {i + 1}:")
#             print(f"  State: {obs['state']}")
#             print(f"  Image shape: {obs['images']['cam_high'].shape}")
            
#             # Display images in real-time
#             display_observations(obs)
            
#             time.sleep(0.2)
        
#         print("\n=== Observation test completed ===")
        
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         cv2.destroyAllWindows()
#         env.close()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PiperEnvironment with real hardware")
    parser.add_argument(
        "--tele-mode",
        action="store_true",
        help="Enable teleoperation mode to save episode data"
    )
    args = parser.parse_args()
    
    # Pass tele_mode to test function
    test_basic_loop(tele_mode=args.tele_mode)
