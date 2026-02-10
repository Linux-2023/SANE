#!/usr/bin/env python3
import h5py
import numpy as np
import cv2
import argparse
import os

def check_hdf5(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    print(f"\n{'='*60}")
    print(f"检查 HDF5 数据文件: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    with h5py.File(file_path, 'r') as f:
        # 1. 打印结构
        print("\n[1] 文件结构:")
        def print_attrs(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}- {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}+ {name}/")
        f.visititems(print_attrs)

        # 2. 基本统计
        num_steps = f['is_intervention'].shape[0]
        interventions = np.array(f['is_intervention'])
        intervention_count = np.sum(interventions)
        
        print("\n[2] 数据统计:")
        print(f"  - 总步数: {num_steps}")
        print(f"  - 人工干预步数: {intervention_count} ({100*intervention_count/num_steps:.1f}%)")
        
        if 'task' in f:
            task = f['task'][0].decode('utf-8')
            print(f"  - 任务描述: {task}")

        # 3. 可视化
        print("\n[3] 可视化预览:")
        print("  操作说明:")
        print("    - [D] 下一帧 | [A] 上一帧")
        print("    - [W] 快进 10 帧 | [S] 快退 10 帧")
        print("    - [Q] 退出预览")
        
        obs_images = f['observations/images']
        cam_names = list(obs_images.keys())
        qpos = f['observations/qpos']
        actions = f['action']

        idx = 0
        window_name = "HDF5 Checker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            display_imgs = []
            for cam_name in cam_names:
                # 获取图像: [C, H, W] -> [H, W, C]
                img = obs_images[cam_name][idx]
                img = np.transpose(img, (1, 2, 0))
                # RGB -> BGR
                img_bgr = img[:, :, ::-1].copy()
                
                # 添加标签
                is_int = interventions[idx]
                color = (0, 0, 255) if is_int else (0, 255, 0)
                status = "INTERVENTION" if is_int else "AUTO"
                cv2.putText(img_bgr, f"{cam_name} ({status})", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                display_imgs.append(img_bgr)

            # 水平拼接
            combined = np.hstack(display_imgs)
            
            # 添加状态信息栏
            info_h = 60
            info_panel = np.zeros((info_h, combined.shape[1], 3), dtype=np.uint8)
            
            cur_q = qpos[idx]
            cur_a = actions[idx]
            
            q_str = "Qpos: " + " ".join([f"{x:5.2f}" for x in cur_q[:6]])
            a_str = "Act : " + " ".join([f"{x:5.2f}" for x in cur_a[:6]])
            
            cv2.putText(info_panel, f"Frame: {idx}/{num_steps-1}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, q_str, (10, 38), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(info_panel, a_str, (10, 53), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            
            final_view = np.vstack([combined, info_panel])
            cv2.imshow(window_name, final_view)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                idx = min(idx + 1, num_steps - 1)
            elif key == ord('a'):
                idx = max(idx - 1, 0)
            elif key == ord('w'):
                idx = min(idx + 10, num_steps - 1)
            elif key == ord('s'):
                idx = max(idx - 10, 0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 HDF5 录制数据")
    parser.add_argument("file", type=str, help="HDF5 文件路径")
    args = parser.parse_args()
    check_hdf5(args.file)

