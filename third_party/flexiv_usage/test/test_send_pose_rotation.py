import sys
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexiv_robot import FlexivRobot

def modify_pose_with_euler(init_pose, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0, degrees=True):
    """
    基于初始pose和欧拉角增量，生成目标pose。
    增量单位：度（默认），或者弧度（degrees=False）。
    欧拉角顺序：XYZ (roll, pitch, yaw)
    """
    target_pose = np.array(init_pose).copy()
    # 修改位置
    target_pose[0] += dx
    target_pose[1] += dy
    target_pose[2] += dz

    # 当前四元数
    quat = target_pose[3:7]
    # 原始姿态
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [qx, qy, qz, qw]
    # 增量旋转
    d_rot = R.from_euler('xyz', [droll, dpitch, dyaw], degrees=degrees)
    # 新的旋转
    new_rot = rot * d_rot
    new_quat = new_rot.as_quat()  # [qx, qy, qz, qw]
    # 替换
    target_pose[3:7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]  # [qw, qx, qy, qz]
    return target_pose

robot = FlexivRobot("Rizon4s-063239")

# 获取初始位姿
init_pose = robot.get_tcp_pose()
print(f"init_pose: {init_pose}")
# 例：绕Z轴旋转10度，平移z+0.01米
target_pose = modify_pose_with_euler(init_pose, droll=-10, dpitch=0, dyaw=0)

print(f"target_pose: {target_pose}")

duration = 10.0   # 10秒

start_time = time.time()
while time.time() - start_time < duration:
    t = time.time() - start_time
    # 发送命令
    robot.send_cartesian_pose(target_pose)
    time.sleep(0.02)  # 50Hz

robot.stop()