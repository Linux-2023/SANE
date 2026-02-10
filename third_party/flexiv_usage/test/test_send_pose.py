import sys
import os
import time
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexiv_robot import FlexivRobot

robot = FlexivRobot("Rizon4s-063239")

# 获取初始位姿
init_pose = robot.get_tcp_pose()
print(f"init_pose: {init_pose}")
target_pose = init_pose.copy()
target_pose[2] += 0.01

duration = 10.0   # 10秒

start_time = time.time()
while time.time() - start_time < duration:
    t = time.time() - start_time
    # 发送命令
    robot.send_cartesian_pose(target_pose)
    
    time.sleep(0.02)  # 50Hz

robot.stop()