import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collision_check.dual_arm_collision_checker import create_checker
import numpy as np

# 创建碰撞检测器（默认使用 capsule 模式）
checker = create_checker(mode="capsule")

# 定义左右臂关节角度（7个关节，单位：弧度）
q_left = np.zeros(7)
q_right = np.zeros(7)

# q_left[1] = 0

import time
start_time = time.time()
# 检测是否碰撞
is_collision = checker.check_collision(q_left, q_right)
end_time = time.time()
print(f"碰撞检测耗时: {end_time - start_time:.6f} 秒")

# 计算最小距离（用于安全裕度）
start_time2 = time.time()
min_dist = checker.compute_min_distance(q_left, q_right)
end_time2 = time.time()
print(f"最小距离计算耗时: {end_time2 - start_time2:.6f} 秒")

# 安全检查（距离 > 阈值）
is_safe = checker.is_safe(q_left, q_right, threshold=0.05) 

print(f"是否碰撞: {is_collision}")
print(f"最小距离: {min_dist}")