"""
双臂碰撞检测与可视化模块

Usage:
    from collision_check import create_checker, DualArmVisualizer
    
    checker = create_checker(mode="capsule")
    viz = DualArmVisualizer()
"""

from .dual_arm_collision_checker import DualArmCollisionChecker, create_checker
from .dual_arm_visualizer import DualArmVisualizer, create_visualizer

__all__ = [
    "DualArmCollisionChecker",
    "create_checker",
    "DualArmVisualizer", 
    "create_visualizer",
]

