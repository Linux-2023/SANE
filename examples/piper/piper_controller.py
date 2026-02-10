#!/usr/bin/env python3
# -*-coding:utf8-*-
import time
from piper_sdk import *

class PiperController:
    """
    一个用于控制 Piper 机械臂的集成控制器类。
    """
    def __init__(self, can_port="can0", gripper_norm=False):
        """
        初始化 Piper 机械臂接口。
        :param can_port: CAN 端口名称，默认为 "can0"。
        """
        self.piper = C_PiperInterface_V2(can_port)
        self.piper.ConnectPort()
        print(f"Connected to Piper robot arm, CAN port: {can_port}")
        self.max_gripper_range = 0.07 # 最大夹爪范围
        self.min_gripper_range = 0.00 # 最小夹爪范围
        self.gripper_norm = gripper_norm
        
    def normalize_gripper(self, range):
        return (range - self.min_gripper_range) / (self.max_gripper_range - self.min_gripper_range)
    
    def denormalize_gripper(self, range):
        return range * (self.max_gripper_range - self.min_gripper_range) + self.min_gripper_range
    
    def enable(self):
        """
        Enable the robot arm.
        """
        self.piper.MotionCtrl_1(0x02,0,0)#恢复
        time.sleep(0.1)
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        time.sleep(0.1)
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        self.control_full_joint([0.0,0.0,0.0,0.0,0.0,0.0,0.05])
        time.sleep(0.1)
        print("EWNABLED: Robot arm is now enabled.")

    def disable(self):
        """
        失能机械臂。
        0.0010995574290074892, -0.0376118453890657, -0.015062191448150209, 0.02258456052596335, 0.45052183992016376, 0.03419100005437573
        """
        self.control_gripper(0.05)
        time.sleep(2)
        self.control_full_joint([0.0,0.0,0.0,0.0,0.0,0.0,0.05])
        time.sleep(2)
        self.control_full_joint([0.001,-0.037,-0.015,0.022,0.450,0.034,0.0001])
        time.sleep(2)
        self.piper.MotionCtrl_1(0x01,0,0)
        while self.piper.DisablePiper():
            time.sleep(0.01)
        print("DISABLED: Robot arm is now disabled.")

    def move_end_pose(self, position, velocity=100):
        """
        控制机械臂末端姿态。
        :param position: 包含 [X, Y, Z, RX, RY, RZ] 的列表，单位为米和度。
        :param velocity: 运动速度。
        """
        factor = 1000  # 转换系数
        X = round(position[0] * factor)
        Y = round(position[1] * factor)
        Z = round(position[2] * factor)
        RX = round(position[3] * factor)
        RY = round(position[4] * factor)
        RZ = round(position[5] * factor)
        
        self.piper.MotionCtrl_2(0x01, 0x00, velocity, 0x00)
        self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

    def move_joint(self, angles, velocity=100):
        """
        控制机械臂关节角度。
        :param angles: 包含 6 个关节角度的列表，单位为弧度。
        :param velocity: 运动速度。
        """
        factor = 57295.7795  # 1000 * 180 / 3.1415926
        
        joint_0 = round(angles[0] * factor)
        joint_1 = round(angles[1] * factor)
        joint_2 = round(angles[2] * factor)
        joint_3 = round(angles[3] * factor)
        joint_4 = round(angles[4] * factor)
        joint_5 = round(angles[5] * factor)
        
        self.piper.MotionCtrl_2(0x01, 0x01, velocity, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

    def control_gripper(self, opening_width, speed=1000):
        """
        控制夹爪。
        :param opening_width: 夹爪开口宽度，单位为米。
        :param speed: 夹爪运动速度。
        """
        # 0.05m = 50mm, 转换系数为 1000 * 1000
        if self.gripper_norm:
            opening_width = self.denormalize_gripper(opening_width)
        # if opening_width < 0.05:
        #     opening_width = 0.001 #opening_width - 0.003
        range_val = round(opening_width * 1000 * 1000)
        self.piper.GripperCtrl(abs(range_val), speed, 0x01, 0)
    
    def control_full_joint(self, joint, velocity=100):
        self.move_joint(joint[0:6], velocity)
        self.control_gripper(joint[6], speed=1000)


    def get_end_pose(self):
        """
        获取末端姿态信息。
        :return: 末端姿态信息。
        """
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        return [ep.RX_axis/1000, ep.RY_axis/1000, ep.RZ_axis/1000, ep.X_axis/1000, ep.Y_axis/1000, ep.Z_axis/1000]

    def get_gripper_status(self):
        """
        获取夹爪状态信息。
        :return: 夹爪状态信息，单位为m。
        """
        gs = self.piper.GetArmGripperMsgs().gripper_state
        gs_percentage = gs.grippers_angle / 1000 / 1000 # 转换为米
        if self.gripper_norm:
            gs_percentage = self.normalize_gripper(gs_percentage) # 转换为百分比
        return gs_percentage

    def get_joint_status(self):
        """
        获取机械臂关节状态信息。
        :return: 机械臂关节状态信息，弧度制。
        """
        js = self.piper.GetArmJointMsgs().joint_state
        return [js.joint_1/57295.7795, js.joint_2/57295.7795, js.joint_3/57295.7795, js.joint_4/57295.7795, js.joint_5/57295.7795, js.joint_6/57295.7795]
    
    def get_full_joint_status(self):
        """
        获取机械臂完整关节状态信息。
        :return: 机械臂完整关节状态信息，包含位置、速度和力矩。
        """
        return self.get_joint_status() + [self.get_gripper_status()]


if __name__ == '__main__':
    # 这是一个如何使用 PiperController 类的示例
    
    # 1. 初始化控制器
    robot = PiperController("can0")

    # 2. 使能机械臂
    robot.enable()

    time.sleep(1)

    joints = [[0,0,0,0,0,0,0.001], [0.2,0.2,-0.2,0.3,-0.2,0.5,0.08]]
    for i in range(10):
        robot.control_full_joint(joints[i % 2])
        print(robot.get_full_joint_status())
        time.sleep(2)
    
    # 9. 失能机械臂
    robot.disable()

    # while True:
    #     print(robot.get_full_joint_status())
    #     time.sleep(1)


    print("示例程序执行完毕。")
