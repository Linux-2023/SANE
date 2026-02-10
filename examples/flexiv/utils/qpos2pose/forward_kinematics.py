"""
Flexiv 机器人正运动学解算器

根据关节角计算末端执行器（TCP）的位姿。

用法:
    from forward_kinematics import ForwardKinematics
    
    fk = ForwardKinematics()
    pose = fk.compute(joint_angles)  # pose: [x, y, z, qw, qx, qy, qz]
"""

import os
from typing import Union
import numpy as np

try:
    import pinocchio as pin
except ImportError as e:
    raise ImportError(
        "请安装 pinocchio 库:\n"
        "  conda install -c conda-forge pinocchio\n"
        f"原始错误: {e}"
    )


class ForwardKinematics:
    """Flexiv 机器人正运动学解算器"""
    
    def __init__(self, urdf_path: str = None, arm_side: str = None):
        """
        初始化正运动学解算器
        
        Args:
            urdf_path: URDF 文件路径。如果为 None，使用默认的 Rizon4s URDF
            arm_side: 机械臂侧，'left' 或 'right'。如果提供，将加载对应的旋转矩阵进行坐标系校准
        """
        if urdf_path is None:
            # 使用默认的 Rizon4s URDF 文件
            # 从 examples/flexiv/utils/ 目录向上查找 third_party/flexiv_usage/urdf/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 向上查找项目根目录，然后定位到 third_party/flexiv_usage/urdf/
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            urdf_path = os.path.join(
                project_root, 
                "third_party", 
                "flexiv_usage", 
                "urdf", 
                "flexiv_Rizon4s_kinematics.urdf"
            )
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
        
        # 加载机器人模型
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 获取末端执行器（flange）的 frame ID
        try:
            self._flange_id = self.model.getFrameId("flange")
        except:
            raise ValueError("URDF 中未找到 'flange' frame")
        
        # 加载旋转矩阵（如果指定了arm_side）
        self._rotation_matrix = None
        if arm_side is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if arm_side.lower() == 'left':
                rotation_file = os.path.join(current_dir, "Left_Rizon4s-063239_rotation_matrix.npy")
            elif arm_side.lower() == 'right':
                rotation_file = os.path.join(current_dir, "Right_Rizon4s-063215_rotation_matrix.npy")
            else:
                raise ValueError(f"arm_side 必须是 'left' 或 'right'，当前为: {arm_side}")
            
            if os.path.exists(rotation_file):
                self._rotation_matrix = np.load(rotation_file)
                print(f"已加载旋转矩阵: {rotation_file}")
            else:
                print(f"警告: 旋转矩阵文件不存在: {rotation_file}，将不使用坐标系旋转")
    
    def compute(self, joint_angles: Union[list, np.ndarray]) -> np.ndarray:
        """
        计算末端执行器位姿
        
        Args:
            joint_angles: 关节角度 [rad]，长度为 7 的数组
            
        Returns:
            末端位姿 [x, y, z, qw, qx, qy, qz]
            - 位置: [x, y, z] 单位: 米
            - 四元数: [qw, qx, qy, qz] (标量在前)
        """
        # 转换为 numpy 数组
        q = np.asarray(joint_angles, dtype=float)
        
        if len(q) != 7:
            raise ValueError(f"关节角数量应为 7，实际为 {len(q)}")
        
        # 计算正运动学
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # 获取末端执行器位姿
        T = self.data.oMf[self._flange_id]
        
        # 提取位置
        position = T.translation
        
        # 提取旋转矩阵
        rotation_matrix = T.rotation
        
        # 如果存在旋转矩阵校准，应用旋转
        if self._rotation_matrix is not None:
            # 应用旋转到位置
            position = self._rotation_matrix @ position
            
            # 应用旋转到旋转矩阵
            rotation_matrix = self._rotation_matrix @ rotation_matrix
        
        # 将旋转矩阵转换为四元数
        # pinocchio 的 SE3 对象可以直接转换为四元数
        quat = pin.Quaternion(rotation_matrix)
        # coeffs() 返回 [x, y, z, w]，需要转换为 [w, x, y, z]
        quat_coeffs = quat.coeffs()  # [x, y, z, w]
        quaternion = np.array([quat_coeffs[3], quat_coeffs[0], quat_coeffs[1], quat_coeffs[2]])  # [w, x, y, z]
        
        # 组合位姿 [x, y, z, qw, qx, qy, qz]
        pose = np.concatenate([position, quaternion])
        
        return pose
    
    def compute_matrix(self, joint_angles: Union[list, np.ndarray]) -> np.ndarray:
        """
        计算末端执行器的 4x4 齐次变换矩阵
        
        Args:
            joint_angles: 关节角度 [rad]，长度为 7 的数组
            
        Returns:
            4x4 齐次变换矩阵
        """
        q = np.asarray(joint_angles, dtype=float)
        
        if len(q) != 7:
            raise ValueError(f"关节角数量应为 7，实际为 {len(q)}")
        
        # 计算正运动学
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # 获取 4x4 齐次变换矩阵
        T = self.data.oMf[self._flange_id]
        
        return T.homogeneous


def main():
    """测试正运动学解算器"""
    print("=" * 60)
    print("Flexiv 机器人正运动学解算器测试")
    print("=" * 60)
    
    try:
        # 初始化正运动学解算器
        print("\n1. 初始化正运动学解算器...")
        fk = ForwardKinematics()
        print("   ✅ 初始化成功")
        
        # 测试用例 1: 全零关节角（Home 位置）
        print("\n2. 测试用例 1: Home 位置（全零关节角）")
        joint_angles_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pose_1 = fk.compute(joint_angles_1)
        print(f"   关节角: {joint_angles_1}")
        print(f"   位置: [{pose_1[0]:.4f}, {pose_1[1]:.4f}, {pose_1[2]:.4f}] m")
        print(f"   四元数: [{pose_1[3]:.4f}, {pose_1[4]:.4f}, {pose_1[5]:.4f}, {pose_1[6]:.4f}]")
        
        # 测试用例 2: 典型关节角
        print("\n3. 测试用例 2: 典型关节角")
        joint_angles_2 = [0.5, -0.3, 0.2, -0.4, 0.1, 0.3, -0.2]
        pose_2 = fk.compute(joint_angles_2)
        print(f"   关节角: {joint_angles_2}")
        print(f"   位置: [{pose_2[0]:.4f}, {pose_2[1]:.4f}, {pose_2[2]:.4f}] m")
        print(f"   四元数: [{pose_2[3]:.4f}, {pose_2[4]:.4f}, {pose_2[5]:.4f}, {pose_2[6]:.4f}]")
        
        # 测试用例 3: 验证四元数归一化
        print("\n4. 验证四元数归一化")
        quat_norm = np.linalg.norm(pose_2[3:7])
        print(f"   四元数模长: {quat_norm:.6f} (应该接近 1.0)")
        if abs(quat_norm - 1.0) < 1e-5:
            print("   ✅ 四元数已归一化")
        else:
            print("   ⚠️  四元数未归一化")
        
        # 测试用例 4: 计算 4x4 变换矩阵
        print("\n5. 测试 4x4 齐次变换矩阵")
        T = fk.compute_matrix(joint_angles_2)
        print(f"   变换矩阵形状: {T.shape}")
        print(f"   位置 (从矩阵): [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}] m")
        print(f"   位置 (从 pose): [{pose_2[0]:.4f}, {pose_2[1]:.4f}, {pose_2[2]:.4f}] m")
        
        # 验证矩阵和 pose 的一致性
        position_match = np.allclose(T[:3, 3], pose_2[:3], atol=1e-6)
        if position_match:
            print("   ✅ 位置一致")
        else:
            print("   ⚠️  位置不一致")
        
        # 测试用例 5: 批量测试多个关节角
        print("\n6. 批量测试多个关节角配置")
        test_configs = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
        for i, angles in enumerate(test_configs, 1):
            pose = fk.compute(angles)
            print(f"   配置 {i}: 位置 = [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}] m")
        
        # 测试用例 6: 错误处理
        print("\n7. 测试错误处理")
        try:
            fk.compute([0.0, 0.0, 0.0])  # 错误的关节角数量
            print("   ⚠️  应该抛出错误但没有")
        except ValueError as e:
            print(f"   ✅ 正确捕获错误: {e}")
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("   请确保 URDF 文件路径正确")
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()


def test_with_robot(robot_sn: str):
    """
    连接机器人并测试正运动学精度
    
    Args:
        robot_sn: 机器人序列号，如 "Rizon4s-063239"
    """
    import sys
    import os
    
    # 添加 flexiv_usage 到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    flexiv_usage_path = os.path.join(project_root, "third_party", "flexiv_usage")
    if flexiv_usage_path not in sys.path:
        sys.path.insert(0, flexiv_usage_path)
    
    from flexiv_robot import FlexivRobot
    
    print("=" * 60)
    print("正运动学精度测试（连接真实机器人）")
    print("=" * 60)
    
    robot = None
    try:
        # 连接机器人
        print(f"\n连接机器人 [{robot_sn}]...")
        robot = FlexivRobot(robot_sn, auto_init=True, verbose=False)
        print("✅ 机器人连接成功")
        
        # 初始化正运动学解算器
        print("\n初始化正运动学解算器...")
        fk = ForwardKinematics()
        print("✅ 正运动学解算器初始化成功")
        
        # 获取真实数据
        print("\n获取机器人当前状态...")
        real_joints = robot.get_joint_positions()  # [rad]
        real_pose = robot.get_tcp_pose()  # [x, y, z, qw, qx, qy, qz]
        
        print(f"真实关节角: {np.round(np.degrees(real_joints), 2)}°")
        print(f"真实位置: [{real_pose[0]:.4f}, {real_pose[1]:.4f}, {real_pose[2]:.4f}] m")
        print(f"真实四元数: [{real_pose[3]:.4f}, {real_pose[4]:.4f}, {real_pose[5]:.4f}, {real_pose[6]:.4f}]")
        
        # 使用正运动学计算位姿
        print("\n使用正运动学计算位姿...")
        computed_pose = fk.compute(real_joints)
        print(f"计算位置: [{computed_pose[0]:.4f}, {computed_pose[1]:.4f}, {computed_pose[2]:.4f}] m")
        print(f"计算四元数: [{computed_pose[3]:.4f}, {computed_pose[4]:.4f}, {computed_pose[5]:.4f}, {computed_pose[6]:.4f}]")
        
        # 计算差异
        print("\n" + "-" * 60)
        print("位姿差异分析")
        print("-" * 60)
        
        # 位置误差
        position_error = np.linalg.norm(real_pose[:3] - computed_pose[:3])
        print(f"位置误差: {position_error*1000:.4f} mm")
        print(f"  X误差: {(real_pose[0] - computed_pose[0])*1000:.4f} mm")
        print(f"  Y误差: {(real_pose[1] - computed_pose[1])*1000:.4f} mm")
        print(f"  Z误差: {(real_pose[2] - computed_pose[2])*1000:.4f} mm")
        
        # 姿态误差（四元数点积）
        # 注意：q 和 -q 表示同一旋转，需要取绝对值
        quat_dot = np.abs(np.dot(real_pose[3:7], computed_pose[3:7]))
        # 限制在 [-1, 1] 范围内，避免数值误差
        quat_dot = np.clip(quat_dot, -1.0, 1.0)
        # 计算角度误差（弧度转度）
        angle_error = 2 * np.arccos(quat_dot)
        print(f"姿态误差: {np.degrees(angle_error):.4f}°")
        print(f"四元数点积: {quat_dot:.6f} (1.0 表示完全一致)")
        
        # 坐标系旋转校准
        print("\n" + "-" * 60)
        print("坐标系旋转校准")
        print("-" * 60)
        
        computed_pos = computed_pose[:3]
        real_pos = real_pose[:3]
        
        # 计算从计算位置到真实位置的旋转矩阵
        # 使用向量对齐方法：计算使计算位置向量对齐到真实位置向量的旋转
        norm_computed = np.linalg.norm(computed_pos)
        norm_real = np.linalg.norm(real_pos)
        
        if norm_computed < 1e-6 or norm_real < 1e-6:
            # 位置接近原点，无法计算旋转
            R_calib = np.eye(3)
            calib_angle = 0.0
            computed_pos_calibrated = computed_pos
        else:
            # 归一化向量
            v1 = computed_pos / norm_computed
            v2 = real_pos / norm_real
            
            # 计算旋转轴和角度
            v_cross = np.cross(v1, v2)
            v_dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
            
            if np.linalg.norm(v_cross) < 1e-6:
                # 向量平行或反平行
                if v_dot > 0:
                    R_calib = np.eye(3)
                    calib_angle = 0.0
                else:
                    # 反平行，需要180度旋转（任意垂直轴）
                    R_calib = -np.eye(3)
                    calib_angle = np.pi
            else:
                # 使用Rodrigues公式计算旋转矩阵
                axis = v_cross / np.linalg.norm(v_cross)
                angle = np.arccos(v_dot)
                
                # 将角度四舍五入到最近的90度倍数
                angle_deg = np.degrees(angle)
                angle_rounded_deg = np.round(angle_deg / 90.0) * 90.0
                calib_angle = np.radians(angle_rounded_deg)
                
                # 使用四舍五入后的角度重新计算旋转矩阵
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R_calib = np.eye(3) + np.sin(calib_angle) * K + (1 - np.cos(calib_angle)) * np.dot(K, K)
            
            # 应用旋转到计算位置（保持原始长度）
            computed_pos_calibrated = R_calib @ computed_pos
        
        # 应用旋转到计算姿态（四元数）
        # 将四元数转换为旋转矩阵，应用校准旋转，再转回四元数
        computed_quat = computed_pose[3:7]  # [qw, qx, qy, qz]
        # 四元数转旋转矩阵（使用scipy或手动计算）
        from scipy.spatial.transform import Rotation
        computed_rot = Rotation.from_quat([computed_quat[1], computed_quat[2], computed_quat[3], computed_quat[0]])  # [qx, qy, qz, qw]
        computed_rot_matrix = computed_rot.as_matrix()
        # 应用校准旋转
        calibrated_rot_matrix = R_calib @ computed_rot_matrix
        calibrated_rot = Rotation.from_matrix(calibrated_rot_matrix)
        calibrated_quat_xyzw = calibrated_rot.as_quat()  # [qx, qy, qz, qw]
        computed_quat_calibrated = np.array([calibrated_quat_xyzw[3], calibrated_quat_xyzw[0], calibrated_quat_xyzw[1], calibrated_quat_xyzw[2]])  # [qw, qx, qy, qz]
        
        # 计算校准后的位置误差
        position_error_calib = np.linalg.norm(real_pos - computed_pos_calibrated)
        print(f"坐标系旋转角度: {np.degrees(calib_angle):.4f}°")
        print(f"校准后位置误差: {position_error_calib*1000:.4f} mm")
        print(f"  校准后X误差: {(real_pos[0] - computed_pos_calibrated[0])*1000:.4f} mm")
        print(f"  校准后Y误差: {(real_pos[1] - computed_pos_calibrated[1])*1000:.4f} mm")
        print(f"  校准后Z误差: {(real_pos[2] - computed_pos_calibrated[2])*1000:.4f} mm")
        print(f"  校准旋转矩阵: {R_calib}")
        
        # 保存旋转矩阵到本地文件
        output_dir = os.path.dirname(os.path.abspath(__file__))
        rotation_matrix_file = os.path.join(output_dir, f"{robot_sn}_rotation_matrix.npy")
        np.save(rotation_matrix_file, R_calib)
        print(f"旋转矩阵已保存到: {rotation_matrix_file}")
        
        # 计算校准后的姿态误差
        quat_dot_calib = np.abs(np.dot(real_pose[3:7], computed_quat_calibrated))
        quat_dot_calib = np.clip(quat_dot_calib, -1.0, 1.0)
        angle_error_calib = 2 * np.arccos(quat_dot_calib)
        print(f"校准后姿态误差: {np.degrees(angle_error_calib):.4f}°")
        print(f"校准后四元数点积: {quat_dot_calib:.6f} (1.0 表示完全一致)")
        
        print("\n" + "=" * 60)
        if position_error < 0.01 and angle_error < 1.0:
            print("✅ 正运动学精度良好")
        elif position_error_calib < 0.01 and angle_error_calib < 1.0:
            print("✅ 校准后精度良好（存在坐标系旋转偏移）")
        else:
            print("⚠️  正运动学精度需要检查")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot is not None:
            try:
                robot.close()
            except:
                pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flexiv 正运动学测试")
    parser.add_argument(
        "--robot-sn", 
        type=str, 
        default='Rizon4s-063239', # 'Rizon4s-063215', # 
        help="机器人序列号（如 Rizon4s-063239）。如果未提供，运行离线测试"
    )
    
    args = parser.parse_args()
    
    if args.robot_sn:
        test_with_robot(args.robot_sn)
    else:
        main()
