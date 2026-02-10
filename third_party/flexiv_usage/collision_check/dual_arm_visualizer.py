"""
Dual Arm Visualizer for Flexiv Rizon 4s

基于 Pinocchio + MeshCat 的双臂可视化工具。
在浏览器中显示双臂 3D 模型，支持实时更新关节角度。

Usage:
    from dual_arm_visualizer import DualArmVisualizer
    
    viz = DualArmVisualizer()
    viz.display(q_left, q_right)  # 显示指定关节角度的双臂姿态

Requirements:
    - pinocchio (pin)
    - meshcat
"""

import os
import time
import numpy as np
from typing import Optional, Tuple, List

try:
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
except ImportError as e:
    raise ImportError(
        "请安装 pinocchio:\n"
        "  uv pip install pin==2.6.21\n"
        f"原始错误: {e}"
    )

try:
    import meshcat
except ImportError:
    raise ImportError(
        "请安装 meshcat:\n"
        "  uv pip install meshcat"
    )


class DualArmVisualizer:
    """
    双臂可视化器
    
    在浏览器中显示 Flexiv Rizon 4s 双臂的 3D 模型。
    """
    
    def __init__(
        self,
        urdf_path: Optional[str] = None,
        open_browser: bool = True
    ):
        """
        初始化可视化器
        
        Args:
            urdf_path: 双臂 URDF 文件路径，默认使用内置的 URDF
            open_browser: 是否自动打开浏览器
        """
        if urdf_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            urdf_path = os.path.join(parent_dir, "urdf", "flexiv_Rizon4s_dual_arm.urdf")
        
        self._urdf_path = os.path.abspath(urdf_path)
        self._mesh_dir = os.path.dirname(self._urdf_path)
        
        # 加载模型
        self._load_model()
        
        # 创建可视化器
        self._create_visualizer(open_browser)
    
    def _load_model(self):
        """加载 URDF 模型"""
        self.model = pin.buildModelFromUrdf(self._urdf_path)
        self.data = self.model.createData()
        
        # 加载可视化几何模型
        self.visual_model = pin.buildGeomFromUrdf(
            self.model,
            self._urdf_path,
            pin.GeometryType.VISUAL,
            package_dirs=[self._mesh_dir]
        )
        
        # 加载碰撞几何模型
        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            self._urdf_path,
            pin.GeometryType.COLLISION,
            package_dirs=[self._mesh_dir]
        )
    
    def _create_visualizer(self, open_browser: bool):
        """创建 MeshCat 可视化器"""
        self.viz = MeshcatVisualizer(
            self.model,
            self.collision_model,
            self.visual_model
        )
        
        # 初始化查看器
        self.viz.initViewer(open=open_browser)
        self.viz.loadViewerModel()
        
        # 获取 URL
        self._viewer_url = self.viz.viewer.url()
        print(f"可视化器已启动: {self._viewer_url}")
    
    @property
    def url(self) -> str:
        """获取可视化器 URL"""
        return self._viewer_url
    
    def _prepare_configuration(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        degrees: bool = False
    ) -> np.ndarray:
        """
        准备完整配置向量
        
        Args:
            q_left: 左臂关节角 (7 DoF)
            q_right: 右臂关节角 (7 DoF)
            degrees: 如果为 True，输入为角度制；否则为弧度制
        
        Returns:
            完整配置向量 (14 DoF)，单位为弧度
        """
        q_left = np.asarray(q_left, dtype=np.float64).flatten()
        q_right = np.asarray(q_right, dtype=np.float64).flatten()
        
        if len(q_left) != 7:
            raise ValueError(f"q_left 必须有 7 个元素，当前有 {len(q_left)} 个")
        if len(q_right) != 7:
            raise ValueError(f"q_right 必须有 7 个元素，当前有 {len(q_right)} 个")
        
        # 如果输入是角度，转换为弧度
        if degrees:
            q_left = np.deg2rad(q_left)
            q_right = np.deg2rad(q_right)
        
        return np.concatenate([q_left, q_right])
    
    def display(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        degrees: bool = False
    ):
        """
        显示指定关节角度的双臂姿态
        
        Args:
            q_left: 左臂关节角 (7 DoF)
            q_right: 右臂关节角 (7 DoF)
            degrees: 如果为 True，输入为角度制；否则为弧度制（默认）
        
        Examples:
            # 弧度输入
            viz.display([0, 0.5, 0, -0.8, 0, 0.5, 0], [0, 0.5, 0, -0.8, 0, 0.5, 0])
            
            # 角度输入
            viz.display([0, 30, 0, -45, 0, 30, 0], [0, 30, 0, -45, 0, 30, 0], degrees=True)
        """
        q = self._prepare_configuration(q_left, q_right, degrees=degrees)
        self.viz.display(q)
    
    def display_trajectory(
        self,
        q_lefts: List[np.ndarray],
        q_rights: List[np.ndarray],
        dt: float = 0.1,
        degrees: bool = False
    ):
        """
        播放轨迹动画
        
        Args:
            q_lefts: 左臂关节角轨迹
            q_rights: 右臂关节角轨迹
            dt: 帧间隔时间（秒）
            degrees: 如果为 True，输入为角度制；否则为弧度制（默认）
        """
        if len(q_lefts) != len(q_rights):
            raise ValueError("左右臂轨迹长度必须相同")
        
        for q_left, q_right in zip(q_lefts, q_rights):
            self.display(q_left, q_right, degrees=degrees)
            time.sleep(dt)
    
    def get_flange_poses(
        self,
        q_left: np.ndarray,
        q_right: np.ndarray,
        degrees: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取法兰盘位姿
        
        Args:
            q_left: 左臂关节角 (7 DoF)
            q_right: 右臂关节角 (7 DoF)
            degrees: 如果为 True，输入为角度制；否则为弧度制（默认）
        
        Returns:
            (left_pose, right_pose): 4x4 SE3 矩阵
        """
        q = self._prepare_configuration(q_left, q_right, degrees=degrees)
        
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        left_flange_id = self.model.getFrameId("left_flange")
        right_flange_id = self.model.getFrameId("right_flange")
        
        left_pose = self.data.oMf[left_flange_id].homogeneous
        right_pose = self.data.oMf[right_flange_id].homogeneous
        
        return left_pose, right_pose
    
    def set_camera(
        self,
        position: Tuple[float, float, float] = (1.5, 1.5, 1.0),
        target: Tuple[float, float, float] = (0, 0, 0.5)
    ):
        """
        设置相机位置
        
        Args:
            position: 相机位置 (x, y, z)
            target: 相机目标点 (x, y, z)
        """
        self.viz.viewer["/Cameras/default/rotated/<object>"].set_transform(
            meshcat.transformations.compose_matrix(
                translate=position
            )
        )
    
    def add_frame(
        self,
        name: str,
        pose: np.ndarray,
        scale: float = 0.1
    ):
        """
        添加坐标系可视化
        
        Args:
            name: 坐标系名称
            pose: 4x4 SE3 矩阵
            scale: 坐标轴长度
        """
        import meshcat.geometry as g
        import meshcat.transformations as tf
        
        # 创建坐标轴
        axes = meshcat.geometry.triad(scale=scale)
        self.viz.viewer[f"frames/{name}"].set_object(axes)
        self.viz.viewer[f"frames/{name}"].set_transform(pose)
    
    def add_sphere(
        self,
        name: str,
        position: Tuple[float, float, float],
        radius: float = 0.02,
        color: Tuple[float, float, float, float] = (1, 0, 0, 0.8)
    ):
        """
        添加球体标记
        
        Args:
            name: 标记名称
            position: 位置 (x, y, z)
            radius: 半径
            color: 颜色 (r, g, b, a)
        """
        import meshcat.geometry as g
        import meshcat.transformations as tf
        
        sphere = g.Sphere(radius)
        material = g.MeshLambertMaterial(
            color=int(color[0]*255) << 16 | int(color[1]*255) << 8 | int(color[2]*255),
            opacity=color[3]
        )
        
        self.viz.viewer[f"markers/{name}"].set_object(sphere, material)
        self.viz.viewer[f"markers/{name}"].set_transform(
            tf.translation_matrix(position)
        )
    
    def clear_markers(self):
        """清除所有标记"""
        self.viz.viewer["markers"].delete()
        self.viz.viewer["frames"].delete()


def create_visualizer(
    urdf_path: Optional[str] = None,
    open_browser: bool = True
) -> DualArmVisualizer:
    """
    工厂函数：创建可视化器
    
    Args:
        urdf_path: URDF 文件路径
        open_browser: 是否自动打开浏览器
    
    Returns:
        DualArmVisualizer 实例
    """
    return DualArmVisualizer(urdf_path, open_browser)


def interactive_demo():
    """交互式演示"""
    print("=" * 50)
    print("双臂可视化器演示")
    print("=" * 50)
    
    viz = create_visualizer()
    
    # 零配置
    print("\n1. 零配置（初始姿态）")
    q_left = np.zeros(7)
    q_right = np.zeros(7)
    viz.display(q_left, q_right)
    
    # 显示法兰盘位置
    left_pose, right_pose = viz.get_flange_poses(q_left, q_right)
    print(f"   左法兰盘位置: {left_pose[:3, 3]}")
    print(f"   右法兰盘位置: {right_pose[:3, 3]}")
    
    input("\n按 Enter 继续...")
    
    # 手臂伸展
    print("\n2. 手臂伸展")
    q_left = np.array([0, 0.5, 0, -0.8, 0, 0.5, 0])
    q_right = np.array([0, 0.5, 0, -0.8, 0, 0.5, 0])
    viz.display(q_left, q_right)
    
    input("\n按 Enter 继续...")
    
    # 轨迹动画
    print("\n3. 轨迹动画")
    n_steps = 50
    q_lefts = []
    q_rights = []
    
    for i in range(n_steps):
        t = i / n_steps * 2 * np.pi
        q_left = np.array([
            0.5 * np.sin(t),
            0.3 + 0.3 * np.sin(t),
            0.3 * np.sin(t * 2),
            -0.5 - 0.3 * np.sin(t),
            0.2 * np.sin(t * 3),
            0.5 + 0.2 * np.sin(t),
            0.3 * np.sin(t * 2)
        ])
        q_right = np.array([
            -0.5 * np.sin(t),
            0.3 + 0.3 * np.sin(t),
            -0.3 * np.sin(t * 2),
            -0.5 - 0.3 * np.sin(t),
            -0.2 * np.sin(t * 3),
            0.5 + 0.2 * np.sin(t),
            -0.3 * np.sin(t * 2)
        ])
        q_lefts.append(q_left)
        q_rights.append(q_right)
    
    viz.display_trajectory(q_lefts, q_rights, dt=0.05)
    
    print("\n演示结束！浏览器窗口保持打开，可以手动查看模型。")
    print(f"可视化器 URL: {viz.url}")
    
    return viz


if __name__ == "__main__":
    interactive_demo()

