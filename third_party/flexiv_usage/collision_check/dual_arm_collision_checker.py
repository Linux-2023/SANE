"""
Dual Arm Collision Checker for Flexiv Rizon 4s

This module provides high-frequency collision detection between two Flexiv Rizon 4s arms
using Pinocchio and HPP-FCL libraries.

Features:
- Supports both mesh (accurate) and capsule (fast) collision detection modes
- Only checks inter-arm collisions (ignores self-collisions within each arm)
- Optimized for real-time control (≥100Hz)

Usage:
    checker = DualArmCollisionChecker(urdf_path, mode="capsule")
    is_collision = checker.check_collision(q_left, q_right)
    min_dist = checker.compute_min_distance(q_left, q_right)

Requirements:
    - pinocchio (conda install -c conda-forge pinocchio)
    - hpp-fcl (conda install -c conda-forge hpp-fcl)
"""

import os
from typing import Optional, Tuple, List
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pinocchio as pin
    from pinocchio import GeometryModel, GeometryObject
    import hppfcl
except ImportError as e:
    raise ImportError(
        "Please install pinocchio and hpp-fcl:\n"
        "  conda install -c conda-forge pinocchio hpp-fcl\n"
        f"Original error: {e}"
    )


# Capsule parameters for each link (radius, length) in meters
# These are approximate bounding capsules for Rizon 4s links
LINK_CAPSULE_PARAMS = {
    "base_link": (0.08, 0.15),
    "link1": (0.07, 0.21),
    "link2": (0.065, 0.20),
    "link3": (0.055, 0.19),
    "link4": (0.055, 0.195),
    "link5": (0.055, 0.19),
    "link6": (0.055, 0.18),
    "link7": (0.045, 0.12),
    "flange": (0.04, 0.05),
    "gripper": (0.05, 0.15),  # 夹爪的胶囊体近似
}


class DualArmCollisionChecker:
    """
    High-frequency collision detector for dual Flexiv Rizon 4s arms.
    
    Attributes:
        mode: Detection mode - "capsule" (fast) or "mesh" (accurate)
        model: Pinocchio model
        data: Pinocchio data
        geom_model: Geometry model for collision detection
        geom_data: Geometry data for collision detection
    """
    
    def __init__(
        self, 
        urdf_path: str, 
        mode: str = "capsule",
        mesh_dir: Optional[str] = None,
        safety_margin: float = 0.0
    ):
        """
        Initialize the dual arm collision checker.
        
        Args:
            urdf_path: Path to the dual-arm URDF file
            mode: Detection mode - "capsule" (fast, ~10x faster) or "mesh" (accurate)
            mesh_dir: Directory containing mesh files. If None, uses URDF directory
            safety_margin: Additional safety margin (meters) for collision detection
        """
        if mode not in ["capsule", "mesh"]:
            raise ValueError(f"Mode must be 'capsule' or 'mesh', got '{mode}'")
        
        self._mode = mode
        self._safety_margin = safety_margin
        self._urdf_path = os.path.abspath(urdf_path)
        
        if mesh_dir is None:
            mesh_dir = os.path.dirname(self._urdf_path)
        self._mesh_dir = mesh_dir
        
        # Build models
        self._build_models()
        
        # Cache link indices for quick lookup
        self._cache_link_indices()
        
        # Configure collision pairs (only inter-arm)
        self._configure_collision_pairs()
    
    def _build_models(self):
        """Build Pinocchio and geometry models from URDF."""
        # Load robot model
        self.model = pin.buildModelFromUrdf(self._urdf_path)
        self.data = self.model.createData()
        
        # Load geometry model based on mode
        if self._mode == "mesh":
            self.geom_model = pin.buildGeomFromUrdf(
                self.model, 
                self._urdf_path,
                pin.GeometryType.COLLISION,
                package_dirs=[self._mesh_dir]
            )
        else:
            # Build capsule geometry model
            self.geom_model = self._build_capsule_geometry()
        
        self.geom_data = pin.GeometryData(self.geom_model)
    
    def _build_capsule_geometry(self) -> GeometryModel:
        """Build geometry model using capsule approximations for each link."""
        geom_model = GeometryModel()
        
        for frame_id in range(self.model.nframes):
            frame = self.model.frames[frame_id]
            frame_name = frame.name
            
            # Skip non-body frames
            if frame.type != pin.FrameType.BODY:
                continue
            
            # Check if this is a left or right arm link
            prefix = None
            link_name = None
            
            if frame_name.startswith("left_"):
                prefix = "left_"
                link_name = frame_name[5:]  # Remove "left_" prefix
            elif frame_name.startswith("right_"):
                prefix = "right_"
                link_name = frame_name[6:]  # Remove "right_" prefix
            else:
                continue
            
            # Get capsule parameters
            if link_name not in LINK_CAPSULE_PARAMS:
                continue
            
            radius, length = LINK_CAPSULE_PARAMS[link_name]
            
            # Create capsule geometry
            capsule = hppfcl.Capsule(radius, length)
            
            # Get parent joint ID (API changed: 'parent' instead of 'parentJoint' in pinocchio 2.x)
            parent_joint = frame.parent
            
            # Capsule placement (centered along z-axis of the link)
            placement = frame.placement.copy()
            
            # Create geometry object
            # Constructor: (name, parent_frame, parent_joint, collision_geometry, placement)
            geom_obj = GeometryObject(
                frame_name + "_capsule",
                frame_id,
                parent_joint,
                capsule,
                placement
            )
            
            geom_model.addGeometryObject(geom_obj)
        
        return geom_model
    
    def _cache_link_indices(self):
        """Cache link frame indices for quick lookup."""
        self._left_link_frames = []
        self._right_link_frames = []
        self._left_geom_ids = []
        self._right_geom_ids = []
        
        for i, geom in enumerate(self.geom_model.geometryObjects):
            name = geom.name
            if name.startswith("left_"):
                self._left_geom_ids.append(i)
            elif name.startswith("right_"):
                self._right_geom_ids.append(i)
    
    def _configure_collision_pairs(self):
        """Configure collision pairs to only check inter-arm collisions."""
        # Remove all existing collision pairs
        self.geom_model.removeAllCollisionPairs()
        
        # Add only inter-arm collision pairs
        # Skip base links as they are fixed and far apart
        for left_id in self._left_geom_ids:
            left_name = self.geom_model.geometryObjects[left_id].name
            # Skip base_link collision with anything (too far)
            if "base_link" in left_name:
                continue
                
            for right_id in self._right_geom_ids:
                right_name = self.geom_model.geometryObjects[right_id].name
                # Skip base_link collision with anything (too far)
                if "base_link" in right_name:
                    continue
                
                self.geom_model.addCollisionPair(
                    pin.CollisionPair(left_id, right_id)
                )
        
        # Recreate geometry data with new collision pairs
        self.geom_data = pin.GeometryData(self.geom_model)
        
        # Store number of collision pairs for reference
        self._n_collision_pairs = len(self.geom_model.collisionPairs)
    
    @property
    def mode(self) -> str:
        """Current detection mode."""
        return self._mode
    
    @property
    def n_collision_pairs(self) -> int:
        """Number of collision pairs being checked."""
        return self._n_collision_pairs
    
    @property
    def joint_names(self) -> List[str]:
        """List of joint names in the model."""
        return [self.model.names[i] for i in range(self.model.njoints)]
    
    def set_mode(self, mode: str):
        """
        Switch detection mode.
        
        Args:
            mode: "capsule" for fast detection, "mesh" for accurate detection
        """
        if mode == self._mode:
            return
        
        if mode not in ["capsule", "mesh"]:
            raise ValueError(f"Mode must be 'capsule' or 'mesh', got '{mode}'")
        
        self._mode = mode
        self._build_models()
        self._cache_link_indices()
        self._configure_collision_pairs()
    
    def _prepare_configuration(
        self, 
        q_left: np.ndarray, 
        q_right: np.ndarray
    ) -> np.ndarray:
        """
        Prepare full configuration vector from left and right arm joint angles.
        
        Args:
            q_left: Left arm joint angles (7 DoF)
            q_right: Right arm joint angles (7 DoF)
            
        Returns:
            Full configuration vector (14 DoF)
        """
        q_left = np.asarray(q_left, dtype=np.float64).flatten()
        q_right = np.asarray(q_right, dtype=np.float64).flatten()
        
        if len(q_left) != 7:
            raise ValueError(f"q_left must have 7 elements, got {len(q_left)}")
        if len(q_right) != 7:
            raise ValueError(f"q_right must have 7 elements, got {len(q_right)}")
        
        # Configuration order matches URDF: left joints first, then right joints
        q = np.concatenate([q_left, q_right])
        return q
    
    def check_collision(
        self, 
        q_left: np.ndarray, 
        q_right: np.ndarray
    ) -> bool:
        """
        Check if the two arms are in collision.
        
        Args:
            q_left: Left arm joint angles (7 DoF) in radians
            q_right: Right arm joint angles (7 DoF) in radians
            
        Returns:
            True if collision detected, False otherwise
        """
        q = self._prepare_configuration(q_left, q_right)
        
        # Update geometry placements
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)
        
        # Check for collisions
        collision = pin.computeCollisions(self.geom_model, self.geom_data, stop_at_first_collision=True)
        
        return collision
    
    def compute_min_distance(
        self, 
        q_left: np.ndarray, 
        q_right: np.ndarray
    ) -> float:
        """
        Compute the minimum distance between the two arms.
        
        Args:
            q_left: Left arm joint angles (7 DoF) in radians
            q_right: Right arm joint angles (7 DoF) in radians
            
        Returns:
            Minimum distance between any pair of links (meters).
            Negative value indicates penetration depth.
        """
        q = self._prepare_configuration(q_left, q_right)
        
        # Update geometry placements
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)
        
        # Compute all distances
        pin.computeDistances(self.geom_model, self.geom_data)
        
        # Find minimum distance
        min_dist = float('inf')
        for i in range(len(self.geom_model.collisionPairs)):
            dist_result = self.geom_data.distanceResults[i]
            if dist_result.min_distance < min_dist:
                min_dist = dist_result.min_distance
        
        return min_dist
    
    def compute_distances(
        self, 
        q_left: np.ndarray, 
        q_right: np.ndarray
    ) -> List[Tuple[str, str, float]]:
        """
        Compute distances for all collision pairs.
        
        Args:
            q_left: Left arm joint angles (7 DoF) in radians
            q_right: Right arm joint angles (7 DoF) in radians
            
        Returns:
            List of (left_link_name, right_link_name, distance) tuples
        """
        q = self._prepare_configuration(q_left, q_right)
        
        # Update geometry placements
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)
        
        # Compute all distances
        pin.computeDistances(self.geom_model, self.geom_data)
        
        results = []
        for i, pair in enumerate(self.geom_model.collisionPairs):
            left_name = self.geom_model.geometryObjects[pair.first].name
            right_name = self.geom_model.geometryObjects[pair.second].name
            dist = self.geom_data.distanceResults[i].min_distance
            results.append((left_name, right_name, dist))
        
        return results
    
    def get_closest_points(
        self, 
        q_left: np.ndarray, 
        q_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get the closest points between the two arms.
        
        Args:
            q_left: Left arm joint angles (7 DoF) in radians
            q_right: Right arm joint angles (7 DoF) in radians
            
        Returns:
            Tuple of (point_on_left_arm, point_on_right_arm, distance)
            Points are in world frame coordinates.
        """
        q = self._prepare_configuration(q_left, q_right)
        
        # Update geometry placements
        pin.updateGeometryPlacements(self.model, self.data, self.geom_model, self.geom_data, q)
        
        # Compute all distances
        pin.computeDistances(self.geom_model, self.geom_data)
        
        # Find minimum distance pair
        min_idx = 0
        min_dist = float('inf')
        for i in range(len(self.geom_model.collisionPairs)):
            dist = self.geom_data.distanceResults[i].min_distance
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        result = self.geom_data.distanceResults[min_idx]
        point_left = result.getNearestPoint1()
        point_right = result.getNearestPoint2()
        
        return np.array(point_left), np.array(point_right), min_dist
    
    def is_safe(
        self, 
        q_left: np.ndarray, 
        q_right: np.ndarray,
        threshold: float = 0.05
    ) -> bool:
        """
        Check if the configuration is safe (distance above threshold).
        
        Args:
            q_left: Left arm joint angles (7 DoF) in radians
            q_right: Right arm joint angles (7 DoF) in radians
            threshold: Minimum safe distance in meters (default 5cm)
            
        Returns:
            True if minimum distance > threshold, False otherwise
        """
        min_dist = self.compute_min_distance(q_left, q_right)
        return min_dist > threshold
    
    def forward_kinematics(
        self, 
        q_left: np.ndarray, 
        q_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for both flanges.
        
        Args:
            q_left: Left arm joint angles (7 DoF) in radians
            q_right: Right arm joint angles (7 DoF) in radians
            
        Returns:
            Tuple of (left_flange_pose, right_flange_pose) as 4x4 SE3 matrices
        """
        q = self._prepare_configuration(q_left, q_right)
        
        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get flange frame IDs
        left_flange_id = self.model.getFrameId("left_flange")
        right_flange_id = self.model.getFrameId("right_flange")
        
        left_pose = self.data.oMf[left_flange_id].homogeneous
        right_pose = self.data.oMf[right_flange_id].homogeneous
        
        return left_pose, right_pose


def create_checker(
    urdf_path: Optional[str] = None,
    mode: str = "capsule"
) -> DualArmCollisionChecker:
    """
    Factory function to create a collision checker with default URDF path.
    
    Args:
        urdf_path: Path to URDF. If None, uses default dual arm URDF
        mode: Detection mode - "capsule" or "mesh"
        
    Returns:
        Configured DualArmCollisionChecker instance
    """
    if urdf_path is None:
        # Use default URDF in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        urdf_path = os.path.join(parent_dir, "urdf", "flexiv_Rizon4s_dual_arm.urdf")
    
    return DualArmCollisionChecker(urdf_path, mode=mode)


if __name__ == "__main__":
    # Quick test
    print("Testing DualArmCollisionChecker...")
    
    checker = create_checker(mode="capsule")
    print(f"Created checker with {checker.n_collision_pairs} collision pairs")
    print(f"Joint names: {checker.joint_names}")
    
    # Test with zero configuration
    q_left = np.zeros(7)
    q_right = np.zeros(7)
    
    collision = checker.check_collision(q_left, q_right)
    min_dist = checker.compute_min_distance(q_left, q_right)
    
    print(f"Zero config - Collision: {collision}, Min distance: {min_dist:.4f}m")
    
    # Test with arms reaching toward each other
    q_left_reach = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])
    q_right_reach = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])
    
    collision = checker.check_collision(q_left_reach, q_right_reach)
    min_dist = checker.compute_min_distance(q_left_reach, q_right_reach)
    
    print(f"Reaching config - Collision: {collision}, Min distance: {min_dist:.4f}m")
    
    print("Test completed!")

