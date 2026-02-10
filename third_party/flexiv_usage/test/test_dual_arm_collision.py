"""
Test suite for DualArmCollisionChecker

Tests:
1. Basic collision detection functionality
2. Mode switching (capsule vs mesh)
3. Performance benchmarks (target: ≥100Hz)
4. Edge cases and safety margins

Usage:
    pytest test/test_dual_arm_collision.py -v
    
Or run directly:
    python test/test_dual_arm_collision.py
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from collision_check.dual_arm_collision_checker import DualArmCollisionChecker, create_checker
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


def get_urdf_path():
    """Get the path to the dual arm URDF."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(script_dir, "urdf", "flexiv_Rizon4s_dual_arm.urdf")


class TestDualArmCollisionChecker:
    """Test cases for DualArmCollisionChecker."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        if not IMPORTS_AVAILABLE:
            print(f"Skipping tests - import error: {IMPORT_ERROR}")
            return
        
        cls.urdf_path = get_urdf_path()
        cls.checker_capsule = None
        cls.checker_mesh = None
        
        try:
            cls.checker_capsule = DualArmCollisionChecker(cls.urdf_path, mode="capsule")
            print(f"Created capsule checker with {cls.checker_capsule.n_collision_pairs} collision pairs")
        except Exception as e:
            print(f"Failed to create capsule checker: {e}")
        
        try:
            cls.checker_mesh = DualArmCollisionChecker(cls.urdf_path, mode="mesh")
            print(f"Created mesh checker with {cls.checker_mesh.n_collision_pairs} collision pairs")
        except Exception as e:
            print(f"Failed to create mesh checker: {e}")
    
    def test_zero_configuration_no_collision(self):
        """Test that zero configuration (home position) has no collision."""
        if self.checker_capsule is None:
            print("SKIP: checker not available")
            return
        
        q_left = np.zeros(7)
        q_right = np.zeros(7)
        
        collision = self.checker_capsule.check_collision(q_left, q_right)
        min_dist = self.checker_capsule.compute_min_distance(q_left, q_right)
        
        print(f"Zero config - Collision: {collision}, Min distance: {min_dist:.4f}m")
        
        assert not collision, "Zero configuration should not have collision"
        assert min_dist > 0.3, f"Zero config distance should be > 0.3m, got {min_dist}"
    
    def test_extreme_reach_collision(self):
        """Test that arms reaching toward each other can detect collision."""
        if self.checker_capsule is None:
            print("SKIP: checker not available")
            return
        
        # Both arms reaching toward center (joint 2 and 4 bent inward)
        q_left = np.array([0.0, 1.2, 0.0, -1.5, 0.0, 1.0, 0.0])
        q_right = np.array([0.0, 1.2, 0.0, -1.5, 0.0, 1.0, 0.0])
        
        min_dist = self.checker_capsule.compute_min_distance(q_left, q_right)
        print(f"Extreme reach - Min distance: {min_dist:.4f}m")
        
        # With 60cm separation, extreme reach should bring arms close
        assert min_dist < 0.5, f"Extreme reach should bring arms close, got {min_dist}m"
    
    def test_mode_switching(self):
        """Test switching between capsule and mesh modes."""
        if self.checker_capsule is None:
            print("SKIP: checker not available")
            return
        
        checker = DualArmCollisionChecker(self.urdf_path, mode="capsule")
        assert checker.mode == "capsule"
        
        q_left = np.zeros(7)
        q_right = np.zeros(7)
        
        # Get capsule distance
        dist_capsule = checker.compute_min_distance(q_left, q_right)
        
        # Switch to mesh mode
        checker.set_mode("mesh")
        assert checker.mode == "mesh"
        
        # Get mesh distance
        dist_mesh = checker.compute_min_distance(q_left, q_right)
        
        print(f"Mode comparison - Capsule: {dist_capsule:.4f}m, Mesh: {dist_mesh:.4f}m")
        
        # Both should report no collision for zero config
        assert dist_capsule > 0
        assert dist_mesh > 0
    
    def test_invalid_joint_angles(self):
        """Test error handling for invalid joint angles."""
        if self.checker_capsule is None:
            print("SKIP: checker not available")
            return
        
        try:
            # Wrong number of joints
            self.checker_capsule.check_collision(np.zeros(5), np.zeros(7))
            assert False, "Should raise ValueError for wrong joint count"
        except ValueError as e:
            print(f"Correctly caught error: {e}")
        
        try:
            self.checker_capsule.check_collision(np.zeros(7), np.zeros(10))
            assert False, "Should raise ValueError for wrong joint count"
        except ValueError as e:
            print(f"Correctly caught error: {e}")
    
    def test_closest_points(self):
        """Test getting closest points between arms."""
        if self.checker_capsule is None:
            print("SKIP: checker not available")
            return
        
        q_left = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.5, 0.0])
        q_right = np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.5, 0.0])
        
        pt_left, pt_right, dist = self.checker_capsule.get_closest_points(q_left, q_right)
        
        print(f"Closest points:")
        print(f"  Left:  {pt_left}")
        print(f"  Right: {pt_right}")
        print(f"  Distance: {dist:.4f}m")
        
        # Verify distance matches point distance
        computed_dist = np.linalg.norm(pt_left - pt_right)
        assert abs(computed_dist - dist) < 0.01, f"Distance mismatch: {computed_dist} vs {dist}"
    
    def test_forward_kinematics(self):
        """Test forward kinematics computation."""
        if self.checker_capsule is None:
            print("SKIP: checker not available")
            return
        
        q_left = np.zeros(7)
        q_right = np.zeros(7)
        
        left_pose, right_pose = self.checker_capsule.forward_kinematics(q_left, q_right)
        
        print(f"Left flange position: {left_pose[:3, 3]}")
        print(f"Right flange position: {right_pose[:3, 3]}")
        
        # Verify shape
        assert left_pose.shape == (4, 4)
        assert right_pose.shape == (4, 4)
        
        # Left arm at x=-0.3, right at x=0.3 (approximately)
        assert left_pose[0, 3] < 0, "Left flange should be at negative x"
        assert right_pose[0, 3] > 0, "Right flange should be at positive x"
    
    def test_is_safe(self):
        """Test safety threshold check."""
        if self.checker_capsule is None:
            print("SKIP: checker not available")
            return
        
        q_left = np.zeros(7)
        q_right = np.zeros(7)
        
        # Zero config should be safe with default threshold
        is_safe = self.checker_capsule.is_safe(q_left, q_right, threshold=0.05)
        assert is_safe, "Zero config should be safe"
        
        # Test with very large threshold
        is_safe_large = self.checker_capsule.is_safe(q_left, q_right, threshold=10.0)
        assert not is_safe_large, "Should not be safe with 10m threshold"


class TestPerformance:
    """Performance benchmarks for collision detection."""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures."""
        if not IMPORTS_AVAILABLE:
            return
        
        cls.urdf_path = get_urdf_path()
        try:
            cls.checker_capsule = DualArmCollisionChecker(cls.urdf_path, mode="capsule")
            cls.checker_mesh = DualArmCollisionChecker(cls.urdf_path, mode="mesh")
        except Exception as e:
            print(f"Failed to create checkers: {e}")
            cls.checker_capsule = None
            cls.checker_mesh = None
    
    def _benchmark(self, checker, name, n_iterations=1000):
        """Run benchmark for a checker."""
        if checker is None:
            print(f"SKIP {name}: checker not available")
            return None
        
        # Generate random configurations
        np.random.seed(42)
        q_lefts = np.random.uniform(-1.5, 1.5, (n_iterations, 7))
        q_rights = np.random.uniform(-1.5, 1.5, (n_iterations, 7))
        
        # Warm up
        for i in range(10):
            checker.check_collision(q_lefts[i], q_rights[i])
        
        # Benchmark collision check
        start_time = time.perf_counter()
        for i in range(n_iterations):
            checker.check_collision(q_lefts[i], q_rights[i])
        elapsed = time.perf_counter() - start_time
        
        freq_collision = n_iterations / elapsed
        time_per_check = elapsed / n_iterations * 1000  # ms
        
        # Benchmark distance computation
        start_time = time.perf_counter()
        for i in range(n_iterations):
            checker.compute_min_distance(q_lefts[i], q_rights[i])
        elapsed_dist = time.perf_counter() - start_time
        
        freq_distance = n_iterations / elapsed_dist
        time_per_dist = elapsed_dist / n_iterations * 1000  # ms
        
        print(f"\n{name} Performance ({n_iterations} iterations):")
        print(f"  Collision check: {freq_collision:.1f} Hz ({time_per_check:.3f} ms/check)")
        print(f"  Distance compute: {freq_distance:.1f} Hz ({time_per_dist:.3f} ms/compute)")
        
        return freq_collision, freq_distance
    
    def test_capsule_performance(self):
        """Benchmark capsule mode performance (target: ≥100Hz)."""
        result = self._benchmark(self.checker_capsule, "Capsule Mode", n_iterations=1000)
        
        if result is not None:
            freq_collision, freq_distance = result
            assert freq_collision >= 100, f"Capsule collision check should be ≥100Hz, got {freq_collision:.1f}Hz"
            print(f"  ✓ Meets 100Hz requirement")
    
    def test_mesh_performance(self):
        """Benchmark mesh mode performance."""
        result = self._benchmark(self.checker_mesh, "Mesh Mode", n_iterations=100)
        
        if result is not None:
            freq_collision, freq_distance = result
            print(f"  Note: Mesh mode is slower but more accurate")
    
    def test_performance_comparison(self):
        """Compare capsule vs mesh performance."""
        if self.checker_capsule is None or self.checker_mesh is None:
            print("SKIP: checkers not available")
            return
        
        n_iterations = 100
        np.random.seed(42)
        q_lefts = np.random.uniform(-1.0, 1.0, (n_iterations, 7))
        q_rights = np.random.uniform(-1.0, 1.0, (n_iterations, 7))
        
        # Capsule
        start = time.perf_counter()
        for i in range(n_iterations):
            self.checker_capsule.check_collision(q_lefts[i], q_rights[i])
        capsule_time = time.perf_counter() - start
        
        # Mesh
        start = time.perf_counter()
        for i in range(n_iterations):
            self.checker_mesh.check_collision(q_lefts[i], q_rights[i])
        mesh_time = time.perf_counter() - start
        
        speedup = mesh_time / capsule_time
        print(f"\nPerformance Comparison:")
        print(f"  Capsule: {n_iterations/capsule_time:.1f} Hz")
        print(f"  Mesh:    {n_iterations/mesh_time:.1f} Hz")
        print(f"  Speedup: {speedup:.1f}x faster with capsule")


def run_all_tests():
    """Run all tests manually."""
    print("=" * 60)
    print("Running DualArmCollisionChecker Tests")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print(f"\n!!! Import error: {IMPORT_ERROR}")
        print("Please install dependencies:")
        print("  conda install -c conda-forge pinocchio hpp-fcl")
        return False
    
    # Check URDF exists
    urdf_path = get_urdf_path()
    if not os.path.exists(urdf_path):
        print(f"\n!!! URDF not found: {urdf_path}")
        return False
    
    print(f"\nUsing URDF: {urdf_path}")
    
    # Run functionality tests
    print("\n" + "-" * 40)
    print("Functionality Tests")
    print("-" * 40)
    
    test_func = TestDualArmCollisionChecker()
    test_func.setup_class()
    
    tests = [
        ("Zero configuration no collision", test_func.test_zero_configuration_no_collision),
        ("Extreme reach collision", test_func.test_extreme_reach_collision),
        ("Mode switching", test_func.test_mode_switching),
        ("Invalid joint angles", test_func.test_invalid_joint_angles),
        ("Closest points", test_func.test_closest_points),
        ("Forward kinematics", test_func.test_forward_kinematics),
        ("Is safe", test_func.test_is_safe),
    ]
    
    passed = 0
    failed = 0
    
    for name, test in tests:
        print(f"\n>>> {name}")
        try:
            test()
            print(f"    PASSED")
            passed += 1
        except AssertionError as e:
            print(f"    FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1
    
    # Run performance tests
    print("\n" + "-" * 40)
    print("Performance Tests")
    print("-" * 40)
    
    test_perf = TestPerformance()
    test_perf.setup_class()
    
    perf_tests = [
        ("Capsule performance", test_perf.test_capsule_performance),
        ("Mesh performance", test_perf.test_mesh_performance),
        ("Performance comparison", test_perf.test_performance_comparison),
    ]
    
    for name, test in perf_tests:
        print(f"\n>>> {name}")
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"    FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

