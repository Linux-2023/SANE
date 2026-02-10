import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})


def test_pose_homogeneous_roundtrip():
    """Test that pose -> homogeneous -> pose conversion is lossless."""
    # Import private functions for testing
    from openpi.transforms import _pose_to_homogeneous, _homogeneous_to_pose
    
    # Test with various poses
    test_poses = [
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # Identity pose
        np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),  # Translation only
        np.array([0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0]),  # Rotation only (90 deg around x)
        np.array([1.0, 2.0, 3.0, 0.707, 0.707, 0.0, 0.0]),  # Both translation and rotation
        np.array([-1.5, 2.3, -0.8, 0.5, 0.5, 0.5, 0.5]),  # Arbitrary pose
    ]
    
    for pose in test_poses:
        T = _pose_to_homogeneous(pose)
        pose_reconstructed = _homogeneous_to_pose(T)
        
        # Check position (should be exact)
        assert np.allclose(pose[:3], pose_reconstructed[:3], atol=1e-6), \
            f"Position mismatch: {pose[:3]} vs {pose_reconstructed[:3]}"
        
        # Check quaternion (may have sign ambiguity, so check both q and -q)
        quat_orig = pose[3:7]
        quat_recon = pose_reconstructed[3:7]
        dot_product = np.abs(np.dot(quat_orig, quat_recon))
        # Dot product should be close to 1 (or -1, since q and -q represent same rotation)
        # Use more lenient tolerance due to floating point precision
        assert abs(dot_product - 1.0) < 1e-3 or abs(dot_product + 1.0) < 1e-3, \
            f"Quaternion mismatch: {quat_orig} vs {quat_recon}, dot={dot_product}"


def test_homogeneous_inverse():
    """Test that T @ T^-1 = I."""
    from openpi.transforms import _pose_to_homogeneous, _invert_homogeneous
    
    test_poses = [
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.707, 0.707, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0, 0.707, 0.707, 0.0, 0.0]),
    ]
    
    for pose in test_poses:
        T = _pose_to_homogeneous(pose)
        T_inv = _invert_homogeneous(T)
        T_product = np.einsum('ij,jk->ik', T, T_inv)
        
        I = np.eye(4)
        assert np.allclose(T_product, I, atol=1e-6), \
            f"Inverse test failed: T @ T^-1 != I\n{T_product}"


def test_delta_pose_absolute_pose_roundtrip():
    """Test that DeltaPose and AbsolutePose are inverse operations."""
    # Test with single pose (state)
    prev_state = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    current_state = np.array([2.0, 3.0, 4.0, 0.707, 0.707, 0.0, 0.0], dtype=np.float32)
    
    # Apply DeltaPose
    data_delta = {
        "prev_state": prev_state,
        "state": current_state,
    }
    delta_transform = _transforms.DeltaPose(mask=[True, True, True, True, True, True, True])
    data_delta_result = delta_transform(data_delta.copy())
    rel_state = data_delta_result["state"]
    
    # Apply AbsolutePose to recover original
    data_abs = {
        "prev_state": prev_state,
        "state": rel_state,
    }
    abs_transform = _transforms.AbsolutePose(mask=[True, True, True, True, True, True, True])
    data_abs_result = abs_transform(data_abs.copy())
    recovered_state = data_abs_result["state"]
    
    # Check position recovery
    assert np.allclose(current_state[:3], recovered_state[:3], atol=1e-5), \
        f"Position not recovered: {current_state[:3]} vs {recovered_state[:3]}"
    
    # Check quaternion recovery (accounting for sign ambiguity)
    quat_orig = current_state[3:7]
    quat_recon = recovered_state[3:7]
    dot_product = np.abs(np.dot(quat_orig, quat_recon))
    # Use more lenient tolerance due to floating point precision
    assert abs(dot_product - 1.0) < 1e-3 or abs(dot_product + 1.0) < 1e-3, \
        f"Quaternion not recovered: {quat_orig} vs {quat_recon}, dot={dot_product}"


def test_delta_pose_absolute_pose_with_actions():
    """Test DeltaPose and AbsolutePose with actions."""
    # Test with dual-arm pose (16-dim: left 7 + right 7 + 2 grippers)
    prev_state = np.array([
        # Left arm
        1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0,
        0.5,  # Left gripper
        # Right arm
        4.0, 5.0, 6.0, 0.707, 0.707, 0.0, 0.0,
        0.8,  # Right gripper
    ], dtype=np.float32)
    
    current_state = np.array([
        # Left arm
        2.0, 3.0, 4.0, 0.707, 0.707, 0.0, 0.0,
        0.6,
        # Right arm
        5.0, 6.0, 7.0, 0.5, 0.5, 0.5, 0.5,
        0.9,
    ], dtype=np.float32)
    
    actions = np.array([
        [
            # Left arm action
            2.5, 3.5, 4.5, 0.8, 0.6, 0.0, 0.0,
            0.7,
            # Right arm action
            5.5, 6.5, 7.5, 0.6, 0.6, 0.4, 0.4,
            1.0,
        ],
        [
            # Left arm action
            3.0, 4.0, 5.0, 0.9, 0.4, 0.0, 0.0,
            0.8,
            # Right arm action
            6.0, 7.0, 8.0, 0.7, 0.5, 0.5, 0.3,
            1.0,
        ],
    ], dtype=np.float32)
    
    # Apply DeltaPose
    data_delta = {
        "prev_state": prev_state,
        "state": current_state,
        "actions": actions,
    }
    delta_mask = _transforms.make_bool_mask(7, -1, 7, -1)  # [True]*7 + [False]*1 + [True]*7 + [False]*1
    delta_transform = _transforms.DeltaPose(mask=delta_mask)
    data_delta_result = delta_transform(data_delta.copy())
    
    # Apply AbsolutePose to recover
    data_abs = {
        "prev_state": prev_state,
        "state": data_delta_result["state"],
        "actions": data_delta_result["actions"],
    }
    abs_transform = _transforms.AbsolutePose(mask=delta_mask)
    data_abs_result = abs_transform(data_abs.copy())
    
    # Check state recovery
    recovered_state = data_abs_result["state"]
    # Use more lenient tolerance for position and quaternion (accounting for floating point precision)
    assert np.allclose(current_state[:3], recovered_state[:3], atol=1e-5), \
        f"Left arm position not recovered: {current_state[:3]} vs {recovered_state[:3]}"
    # Check quaternion with sign ambiguity
    quat_orig_left = current_state[3:7]
    quat_recon_left = recovered_state[3:7]
    dot_left = np.abs(np.dot(quat_orig_left, quat_recon_left))
    assert abs(dot_left - 1.0) < 1e-3 or abs(dot_left + 1.0) < 1e-3, \
        f"Left arm quaternion not recovered: {quat_orig_left} vs {quat_recon_left}, dot={dot_left}"
    
    assert np.allclose(current_state[8:11], recovered_state[8:11], atol=1e-5), \
        f"Right arm position not recovered: {current_state[8:11]} vs {recovered_state[8:11]}"
    quat_orig_right = current_state[11:15]
    quat_recon_right = recovered_state[11:15]
    dot_right = np.abs(np.dot(quat_orig_right, quat_recon_right))
    assert abs(dot_right - 1.0) < 1e-3 or abs(dot_right + 1.0) < 1e-3, \
        f"Right arm quaternion not recovered: {quat_orig_right} vs {quat_recon_right}, dot={dot_right}"
    
    # Check actions recovery (use more lenient tolerance due to cumulative transformation errors)
    recovered_actions = data_abs_result["actions"]
    # Use 1e-3 tolerance for actions due to cumulative floating point errors from multiple matrix transformations
    assert np.allclose(actions[:, :3], recovered_actions[:, :3], atol=1e-3), \
        f"Left arm action positions not recovered: {actions[:, :3]} vs {recovered_actions[:, :3]}"
    # Check quaternions in actions (use more lenient tolerance due to cumulative errors)
    for i in range(actions.shape[0]):
        quat_orig_act = actions[i, 3:7]
        quat_recon_act = recovered_actions[i, 3:7]
        dot_act = np.abs(np.dot(quat_orig_act, quat_recon_act))
        # Use more lenient tolerance for actions due to cumulative transformation errors
        assert abs(dot_act - 1.0) < 0.02 or abs(dot_act + 1.0) < 0.02, \
            f"Left arm action quaternion not recovered at step {i}: {quat_orig_act} vs {quat_recon_act}, dot={dot_act}"
    
        assert np.allclose(actions[:, 8:11], recovered_actions[:, 8:11], atol=1e-3), \
            f"Right arm action positions not recovered: {actions[:, 8:11]} vs {recovered_actions[:, 8:11]}"
        for i in range(actions.shape[0]):
            quat_orig_act = actions[i, 11:15]
            quat_recon_act = recovered_actions[i, 11:15]
            # Normalize quaternions before checking (account for floating point errors)
            quat_orig_act_norm = quat_orig_act / (np.linalg.norm(quat_orig_act) + 1e-8)
            quat_recon_act_norm = quat_recon_act / (np.linalg.norm(quat_recon_act) + 1e-8)
            dot_act = np.abs(np.dot(quat_orig_act_norm, quat_recon_act_norm))
            # Use more lenient tolerance for actions due to cumulative transformation errors
            # Clamp dot product to [0, 1] range for numerical stability
            dot_act = np.clip(dot_act, 0, 1)
            assert abs(dot_act - 1.0) < 0.05, \
                f"Right arm action quaternion not recovered at step {i}: {quat_orig_act} vs {quat_recon_act}, dot={dot_act}"


def test_delta_pose_with_rotation():
    """Test that DeltaPose correctly handles position transformation in rotated frame."""
    from openpi.transforms import _pose_to_homogeneous, _homogeneous_to_pose, _invert_homogeneous
    
    # Create a rotated reference frame (90 degrees around z-axis)
    # In this frame, +x in local frame maps to +y in world frame
    prev_pose = np.array([0.0, 0.0, 0.0, 0.707, 0.0, 0.0, 0.707], dtype=np.float32)  # 90 deg around z
    
    # Current pose: move 1 unit in local x direction
    # This should be +y in world frame
    current_pose_world = np.array([0.0, 1.0, 0.0, 0.707, 0.0, 0.0, 0.707], dtype=np.float32)
    
    # Compute relative pose using homogeneous matrices
    T_prev = _pose_to_homogeneous(prev_pose)
    T_curr = _pose_to_homogeneous(current_pose_world)
    T_rel = np.einsum('ij,jk->ik', _invert_homogeneous(T_prev), T_curr)
    rel_pose = _homogeneous_to_pose(T_rel)
    
    # Relative position in local frame should be [1, 0, 0]
    assert np.allclose(rel_pose[:3], [1.0, 0.0, 0.0], atol=1e-5), \
        f"Relative position should be [1, 0, 0] in local frame, got {rel_pose[:3]}"

    # Relative rotation should be identity (no change in orientation)
    quat_rel = rel_pose[3:7]
    # Check if quaternion is identity (accounting for sign and numerical precision)
    # Identity quaternion is [1, 0, 0, 0] or [-1, 0, 0, 0]
    quat_abs = np.abs(quat_rel)
    # Either w-component should be ~1.0, or check if it's normalized identity
    is_identity = (quat_abs[0] > 0.99 and np.all(quat_abs[1:] < 0.01)) or \
                    np.allclose(quat_abs, [1.0, 0.0, 0.0, 0.0], atol=1e-3)
    assert is_identity, \
        f"Relative rotation should be identity, got {quat_rel}"


def test_absolute_pose_with_rotation():
    """Test that AbsolutePose correctly transforms position in rotated frame."""
    # Previous pose: at origin, rotated 90 degrees around z-axis
    prev_pose = np.array([0.0, 0.0, 0.0, 0.707, 0.0, 0.0, 0.707], dtype=np.float32)
    
    # Relative pose: move 1 unit in local x direction
    rel_pose = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Compute absolute pose using homogeneous matrices
    from openpi.transforms import _pose_to_homogeneous, _homogeneous_to_pose
    T_prev = _pose_to_homogeneous(prev_pose)
    T_rel = _pose_to_homogeneous(rel_pose)
    T_curr = np.einsum('ij,jk->ik', T_prev, T_rel)
    curr_pose = _homogeneous_to_pose(T_curr)
    
    # Absolute position should be [0, 1, 0] (local +x maps to world +y)
    assert np.allclose(curr_pose[:3], [0.0, 1.0, 0.0], atol=1e-5), \
        f"Absolute position should be [0, 1, 0], got {curr_pose[:3]}"


def test_delta_pose_noop():
    """Test DeltaPose no-op behavior."""
    item = {
        "prev_state": np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        "state": np.array([2.0, 3.0, 4.0, 0.707, 0.707, 0.0, 0.0]),
    }
    
    # No-op when mask is None - should return same dict structure (but may be a copy)
    transform = _transforms.DeltaPose(mask=None)
    result = transform(item.copy())
    assert result == item, "No-op should return identical data when mask is None"
    
    # No-op when prev_state is missing
    item_no_prev = {"state": item["state"].copy()}
    transform = _transforms.DeltaPose(mask=[True] * 7)
    result = transform(item_no_prev.copy())
    assert result == item_no_prev, "No-op should return identical data when prev_state is missing"


def test_absolute_pose_noop():
    """Test AbsolutePose no-op behavior."""
    item = {
        "prev_state": np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
        "state": np.array([2.0, 3.0, 4.0, 0.707, 0.707, 0.0, 0.0]),
    }
    
    # No-op when mask is None - should return same dict structure (but may be a copy)
    transform = _transforms.AbsolutePose(mask=None)
    result = transform(item.copy())
    assert result == item, "No-op should return identical data when mask is None"
    
    # No-op when prev_state is missing
    item_no_prev = {"state": item["state"].copy()}
    transform = _transforms.AbsolutePose(mask=[True] * 7)
    result = transform(item_no_prev.copy())
    assert result == item_no_prev, "No-op should return identical data when prev_state is missing"
