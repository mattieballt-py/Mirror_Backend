"""
SO-100 Robot Forward Kinematics

This module provides forward kinematics calculations to compute the camera pose
from joint positions of the SO-100 robotic arm.
"""
import numpy as np
from typing import Tuple
import torch

from src.robot.camera_params import CAMERA_OFFSET, CAMERA_ROTATION


def compute_camera_pose(joint_positions: np.ndarray) -> np.ndarray:
    """
    Compute camera pose (world to camera transformation) from joint positions.
    
    This is a placeholder implementation. Replace with actual SO-100 kinematics.
    
    Args:
        joint_positions: Array of 6 joint angles in radians [q1, q2, q3, q4, q5, q6]
        
    Returns:
        4x4 homogeneous transformation matrix (world to camera)
    """
    # Placeholder: Simple identity-based pose
    # In reality, you would compute this using your actual robot's DH parameters
    
    # For now, return a reasonable default pose
    # This should be replaced with actual forward kinematics
    
    pose = np.eye(4, dtype=np.float32)
    
    # Simple example: x offset based on joint 2, y based on joint 3
    pose[0, 3] = float(joint_positions[1]) * 0.1  # x translation
    pose[1, 3] = float(joint_positions[2]) * 0.1  # y translation
    pose[2, 3] = 0.5  # z roughly 50cm away
    
    # Apply camera offset
    pose[0, 3] += CAMERA_OFFSET[0]
    pose[1, 3] += CAMERA_OFFSET[1]
    pose[2, 3] += CAMERA_OFFSET[2]
    
    return pose


def compute_camera_pose_batch(
    joint_positions_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera poses for a batch of joint positions.
    
    Args:
        joint_positions_batch: Array of shape (N, 6) with N joint configurations
        
    Returns:
        Tuple of (poses, view_matrices) each of shape (N, 4, 4)
    """
    batch_size = joint_positions_batch.shape[0]
    poses = np.array([
        compute_camera_pose(joint_positions_batch[i])
        for i in range(batch_size)
    ], dtype=np.float32)
    
    # View matrices are inverse of poses
    view_matrices = np.linalg.inv(poses)
    
    return poses, view_matrices


def joints_to_camera_frame(points: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """
    Transform 3D points from world frame to camera frame.
    
    Args:
        points: Array of shape (N, 3) in world coordinates
        camera_pose: 4x4 camera pose matrix (world to camera)
        
    Returns:
        Points in camera frame, shape (N, 3)
    """
    # Add homogeneous coordinate
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Transform to camera frame
    points_camera_h = (camera_pose @ points_h.T).T
    
    # Remove homogeneous coordinate
    return points_camera_h[:, :3]


def camera_frame_to_joints(points: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """
    Transform 3D points from camera frame to world frame (inverse of above).
    
    Args:
        points: Array of shape (N, 3) in camera coordinates
        camera_pose: 4x4 camera pose matrix (world to camera)
        
    Returns:
        Points in world frame, shape (N, 3)
    """
    # Inverse pose (camera to world)
    inv_pose = np.linalg.inv(camera_pose)
    
    # Add homogeneous coordinate
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Transform to world frame
    points_world_h = (inv_pose @ points_h.T).T
    
    # Remove homogeneous coordinate
    return points_world_h[:, :3]


def tensor_compute_camera_pose(joint_positions: torch.Tensor) -> torch.Tensor:
    """
    PyTorch version of forward kinematics for batched GPU computation.
    
    Args:
        joint_positions: Tensor of shape (batch_size, 6) on GPU
        
    Returns:
        Camera poses as (batch_size, 4, 4) tensors
    """
    batch_size = joint_positions.shape[0]
    device = joint_positions.device
    
    # Initialize identity matrices
    poses = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    # Simple example implementation
    poses[:, 0, 3] = joint_positions[:, 1] * 0.1  # x
    poses[:, 1, 3] = joint_positions[:, 2] * 0.1  # y
    poses[:, 2, 3] = 0.5  # z
    
    return poses
