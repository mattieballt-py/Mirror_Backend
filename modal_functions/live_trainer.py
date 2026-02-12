"""
Modal GPU class for live Gaussian Splat training
This stays warm on Modal infrastructure and receives frames from the robot.
"""
import modal
import numpy as np
import cv2
from typing import Dict, Optional

from src.training.incremental_trainer import IncrementalGaussianSplat
from src.robot.camera_params import SO100_INTRINSICS
from src.robot.so100_kinematics import compute_camera_pose


class LiveSplatTrainer:
    """Modal GPU class that stays warm and processes frames"""
    
    def __init__(self):
        """Initialize the Gaussian Splat trainer on GPU"""
        self.trainer = IncrementalGaussianSplat(
            intrinsics=SO100_INTRINSICS,
            num_gaussians=100000
        )
        self.frame_count = 0
        self.training_active = False
    
    @modal.method()
    def add_frame(
        self,
        image_bytes: bytes,
        joint_positions: list,
    ) -> Dict[str, any]:
        """
        Receive a frame from the robot and train the Gaussian Splat.
        
        Args:
            image_bytes: Encoded image as bytes (PNG/JPEG)
            joint_positions: List of 6 joint angles in radians
            
        Returns:
            Dict with training status and statistics
        """
        try:
            # Decode image from bytes
            nparr = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to decode image",
                    "frame_count": self.frame_count,
                }
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Compute camera pose from joint angles
            joint_array = np.array(joint_positions, dtype=np.float32)
            camera_pose = compute_camera_pose(joint_array)
            
            # Add observation to trainer
            self.trainer.add_observation(image, camera_pose)
            
            self.frame_count += 1
            self.training_active = True
            
            return {
                "success": True,
                "frame_count": self.frame_count,
                "gaussian_count": self.trainer.get_gaussian_count(),
                "message": f"Processed frame {self.frame_count}",
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "frame_count": self.frame_count,
            }
    
    @modal.method()
    def export_splat(self) -> bytes:
        """
        Export the current Gaussian Splat model as PLY format.
        
        Returns:
            PLY file content as bytes
        """
        try:
            return self.trainer.to_ply_bytes()
        except Exception as e:
            print(f"Error exporting splat: {e}")
            return b""
    
    @modal.method()
    def get_status(self) -> Dict[str, any]:
        """
        Get current training status.
        
        Returns:
            Dict with frame count, Gaussian count, etc.
        """
        return {
            "frame_count": self.frame_count,
            "gaussian_count": self.trainer.get_gaussian_count(),
            "training_active": self.training_active,
        }
    
    @modal.method()
    def reset_model(self) -> Dict[str, any]:
        """
        Reset the Gaussian Splat model and begin fresh training.
        
        Returns:
            Confirmation dict
        """
        self.trainer = IncrementalGaussianSplat(
            intrinsics=SO100_INTRINSICS,
            num_gaussians=100000
        )
        self.frame_count = 0
        self.training_active = False
        
        return {
            "success": True,
            "message": "Model reset successfully",
        }
