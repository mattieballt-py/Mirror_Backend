"""
Modal GPU class for live Gaussian Splat training
This stays warm on Modal infrastructure and receives frames from the robot.
"""
import modal
import numpy as np
import cv2
from typing import Dict, Optional
import json
import os

from src.training.incremental_trainer import IncrementalGaussianSplat
from src.robot.camera_params import SO100_INTRINSICS
from src.robot.so100_kinematics import compute_camera_pose
from src.storage.r2_uploader import R2Uploader


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
        self.chunk_count = 0
        self.uploaded_chunks = []
        
        # Initialize R2 uploader from environment variables
        self.uploader = R2Uploader(
            bucket_name=os.getenv('R2_BUCKET_NAME', 'gsplat-scenes'),
            access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            endpoint_url=os.getenv('R2_ENDPOINT_URL'),
            public_url=os.getenv('R2_PUBLIC_URL'),
        )
        
        # Chunk export settings
        self.frames_per_chunk = 50  # Export PLY every 50 frames
    
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
            
            # Check if it's time to export a chunk
            if self.frame_count % self.frames_per_chunk == 0:
                self._export_and_upload_chunk()
            
            return {
                "success": True,
                "frame_count": self.frame_count,
                "gaussian_count": self.trainer.get_gaussian_count(),
                "chunks_uploaded": len(self.uploaded_chunks),
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
    
    def _export_and_upload_chunk(self) -> None:
        """
        Export current model as a PLY chunk and upload to R2.
        Called periodically during training.
        """
        try:
            ply_bytes = self.trainer.to_ply_bytes()
            chunk_name = f"splat_{self.chunk_count:03d}.ply"
            
            # Upload chunk to R2
            url = self.uploader.upload_ply(ply_bytes, chunk_name)
            self.uploaded_chunks.append(chunk_name)
            self.chunk_count += 1
            
            # Update status.json
            progress = min(self.frame_count / 10000.0, 1.0)  # Assume 10k frames = 100%
            status = {
                "progress": progress,
                "chunks": self.uploaded_chunks,
                "complete": False,
                "frame_count": self.frame_count,
                "gaussian_count": self.trainer.get_gaussian_count(),
            }
            self.uploader.upload_status(status)
            print(f"Uploaded chunk {chunk_name}, progress: {progress:.1%}")
            
        except Exception as e:
            print(f"Error exporting/uploading chunk: {e}")
    
    @modal.method()
    def export_final(self) -> Dict[str, any]:
        """
        Export final model and mark training as complete.
        
        Returns:
            Dict with final status
        """
        try:
            # Export final chunk
            ply_bytes = self.trainer.to_ply_bytes()
            chunk_name = f"splat_{self.chunk_count:03d}.ply"
            url = self.uploader.upload_ply(ply_bytes, chunk_name)
            self.uploaded_chunks.append(chunk_name)
            
            # Mark as complete
            status = {
                "progress": 1.0,
                "chunks": self.uploaded_chunks,
                "complete": True,
                "frame_count": self.frame_count,
                "gaussian_count": self.trainer.get_gaussian_count(),
            }
            self.uploader.upload_status(status)
            
            return {
                "success": True,
                "message": "Training complete",
                "chunks": self.uploaded_chunks,
                "frame_count": self.frame_count,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    @modal.method()
    def get_status(self) -> Dict[str, any]:
        """
        Get current training status.
        
        Returns:
            Dict with frame count, Gaussian count, chunks, etc.
        """
        progress = min(self.frame_count / 10000.0, 1.0)
        return {
            "frame_count": self.frame_count,
            "gaussian_count": self.trainer.get_gaussian_count(),
            "training_active": self.training_active,
            "chunks_uploaded": self.uploaded_chunks,
            "progress": progress,
            "chunk_count": self.chunk_count,
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
