"""
Incremental Gaussian Splatting trainer using gsplat library.
"""
import torch
import random
from typing import Tuple, List
import numpy as np
import os
import json
import boto3
from datetime import datetime

from src.training.gaussian_model import GaussianModel


class IncrementalGaussianSplat:
    """Custom incremental training logic using gsplat"""
    
    def __init__(self, intrinsics: np.ndarray, num_gaussians: int = 100000):
        """
        Initialize the Gaussian Splat trainer.
        
        Args:
            intrinsics: 3x3 camera intrinsics matrix (K matrix)
            num_gaussians: Initial number of Gaussians
        """
        # Initialize Gaussian parameters
        self.gaussians = GaussianModel(num_gaussians=num_gaussians)
        self.gaussians = self.gaussians.cuda()
        
        # Camera intrinsics
        self.K = torch.tensor(intrinsics, dtype=torch.float32).cuda()
        
        # Optimizer for Gaussians
        self.optimizer = torch.optim.Adam(
            self.gaussians.parameters(),
            lr=0.0016
        )
        
        # Scheduler (optional)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=0.99
        )
        
        # Frame buffer
        self.frames: List[torch.Tensor] = []
        self.poses: List[torch.Tensor] = []
        self.frame_count = 0
        
        # Training config
        self.densification_interval = 100
        self.prune_threshold = 0.005
        
        # R2 upload tracking
        self.job_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.chunk_count = 0
        self.uploaded_chunks = []
        self.total_frames = 1000  # Can be updated dynamically
        
        # Initialize boto3 S3 client for R2
        self.s3_client = None
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ.get('CF_R2_KEY'),
                aws_secret_access_key=os.environ.get('CF_R2_SECRET'),
                endpoint_url=os.environ.get('CF_R2_ENDPOINT')
            )
            self.r2_bucket = os.environ.get('R2_BUCKET_NAME', 'gsplat-scenes')
            print(f"✓ R2 client initialized for job {self.job_id}")
        except Exception as e:
            print(f"✗ Failed to initialize R2 client: {e}")
            self.s3_client = None
    
    def add_observation(self, image: np.ndarray, camera_pose: np.ndarray) -> None:
        """
        Add a new observation (frame + camera pose) to trainer.
        
        Args:
            image: RGB image as numpy array (H, W, 3) in range [0, 255]
            camera_pose: 4x4 camera pose matrix (world to camera)
        """
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().cuda() / 255.0
        pose_tensor = torch.from_numpy(camera_pose).float().cuda()
        
        self.frames.append(image_tensor)
        self.poses.append(pose_tensor)
        self.frame_count += 1
        
        print(f"Frame processed: {self.frame_count}")
        
        # Upload chunk every 5 frames
        if self.frame_count % 5 == 0:
            self._export_and_upload_chunk(self.frame_count)
        
        # Refine every 10 frames
        if len(self.frames) % 10 == 0:
            loss = self._refine(iterations=100)
            print(f"✓ Training iteration complete: {len(self.frames)} frames, avg loss: {loss:.6f}")
    
    def _refine(self, iterations: int = 100) -> float:
        """
        Training loop: use gsplat's rasterization to optimize Gaussians.
        
        Args:
            iterations: Number of optimization iterations
            
        Returns:
            Average loss over iterations
        """
        try:
            from gsplat import rasterization
        except ImportError:
            print("Warning: gsplat not installed. Skipping training step.")
            return 0.0
        
        total_loss = 0.0
        
        for i in range(iterations):
            # Sample random frame
            idx = random.randint(0, len(self.frames) - 1)
            image = self.frames[idx]
            pose = self.poses[idx]
            
            # Get Gaussian parameters
            params = self.gaussians.get_all_params()
            
            # Prepare batched inputs for gsplat 1.x
            viewmats = pose.unsqueeze(0)  # (1, 4, 4)
            Ks = self.K.unsqueeze(0)  # (1, 3, 3)
            
            try:
                # Use gsplat 1.x rasterization API
                rendered_colors, alphas, info = rasterization(
                    means=params['means'],
                    quats=params['quats'],
                    scales=params['scales'],
                    opacities=params['opacities'],
                    colors=params['colors'],
                    viewmats=viewmats,
                    Ks=Ks,
                    width=image.shape[1],
                    height=image.shape[0],
                    near_plane=0.01,
                    far_plane=100.0,
                )
                
                # Extract single image from batch (shape: (1, H, W, 3) -> (H, W, 3))
                rendered = rendered_colors[0]
                
                # Compute loss (MSE between rendered and ground truth)
                loss = ((rendered - image) ** 2).mean()
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            except Exception as e:
                print(f"✗ Error during rasterization iteration {i}: {e}")
                import traceback
                traceback.print_exc()
                self.optimizer.zero_grad()
                continue
            
            # Densification (using simple heuristic)
            if i % self.densification_interval == 0 and i > 0:
                self._densify_or_prune()
        
        # Step scheduler
        self.scheduler.step()
        
        return total_loss / max(iterations, 1)
    
    def _densify_or_prune(self) -> None:
        """Simple densification/pruning based on gradient magnitude"""
        with torch.no_grad():
            # Simple heuristic: prune Gaussians with low opacity
            opacities = self.gaussians.opacities.data
            
            # Count high-opacity Gaussians
            high_opacity_mask = opacities.squeeze() > self.prune_threshold
            num_high_opacity = high_opacity_mask.sum().item()
            
            # Only prune if we have excess Gaussians
            if num_high_opacity < self.gaussians.num_gaussians * 0.5:
                # Could implement splitting here
                pass
    
    def to_ply_bytes(self) -> bytes:
        """
        Export Gaussian Splat as PLY format bytes in @mkkellogg/gaussian-splats-3d format.
        
        Returns:
            PLY file content as bytes
        """
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            print("Warning: plyfile not installed")
            return b""
        
        with torch.no_grad():
            params = self.gaussians.get_all_params()
        
        # Guard: Check if parameters have been trained
        means = params['means'].detach().cpu().numpy()
        if means.shape[0] == 0:
            print("✗ Cannot export PLY: no Gaussian points")
            return b""
        
        print(f"Exporting PLY: {means.shape[0]} gaussians")
        
        # Detach all parameters from GPU
        colors = params['colors'].detach().cpu().numpy()  # (N, 3) in [0, 1]
        opacities = params['opacities'].detach().cpu().numpy()  # (N,)
        scales = params['scales'].detach().cpu().numpy()  # (N, 3) log-scale
        quats = params['quats'].detach().cpu().numpy()  # (N, 4)
        
        # Convert RGB colors to spherical harmonics DC coefficients
        # SH_C0 = 0.28209479177387814 (0th order SH coefficient)
        SH_C0 = 0.28209479177387814
        f_dc = (colors - 0.5) / SH_C0  # (N, 3)
        
        # Convert opacity to pre-sigmoid (inverse sigmoid)
        # If opacity is already in [0, 1], convert to logit space
        # logit(x) = log(x / (1 - x))
        opacities_clamped = np.clip(opacities, 1e-6, 1 - 1e-6)
        opacities_logit = np.log(opacities_clamped / (1 - opacities_clamped))
        
        # Normalize quaternions
        quats_norm = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)
        
        num_points = means.shape[0]
        
        # Build vertex data with proper @mkkellogg/gaussian-splats-3d format
        # Position (x, y, z), normals (nx, ny, nz), SH DC (f_dc_0, f_dc_1, f_dc_2),
        # 45 higher-order SH coefficients (f_rest_0 ... f_rest_44),
        # opacity, scale (scale_0, scale_1, scale_2), rotation (rot_0, rot_1, rot_2, rot_3)
        
        dtype_full = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
        # Add 45 higher-order SH coefficients (for 3 channels, 15 coefficients each)
        for i in range(45):
            dtype_full.append((f'f_rest_{i}', 'f4'))
        dtype_full.extend([
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ])
        
        # Build structured array
        vertex_data = np.zeros(num_points, dtype=dtype_full)
        vertex_data['x'] = means[:, 0]
        vertex_data['y'] = means[:, 1]
        vertex_data['z'] = means[:, 2]
        vertex_data['nx'] = 0.0
        vertex_data['ny'] = 0.0
        vertex_data['nz'] = 0.0
        vertex_data['f_dc_0'] = f_dc[:, 0]
        vertex_data['f_dc_1'] = f_dc[:, 1]
        vertex_data['f_dc_2'] = f_dc[:, 2]
        # Higher-order SH coefficients set to 0
        for i in range(45):
            vertex_data[f'f_rest_{i}'] = 0.0
        vertex_data['opacity'] = opacities_logit
        vertex_data['scale_0'] = scales[:, 0]
        vertex_data['scale_1'] = scales[:, 1]
        vertex_data['scale_2'] = scales[:, 2]
        vertex_data['rot_0'] = quats_norm[:, 0]
        vertex_data['rot_1'] = quats_norm[:, 1]
        vertex_data['rot_2'] = quats_norm[:, 2]
        vertex_data['rot_3'] = quats_norm[:, 3]
        
        # Create PLY element
        vertex_el = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_el])
        
        # Write to bytes
        import io
        output = io.BytesIO()
        ply_data.write(output)
        ply_bytes = output.getvalue()
        
        print(f"✓ PLY export complete: {len(ply_bytes)} bytes ({len(ply_bytes) / 1024 / 1024:.2f} MB)")
        return ply_bytes
    
    def get_gaussian_count(self) -> int:
        """Return current number of Gaussians"""
        return self.gaussians.num_gaussians
    
    def _export_and_upload_chunk(self, frame_idx: int) -> None:
        """
        Export current Gaussian parameters to PLY and upload to R2.
        
        Args:
            frame_idx: Current frame index for tracking progress
        """
        if self.s3_client is None:
            print("✗ R2 client not initialized, skipping upload")
            return
        
        try:
            # Guard: Check if we have enough frames for training
            if len(self.frames) == 0:
                print(f"✗ No frames available for export at frame {frame_idx}")
                return
            
            # Export PLY bytes
            ply_bytes = self.to_ply_bytes()
            
            if not ply_bytes or len(ply_bytes) == 0:
                print(f"✗ PLY export returned empty bytes at frame {frame_idx}")
                return
            
            # Validate file size
            size_kb = len(ply_bytes) / 1024
            if size_kb < 100:
                print(f"⚠️  WARNING: PLY suspiciously small: {len(ply_bytes)} bytes ({size_kb:.1f} KB) — may be empty or untrained")
            
            # Create chunk filename
            chunk_filename = f"splat_{self.chunk_count:03d}.ply"
            chunk_key = f"{self.job_id}/{chunk_filename}"
            
            # Write PLY to /tmp for debugging
            os.makedirs(f"/tmp/{self.job_id}", exist_ok=True)
            local_path = f"/tmp/{self.job_id}/{chunk_filename}"
            with open(local_path, 'wb') as f:
                f.write(ply_bytes)
            print(f"✓ PLY written locally: {local_path} ({len(ply_bytes)} bytes)")
            
            # Upload to R2
            try:
                self.s3_client.put_object(
                    Bucket=self.r2_bucket,
                    Key=chunk_key,
                    Body=ply_bytes,
                    ContentType='application/octet-stream'
                )
                self.uploaded_chunks.append(chunk_key)
                self.chunk_count += 1
                print(f"✓ Chunk uploaded to R2: {chunk_key}")
                
            except Exception as upload_error:
                print(f"✗ Failed to upload {chunk_key} to R2:")
                print(f"   Error type: {type(upload_error).__name__}")
                print(f"   Error message: {str(upload_error)}")
                import traceback
                traceback.print_exc()
                return
            
            # Update status.json in R2
            self._update_status_json(frame_idx, complete=False)
            
        except Exception as e:
            print(f"✗ Error in _export_and_upload_chunk at frame {frame_idx}:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_status_json(self, frame_idx: int, complete: bool = False) -> None:
        """
        Update status.json in R2 with current progress.
        
        Args:
            frame_idx: Current frame index
            complete: Whether training is complete
        """
        if self.s3_client is None:
            return
        
        try:
            progress = min(frame_idx / self.total_frames, 1.0)
            status = {
                "progress": progress,
                "chunks": self.uploaded_chunks,
                "complete": complete,
                "frame_count": frame_idx,
                "gaussian_count": self.get_gaussian_count(),
                "job_id": self.job_id
            }
            
            status_key = f"{self.job_id}/status.json"
            status_json = json.dumps(status, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.r2_bucket,
                Key=status_key,
                Body=status_json.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"✓ Status updated: {progress:.1%} complete, {len(self.uploaded_chunks)} chunks")
            
        except Exception as e:
            print(f"✗ Failed to update status.json: {e}")
            import traceback
            traceback.print_exc()
    
    def finalize_training(self) -> None:
        """
        Upload final PLY and mark training as complete.
        Call this when all frames are processed.
        """
        if self.s3_client is None:
            print("✗ R2 client not initialized, cannot finalize")
            return
        
        try:
            # Export final PLY
            ply_bytes = self.to_ply_bytes()
            
            if not ply_bytes:
                print("✗ Cannot export final PLY: empty data")
                return
            
            # Upload final PLY
            final_key = f"{self.job_id}/splat_final.ply"
            self.s3_client.put_object(
                Bucket=self.r2_bucket,
                Key=final_key,
                Body=ply_bytes,
                ContentType='application/octet-stream'
            )
            self.uploaded_chunks.append(final_key)
            print(f"✓ Final PLY uploaded to R2: {final_key} ({len(ply_bytes)} bytes)")
            
            # Mark complete in status.json
            self._update_status_json(self.frame_count, complete=True)
            print(f"✓ Training finalized: {self.frame_count} frames, {len(self.uploaded_chunks)} chunks")
            
        except Exception as e:
            print(f"✗ Error finalizing training:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            import traceback
            traceback.print_exc()
