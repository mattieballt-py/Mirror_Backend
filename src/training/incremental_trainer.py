"""
Incremental Gaussian Splatting trainer using gsplat library.
"""
import torch
import random
from typing import Tuple, List
import numpy as np

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
        
        # Refine every 10 frames
        if len(self.frames) % 10 == 0:
            self._refine(iterations=100)
    
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
            
            try:
                # Use gsplat's rasterization
                rendered = rasterization(
                    means=params['means'],
                    quats=params['quats'],
                    scales=params['scales'],
                    opacities=params['opacities'],
                    colors=params['colors'],
                    viewmat=pose,
                    K=self.K,
                    width=image.shape[1],
                    height=image.shape[0],
                )
                
                # Compute loss (MSE between rendered and ground truth)
                loss = ((rendered[..., :3] - image) ** 2).mean()
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            except Exception as e:
                print(f"Warning during rasterization: {e}")
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
        Export Gaussian Splat as PLY format bytes.
        
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
        
        # Detach from GPU
        means = params['means'].cpu().numpy()
        colors = (params['colors'].cpu().numpy() * 255).astype(np.uint8)
        opacities = params['opacities'].cpu().numpy()
        scales = params['scales'].cpu().numpy()
        quats = params['quats'].cpu().numpy()
        
        # Create vertex data
        vertex_data = np.array([
            tuple(list(means[i]) + list(colors[i]) + [opacities[i, 0]] +
                  list(scales[i]) + list(quats[i]))
            for i in range(len(means))
        ],
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('alpha', 'f4'),
                ('scale_x', 'f4'), ('scale_y', 'f4'), ('scale_z', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ]
        )
        
        # Create PLY element
        vertex_el = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_el])
        
        # Write to bytes
        import io
        output = io.BytesIO()
        ply_data.write(output)
        return output.getvalue()
    
    def get_gaussian_count(self) -> int:
        """Return current number of Gaussians"""
        return self.gaussians.num_gaussians
