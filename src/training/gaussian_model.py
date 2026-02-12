"""
Wrapper around gsplat for Gaussian Splat model management.
"""
import torch
import torch.nn as nn


class GaussianModel(nn.Module):
    """Simple wrapper for Gaussian Splat parameters"""
    
    def __init__(self, num_gaussians: int = 100000):
        super().__init__()
        
        self.num_gaussians = num_gaussians
        
        # Register parameters
        self.register_parameter(
            'means',
            nn.Parameter(
                torch.randn(num_gaussians, 3) * 0.1,
                requires_grad=True
            )
        )
        
        self.register_parameter(
            'quats',
            nn.Parameter(
                torch.randn(num_gaussians, 4),
                requires_grad=True
            )
        )
        
        self.register_parameter(
            'scales',
            nn.Parameter(
                torch.randn(num_gaussians, 3) - 2,
                requires_grad=True
            )
        )
        
        self.register_parameter(
            'opacities',
            nn.Parameter(
                torch.ones(num_gaussians, 1) * 0.5,
                requires_grad=True
            )
        )
        
        self.register_parameter(
            'colors',
            nn.Parameter(
                torch.rand(num_gaussians, 3),
                requires_grad=True
            )
        )
    
    def forward(self):
        """Return all Gaussian parameters"""
        return {
            'means': self.means,
            'quats': self.quats,
            'scales': self.scales,
            'opacities': self.opacities,
            'colors': self.colors,
        }
    
    def get_all_params(self):
        """Return parameters as dict"""
        return {
            'means': self.means.data,
            'quats': self.quats.data,
            'scales': self.scales.data,
            'opacities': self.opacities.data,
            'colors': self.colors.data,
        }
