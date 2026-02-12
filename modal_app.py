"""
Modal App Entry Point

This is the main file that deploys to Modal.
It defines the Docker image, imports Modal functions, and registers classes.
"""
import modal

# Define the Docker image Modal will use for execution
# This image includes all dependencies (torch, gsplat, numpy, opencv, etc.)
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.0.0",
        "gsplat>=0.1.0",
        "numpy",
        "opencv-python",
        "boto3",
        "plyfile",
    )
)

# Import the Modal class after image is defined
from modal_functions.live_trainer import LiveSplatTrainer

# Create the Modal app
app = modal.App("so100-live-splat")


# Register the GPU class with Modal
@app.cls(
    image=image,
    gpu="A100",  # Using A100 GPU for training
    keep_warm=1,  # Keep 1 instance warm between calls
    concurrency_limit=1,  # Only 1 concurrent request
)
class SplatTrainerGPU:
    """GPU-accelerated Gaussian Splat trainer"""
    
    def __init__(self):
        """Initialize on Modal GPU"""
        self.trainer = LiveSplatTrainer()
    
    @modal.method()
    def add_frame(self, image_bytes: bytes, joint_positions: list) -> dict:
        """Process a frame from the robot"""
        return self.trainer.add_frame(image_bytes, joint_positions)
    
    @modal.method()
    def export_splat(self) -> bytes:
        """Export the trained splat as PLY"""
        return self.trainer.export_splat()
    
    @modal.method()
    def get_status(self) -> dict:
        """Get training status"""
        return self.trainer.get_status()
    
    @modal.method()
    def reset_model(self) -> dict:
        """Reset the model"""
        return self.trainer.reset_model()


# Optional: Local test function (runs on CPU)
@app.local_entrypoint()
def test_local():
    """Quick local test - can be run with: modal run modal_app.py"""
    print("Modal app loaded successfully!")
    print("To deploy: modal deploy modal_app.py")
    print("To run: modal run modal_app.py")
