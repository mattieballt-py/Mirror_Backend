"""
Modal App Entry Point for Gaussian Splat Training API

Uses Modal 1.x API with web_endpoint for HTTP routes
"""
import modal
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
import os

# Define the Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "gsplat>=0.1.0",
        "numpy",
        "opencv-python-headless",
        "boto3",
        "plyfile",
        "fastapi",
        "uvicorn",
        "python-multipart",
    )
    .add_local_python_source("modal_functions", "src")
)

# Create the Modal app
app = modal.App("so100-live-splat", image=image)

# Global state for the trainer
trainer = None
current_joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# GPU-based trainer class
@app.cls(image=image, gpu="A100", timeout=3600)
class SplatTrainerContainer:
    """GPU container for Gaussian Splat training"""
    
    @modal.enter()
    def setup(self):
        """Initialize trainer when container starts"""
        # Import here to avoid module-level boto3 issues
        from modal_functions.live_trainer import LiveSplatTrainer
        self.trainer = LiveSplatTrainer()
        self.joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    @modal.method()
    def add_frame(self, image_bytes: bytes, joint_positions: list) -> dict:
        """Add frame for training"""
        result = self.trainer.add_frame(image_bytes, joint_positions)
        print(f"Frame processed: {result['frame_count']}")
        return result
    
    @modal.method()
    def get_ply(self) -> bytes:
        """Get current PLY"""
        return self.trainer.export_splat()
    
    @modal.method()
    def get_status(self) -> dict:
        """Get training status"""
        return self.trainer.get_status()
    
    @modal.method()
    def export_final(self) -> dict:
        """Finalize and export"""
        return self.trainer.export_final()
    
    @modal.method()
    def update_joints(self, joint_positions: list) -> dict:
        """Update joint angles"""
        self.joint_angles = joint_positions
        return {"success": True, "joint_angles": self.joint_angles}
    
    @modal.method()
    def reset(self) -> dict:
        """Reset trainer"""
        from modal_functions.live_trainer import LiveSplatTrainer
        self.trainer = LiveSplatTrainer()
        self.joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return {"success": True}


# Create a single trainer instance
trainer_instance = SplatTrainerContainer()


# ===== WEB ENDPOINTS (lightweight, no GPU) =====

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def upload_frame(request: Request):
    """Receive frame from frontend and send to GPU trainer"""
    global current_joint_angles
    
    try:
        form = await request.form()
        file = form.get("file")
        
        if not file:
            return JSONResponse({"success": False, "error": "No file"}, status_code=400)
        
        image_bytes = await file.read()
        
        # Call GPU trainer
        result = trainer_instance.add_frame.remote(image_bytes, current_joint_angles)
        
        print(f"✓ Frame {result['frame_count']} processed")
        
        return JSONResponse({
            "success": result.get("success", True),
            "frame_count": result.get("frame_count", 0),
            "chunks_uploaded": result.get("chunks_uploaded", 0),
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def update_joint_angles(request: Request):
    """Update joint angles from LeRobot"""
    global current_joint_angles
    
    try:
        data = await request.json()
        joint_angles = data.get("joint_angles", [])
        
        if len(joint_angles) != 6:
            return JSONResponse(
                {"success": False, "error": f"Expected 6 joints, got {len(joint_angles)}"},
                status_code=400
            )
        
        current_joint_angles = joint_angles
        trainer_instance.update_joints.remote(joint_angles)
        
        print(f"✓ Joints updated: {current_joint_angles}")
        
        return JSONResponse({"success": True, "joint_angles": current_joint_angles})
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
async def get_ply(request: Request):
    """Stream current PLY file"""
    try:
        ply_bytes = trainer_instance.get_ply.remote()
        
        return StreamingResponse(
            iter([ply_bytes]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=splat.ply"}
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
async def status(request: Request):
    """Get training status"""
    try:
        st = trainer_instance.get_status.remote()
        return JSONResponse(st)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def stop_training(request: Request):
    """Finalize training"""
    try:
        result = trainer_instance.export_final.remote()
        print(f"✓ Training finalized")
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def reset(request: Request):
    """Reset trainer"""
    try:
        result = trainer_instance.reset.remote()
        print(f"✓ Trainer reset")
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
async def health(request: Request):
    """Health check"""
    return JSONResponse({"status": "healthy"})







