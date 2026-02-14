"""
FastAPI web server for Gaussian Splat training via HTTP.
This runs on Modal and handles frame uploads from the React frontend.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import json
import numpy as np

# Global trainer instance (initialized on startup)
trainer = None

app = FastAPI(title="Gaussian Splat Trainer API")

# Enable CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Current joint angles (updated via endpoint, used for frame processing)
current_joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@app.on_event("startup")
async def startup_event():
    """Initialize trainer on startup - import here to avoid issues"""
    global trainer
    from modal_functions.live_trainer import LiveSplatTrainer
    
    trainer = LiveSplatTrainer()
    print("✓ Trainer initialized on Modal GPU")


@app.post("/upload-frame")
async def upload_frame(file: UploadFile = File(...)):
    """
    Accept a frame from the frontend.
    
    Args:
        file: Image file (JPEG/PNG)
        
    Returns:
        JSON with status
    """
    global trainer, current_joint_angles
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Process frame with current joint angles
        result = trainer.add_frame(image_bytes, current_joint_angles)
        
        # Log to console for debugging
        print(f"Frame received: {result['frame_count']} frames, {result['gaussian_count']} gaussians")
        
        # Return minimal response
        return JSONResponse({
            "success": result.get("success", True),
            "frame_count": result.get("frame_count", 0),
            "chunks_uploaded": result.get("chunks_uploaded", 0),
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@app.post("/update-joint-angles")
async def update_joint_angles(data: dict):
    """
    Update the current joint angles.
    Called by the LeRobot process.
    
    Args:
        data: JSON with "joint_angles" key (list of 6 floats in radians)
        
    Returns:
        Confirmation
    """
    global current_joint_angles
    
    try:
        joint_angles = data.get("joint_angles", [])
        
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")
        
        current_joint_angles = joint_angles
        print(f"Joint angles updated: {current_joint_angles}")
        
        return JSONResponse({"success": True, "joint_angles": current_joint_angles})
        
    except Exception as e:
        print(f"Error updating joint angles: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=400
        )


@app.get("/status")
async def get_status():
    """
    Get current training status.
    
    Returns:
        JSON with progress, frame count, chunks, etc.
    """
    global trainer
    
    if trainer is None:
        return JSONResponse(
            {"error": "Trainer not initialized"},
            status_code=500
        )
    
    status = trainer.get_status()
    return JSONResponse(status)


@app.get("/get-ply")
async def get_ply():
    """
    Get current PLY file as bytes (in-memory, real-time updated).
    Frontend polls this to get the incrementally growing PLY.
    
    Returns:
        PLY file as binary stream
    """
    global trainer
    
    if trainer is None:
        raise HTTPException(status_code=500, detail="Trainer not initialized")
    
    try:
        ply_bytes = trainer.export_splat()
        
        return StreamingResponse(
            iter([ply_bytes]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=splat.ply"}
        )
        
    except Exception as e:
        print(f"Error exporting PLY: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop-training")
async def stop_training():
    """
    Finalize training and export final model.
    Called when user clicks "Stop Capture".
    
    Returns:
        Final status with all chunks
    """
    global trainer
    
    if trainer is None:
        return JSONResponse(
            {"error": "Trainer not initialized"},
            status_code=500
        )
    
    try:
        result = trainer.export_final()
        print(f"Training finalized: {result}")
        return JSONResponse(result)
        
    except Exception as e:
        print(f"Error finalizing training: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@app.post("/reset")
async def reset():
    """
    Reset the trainer and start fresh.
    
    Returns:
        Confirmation
    """
    global trainer
    
    try:
        trainer.reset_model()
        print("✓ Trainer reset")
        return JSONResponse({"success": True, "message": "Trainer reset"})
        
    except Exception as e:
        print(f"Error resetting trainer: {e}")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse({"status": "healthy", "trainer_ready": trainer is not None})
