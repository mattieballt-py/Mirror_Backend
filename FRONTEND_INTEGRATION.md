# FastAPI + Modal Deployment & Frontend Integration Guide

## Architecture Overview

### Modal HTTP Endpoint: Inside vs Separate

**WHAT WE'RE USING: Separate Web Server**

- **Separate** (what we have now):
  - FastAPI app runs as an independent Modal web endpoint (`@app_instance.asgi_app()`)
  - Handles all HTTP requests (frame uploads, status checks, joint angle updates)
  - Instantiates `LiveSplatTrainer` globally on startup
  - **Pros**: Cleaner separation, easier to scale, better error isolation
  - **Cons**: Slightly more code
  
- **Inside** (alternative):
  - FastAPI methods would be Modal `@modal.method()` on the GPU class
  - Requires client to call via Modal SDK (not HTTP)
  - **Pros**: Simpler if not using HTTP
  - **Cons**: Can't call from browser, requires Modal authentication

**We chose Separate** because you need HTTP endpoints for:
- Browser fetch requests from React frontend
- POST requests from LeRobot process for joint angles
- Direct HTTP streaming of PLY files

---

## Deployment Steps

### 1. Set Environment Variables

```bash
cd /Users/hedgehog/Desktop/MechEng_Degree/Coding_Things/Mirror_Backend

export AWS_ACCESS_KEY_ID="3fc30c66cf3b5e0f99bc48d13e67ea14"
export AWS_SECRET_ACCESS_KEY="e4e8ecced9e81e6464b6b811a0fc5d713d8c3023aa0b0799d4ec7b19f89a35d3"
export R2_ENDPOINT_URL="https://225d6b271578fbbf68da8d2044e5cff4.r2.cloudflarestorage.com"
export R2_BUCKET_NAME="gsplat-scenes"
export R2_PUBLIC_URL="https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev"
```

### 2. Create Modal Secret (Optional but Recommended)

Instead of exporting during deployment, create a Modal secret:

```bash
modal secret create r2-credentials \
  AWS_ACCESS_KEY_ID="3fc30c66cf3b5e0f99bc48d13e67ea14" \
  AWS_SECRET_ACCESS_KEY="e4e8ecced9e81e6464b6b811a0fc5d713d8c3023aa0b0799d4ec7b19f89a35d3" \
  R2_ENDPOINT_URL="https://225d6b271578fbbf68da8d2044e5cff4.r2.cloudflarestorage.com" \
  R2_BUCKET_NAME="gsplat-scenes" \
  R2_PUBLIC_URL="https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev"
```

Then update `modal_app.py` to reference the secret (if using this approach):
```python
image = (
    modal.Image.debian_slim()
    ...
)

@app_instance.asgi_app(secrets=[modal.Secret.from_name("r2-credentials")])
def fastapi_app():
    return app
```

### 3. Deploy to Modal

```bash
modal deploy modal_app.py
```

**Output will show:**
```
✓ Created Async App 'so100-live-splat'
https://YOUR_USERNAME--so100-live-splat.modal.run
```

**Copy that URL** — it's your API endpoint!

---

## Frontend Integration

### In Your React App

1. **Create `.env.local`:**
   ```
   VITE_API_URL=https://YOUR_USERNAME--so100-live-splat.modal.run
   ```

2. **In your component:**
   ```javascript
   const API_URL = import.meta.env.VITE_API_URL;
   
   const uploadFrame = async (canvas) => {
     canvas.toBlob(async (blob) => {
       const formData = new FormData();
       formData.append('file', blob, 'frame.jpg');
       
       const response = await fetch(`${API_URL}/upload-frame`, {
         method: 'POST',
         body: formData,
       });
       
       const result = await response.json();
       console.log(result); // {success: true, frame_count: 5, chunks_uploaded: 1}
     }, 'image/jpeg', 0.8);
   };
   
   // Get current PLY (real-time, incrementally updated)
   const downloadPLY = async () => {
     const response = await fetch(`${API_URL}/get-ply`);
     const blob = await response.blob();
     // Feed to Three.js viewer for real-time rendering
   };
   
   // Poll status every 3 seconds
   const pollStatus = setInterval(async () => {
     const response = await fetch(`${API_URL}/status`);
     const status = await response.json();
     console.log(status);
     // {frame_count: 50, gaussian_count: 100000, chunks_uploaded: 1, progress: 0.5}
   }, 3000);
   
   // Stop training
   const stopCapture = async () => {
     const response = await fetch(`${API_URL}/stop-training`, {
       method: 'POST',
     });
     const result = await response.json();
     console.log(result); // {success: true, chunks: [...], complete: true}
     clearInterval(pollStatus);
   };
   ```

---

## Backend API Endpoints

All endpoints return JSON (and PLY data for `/get-ply`):

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/upload-frame` | POST | Form file (image) | `{success, frame_count, chunks_uploaded}` |
| `/update-joint-angles` | POST | `{"joint_angles": [0.0, ...]}` | `{success, joint_angles}` |
| `/status` | GET | — | Frame count, gaussian count, progress, chunks |
| `/get-ply` | GET | — | Binary PLY file (real-time) |
| `/stop-training` | POST | — | `{success, chunks, complete}` |
| `/reset` | POST | — | `{success}` |
| `/health` | GET | — | `{status, trainer_ready}` |

---

## LeRobot Integration

### Prompt for Your LeRobot Repo

Give this to the LeRobot process to send joint angles to the backend:

```
You're running LeRobot on a SO-101 arm. After each observation/inference step,
send the current joint positions to the Gaussian Splat backend.

POST to: http://YOUR_BACKEND_URL/update-joint-angles
(Replace YOUR_BACKEND_URL with your Modal endpoint)

Payload format:
{
  "joint_angles": [joint1_rad, joint2_rad, joint3_rad, joint4_rad, joint5_rad, joint6_rad]
}

Example with curl:
curl -X POST http://YOUR_BACKEND_URL/update-joint-angles \
  -H "Content-Type: application/json" \
  -d '{"joint_angles": [0.5, 1.2, -0.3, 0.0, 0.1, 0.5]}'

Python example:
import requests

def send_joint_angles(joint_positions):
    response = requests.post(
        'http://YOUR_BACKEND_URL/update-joint-angles',
        json={'joint_angles': joint_positions.tolist()}  # Convert numpy array to list
    )
    return response.json()

# Call this after each policy inference step
joint_state = policy.forward(...)  # Your LeRobot inference
send_joint_angles(joint_state)
```

---

## Real-Time PLY Streaming Flow

1. **Frontend captures frame** → POST `/upload-frame`
2. **Backend trains** → Adds frame, updates in-memory PLY
3. **Frontend polls** `/get-ply` every 0.5–1 second
4. **Three.js renders** the streamed PLY in real-time
5. Every 50 frames: Backend exports **chunk** → uploads to R2
6. Frontend can also watch `status.json` on R2 for chunk completion
7. **User clicks Stop** → POST `/stop-training` → finalizes all chunks
8. **Training complete** → All chunks persist on R2

---

## Debugging

Check Modal logs:
```bash
modal run modal_app.py::test_local
modal logs your-workspace/so100-live-splat
```

Check endpoint health:
```bash
curl https://YOUR_USERNAME--so100-live-splat.modal.run/health
```

---

## Production Notes

- **CORS**: Currently `allow_origins=["*"]` for development. Set to your React domain in production.
- **Joint angles**: Default is `[0, 0, 0, 0, 0, 0]` if LeRobot process doesn't send updates.
- **Streaming PLY**: Grows with each frame; consider file size & network for large training sessions.
- **R2 chunks**: Every 50 frames, exports a separate `.ply` file for long-term storage.
