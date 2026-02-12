# Deploying to Modal with R2 Bucket Integration

## Environment Variables
Set these before deploying to Modal:

```bash
export AWS_ACCESS_KEY_ID="3fc30c66cf3b5e0f99bc48d13e67ea14"
export AWS_SECRET_ACCESS_KEY="e4e8ecced9e81e6464b6b811a0fc5d713d8c3023aa0b0799d4ec7b19f89a35d3"
export R2_ENDPOINT_URL="https://225d6b271578fbbf68da8d2044e5cff4.r2.cloudflarestorage.com"
export R2_BUCKET_NAME="gsplat-scenes"
export R2_PUBLIC_URL="https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev"
```

## Deployment Steps

1. **Set environment variables** (see above)
2. **Deploy to Modal:**
   ```bash
   modal deploy modal_app.py
   ```

3. **Create Modal secret** (so Modal containers have R2 credentials):
   ```bash
   modal secret create r2-credentials \
     AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
     AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
     R2_ENDPOINT_URL=$R2_ENDPOINT_URL \
     R2_BUCKET_NAME=$R2_BUCKET_NAME \
     R2_PUBLIC_URL=$R2_PUBLIC_URL
   ```

4. **Update modal_app.py** to reference the secret:
   ```python
   @app.cls(
       image=image,
       gpu="A100",
       secrets=[modal.Secret.from_name("r2-credentials")],
       min_containers=1,
       max_containers=1,
   )
   ```

## How It Works

### Frame Processing
1. React frontend sends video frame + joint angles to `/add_frame`
2. Modal trains Gaussian Splat on the frame
3. Every 50 frames, automatic PLY chunk export + R2 upload
4. `status.json` updated with progress and chunk list

### React Frontend Integration
Poll `https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev/status.json` every 3-5 seconds:

```javascript
const fetchStatus = async () => {
  const response = await fetch('https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev/status.json');
  const status = await response.json();
  console.log(status);
  // {
  //   "progress": 0.4,
  //   "chunks": ["splat_000.ply", "splat_001.ply"],
  //   "complete": false,
  //   "frame_count": 200,
  //   "gaussian_count": 100000
  // }
};
```

### Loading Chunks in Three.js
Each chunk file is a PLY at:
```
https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev/splat_000.ply
https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev/splat_001.ply
```

## API Endpoints (Modal Methods)

### `add_frame(image_bytes, joint_positions)`
Send a frame for training.
- `image_bytes`: PNG/JPEG encoded image bytes
- `joint_positions`: List of 6 joint angles in radians
- **Response:** Status dict with frame count and chunks uploaded

### `get_status()`
Get current training status.
- **Response:** Dict with progress, chunks, frame count, etc.

### `export_final()`
Finalize training and mark complete in status.json.
- **Response:** Dict with final chunk list and confirmation

### `reset_model()`
Reset the Gaussian Splat model and start fresh.
- **Response:** Confirmation dict
