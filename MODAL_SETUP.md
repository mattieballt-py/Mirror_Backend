# Modal Setup Guide

To get your FastAPI backend running on Modal, follow these steps:

## 1. Install Modal CLI

```bash
pip install modal
```

## 2. Authenticate with Modal

```bash
modal token new
```

This will:
- Open your browser to create a Modal account (or log in)
- Display an API token in the browser
- Paste it into the terminal when prompted
- Saves credentials locally to `~/.modal.toml`

## 3. Verify Installation

```bash
modal version
modal auth verify
```

## 4. Set Environment Variables Before Deployment

```bash
cd /Users/hedgehog/Desktop/MechEng_Degree/Coding_Things/Mirror_Backend

export AWS_ACCESS_KEY_ID="3fc30c66cf3b5e0f99bc48d13e67ea14"
export AWS_SECRET_ACCESS_KEY="e4e8ecced9e81e6464b6b811a0fc5d713d8c3023aa0b0799d4ec7b19f89a35d3"
export R2_ENDPOINT_URL="https://225d6b271578fbbf68da8d2044e5cff4.r2.cloudflarestorage.com"
export R2_BUCKET_NAME="gsplat-scenes"
export R2_PUBLIC_URL="https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev"
```

## 5. Deploy to Modal

```bash
modal deploy modal_app.py
```

**Output will show:**
```
✓ Created web endpoint 'so100-live-splat'
https://YOUR_USERNAME--so100-live-splat.modal.run
```

Copy that URL — it's your backend API endpoint!

## 6. Test the Deployment

```bash
# Check health
curl https://YOUR_USERNAME--so100-live-splat.modal.run/health

# Should return:
# {"status":"healthy","trainer_ready":true}
```

## 7. View Logs (Debugging)

```bash
# Real-time logs
modal logs YOUR_USERNAME/so100-live-splat

# Or run locally to test
modal run modal_app.py::test_local
```

## 8. In Your React Frontend

Create `.env.local`:
```
VITE_API_URL=https://YOUR_USERNAME--so100-live-splat.modal.run
```

Then use it:
```javascript
const API_URL = import.meta.env.VITE_API_URL;

// POST frame
fetch(`${API_URL}/upload-frame`, {
  method: 'POST',
  body: formData
});

// Poll status
fetch(`${API_URL}/status`);

// Get PLY
fetch(`${API_URL}/get-ply`);
```

## Troubleshooting

**Issue**: `modal: command not found`
- Solution: Make sure you installed it: `pip install modal`

**Issue**: Auth fails
- Solution: Run `modal token new` again

**Issue**: Deployment fails with import errors
- Solution: Check `requirements.txt` has all dependencies (torch, gsplat, fastapi, uvicorn, etc.)

**Issue**: Endpoint returns 502 Bad Gateway
- Solution: Check logs with `modal logs YOUR_USERNAME/so100-live-splat`

**Issue**: LeRobot can't reach endpoint
- Solution: Use full HTTPS URL: `https://YOUR_USERNAME--so100-live-splat.modal.run/update-joint-angles`

---

## Optional: Use Modal Secrets for Credentials

Instead of exporting environment variables each time, create a Modal secret:

```bash
modal secret create r2-credentials \
  AWS_ACCESS_KEY_ID="3fc30c66cf3b5e0f99bc48d13e67ea14" \
  AWS_SECRET_ACCESS_KEY="e4e8ecced9e81e6464b6b811a0fc5d713d8c3023aa0b0799d4ec7b19f89a35d3" \
  R2_ENDPOINT_URL="https://225d6b271578fbbf68da8d2044e5cff4.r2.cloudflarestorage.com" \
  R2_BUCKET_NAME="gsplat-scenes" \
  R2_PUBLIC_URL="https://pub-8483e6a1db1342bda70ce67e0a39a8cc.r2.dev"
```

Then update `modal_app.py`:
```python
@app_instance.asgi_app(secrets=[modal.Secret.from_name("r2-credentials")])
def fastapi_app():
    return app
```

Then deploy without exporting:
```bash
modal deploy modal_app.py
```

---

## That's it!

Once you see your Modal URL, your backend is live and ready for:
- React frontend to POST frames
- LeRobot to POST joint angles
- Real-time PLY streaming to the browser
