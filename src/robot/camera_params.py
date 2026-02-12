"""
SO-100 Robot Camera Parameters
"""
import numpy as np

# SO-100 Camera Intrinsics
# Adjust these values based on your actual camera calibration
SO100_INTRINSICS = np.array([
    [800.0,   0.0, 320.0],  # fx, 0, cx
    [  0.0, 800.0, 240.0],  # 0, fy, cy
    [  0.0,   0.0,   1.0]   # 0, 0, 1
], dtype=np.float32)

# Image resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Camera mounting parameters
# Offset from end-effector to camera lens (in meters)
CAMERA_OFFSET = np.array([0.0, 0.0, 0.1], dtype=np.float32)  # 10cm forward

# Camera rotation relative to end-effector (in radians)
# Assumes camera is mounted looking forward
CAMERA_ROTATION = np.array([0.0, 0.0, 0.0], dtype=np.float32)
