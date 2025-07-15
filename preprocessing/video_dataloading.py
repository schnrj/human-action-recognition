# File: preprocessing/video_dataloading.py

import cv2
import os
import numpy as np
import scipy.io

def load_video_frames(video_path, max_frames=None):
    """
    Loads frames from a video file using OpenCV.
    Args:
        video_path: Path to the .avi or .mp4 video file
        max_frames: Optional. Limit number of frames loaded.
    Returns:
        frames: list of np.array frames (H, W, 3)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return np.array(frames)  # shape: (num_frames, H, W, 3)

def load_rgb_video(rgb_path, max_frames=None):
    return load_video_frames(rgb_path, max_frames)

def load_depth_video(depth_mat_path, key='d_depth', max_frames=None):
    """
    Load depth frames from a .mat file.
    Assumes the key inside .mat file is 'd_depth' or similar.
    """
    if not os.path.exists(depth_mat_path):
        raise FileNotFoundError(f"Depth file not found: {depth_mat_path}")

    mat = scipy.io.loadmat(depth_mat_path)
    print("MAT keys:", mat.keys())

    if key not in mat:
        raise KeyError(f"Key '{key}' not found in MAT file.")

    frames = mat[key]  # shape: (T, H, W) or (H, W, T)
    if frames.ndim == 3 and frames.shape[0] < frames.shape[-1]:
        frames = np.transpose(frames, (2, 0, 1))  # ensure shape (T, H, W)

    if max_frames:
        frames = frames[:max_frames]

    frames = np.expand_dims(frames, axis=-1)  # shape: (T, H, W, 1)
    return frames.astype(np.uint8)
