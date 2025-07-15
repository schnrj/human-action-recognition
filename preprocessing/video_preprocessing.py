# File: preprocessing/video_preprocessing.py

import numpy as np
import cv2

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def resize_and_center_crop(frame, target_size=(224, 224)):
    h, w, _ = frame.shape
    scale = max(target_size[0] / h, target_size[1] / w)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

    center_h = resized.shape[0] // 2
    center_w = resized.shape[1] // 2
    half_th = target_size[0] // 2
    half_tw = target_size[1] // 2

    cropped = resized[
        center_h - half_th:center_h + half_th,
        center_w - half_tw:center_w + half_tw
    ]
    return cropped

def normalize_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    for c in range(3):
        frame[:, :, c] = (frame[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return frame

def preprocess_rgb_frames(frames, target_size=(224, 224), sample_count=16):
    """
    C. 3. RGB Modality
    3.1 Frame Sampling: Uniform sampling of N frames
    3.2 a) Resize and Center Crop
        b) Normalization (ImageNet stats)
        c) Stacking: return shape (N, H, W, 3)
    3.3 Final Format: B x 3 x H x W (can be transposed later)
    """
    total = len(frames)
    if total < sample_count:
        raise ValueError("Video has fewer frames than sample count.")

    indices = np.linspace(0, total - 1, sample_count).astype(int)
    sampled = [frames[i] for i in indices]

    processed = []
    for frame in sampled:
        resized = resize_and_center_crop(frame, target_size)
        normalized = normalize_frame(resized)
        processed.append(normalized)

    return np.array(processed)  # shape: (sample_count, H, W, 3)

def preprocess_depth_frames(frames, target_size=(224, 224), sample_count=16):
    """
    D. 4. Depth Modality
    4.1 a) Normalization (min-max scaling)
        b) Resizing
        c) Optional: SFI/Prewitt (not included here)
    4.2 Final Format: B x 1 x H x W (can be expanded and transposed)
    """
    total = len(frames)
    if total < sample_count:
        raise ValueError("Video has fewer frames than sample count.")

    indices = np.linspace(0, total - 1, sample_count).astype(int)
    sampled = [frames[i] for i in indices]

    processed = []
    for frame in sampled:
        resized = cv2.resize(frame.squeeze(), target_size)
        norm = cv2.normalize(resized, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        processed.append(norm)

    return np.expand_dims(np.array(processed), axis=-1)  # shape: (sample_count, H, W, 1)
