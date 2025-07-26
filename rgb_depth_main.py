import matplotlib.pyplot as plt
import numpy as np
import torch
from preprocessing.video_dataloading import load_rgb_video, load_depth_video
from preprocessing.video_preprocessing import preprocess_rgb_frames, preprocess_depth_frames

# --- File paths ---
rgb_path = "dataset/RGB-part1/a1_s1_t1_color.avi"
depth_path = "dataset/Depth/Depth/a1_s1_t1_depth.mat"

# --- Load & preprocess video frames ---
rgb_frames = load_rgb_video(rgb_path)
depth_frames = load_depth_video(depth_path, key='d_depth')

sample_count = 16
rgb_processed = preprocess_rgb_frames(rgb_frames, sample_count=sample_count)
depth_processed = preprocess_depth_frames(depth_frames, sample_count=sample_count)

# --- Average across time, channel-first & model-ready conversion ---
rgb_avg = np.mean(rgb_processed, axis=0)          # (224, 224, 3)
depth_avg = np.mean(depth_processed, axis=0)      # (224, 224, 1)

rgb_tensor = np.transpose(rgb_avg, (2, 0, 1))     # (3, 224, 224)
depth_tensor = np.transpose(depth_avg, (2, 0, 1)) # (1, 224, 224)

x_rgb = torch.tensor(rgb_tensor, dtype=torch.float32).unsqueeze(0)     # (1, 3, 224, 224)
x_depth = torch.tensor(depth_tensor, dtype=torch.float32).unsqueeze(0) # (1, 1, 224, 224)

print("Final x_rgb tensor shape for model:", x_rgb.shape)
print("Final x_depth tensor shape for model:", x_depth.shape)

torch.save(x_rgb, 'x_rgb.pt')
torch.save(x_depth, 'x_depth.pt')
