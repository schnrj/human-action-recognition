# # File: rgb_depth_main.py

# import matplotlib.pyplot as plt
# from preprocessing.video_dataloading import load_rgb_video, load_depth_video
# from preprocessing.video_preprocessing import preprocess_rgb_frames, preprocess_depth_frames
# import numpy as np

# # --- File paths ---
# rgb_path = "dataset/RGB-part1/a1_s1_t1_color.avi"
# depth_path = "dataset/Depth/Depth/a1_s1_t1_depth.mat"

# # --- Load video frames ---
# print("Loading videos...")
# rgb_frames = load_rgb_video(rgb_path)
# depth_frames = load_depth_video(depth_path, key='d_depth')

# print("Original RGB video shape:", rgb_frames.shape)   # (T, H, W, 3)
# print("Original Depth shape:", depth_frames.shape)     # (T, H, W, 1)

# # --- Preprocess ---
# sample_count = 16
# rgb_processed = preprocess_rgb_frames(rgb_frames, sample_count=sample_count)
# depth_processed = preprocess_depth_frames(depth_frames, sample_count=sample_count)

# print("Processed RGB shape:", rgb_processed.shape)     # (16, 224, 224, 3)
# print("Processed Depth shape:", depth_processed.shape) # (16, 224, 224, 1)

# # --- Show Sample Frames ---
# fig, axs = plt.subplots(2, sample_count, figsize=(sample_count * 2, 4))
# for i in range(sample_count):
#     axs[0, i].imshow((rgb_processed[i] * 0.229 + 0.485).clip(0,1))  # approx. unnormalize for display
#     axs[0, i].axis('off')
#     axs[0, i].set_title(f"RGB {i+1}")

#     axs[1, i].imshow(depth_processed[i, :, :, 0], cmap='gray')
#     axs[1, i].axis('off')
#     axs[1, i].set_title(f"Depth {i+1}")

# plt.suptitle("Sample Preprocessed RGB and Depth Frames")
# plt.tight_layout()
# plt.show()
# File: rgb_depth_main.py

import matplotlib.pyplot as plt
from preprocessing.video_dataloading import load_rgb_video, load_depth_video
from preprocessing.video_preprocessing import preprocess_rgb_frames, preprocess_depth_frames
import numpy as np

# --- File paths ---
rgb_path = "dataset/RGB-part1/a1_s1_t1_color.avi"
depth_path = "dataset/Depth/Depth/a1_s1_t1_depth.mat"

# --- Load video frames ---
print("Loading videos...")
rgb_frames = load_rgb_video(rgb_path)
depth_frames = load_depth_video(depth_path, key='d_depth')

print("Original RGB video shape:", rgb_frames.shape)   # (T, H, W, 3)
print("Original Depth shape:", depth_frames.shape)     # (T, H, W, 1)

# --- Preprocess ---
sample_count = 16
rgb_processed = preprocess_rgb_frames(rgb_frames, sample_count=sample_count)
depth_processed = preprocess_depth_frames(depth_frames, sample_count=sample_count)

print("Processed RGB shape:", rgb_processed.shape)     # (16, 224, 224, 3)
print("Processed Depth shape:", depth_processed.shape) # (16, 224, 224, 1)

# --- Stack Time to Image Representations ---
rgb_avg = np.mean(rgb_processed, axis=0)    # shape: (224, 224, 3)
depth_avg = np.mean(depth_processed, axis=0)  # shape: (224, 224, 1)

# Transpose to channel-first format
rgb_tensor = np.transpose(rgb_avg, (2, 0, 1))     # (3, H, W)
depth_tensor = np.transpose(depth_avg, (2, 0, 1)) # (1, H, W)

print("RGB stack tensor shape:", rgb_tensor.shape)
print("Depth stack tensor shape:", depth_tensor.shape)

# --- Show Averaged Frames ---
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow((rgb_avg * 0.229 + 0.485).clip(0, 1))  # approx unnormalize
plt.title("Averaged RGB")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth_avg[:, :, 0], cmap='gray')
plt.title("Averaged Depth")
plt.axis('off')

plt.suptitle("Collapsed Time-Stacked Representations")
plt.tight_layout()
plt.show()
