import torch
import numpy as np

# --- File paths ---
rgb_path = "dataset/RGB-part1/a1_s1_t1_color.avi"
depth_path = "dataset/Depth/Depth/a1_s1_t1_depth.mat"
imu_path = "dataset/Inertial/a1_s1_t1_inertial.mat"

# --- Import your pipeline modules ---
from preprocessing.video_dataloading import load_rgb_video, load_depth_video
from preprocessing.video_preprocessing import preprocess_rgb_frames, preprocess_depth_frames
from preprocessing.dataloading import load_data             # For IMU
from preprocessing.preprocessing import preprocess_imu_data
from preprocessing.cwt_transform import (
    kaiser_lowpass_filter,
    z_score_normalize,
    to_time_frequency_image
)

# --- RGB ---
rgb_frames = load_rgb_video(rgb_path)
rgb_processed = preprocess_rgb_frames(rgb_frames, sample_count=16)
rgb_avg = np.mean(rgb_processed, axis=0)                  # (224, 224, 3)
rgb_tensor = np.transpose(rgb_avg, (2, 0, 1))             # (3, 224, 224)
x_rgb = torch.tensor(rgb_tensor, dtype=torch.float32).unsqueeze(0)  # (1, 3, 224, 224)
torch.save(x_rgb, "x_rgb.pt")

# --- Depth ---
depth_frames = load_depth_video(depth_path, key='d_depth')
depth_processed = preprocess_depth_frames(depth_frames, sample_count=16)
depth_avg = np.mean(depth_processed, axis=0)              # (224, 224, 1)
depth_tensor = np.transpose(depth_avg, (2, 0, 1))         # (1, 224, 224)
x_depth = torch.tensor(depth_tensor, dtype=torch.float32).unsqueeze(0) # (1, 1, 224, 224)
torch.save(x_depth, "x_depth.pt")

# --- IMU ---
data = load_data()                                # Provide the path for the IMU file
cleaned_data = preprocess_imu_data(data)
filtered_data = kaiser_lowpass_filter(cleaned_data)
normalized_data = z_score_normalize(filtered_data)
cwt_image = to_time_frequency_image(
    normalized_data, selected_channels=[0,1,2], H=64, W=64
)
# Upsample to (3, 224, 224)
from skimage.transform import resize
upsampled_cwt = np.stack([
    resize(cwt_image[i], (224, 224), mode='reflect', anti_aliasing=True)
    for i in range(cwt_image.shape[0])
])
x_imu = torch.tensor(upsampled_cwt, dtype=torch.float32).unsqueeze(0) # (1, 3, 224, 224)
torch.save(x_imu, "x_imu.pt")

# --- Extract and save label ---
def extract_label_from_filename(filename):
    import re
    match = re.search(r'a(\d+)', filename)
    if match:
        action_num = int(match.group(1))
        return action_num - 1  # zero-based for dataset with action labels [0, 26]
    else:
        raise ValueError("Could not extract label from filename")

label_value = extract_label_from_filename(rgb_path)  # Use RGB file for action index
label = torch.tensor([label_value], dtype=torch.long)
torch.save(label, "label.pt")
