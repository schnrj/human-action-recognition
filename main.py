# File: main.py

import matplotlib.pyplot as plt
import numpy as np

from preprocessing.dataloading import load_data
from preprocessing.preprocessing import preprocess_imu_data
from preprocessing.feature_engineering import (
    compute_statistical_features,
    compute_smv,
    compute_frequency_features
)
from preprocessing.cwt_transform import (
    kaiser_lowpass_filter,
    z_score_normalize,
    to_time_frequency_image,
    to_gaf_image
)

# Step 1: Load IMU Data
data = load_data()
print("Original shape:", data.shape)

# Step 2: Preprocess IMU Data
cleaned_data = preprocess_imu_data(data)
print("Cleaned shape:", cleaned_data.shape)

# Step 3: Plot Raw and Preprocessed Data
channel_names = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

plt.figure(figsize=(10, 4))
for i in range(data.shape[1]):
    plt.plot(data[:, i], label=channel_names[i])
plt.title("Raw Inertial Data")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
for i in range(cleaned_data.shape[1]):
    plt.plot(cleaned_data[:, i], label=channel_names[i])
plt.title("Preprocessed Inertial Data")
plt.legend()
plt.tight_layout()
plt.show()

# Step 4: Feature Engineering
stat_features = compute_statistical_features(cleaned_data)
smv = compute_smv(cleaned_data)
freq_features = compute_frequency_features(cleaned_data)

print("Statistical features (36):", stat_features.shape)
print("SMV shape (180,):", smv.shape)
print("Frequency features (12):", freq_features.shape)

# Step 5: Combine Feature Vector
final_features = stat_features.tolist() + freq_features.tolist()
print("Final feature vector length:", len(final_features))

# Step 6: Denoising + Normalization for Image Generation
filtered_data = kaiser_lowpass_filter(data)
normalized_data = z_score_normalize(filtered_data)

# Step 7: Convert to Time-Frequency Image (CWT)
cwt_image = to_time_frequency_image(
    normalized_data, selected_channels=[0, 1, 2], H=64, W=64
)  # Shape: (3, 64, 64)
print("CWT image shape (should be 3, 64, 64):", cwt_image.shape)

# Step 8: Display CWT Image
plt.figure(figsize=(8, 3))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(cwt_image[i], cmap='viridis', aspect='auto')
    plt.title(f'CWT Channel {i+1}')
    plt.axis('off')
plt.suptitle('Time-Frequency Images from CWT (Accelerometer)')
plt.tight_layout()
plt.show()

# Step 9: Convert to Time-Frequency Image (GAF)
gaf_image = to_gaf_image(
    normalized_data, selected_channels=[0, 1, 2], H=64, W=64
)  # Shape: (3, 64, 64)
print("GAF image shape (should be 3, 64, 64):", gaf_image.shape)

# Step 10: Display GAF Image
plt.figure(figsize=(8, 3))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(gaf_image[i], cmap='viridis', aspect='auto')
    plt.title(f'GAF Channel {i+1}')
    plt.axis('off')
plt.suptitle('Time-Frequency Images from GAF (Accelerometer)')
plt.tight_layout()
plt.show()

# Step 11: Prepare IMU_CWT image for model input
# Use all three channels if you want (3, 64, 64), or average if you want (1, 64, 64)
imu_cwt = cwt_image  # Shape: (3, 64, 64)
print("IMU CWT tensor shape (for model, before upsampling):", imu_cwt.shape)

# Step 12: Optional Upsampling to (3, 224, 224) for model compatibility
from skimage.transform import resize

upsampled_cwt = np.stack([
    resize(imu_cwt[i], (224, 224), mode='reflect', anti_aliasing=True)
    for i in range(imu_cwt.shape[0])
])
print("Upsampled IMU CWT shape (for model):", upsampled_cwt.shape)  # Should be (3, 224, 224)
