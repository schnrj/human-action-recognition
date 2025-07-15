import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def preprocess_imu_data(data, target_length=180):
    """
    Preprocess IMU time-series data:
    - Handles NaNs via interpolation
    - Removes outliers (z-score > 3)
    - Applies Savitzky-Golay smoothing
    - Resamples to target_length
    - Z-score normalization
    Returns: (target_length, num_channels) numpy array
    """
    # Step 1: Handle NaNs
    if np.isnan(data).any():
        for i in range(data.shape[1]):
            nans = np.isnan(data[:, i])
            if nans.any():
                data[nans, i] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(~nans),
                    data[~nans, i]
                )

    # Step 2: Handle outliers (z-score > 3)
    z_scores = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
    outliers = z_scores > 3
    if np.any(outliers):
        for i in range(data.shape[1]):
            channel = data[:, i]
            out_idx = np.where(outliers[:, i])[0]
            for idx in out_idx:
                left = max(idx - 1, 0)
                right = min(idx + 1, len(channel) - 1)
                channel[idx] = (channel[left] + channel[right]) / 2
            data[:, i] = channel

    # Step 3: Signal smoothing
    data = savgol_filter(data, window_length=9, polyorder=2, axis=0)

    # Step 4: Resampling
    x_old = np.linspace(0, 1, data.shape[0])
    x_new = np.linspace(0, 1, target_length)
    data_resampled = np.zeros((target_length, data.shape[1]))
    for i in range(data.shape[1]):
        f = interp1d(x_old, data[:, i], kind='linear')
        data_resampled[:, i] = f(x_new)

    # Step 5: Z-score Normalization
    data_resampled = (data_resampled - data_resampled.mean(axis=0)) / data_resampled.std(axis=0)

    return data_resampled

def upsample_imu_cwt(imu_cwt, target_size=(224, 224)):
    """
    Upsample a 2D IMU CWT spectrogram to match the spatial size of RGB/Depth images.
    Args:
        imu_cwt: numpy array of shape (1, H, W) or (H, W)
        target_size: tuple, e.g., (224, 224)
    Returns:
        numpy array of shape (1, target_H, target_W)
    """
    import torch
    import torch.nn.functional as F

    # Ensure input shape is (1, H, W)
    if imu_cwt.ndim == 2:
        imu_cwt = imu_cwt[np.newaxis, ...]
    imu_cwt_tensor = torch.tensor(imu_cwt, dtype=torch.float32).unsqueeze(0)  # (1, 1, H, W)
    upsampled = F.interpolate(imu_cwt_tensor, size=target_size, mode='bilinear', align_corners=False)
    upsampled = upsampled.squeeze(0).numpy()  # (1, target_H, target_W)
    return upsampled

# Example usage:
if __name__ == "__main__":
    # Example IMU data loading
    # imu_data = np.load('path_to_imu.npy')  # shape: (N, 6)
    # imu_processed = preprocess_imu_data(imu_data, target_length=180)

    # Example IMU CWT (after CWT transform, shape: (1, 64, 64))
    # imu_cwt = np.load('imu_cwt.npy')
    # imu_cwt_upsampled = upsample_imu_cwt(imu_cwt, target_size=(224, 224))
    pass
