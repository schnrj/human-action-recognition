# File: preprocessing/cwt_transform.py

import numpy as np
import scipy.signal as signal
import pywt
from skimage.transform import resize

# --- 1. Denoising using Kaiser Window ---
def kaiser_lowpass_filter(data, beta=14, cutoff=0.3):
    numtaps = 51
    taps = signal.firwin(numtaps, cutoff=cutoff, window=('kaiser', beta))
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        if data.shape[0] <= 3 * numtaps:
            raise ValueError("Signal too short for filtering with current numtaps.")
        filtered[:, i] = signal.filtfilt(taps, [1.0], data[:, i])
    return filtered

# --- 2. Z-score Normalization ---
def z_score_normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std

# --- 3. Continuous Wavelet Transform ---
def cwt_transform(signal_1d, wavelet='morl', scales=np.arange(1, 64)):
    coef, _ = pywt.cwt(signal_1d, scales, wavelet)
    return np.abs(coef)  # Shape: (scales, time)

# --- 4. Time-Frequency Image Conversion (CWT) ---
def to_time_frequency_image(data, selected_channels=[0, 1, 2], H=64, W=64):
    image_channels = []
    for i in selected_channels:
        cwt_img = cwt_transform(data[:, i])
        resized = resize(cwt_img, (H, W), mode='constant', preserve_range=True)
        image_channels.append(resized)
    image = np.stack(image_channels, axis=0)  # Shape: (C, H, W)
    return image

# --- 5. Gramian Angular Field (GAF) Transformation ---
def to_gaf_image(data, selected_channels=[0, 1, 2], H=64, W=64):
    """
    Converts normalized time-series to GAF image per channel.
    data: (timesteps, channels)
    returns: (C, H, W)
    """
    def min_max_scale(ts):
        return (ts - np.min(ts)) / (np.max(ts) - np.min(ts)) * 2 - 1

    def compute_gaf(ts):
        ts_scaled = min_max_scale(ts)
        phi = np.arccos(ts_scaled)
        gaf = np.cos(phi[:, None] + phi[None, :])
        return gaf

    image_channels = []
    for i in selected_channels:
        gaf_img = compute_gaf(data[:, i])
        resized = resize(gaf_img, (H, W), mode='constant', preserve_range=True)
        image_channels.append(resized)
    image = np.stack(image_channels, axis=0)  # Shape: (C, H, W)
    return image
