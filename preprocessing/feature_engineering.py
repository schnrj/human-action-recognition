import numpy as np
import scipy.stats

def compute_statistical_features(data):
    features = []
    for i in range(data.shape[1]):
        channel = data[:, i]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.min(channel),
            np.max(channel),
            scipy.stats.skew(channel),
            scipy.stats.kurtosis(channel)
        ])
    return np.array(features)

def compute_smv(data):
    acc_x = data[:, 0]
    acc_y = data[:, 1]
    acc_z = data[:, 2]
    smv = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    return smv

def compute_frequency_features(data):
    fft_features = []
    for i in range(data.shape[1]):
        spectrum = np.abs(np.fft.fft(data[:, i]))
        fft_features.append(np.max(spectrum[1:len(spectrum)//2]))  # Dominant freq amplitude
        fft_features.append(np.sum(spectrum[1:len(spectrum)//2]**2))  # Spectral energy
    return np.array(fft_features)
