# File: test_model.py

import sys
import os
# Ensure project root is on Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize

from models.pyramid_model import PyramidAttentionModel
from preprocessing.dataloading import load_data
from preprocessing.preprocessing import preprocess_imu_data
from preprocessing.cwt_transform import (
    kaiser_lowpass_filter,
    z_score_normalize,
    to_time_frequency_image,
)

def prepare_combined_input():
    """
    Prepare combined input tensor by loading and preprocessing real IMU data,
    generating CWT, upsampling, and concatenating with RGB and Depth tensors.
    Returns:
        input_tensor (torch.Tensor): Shape (1, 7, 224, 224)
    """
    # Load and preprocess IMU data
    imu_data = load_data()                             # Shape: (T, 6)
    cleaned_imu = preprocess_imu_data(imu_data)        # Shape: (180, 6)

    # Generate CWT image for accelerometer channels [0,1,2]
    filtered = kaiser_lowpass_filter(imu_data)
    normalized = z_score_normalize(filtered)
    cwt_image = to_time_frequency_image(
        normalized, selected_channels=[0, 1, 2], H=64, W=64
    )  # Shape: (3, 64, 64)

    # Upsample IMU CWT to (3, 224, 224)
    upsampled_cwt = np.stack([
        resize(cwt_image[i], (224, 224), mode='reflect', anti_aliasing=True)
        for i in range(cwt_image.shape[0])
    ]).astype(np.float32)

    # Placeholder: load or simulate Depth and RGB tensors
    # Replace these with real data-loading code as needed
    rgb_tensor = np.random.rand(3, 224, 224).astype(np.float32)
    depth_tensor = np.random.rand(1, 224, 224).astype(np.float32)

    # Concatenate modalities: IMU (3) + Depth (1) + RGB (3) = 7 channels
    combined = np.concatenate([upsampled_cwt, depth_tensor, rgb_tensor], axis=0)
    input_tensor = torch.tensor(combined).unsqueeze(0)  # Shape: (1, 7, 224, 224)

    return input_tensor

def test_model_inference():
    """
    Test model inference to ensure forward pass works without errors.
    """
    print("=" * 60)
    print("Testing PyTorch Model Inference")
    print("=" * 60)

    input_tensor = prepare_combined_input()
    print(f"Input tensor shape: {input_tensor.shape}")

    # Initialize model with matching channel count and splits
    NUM_CLASSES = 27
    NUM_CHANNELS = input_tensor.shape[1]  # 7
    model = PyramidAttentionModel(
        input_channels=NUM_CHANNELS,
        n_classes=NUM_CLASSES,
        num_splits=2
    )
    model.eval()

    with torch.no_grad():
        gap_pred, dct_pred = model(input_tensor)

    print(f"GAP output shape: {gap_pred.shape}")
    print(f"DCT output shape: {dct_pred.shape}")
    print(f"Max GAP probability: {torch.softmax(gap_pred, dim=1).max():.4f}")
    print(f"Max DCT probability: {torch.softmax(dct_pred, dim=1).max():.4f}")

    gap_class = torch.argmax(gap_pred, dim=1).item()
    dct_class = torch.argmax(dct_pred, dim=1).item()
    print(f"GAP predicted class: {gap_class}")
    print(f"DCT predicted class: {dct_class}")

    print("✅ Inference test passed!\n")

def test_model_training_step():
    """
    Test a single training step to verify backpropagation works.
    """
    print("=" * 60)
    print("Testing Model Training Step")
    print("=" * 60)

    input_tensor = prepare_combined_input()
    target = torch.tensor([0])  # Dummy target; adjust if necessary

    NUM_CLASSES = 27
    NUM_CHANNELS = input_tensor.shape[1]
    model = PyramidAttentionModel(
        input_channels=NUM_CHANNELS,
        n_classes=NUM_CLASSES,
        num_splits=2
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    optimizer.zero_grad()
    gap_pred, dct_pred = model(input_tensor)
    loss = criterion(gap_pred, target)
    print(f"Training loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()
    print("✅ Training step test passed!\n")

if __name__ == "__main__":
    test_model_inference()
    test_model_training_step()
    print("🎉 All tests completed successfully!")
