import torch
import torch.nn as nn
from models.multimodal_mamba import MultimodalMamba, StageConfig  # Adjust import path as needed

# Load tensors
x_rgb   = torch.load("x_rgb.pt")      # (1, 3, 224, 224)
x_depth = torch.load("x_depth.pt")    # (1, 1, 224, 224)
x_imu   = torch.load("x_imu.pt")      # (1, 3, 224, 224)
label   = torch.load("label.pt")      # (1), torch.long

# Concatenate all modalities
x_input = torch.cat([x_rgb, x_depth, x_imu], dim=1)  # (1, 7, 224, 224)

# Model setup
NUM_CLASSES = 27
model = MultimodalMamba(
    in_channels=7,
    num_classes=NUM_CLASSES,
    stages=[StageConfig(1,64), StageConfig(1,128)],   # Or default config
    hmcn_k=[3,5,7,9], patch=7, d_model=256, d_hidden=128, use_mamba=False
)

# Move to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x_input = x_input.to(device)
label = label.to(device)

# Run one epoch step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
optimizer.zero_grad()
logits = model(x_input)  # [1, num_classes]
loss = criterion(logits, label)
loss.backward()
optimizer.step()
print(f'Loss after one epoch step: {loss.item():.4f}')
