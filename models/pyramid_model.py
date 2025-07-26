import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

# --- Lightweight ECAM block ---
class EfficientCrossAttentionModule(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Conv1d(channel_dim, channel_dim, kernel_size=1)
        self.ln = nn.LayerNorm(channel_dim)
    def forward(self, Q, K, V, orig):
        # Q, K, V: [B, N, C]
        attn = torch.matmul(self.q_proj(Q), self.k_proj(K).transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        weights = F.softmax(attn, dim=-1)
        context = torch.matmul(weights, self.v_proj(V))   # [B, N, C]
        context = context.permute(0, 2, 1)
        context = self.out_proj(context).permute(0, 2, 1) # [B, N, C]
        fused = self.ln(orig + context)
        return fused

# --- HMCN backbone ---
class PyramidMultiScaleCNN(nn.Module):
    def __init__(self, input_channels, groups=4):
        super().__init__()
        self.groups = groups
        padded = input_channels
        if input_channels % groups != 0:
            padded += groups - (input_channels % groups)
        self.C_prime = padded // groups
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(self.C_prime, self.C_prime, kernel_size=3, padding=1, groups=self.C_prime)
            for _ in range(groups)
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv2d(self.C_prime, self.C_prime, kernel_size=1)
            for _ in range(groups)
        ])
    def convdwp(self, x, idx):
        x = self.depthwise_convs[idx](x)
        x = self.pointwise_convs[idx](x)
        return F.relu(x)
    def forward(self, x):
        B, C, H, W = x.shape
        if C % self.groups != 0:
            pad_len = self.groups - (C % self.groups)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        splits = torch.split(x, self.C_prime, dim=1)
        outputs = [self.convdwp(split, idx) for idx, split in enumerate(splits)]
        return torch.cat(outputs, dim=1)

# ---- FINAL MULTIMODAL FUSION MODEL ----
class MultimodalFrequencyAwareHAR(nn.Module):
    def __init__(self, rgb_channels, depth_channels, imu_channels, num_classes=27, groups=4):
        super().__init__()
        self.hmcn_rgb = PyramidMultiScaleCNN(rgb_channels, groups=groups)
        self.hmcn_depth = PyramidMultiScaleCNN(depth_channels, groups=groups)
        self.hmcn_imu = PyramidMultiScaleCNN(imu_channels, groups=groups)
        self.groups = groups
        self.feature_channels = self.hmcn_rgb.C_prime * groups
        # Efficient spatial reduction (critical for ECAM memory!)
        self.reduce = nn.AdaptiveAvgPool2d((14, 14))  # Reduce feature map size before tokens
        self.ecam_depth = EfficientCrossAttentionModule(self.feature_channels)
        self.ecam_imu   = EfficientCrossAttentionModule(self.feature_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.feature_channels * 3, num_classes)

    def forward(self, x_rgb, x_depth, x_imu):
        # --- Step1: Feature extraction
        F_rgb   = self.hmcn_rgb(x_rgb)       # [B, C', H', W']
        F_depth = self.hmcn_depth(x_depth)
        F_imu   = self.hmcn_imu(x_imu)

        # --- Step2: Spatial reduction to manageable grid size for ECAM
        F_rgb   = self.reduce(F_rgb)
        F_depth = self.reduce(F_depth)
        F_imu   = self.reduce(F_imu)

        B, C, H, W = F_rgb.shape
        N = H * W

        # --- Step3: Tokenize (flatten spatial, for attention)
        F_rgb_tok   = F_rgb.flatten(2).transpose(1,2)   # [B, N, C]
        F_depth_tok = F_depth.flatten(2).transpose(1,2) # [B, N, C]
        F_imu_tok   = F_imu.flatten(2).transpose(1,2)   # [B, N, C]

        # --- Step4: Cross-attention fusion (Depth←RGB and IMU←RGB)
        Ffused_depth_tok = self.ecam_depth(F_depth_tok, F_rgb_tok, F_rgb_tok, F_depth_tok)
        Ffused_imu_tok   = self.ecam_imu(F_imu_tok, F_rgb_tok, F_rgb_tok, F_imu_tok)

        # --- Step5: Untokenize
        Ffused_depth = Ffused_depth_tok.transpose(1,2).reshape(B, C, H, W)
        Ffused_imu   = Ffused_imu_tok.transpose(1,2).reshape(B, C, H, W)

        # --- Step6: Late fusion and classification
        F_final = torch.cat([F_rgb, Ffused_depth, Ffused_imu], dim=1) # [B, C*3, H, W]
        pooled  = self.gap(F_final).flatten(1)
        logits = self.classifier(pooled)
        return logits
