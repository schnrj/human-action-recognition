"""
MultimodalMamba for Human Action Recognition (HAR)

This module implements a multimodal deep learning model combining RGB, Depth, and IMU data for HAR tasks.

Key components:
- Hierarchical Multiheaded Convolution Network (HMCN) encoder block for multi-scale feature extraction.
- Frequency-Based Attention Module (FBAM) with wavelet channel and spatial attention.
- Lightweight Vim block based on bidirectional state-space models for sequence modeling.
- FBAM-Mamba block bridging 2D and 1D representations using patched Conv1d layers.
- Multistage backbone and classification head.

References:
- Wavelet attention: https://github.com/yutinyang/DWAN
- State-space Mamba: https://github.com/state-spaces/mamba

The implementation requires only PyTorch and einops.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ------------------------------------------------------------
# Utility layers
# ------------------------------------------------------------


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution (depthwise with pointwise conv) for efficient spatial processing."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.dw = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution over sequence length for temporal feature extraction."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=channels, bias=bias
        )

    def forward(self, x):
        return self.dw(x)


# ------------------------------------------------------------
# HMCN: Hierarchical Multiheaded Convolution Network encoder block
# ------------------------------------------------------------


class HMCNBlock(nn.Module):
    """
    Hierarchical multi-scale convolution block with split-propagate-concat pattern.

    Args:
        in_ch (int): Number of input channels to the first stage.
        width (int): Internal channel width for all stages.
        stages_k (List[int]): List of kernel sizes for each hierarchical block.
    """

    def __init__(self, in_ch: int, width: int, stages_k: List[int]):
        super().__init__()
        self.width = width
        self.stages_k = stages_k
        # Number of output channels after concat of splits
        self.out_ch = width * len(stages_k) // 2


        convs = []
        for idx, k in enumerate(stages_k):
            # First block: input channels = in_ch (e.g. 64)
            # Subsequent blocks: input channels = half of width (because input is part_b of y split)
            in_channels_block = in_ch if idx == 0 else width // 2
            convs.append(
                nn.Sequential(
                    DepthwiseSeparableConv2d(in_channels_block, width, k),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
            )
        self.convs = nn.ModuleList(convs)
        self.skip = nn.Identity()
        self.skip_proj = nn.Conv2d(in_ch, self.out_ch, kernel_size=1)  # Project skip to out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical blocks with channel splitting.

        Args:
            x (torch.Tensor): Tensor of shape [B, in_ch, H, W]

        Returns:
            torch.Tensor: Tensor of shape [B, width * len(stages_k), H, W]
        """
        outs = []
        y_prev = None
        for i, block in enumerate(self.convs):
            if i == 0:
                y = block(x)
            else:
                # Use previous half output as input to next block to avoid channel explosion
                y = block(y_prev)

            c = y.shape[1]
            c_half = c // 2
            part_a, part_b = y[:, :c_half], y[:, c_half:]
            outs.append(part_a)
            y_prev = part_b

        out = torch.cat(outs, dim=1)
        skip = F.interpolate(self.skip_proj(x), size=out.shape[-2:], mode="nearest")
        # Residual connection interpolated to output size
        return out + skip



# ------------------------------------------------------------
# Haar DWT implemented as fixed depthwise Conv2d filters
# ------------------------------------------------------------


class HaarDWT(nn.Module):
    """
    Single-level 2D Haar Discrete Wavelet Transform implemented as fixed depthwise Conv2d filters.

    This module applies Haar wavelet filters (LL, LH, HL, HH) to input feature maps,
    performing a depthwise convolution with stride 2, which halves the spatial resolution.

    The implementation dynamically handles variable input channel sizes by preparing
    a kernel buffer with a maximum channel capacity and slicing it according to actual input channels.

    Args:
        max_channels (int): Maximum expected number of input channels.
                            The internal kernel buffer is pre-allocated for this many channels.
                            Must be >= maximum input channel count seen in forward calls.
    """

    def __init__(self, max_channels: int = 512):
        super().__init__()
        # Haar filters normalized by 1/sqrt(2)
        h = torch.tensor([1.0, 1.0]) / 1.41421356237  # Low-pass
        g = torch.tensor([1.0, -1.0]) / 1.41421356237  # High-pass

        # Create 2D Haar kernels via outer products
        ll = torch.einsum("i,j->ij", h, h)  # Approximation (low-low)
        lh = torch.einsum("i,j->ij", h, g)  # Low-high
        hl = torch.einsum("i,j->ij", g, h)  # High-low
        hh = torch.einsum("i,j->ij", g, g)  # High-high

        # Stack kernels: shape (4, 2, 2)
        K = torch.stack([ll, lh, hl, hh], dim=0)

        # Register kernel buffer repeated max_channels times for depthwise conv
        # Shape: (max_channels, 4, 2, 2)
        self.register_buffer("K", K[None, :, :, :].repeat(max_channels, 1, 1, 1))
        self.max_channels = max_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply single-level Haar Discrete Wavelet Transform on input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
                              where C <= max_channels.

        Returns:
            Tuple of four sub-band feature tensors (LL, LH, HL, HH), each of shape (B, C, H//2, W//2).
        """
        B, C, H, W = x.shape
        assert C <= self.max_channels, f"Input channel {C} exceeds max_channels {self.max_channels}"

        # Slice the kernel buffer dynamically according to input channels and reshape for conv2d
        weight = self.K[:C].reshape(C * 4, 1, 2, 2)  # (C*4, 1, 2, 2)

        # Depthwise convolution: groups=C applies one filter group per channel
        x_dw = F.conv2d(x, weight, stride=2, padding=0, groups=C)  # (B, C*4, H/2, W/2)

        # Reshape to separate wavelet components per channel
        x_dw = x_dw.view(B, C, 4, H // 2, W // 2)

        # Split into four wavelet sub-bands
        LL = x_dw[:, :, 0]
        LH = x_dw[:, :, 1]
        HL = x_dw[:, :, 2]
        HH = x_dw[:, :, 3]

        return LL, LH, HL, HH


# ------------------------------------------------------------
# Frequency-Based Attention Module (FBAM)
# ------------------------------------------------------------


class WaveletChannelAttention(nn.Module):
    """Wavelet-based channel attention module with dynamic channel handling."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        print(f"[DEBUG] Initializing WaveletChannelAttention: channels={channels}, hidden={hidden}")
        self.dwt = HaarDWT(max_channels=channels)  # Ensure HaarDWT max_channels matches
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"[DEBUG] WaveletChannelAttention forward input shape: {x.shape}")
        LL, LH, HL, HH = self.dwt(x)
        agg = LL + LH + HL + HH
        w = F.adaptive_avg_pool2d(agg, 1)
        print(f"[DEBUG] w shape before fc layers: {w.shape}")
        w = self.fc2(F.relu(self.fc1(w), inplace=True))
        return torch.sigmoid(w)


class WaveletSpatialAttention(nn.Module):
    """Wavelet-based spatial attention.

    Args:
        channels (int): Number of input channels.
        inter_channels (Optional[int]): Intermediate channel size for spatial attention.
    """

    def __init__(self, channels: int, inter_channels: Optional[int] = None):
        super().__init__()
        self.dwt = HaarDWT(max_channels=channels)
        if inter_channels is None:
            inter_channels = max(channels // 4, 8)

        self.reduce = nn.Conv2d(channels * 2, inter_channels, kernel_size=1)
        self.proj = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"[DEBUG] WaveletSpatialAttention forward input shape: {x.shape}")
        LL, LH, HL, HH = self.dwt(x)

        highs = LH + HL + HH
        s = torch.cat([LL, highs], dim=1)
        s = F.relu(self.reduce(s), inplace=True)
        s = self.proj(s)
        s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(s)


class FBAM(nn.Module):
    """Frequency-Based Attention Module combining channel and spatial attention.

    Args:
        channels (int): Number of input channels.
        reduction (int): Channel reduction ratio.
        spatial_inter (Optional[int]): Intermediate spatial channel size.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_inter: Optional[int] = None):
        super().__init__()
        print(f"[DEBUG] Initializing FBAM with channels={channels}")
        self.wca = WaveletChannelAttention(channels, reduction)
        self.wsa = WaveletSpatialAttention(channels, spatial_inter)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable scalar for combining attentions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_c = self.wca(x) * x
        attn_s = self.wsa(x) * x
        a = torch.clamp(self.alpha, 0.0, 1.0)
        return a * attn_c + (1 - a) * attn_s


# ------------------------------------------------------------
# Lightweight Vim block (bidirectional state-space style)
# ------------------------------------------------------------


class LowRankLinear(nn.Module):
    """Low-rank linear projection to reduce parameters.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        rank_ratio (float): Ratio for low rank (0,1].
        bias (bool): Whether to use bias.
    """

    def __init__(self, in_dim: int, out_dim: int, rank_ratio: float = 0.5, bias: bool = True):
        super().__init__()
        r = max(1, int(min(in_dim, out_dim) * rank_ratio))
        self.U = nn.Linear(in_dim, r, bias=False)
        self.V = nn.Linear(r, out_dim, bias=bias)

    def forward(self, x):
        return self.V(self.U(x))


class LightweightVimBlock(nn.Module):
    """Bidirectional SSM-style block with low-rank projections and gated convolution.

    Args:
        d_model (int): Input token dimension.
        d_hidden (int): Internal hidden dimension.
        use_mamba (bool): Whether to use mamba_ssm package.
        rank_ratio (float): Rank ratio for projections.
    """

    def __init__(self, d_model: int, d_hidden: int = 128, use_mamba: bool = False, rank_ratio: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_hidden)
        self.dwconv = DepthwiseConv1d(d_hidden, 3)
        self.use_mamba = use_mamba

        if use_mamba:
            try:
                from mamba_ssm import Mamba

                self.fwd = Mamba(d_hidden)
                self.bwd = Mamba(d_hidden)
                self._has_mamba = True
            except Exception:
                self._has_mamba = False
        else:
            self._has_mamba = False

        self.G = LowRankLinear(d_hidden, d_hidden, rank_ratio)
        self.J = LowRankLinear(d_hidden, d_hidden, rank_ratio)
        self.Delta = LowRankLinear(d_hidden, d_hidden, rank_ratio)
        self.gate_fc = nn.Linear(2 * d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, d_model)

    def simple_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lightweight state-space model style update using gated depthwise convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, L).

        Returns:
            torch.Tensor: Output tensor with the same shape (B, C, L).
        """
        z = self.dwconv(x)
        g = torch.sigmoid(self.G(z.transpose(1, 2))).transpose(1, 2)
        j = torch.tanh(self.J(z.transpose(1, 2))).transpose(1, 2)
        d = torch.relu(self.Delta(z.transpose(1, 2))).transpose(1, 2)
        y = g * j + (1 - g) * (x + d)
        return y

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens (torch.Tensor): Input tokens of shape (B, L, D_model).

        Returns:
            torch.Tensor: Output tokens of shape (B, L, D_model).
        """
        x = self.norm(tokens)
        x = self.proj(x)
        x1 = x.transpose(1, 2)

        if self._has_mamba:
            y_f = self.fwd(x1.transpose(1, 2)).transpose(1, 2)
            y_b = torch.flip(self.bwd(torch.flip(x1.transpose(1, 2), dims=[1])), dims=[1]).transpose(1, 2)
        else:
            y_f = self.simple_ssm(x1)
            y_b = torch.flip(self.simple_ssm(torch.flip(x1, dims=[-1])), dims=[-1])

        y_f = y_f.transpose(1, 2)
        y_b = y_b.transpose(1, 2)

        g = torch.sigmoid(self.gate_fc(torch.cat([y_f, y_b], dim=-1)))
        y = g * y_f + (1 - g) * y_b
        out = self.out(y)

        return tokens + out


# ------------------------------------------------------------
# FBAM-Mamba: Bridges 2D feature maps and 1D token sequences
# ------------------------------------------------------------


class FBAMMambaBlock(nn.Module):
    """
    Blocks that apply FBAM attention, patch based tokenization, Vim block, and reconstruct 2D features.

    Args:
        channels (int): Number of input feature channels.
        patch (int): Size of square patch for tokenization.
        d_model (int): Token embedding dimension.
        d_hidden (int): Hidden dimension in Vim block.
        use_mamba (bool): Use mamba_ssm package if available.
        rank_ratio (float): Rank ratio for LowRankLinear projections.
    """

    def __init__(
        self,
        channels: int,
        patch: int = 7,
        d_model: int = 256,
        d_hidden: int = 128,
        use_mamba: bool = False,
        rank_ratio: float = 0.5,
    ):
        super().__init__()
        
        print(f"[DEBUG] Initializing FBAMMambaBlock with channels={channels}")

        self.fbam = FBAM(channels)
        self.patch = patch
        # Use groups=1 in Conv1d to avoid channel divisibility issues in vanilla PyTorch
        self.to_seq_dw = nn.Conv1d(
            patch * patch * channels, d_model, kernel_size=3, padding=1, groups=1
        )
        self.vim = LightweightVimBlock(d_model=d_model, d_hidden=d_hidden, use_mamba=use_mamba, rank_ratio=rank_ratio)
        self.from_seq = nn.Conv1d(d_model, patch * patch * channels, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        print(f"[DEBUG] FBAMMambaBlock input shape: {x.shape}")

        # Ensure square spatial dims before FBAM
        if H != W:
            min_HW = min(H, W)
            print(f"[DEBUG] Interpolating from ({H}, {W}) to square ({min_HW}, {min_HW})")
            x = F.interpolate(x, size=(min_HW, min_HW), mode="bilinear", align_corners=False)

        # Assert square spatial dims
        assert x.shape[-2] == x.shape[-1], f"Non-square input to FBAMMambaBlock: {x.shape}"

        x = self.fbam(x)  # Apply channel & spatial attention

        P = self.patch
        Hp = (x.shape[-2] // P) * P
        Wp = (x.shape[-1] // P) * P

        # Resize height/width to multiples of patch size for perfect tokenization, if needed
        if Hp != x.shape[-2] or Wp != x.shape[-1]:
            print(f"[DEBUG] Interpolating to multiples of patch size: {(Hp, Wp)}")
            x = F.interpolate(x, size=(Hp, Wp), mode="bilinear", align_corners=False)

        # Patchify: from 2D feature maps to sequence tokens
        tokens = rearrange(x, "b c (hp p1) (wp p2) -> b (hp wp) (p1 p2 c)", p1=P, p2=P)

        # Project tokens via depthwise conv1d (groups=1 for compatibility)
        seq = tokens.transpose(1, 2)
        seq = self.to_seq_dw(seq)
        seq = seq.transpose(1, 2)

        # Process with Vim block (state-space sequence model)
        seq = self.vim(seq)

        # Reconstruct to 2D feature maps
        seq = seq.transpose(1, 2)
        seq = self.from_seq(seq)
        tokens = seq.transpose(1, 2)

        x = rearrange(
            tokens,
            "b (hpwp_h hpwp_w) (p p2 c) -> b c (hpwp_h p) (hpwp_w p2)",
            p=P,
            p2=P,
            c=x.shape[1],
            hpwp_h=Hp // P,
            hpwp_w=Wp // P,
        )
        print(f"[DEBUG] FBAMMambaBlock output shape: {x.shape}")
        return x


# ------------------------------------------------------------
# Model stages and backbone
# ------------------------------------------------------------


@dataclass
class StageConfig:
    blocks: int
    channels: int


class MultimodalMamba(nn.Module):
    """
    Main MultimodalMamba model for HAR combining RGB, Depth, and IMU features.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of target classes.
        stages (List[StageConfig]): Model stages configuration.
        hmcn_k (List[int]): Kernel sizes for HMCNBlock.
        patch (int): Patch size for FBAMMambaBlock.
        d_model (int): Token embedding dimension.
        d_hidden (int): Hidden size for Vim blocks.
        fbam_shrink (float): Optional shrink parameter for FBAM (not used directly).
        use_mamba (bool): Enable mamba_ssm-based Vim blocks if available.
        rank_ratio (float): Low-rank approximation ratio in Vim block projections.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stages: List[StageConfig] = (
            StageConfig(blocks=1, channels=64),
            StageConfig(blocks=1, channels=128),
            StageConfig(blocks=1, channels=256),
        ),
        hmcn_k: List[int] = (3, 5, 7, 9),
        patch: int = 7,
        d_model: int = 256,
        d_hidden: int = 128,
        fbam_shrink: float = 0.25,
        use_mamba: bool = False,
        rank_ratio: float = 0.5,
    ):
        super().__init__()
        self.stages_cfg = stages
        self.stem = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, stages[0].channels, 3),
            nn.BatchNorm2d(stages[0].channels),
            nn.ReLU(inplace=True),
        )
        self.hmcn = HMCNBlock(
            in_ch=stages[0].channels, width=stages[0].channels, stages_k=list(hmcn_k)
        )

        blocks = []
        in_ch = stages[0].channels * len(hmcn_k) // 2 # concat of splits from HMCN

        for si, s in enumerate(stages):
            # Downsample between stages except the first
            if si > 0:
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, s.channels, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(s.channels),
                        nn.ReLU(inplace=True),
                    )
                )
                in_ch = s.channels

            # Add FBAM-Mamba blocks at each stage
            for _ in range(s.blocks):
                print(f"[DEBUG] Adding FBAMMambaBlock at stage {si} with channels={in_ch}")
                blocks.append(
                    FBAMMambaBlock(
                        in_ch,
                        patch=patch,
                        d_model=d_model,
                        d_hidden=d_hidden,
                        use_mamba=use_mamba,
                        rank_ratio=rank_ratio,
                    )
                )

        self.backbone = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full model.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W), 
                              where C is number of input channels (e.g., 7).

        Returns:
            torch.Tensor: Logits tensor with shape (B, num_classes).
        """
        print(f"[DEBUG] MultimodalMamba input shape: {x.shape}")
        x = self.stem(x)
        print(f"[DEBUG] After stem: {x.shape}")
        x = self.hmcn(x)
        print(f"[DEBUG] After hmcn: {x.shape}")
        x = self.backbone(x)
        print(f"[DEBUG] After backbone: {x.shape}")
        logits = self.head(x)
        print(f"[DEBUG] Output logits shape: {logits.shape}")
        return logits


# ------------------------------------------------------------
# Example usage for debug or standalone test
# ------------------------------------------------------------

if __name__ == "__main__":
    B, C, H, W = 2, 6, 112, 112  # Sample batch with 6 input channels
    num_classes = 10
    model = MultimodalMamba(
        in_channels=C,
        num_classes=num_classes,
        stages=[StageConfig(1, 64), StageConfig(1, 128)],
        hmcn_k=[3, 5, 7, 9],
        patch=7,
        d_model=256,
        d_hidden=128,
        use_mamba=False,
    )
    x = torch.randn(B, C, H, W)
    y = model(x)
    print("logits:", y.shape)  # Expected output shape: (B, num_classes)
