

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
# Model Analysis Utilities
# ------------------------------------------------------------


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def print_model_summary(model: nn.Module, input_shape: tuple = (1, 6, 112, 112)):
    """Print comprehensive model summary including parameters and layer details."""
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Parameter count
    param_stats = count_parameters(model)
    print(f"Total Parameters: {param_stats['total_parameters']:,}")
    print(f"Trainable Parameters: {param_stats['trainable_parameters']:,}")
    print(f"Non-trainable Parameters: {param_stats['non_trainable_parameters']:,}")
    print(f"Model Size (MB): {param_stats['total_parameters'] * 4 / 1024 / 1024:.2f}")
    print()
    
    # Layer-wise parameter breakdown
    print("LAYER-WISE PARAMETER BREAKDOWN:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Parameters':<15} {'Shape'}")
    print("-" * 80)
    
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:<40} {param_count:<15,} {str(list(param.shape))}")
    
    print("-" * 80)
    print(f"{'TOTAL':<40} {total_params:<15,}")
    print("=" * 80)


class FLOPCounter:
    """FLOP counter for various neural network operations."""
    
    @staticmethod
    def conv2d_flops(input_shape, kernel_size, out_channels, groups=1, stride=1):
        """Calculate FLOPs for 2D convolution."""
        batch_size, in_channels, input_height, input_width = input_shape
        kernel_height, kernel_width = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        output_height = input_height // stride
        output_width = input_width // stride
        
        # FLOPs = batch_size * output_height * output_width * out_channels * 
        #         (in_channels / groups) * kernel_height * kernel_width
        flops = (batch_size * output_height * output_width * out_channels * 
                (in_channels // groups) * kernel_height * kernel_width)
        return flops, (batch_size, out_channels, output_height, output_width)
    
    @staticmethod
    def conv1d_flops(input_shape, kernel_size, out_channels, groups=1, stride=1):
        """Calculate FLOPs for 1D convolution."""
        batch_size, in_channels, input_length = input_shape
        output_length = input_length // stride
        
        flops = (batch_size * output_length * out_channels * 
                (in_channels // groups) * kernel_size)
        return flops, (batch_size, out_channels, output_length)
    
    @staticmethod
    def linear_flops(input_shape, out_features):
        """Calculate FLOPs for linear layer."""
        batch_size = input_shape[0]
        in_features = input_shape[-1]
        flops = batch_size * in_features * out_features
        return flops, (*input_shape[:-1], out_features)
    
    @staticmethod
    def pool_flops(input_shape, kernel_size):
        """Calculate FLOPs for pooling operations (minimal)."""
        return input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], input_shape


def estimate_model_flops(model: nn.Module, input_shape: tuple = (1, 6, 112, 112)):
    """Estimate FLOPs for the entire model through a forward pass."""
    print("\nFLOPS ESTIMATION:")
    print("=" * 50)
    
    # Create sample input
    sample_input = torch.randn(*input_shape)
    
    # Hook to capture intermediate shapes and estimate FLOPs
    total_flops = 0
    layer_flops = {}
    
    def flop_hook(name, module):
        def hook_fn(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Conv2d):
                input_shape = input[0].shape
                kernel_size = module.kernel_size
                out_channels = module.out_channels
                groups = module.groups
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                
                flops, _ = FLOPCounter.conv2d_flops(input_shape, kernel_size, out_channels, groups, stride)
                layer_flops[name] = flops
                total_flops += flops
                
            elif isinstance(module, nn.Conv1d):
                input_shape = input[0].shape
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                out_channels = module.out_channels
                groups = module.groups
                stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                
                flops, _ = FLOPCounter.conv1d_flops(input_shape, kernel_size, out_channels, groups, stride)
                layer_flops[name] = flops
                total_flops += flops
                
            elif isinstance(module, nn.Linear):
                input_shape = input[0].shape
                out_features = module.out_features
                
                flops, _ = FLOPCounter.linear_flops(input_shape, out_features)
                layer_flops[name] = flops
                total_flops += flops
        
        return hook_fn
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            hook = module.register_forward_hook(flop_hook(name, module))
            hooks.append(hook)
    
    # Forward pass to estimate FLOPs
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print FLOP statistics
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Total GFLOPs: {total_flops / 1e9:.2f}")
    print(f"Total MFLOPs: {total_flops / 1e6:.2f}")
    
    # Print top FLOP consumers
    if layer_flops:
        print("\nTop 10 FLOP consuming layers:")
        print("-" * 50)
        sorted_layers = sorted(layer_flops.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, flops in sorted_layers:
            print(f"{name:<30}: {flops:>15,} ({flops/total_flops*100:.1f}%)")
    
    return total_flops


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
        self.dwt = HaarDWT(max_channels=channels)  # Ensure HaarDWT max_channels matches
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        LL, LH, HL, HH = self.dwt(x)
        agg = LL + LH + HL + HH
        w = F.adaptive_avg_pool2d(agg, 1)
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

        # Ensure square spatial dims before FBAM
        if H != W:
            min_HW = min(H, W)
            x = F.interpolate(x, size=(min_HW, min_HW), mode="bilinear", align_corners=False)

        # Assert square spatial dims
        assert x.shape[-2] == x.shape[-1], f"Non-square input to FBAMMambaBlock: {x.shape}"

        x = self.fbam(x)  # Apply channel & spatial attention

        P = self.patch
        Hp = (x.shape[-2] // P) * P
        Wp = (x.shape[-1] // P) * P

        # Resize height/width to multiples of patch size for perfect tokenization, if needed
        if Hp != x.shape[-2] or Wp != x.shape[-1]:
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
        x = self.stem(x)
        x = self.hmcn(x)
        x = self.backbone(x)
        logits = self.head(x)
        return logits

    def analyze_model(self, input_shape: tuple = (1, 6, 112, 112), verbose: bool = True):
        """
        Comprehensive model analysis including parameters and FLOPs.
        
        Args:
            input_shape (tuple): Input tensor shape (B, C, H, W)
            verbose (bool): Whether to print detailed analysis
            
        Returns:
            dict: Analysis results containing parameters and FLOPs
        """
        if verbose:
            print_model_summary(self, input_shape)
            
        param_stats = count_parameters(self)
        flops = estimate_model_flops(self, input_shape)
        
        analysis_results = {
            'parameters': param_stats,
            'flops': flops,
            'gflops': flops / 1e9,
            'mflops': flops / 1e6,
            'model_size_mb': param_stats['total_parameters'] * 4 / 1024 / 1024
        }
        
        if verbose:
            print(f"\nMODEL ANALYSIS SUMMARY:")
            print(f"Total Parameters: {param_stats['total_parameters']:,}")
            print(f"Model Size: {analysis_results['model_size_mb']:.2f} MB")
            print(f"FLOPs: {flops:,}")
            print(f"GFLOPs: {analysis_results['gflops']:.2f}")
        
        return analysis_results


# ------------------------------------------------------------
# Example usage for debug or standalone test with analysis
# ------------------------------------------------------------

if __name__ == "__main__":
    print("MultimodalMamba Model Analysis")
    print("=" * 50)
    
    # Model configuration
    B, C, H, W = 2, 6, 112, 112  # Sample batch with 6 input channels
    num_classes = 10
    
    # Create model with different configurations to compare
    configs = [
        {
            'name': 'Small Model',
            'config': {
                'stages': [StageConfig(1, 32), StageConfig(1, 64)],
                'd_model': 128,
                'd_hidden': 64,
                'patch': 7
            }
        },
        {
            'name': 'Medium Model', 
            'config': {
                'stages': [StageConfig(1, 64), StageConfig(1, 128)],
                'd_model': 256,
                'd_hidden': 128,
                'patch': 7
            }
        },
        {
            'name': 'Large Model',
            'config': {
                'stages': [StageConfig(1, 64), StageConfig(1, 128), StageConfig(1, 256)],
                'd_model': 256,
                'd_hidden': 128,
                'patch': 7
            }
        }
    ]
    
    comparison_results = []
    
    for config_info in configs:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {config_info['name']}")
        print(f"{'='*60}")
        
        model = MultimodalMamba(
            in_channels=C,
            num_classes=num_classes,
            hmcn_k=[3, 5, 7, 9],
            use_mamba=False,
            **config_info['config']
        )
        
        # Perform analysis
        results = model.analyze_model(input_shape=(B, C, H, W), verbose=True)
        results['name'] = config_info['name']
        comparison_results.append(results)
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        x = torch.randn(B, C, H, W)
        try:
            with torch.no_grad():
                y = model(x)
            print(f"✓ Forward pass successful! Output shape: {y.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'Parameters':<12} {'Size(MB)':<10} {'GFLOPs':<10} {'MFLOPs':<10}")
    print("-" * 80)
    
    for result in comparison_results:
        print(f"{result['name']:<15} "
              f"{result['parameters']['total_parameters']:<12,} "
              f"{result['model_size_mb']:<10.2f} "
              f"{result['gflops']:<10.2f} "
              f"{result['mflops']:<10.1f}")
    
    print(f"{'='*80}")
    
    # Additional efficiency metrics
    print(f"\nEFFICIENCY METRICS:")
    print("-" * 40)
    for result in comparison_results:
        params_m = result['parameters']['total_parameters'] / 1e6
        efficiency = result['gflops'] / params_m if params_m > 0 else 0
        print(f"{result['name']:<15}: {efficiency:.2f} GFLOPs/M params")
    
    print(f"\nAnalysis complete! Use model.analyze_model() method for detailed analysis of any configuration.")