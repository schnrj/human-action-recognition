import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCrossAttentionModule(nn.Module):
    """
    Implements lightweight cross-attention: for each target modality, tokens attend over RGB tokens.
    """
    def __init__(self, channel_dim):
        super().__init__()
        self.q_proj = nn.Linear(channel_dim, channel_dim)
        self.k_proj = nn.Linear(channel_dim, channel_dim)
        self.v_proj = nn.Linear(channel_dim, channel_dim)
        self.out_proj = nn.Conv1d(channel_dim, channel_dim, kernel_size=1)
        self.ln = nn.LayerNorm(channel_dim)

    def forward(self, Q, K, V, orig):
        # Q, K, V: [B, n_tokens, C]
        attn = torch.matmul(self.q_proj(Q), self.k_proj(K).transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
        weights = F.softmax(attn, dim=-1)
        context = torch.matmul(weights, self.v_proj(V))      # [B, n_tokens, C]
        context = context.permute(0, 2, 1)
        context = self.out_proj(context).permute(0, 2, 1)    # [B, n_tokens, C]
        fused = self.ln(orig + context)                      # residual connection and norm
        return fused
