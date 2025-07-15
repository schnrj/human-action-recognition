

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fft import dct, idct
import pywt

def multi_frequency_compression(input_tensor, k=2, max_u=5, max_v=5):
    """
    Multi-frequency compression using DCT and selective coefficient retention.
    
    Args:
        input_tensor: Input tensor of shape (C, H, W)
        k: Number of parts for DCT compression
        max_u: Maximum frequency component along height
        max_v: Maximum frequency component along width
    
    Returns:
        compressed_tensor: Compressed frequency representation
    """
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.detach().cpu().numpy()
    
    C, H, W = input_tensor.shape
    
    # Initialize output tensor
    compressed_features = []
    
    for c in range(C):
        channel_data = input_tensor[c]
        
        # Apply 2D DCT
        dct_coeffs = dct(dct(channel_data, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        # Keep only the most important coefficients (top-left corner)
        compressed_coeffs = np.zeros_like(dct_coeffs)
        compressed_coeffs[:min(max_u, H), :min(max_v, W)] = dct_coeffs[:min(max_u, H), :min(max_v, W)]
        
        # Flatten and keep only non-zero coefficients
        non_zero_coeffs = compressed_coeffs[compressed_coeffs != 0]
        
        # If we have too few coefficients, pad with zeros
        if len(non_zero_coeffs) < max_u * max_v:
            padded_coeffs = np.zeros(max_u * max_v)
            padded_coeffs[:len(non_zero_coeffs)] = non_zero_coeffs
            compressed_features.append(padded_coeffs)
        else:
            compressed_features.append(non_zero_coeffs[:max_u * max_v])
    
    # Stack all channels
    result = np.array(compressed_features)
    
    return torch.tensor(result.flatten(), dtype=torch.float32)

def channel_attention(input_tensor, reduction_ratio=16):
    """
    Channel attention mechanism.
    
    Args:
        input_tensor: Input tensor of shape (B, C, H, W)
        reduction_ratio: Reduction ratio for the attention mechanism
    
    Returns:
        attention_weights: Channel attention weights
    """
    if len(input_tensor.shape) == 3:
        # Add batch dimension if missing
        input_tensor = input_tensor.unsqueeze(0)
    
    batch_size, channels, height, width = input_tensor.shape
    
    # Global Average Pooling
    avg_pool = F.adaptive_avg_pool2d(input_tensor, (1, 1))  # (B, C, 1, 1)
    
    # Global Max Pooling
    max_pool = F.adaptive_max_pool2d(input_tensor, (1, 1))  # (B, C, 1, 1)
    
    # Shared MLP
    hidden_dim = max(1, channels // reduction_ratio)
    
    # Create temporary linear layers for attention computation
    fc1 = nn.Linear(channels, hidden_dim).to(input_tensor.device)
    fc2 = nn.Linear(hidden_dim, channels).to(input_tensor.device)
    
    # Process average pooled features
    avg_out = avg_pool.view(batch_size, channels)
    avg_out = F.relu(fc1(avg_out))
    avg_out = fc2(avg_out)
    
    # Process max pooled features
    max_out = max_pool.view(batch_size, channels)
    max_out = F.relu(fc1(max_out))
    max_out = fc2(max_out)
    
    # Combine and apply sigmoid
    attention_weights = torch.sigmoid(avg_out + max_out)
    
    return attention_weights.view(batch_size, channels, 1, 1)

class DWTProcessor:
    """
    Discrete Wavelet Transform processor for the methodology described.
    """
    
    @staticmethod
    def dwt_2d(input_tensor, wavelet='db4'):
        """
        2D Discrete Wavelet Transform.
        
        Args:
            input_tensor: Input tensor of shape (B, C, H, W)
            wavelet: Wavelet type
        
        Returns:
            Dictionary with LL, LH, HL, HH subbands
        """
        if isinstance(input_tensor, torch.Tensor):
            input_np = input_tensor.detach().cpu().numpy()
        else:
            input_np = input_tensor
        
        batch_size, channels, height, width = input_np.shape
        
        # Initialize output dictionaries
        subbands = {'LL': [], 'LH': [], 'HL': [], 'HH': []}
        
        for b in range(batch_size):
            batch_subbands = {'LL': [], 'LH': [], 'HL': [], 'HH': []}
            
            for c in range(channels):
                # Apply 2D DWT
                coeffs = pywt.dwt2(input_np[b, c], wavelet)
                cA, (cH, cV, cD) = coeffs
                
                batch_subbands['LL'].append(cA)
                batch_subbands['LH'].append(cH)
                batch_subbands['HL'].append(cV)
                batch_subbands['HH'].append(cD)
            
            # Stack channels
            for key in subbands.keys():
                if not subbands[key]:
                    subbands[key] = [np.stack(batch_subbands[key])]
                else:
                    subbands[key].append(np.stack(batch_subbands[key]))
        
        # Convert to tensors and stack batches
        for key in subbands.keys():
            subbands[key] = torch.tensor(np.stack(subbands[key]), dtype=torch.float32)
        
        return subbands
    
    @staticmethod
    def idwt_2d(subbands, wavelet='db4'):
        """
        Inverse 2D Discrete Wavelet Transform.
        
        Args:
            subbands: Dictionary with LL, LH, HL, HH subbands
            wavelet: Wavelet type
        
        Returns:
            Reconstructed tensor
        """
        LL = subbands['LL'].detach().cpu().numpy()
        LH = subbands['LH'].detach().cpu().numpy()
        HL = subbands['HL'].detach().cpu().numpy()
        HH = subbands['HH'].detach().cpu().numpy()
        
        batch_size, channels = LL.shape[:2]
        
        reconstructed = []
        
        for b in range(batch_size):
            batch_reconstructed = []
            
            for c in range(channels):
                # Reconstruct from subbands
                coeffs = (LL[b, c], (LH[b, c], HL[b, c], HH[b, c]))
                reconstructed_channel = pywt.idwt2(coeffs, wavelet)
                batch_reconstructed.append(reconstructed_channel)
            
            reconstructed.append(np.stack(batch_reconstructed))
        
        return torch.tensor(np.stack(reconstructed), dtype=torch.float32)

def frequency_attention_gate(rgb_subbands, reduction_ratio=16):
    """
    Generate frequency attention gate using RGB subbands as described in the methodology.
    
    Args:
        rgb_subbands: Dictionary with RGB subbands (LL, LH, HL, HH)
        reduction_ratio: Reduction ratio for attention
    
    Returns:
        attention_vector: Frequency attention vector
    """
    # Combine all RGB subbands
    Frgb_wave = rgb_subbands['LL'] + rgb_subbands['LH'] + rgb_subbands['HL'] + rgb_subbands['HH']
    
    # Global Average Pooling
    g = F.adaptive_avg_pool2d(Frgb_wave, (1, 1))
    g = g.view(g.size(0), -1)
    
    # Create attention network
    channels = g.size(1)
    hidden_dim = max(1, channels // reduction_ratio)
    
    # Temporary linear layers
    W1 = nn.Linear(channels, hidden_dim).to(g.device)
    W2 = nn.Linear(hidden_dim, channels).to(g.device)
    
    # Apply attention mechanism
    alpha = torch.sigmoid(W2(F.relu(W1(g))))
    
    return alpha

def apply_frequency_modulation(subbands, attention_vector):
    """
    Apply frequency attention modulation to subbands.
    
    Args:
        subbands: Dictionary with subbands
        attention_vector: Attention vector for modulation
    
    Returns:
        modulated_subbands: Modulated subbands
    """
    modulated_subbands = {}
    
    # Reshape attention vector for broadcasting
    alpha_reshaped = attention_vector.view(attention_vector.size(0), attention_vector.size(1), 1, 1)
    
    for key in subbands.keys():
        modulated_subbands[key] = subbands[key] * alpha_reshaped
    
    return modulated_subbands

# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    print("Testing DCT multi-frequency compression...")
    
    # Create a test tensor
    test_tensor = torch.randn(64, 32, 32)
    
    # Test multi-frequency compression
    compressed = multi_frequency_compression(test_tensor, k=2, max_u=5, max_v=5)
    print(f"Compressed tensor shape: {compressed.shape}")
    
    # Test channel attention
    test_input = torch.randn(1, 64, 32, 32)
    attention_weights = channel_attention(test_input)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test DWT
    dwt_processor = DWTProcessor()
    subbands = dwt_processor.dwt_2d(test_input)
    print("DWT subbands shapes:")
    for key, value in subbands.items():
        print(f"  {key}: {value.shape}")
    
    # Test reconstruction
    reconstructed = dwt_processor.idwt_2d(subbands)
    print(f"Reconstructed tensor shape: {reconstructed.shape}")
    
    print("All tests completed successfully!")
