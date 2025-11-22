# fft_ia/utils.py
import torch

def next_power_of_2(n):
    """Return smallest power of 2 >= n"""
    if n == 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1

def pad_to_power_of_2(x, dim=1):
    """Pad sequence length to next power of 2 (required for FFT-IA)"""
    seq_len = x.shape[dim]
    target = next_power_of_2(seq_len)
    if target == seq_len:
        return x, seq_len
    pad = target - seq_len
    x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    return x, target

def unpad(x, original_len):
    """Remove padding after forward pass"""
    return x[:, :original_len] if x.shape[1] > original_len else x
