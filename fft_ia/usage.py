from fft_ia.core import FFTInspiredAttention

attn = FFTInspiredAttention(dim=512, heads=8)
out = attn(x)  # x: (B, N, D), N=2^k
