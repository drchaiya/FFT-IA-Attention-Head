from fft_ia import FFTInspiredAttention
from fft_ia.utils import pad_to_power_of_2, unpad

layer = FFTInspiredAttention(dim=512, heads=8)

x = torch.randn(1, 3715, 512)           # any length
x_pad, n_pad = pad_to_power_of_2(x)     # → 4096
out_pad = layer(x_pad)
out = unpad(out_pad, 3715)              # back to original
