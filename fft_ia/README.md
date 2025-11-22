# FFT-IA

FFT-Inspired Attention.

Replaces the O(N²) self-attention layer in any Transformer with a fixed, radix-2 butterfly network.

- True O(N log N) in sequence length  
- Exact local softmax (no approximation)  
- Dynamic Q/K re-projection every stage (content-aware)  
- Sequence length must be power of 2 (auto-padded)  
- Works today. Drop-in replacement.

That's it.

No kernels.  
No low-rank.  
No learned sparsity.  
No "efficient attention" tricks.

Just structural pruning inspired by the 1965 FFT.

### Install
```bash
pip install git+https://github.com/drchaiya/FFT-IA-Attention-Head.git
