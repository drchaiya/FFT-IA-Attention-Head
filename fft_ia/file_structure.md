fft-ia/
├── fft_ia/
│   ├── __init__.py
│   ├── core.py              # Main FFT-IA layer (100% paper-faithful)
│   ├── butterfly.py         # Radix-2 butterfly indexing + local attention
│   ├── fused_kernel.py      # Optional Triton fused kernel stub (for wall-clock speed)
│   └── utils.py
├── examples/
│   └── train_nano.py        # TinyLlama-style training demo (N=4096 works)
├── tests/
│   └── test_fft_ia.py
├── requirements.txt
├── README.md
└── pyproject.toml
