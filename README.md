# GPU Infrastructure Labs

**Hands-on labs for understanding GPUs the way an LLM systems engineer has to** — read the hardware, hit the memory wall, and measure the roofline and the interconnect, all on real cards.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.7-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Notebooks](https://img.shields.io/badge/format-Jupyter-orange)

These three labs take you from *"what GPU am I on?"* to *"why is my model out of memory, why is generation slow, and why is splitting it across two cards expensive?"* — every answer measured off your **own** hardware, not read from a slide.

Built for the **Tasar'u AI Infrastructure Bootcamp** (Tuwaiq Academy) and tuned for a **2× NVIDIA RTX 5090** server (Blackwell, 32 GB each, PCIe 5.0, no NVLink) — but they run on any NVIDIA GPU, and the notebooks auto-measure whatever card they find.

---

## The labs

| # | Lab | What you'll do | Runs on Colab? |
|---|---|---|---|
| **A** | [**Introspection**](Lab_A_Introspection/) — *read your GPU* | `nvidia-smi`, `nvidia-smi topo -m`, decode every spec (VRAM, bandwidth, PCIe, precision, NVLink) | ✅ (single GPU) |
| **B** | [**The VRAM Wall**](Lab_B_VRAM_Wall/Lab_B_VRAM_Wall.ipynb) — *make it OOM, then fit* | Measure a model's memory, watch the KV cache grow, force an out-of-memory error, quantize to fit | ✅ Yes |
| **C** | [**Roofline & Interconnect**](Lab_C_Roofline_P2P/Lab_C_Roofline_and_P2P.ipynb) — *feed the matmul* | Measure peak compute/bandwidth + ridge point, compute-bound vs memory-bound, prefill vs decode, PCIe vs HBM | ⚠️ Partial* |

<sub>*Lab C §1–3 (roofline, prefill/decode, batching) run on any single GPU. §4–5 measure **two-GPU** PCIe bandwidth and need a real multi-GPU box — Colab has only one GPU, so those cells skip gracefully.*</sub>

Open Lab B in Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/Lab_B_VRAM_Wall/Lab_B_VRAM_Wall.ipynb)

---

## What you'll actually learn

- **Read a GPU like an engineer** — the four numbers that decide everything: capacity · bandwidth · precision floor · link.
- **The VRAM wall** — `weights + KV cache + activations + overhead`, why long contexts OOM, and how quantization (FP16 → FP8 → INT4) buys headroom.
- **The roofline** — arithmetic intensity, the ridge point, and *why* a matrix-vector op (decode) wastes the GPU while a matrix-matrix op (prefill) saturates it.
- **Prefill vs decode** — why generation is memory-bound and why batching recovers throughput.
- **The interconnect** — measuring PCIe vs on-chip HBM, and why *no NVLink* means you should reach for data/pipeline parallelism, not tensor parallelism.

---

## Quick start

```bash
git clone https://github.com/Lexith-AI/tasaru-gpu-labs.git
cd tasaru-gpu-labs

# 1) PyTorch — install the CUDA build that matches your GPU.
#    RTX 5090 / Blackwell (sm_120) needs the cu128 wheels:
pip install --upgrade "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128

# 2) The rest
pip install -r requirements.txt

# 3) Lab A is terminal commands — just open its README and run them.
#    Labs B & C are notebooks:
jupyter lab   # or jupyter notebook / VS Code
```

In **Lab B** and **Lab C** set one line near the top to a model available on your machine:

```python
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # a 7–8B model is ideal for a 32 GB card
```

> **On Colab?** Skip the `cu128` install line (Colab's torch already matches its GPU), and use a smaller model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`) so it loads fast.

---

## Repository structure

```
tasaru-gpu-labs/
├── Lab_A_Introspection/
│   └── README.md                      # terminal commands: read your GPU
├── Lab_B_VRAM_Wall/
│   └── Lab_B_VRAM_Wall.ipynb          # memory math, KV cache, OOM, quantization
├── Lab_C_Roofline_P2P/
│   └── Lab_C_Roofline_and_P2P.ipynb   # roofline, prefill/decode, PCIe vs HBM
├── requirements.txt
├── LICENSE
└── README.md
```

Each lab ends with reflection questions that make you read *your own* measured numbers.

---

## Credits

Created for **Tasar'u** — the first AI Infrastructure bootcamp in Saudi Arabia — at **Tuwaiq Academy**.
Hardware used for development: 2× NVIDIA RTX 5090 (Blackwell).

## License

[MIT](LICENSE) — free to use, adapt, and teach with. Attribution appreciated.
