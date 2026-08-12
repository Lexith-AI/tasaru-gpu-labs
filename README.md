# Tasar'u — GPU & AI Infrastructure Labs

**Hands-on labs and a live challenge for the [Tasar'u](https://tuwaiq.edu.sa) AI Infrastructure bootcamp** (Tuwaiq Academy) — the first of its kind in Saudi Arabia.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.7-ee4c2c)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

The path goes from a single GPU to a datacenter: **read** the hardware, **program & profile** it with CUDA, make PyTorch **training fast**, **scale it across many GPUs**, and **design a frontier-scale training plan** in a competitive arena. Every answer is measured off real hardware — or free cloud GPUs — not read from a slide. Developed on a **2× NVIDIA RTX 5090** server, but the labs run on any NVIDIA GPU (and most run free on Colab / Kaggle).

---

## Contents

| Folder | Topic |
|---|---|
| [`01_gpu_infrastructure/`](01_gpu_infrastructure) | GPU introspection, the **VRAM wall**, the **roofline** & interconnect — plus the GPU→LLM worksheet and a performance problem set |
| [`02_cuda_and_profiling/`](02_cuda_and_profiling) | First **CUDA kernels** → making them fast → **profiling** (Nsight), a FlashAttention-on-LLM lab, and the `gpu_report` characterization tool |
| [`03_fast_training/`](03_fast_training) | Fast **PyTorch training** on one GPU — mixed precision, `torch.compile`, data pipelines, finding bottlenecks |
| [`04_multi_gpu/`](04_multi_gpu) | **DDP · FSDP · NCCL**, tensor-parallel from scratch, the **3D-parallelism** planning exercise, the cluster flowchart, and a one-click **Kaggle (2× T4)** notebook |
| [`05_challenge/`](05_challenge) | The **3D Parallelism Arena** — a live, auto-scored Gradio leaderboard — plus the full *Scale Out* assignment |

---

## Running the labs

- **No GPU?** Most labs run free on **Google Colab** or **Kaggle**. The multi-GPU labs run on Kaggle's free **2× T4**: open [`04_multi_gpu/Kaggle_Multi_GPU_Labs.ipynb`](04_multi_gpu/Kaggle_Multi_GPU_Labs.ipynb), set **Accelerator → GPU T4 ×2**, and **Run All**.
- **Multi-GPU** scripts launch with `torchrun --standalone --nproc_per_node=2 …`.
- **The challenge** needs only `pip install gradio` — no GPU at all: `python 05_challenge/arena_app.py`.

For a matching CUDA build (e.g. RTX 5090 / Blackwell `sm_120`):
```bash
pip install --upgrade "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Credits

Created for **Tasar'u** — the first AI Infrastructure bootcamp in Saudi Arabia — at **Tuwaiq Academy**.
Developed on 2× NVIDIA RTX 5090 (Blackwell).

## License
[MIT](LICENSE) — free to use, adapt, and teach with. Attribution appreciated.
