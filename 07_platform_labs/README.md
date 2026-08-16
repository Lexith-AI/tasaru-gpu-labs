# 07 · Platform Labs — NVIDIA Platform & Cert Prep

Four runnable notebooks that turn the **NVIDIA Platform** slides into things you *do* on
**free GPUs** (Google Colab, 1× T4 · Kaggle, 2× T4). Every lab is an **LLM workload** and ends
with reflection questions mapped to the deck and the **NCA-AIIO** exam.

| # | Lab | Open | Runs on | Objective |
|---|-----|------|---------|-----------|
| 1 | [GPU Partitioning Playground](01_gpu_partitioning) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/01_gpu_partitioning/01_gpu_partitioning_playground.ipynb) | Colab 1×T4 · Kaggle 2×T4 | MIG / vGPU / time-slicing via software stand-ins; whole-GPU assignment; split a model across 2 GPUs |
| 2 | [Inference & Optimization](02_inference_optimization) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/02_inference_optimization/02_inference_optimization.ipynb) | 1× T4 | Baseline → 4-bit quant (fit a 7B) → batching/KV → vLLM → benchmark |
| 3 | [QLoRA Finetune + Eval](03_qlora_finetune_eval) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/03_qlora_finetune_eval/03_qlora_finetune_eval.ipynb) | 1× T4 · Kaggle 2×T4 | QLoRA finetune a 4-bit base, then eval base-vs-tuned; scale to FSDP |
| 4 | [ONNX + Netron + Serving](04_onnx_netron_serving) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/04_onnx_netron_serving/04_onnx_netron_serving.ipynb) | CPU/GPU | ONNX export → Netron graph → ONNX Runtime vs PyTorch (speed + accuracy) |

## The partitioning story (Lab 1)
Real **MIG** and **vGPU** need an A100/H100 — not on free tiers. Lab 1 teaches the concepts with
what a T4 allows: **memory-fraction caps** (poor-man's MIG), a **time-slicing** concurrency
benchmark, **`CUDA_VISIBLE_DEVICES`** whole-GPU assignment, and **`device_map`** model splitting.
Each experiment maps back to the real NVIDIA feature.

## Notes
- Models are small so labs finish in minutes; each notebook marks where to **swap in a 7B**.
- **vLLM on T4** (Lab 2) is version-sensitive and degrades to a concept explanation if needed.
- Builds on [`04_multi_gpu/`](../04_multi_gpu) and [`06_finetuning/`](../06_finetuning) for the
  multi-GPU / FSDP pieces rather than duplicating them.

Each notebook has its own README with objective, where to run, and how to run.
