# 🧩 GPU Partitioning Playground

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/01_gpu_partitioning/01_gpu_partitioning_playground.ipynb)

## Objective
Learn **how one GPU is shared and how a model is split across many** — the ideas behind
**MIG**, **vGPU**, **time-slicing**, and **model parallelism** from the NVIDIA Platform module.
Real MIG/vGPU need an A100/H100, so this lab runs faithful **software stand-ins** on a free
T4 and shows exactly how each maps back to the real feature.

By the end you can explain, with numbers you produced:
- how a memory cap isolates a "GPU instance" (poor-man's MIG) — and what it *doesn't* guarantee
- the **time-slicing tax**: why 2× clients ≠ 2× throughput on one GPU
- **whole-GPU assignment** with `CUDA_VISIBLE_DEVICES` (what a scheduler like Run:ai does)
- **splitting one model across two GPUs** with `device_map`

## Where to run
- **Google Colab** (1× T4) → sections 1–3.
- **Kaggle** with **Accelerator → GPU T4 ×2** → also unlocks sections 4–5 (two-GPU work).

## How to run
1. Click **Open in Colab** above (or upload the notebook to Kaggle).
2. Colab: **Runtime → Change runtime type → T4 GPU**. Kaggle: **Settings → Accelerator → GPU T4 ×2**.
3. **Run all cells** top to bottom (~5–8 min). Sections 4–5 auto-skip on a single GPU.

## Maps to
Slides: *Partitioning — MIG / vGPU / time-slicing*, *Compute platforms*, *Orchestration (Run:ai)*.
Exam domain: **AI Infrastructure** (+ a bit of **AI Operations**).

## What's inside
| § | You run | Real feature |
|---|---------|--------------|
| 2 | memory-fraction cap + a deliberate OOM | **MIG** (hardware isolation) |
| 2b | 4 capped tenants on one GPU | **MIG / vGPU** multi-tenant |
| 3 | 1→2→4 client throughput/latency curve | **time-slicing** |
| 4 | `CUDA_VISIBLE_DEVICES` per tenant *(2 GPUs)* | **vGPU / Run:ai / K8s device plugin** |
| 5 | `device_map` splitting a model *(2 GPUs)* | **tensor/pipeline parallelism** |

Ends with reflection questions that map each experiment back to the exam.
