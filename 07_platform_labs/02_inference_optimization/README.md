# ⚡ Inference & Optimization Lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/02_inference_optimization/02_inference_optimization.ipynb)

## Objective
Walk the real serving path: **baseline → quantize → batch → serve → benchmark.** You'll make a
model that *doesn't fit* on a T4 fit, and see why production uses a dedicated serving engine.

By the end you can:
- measure inference **throughput (tok/s)** and **peak memory**
- use **4-bit (NF4) quantization** to run a **7B model on a single 16 GB T4**
- show what **batching** and the **KV cache** do to throughput/latency
- serve with **vLLM** (continuous batching) and compare to a plain 🤗 loop

## Where to run
- **Google Colab** or **Kaggle**, a single **T4 (16 GB)** is enough. GPU required.

## How to run
1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Run all cells** (~8–12 min, mostly model downloads).
4. The vLLM section (⚠️ Turing-sensitive) degrades to a concept explanation if it won't install —
   that's expected; the rest of the lab is unaffected.

## Maps to
Slides: *Software stack — TensorRT / Triton / NIM*, *Partitioning* (quantization = fit-to-GPU).
Exam domain: **Essential AI Knowledge** + **AI Operations**.

## What's inside
| § | You run | Lesson |
|---|---------|--------|
| 1 | fp16 baseline generation | tok/s + memory |
| 2 | 4-bit NF4 → **7B on a T4** | quantization (the TensorRT idea) |
| 3 | throughput vs batch size; KV-cache on/off | in-flight batching / paged KV |
| 4 | vLLM offline engine | Triton / NIM continuous batching |
| 5 | benchmark table | AI-Operations mini-report |

> **Scale it up:** swap the base model for `Qwen/Qwen2.5-7B-Instruct` where the notebook marks it.
