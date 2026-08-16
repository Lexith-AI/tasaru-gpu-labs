# 🎯 QLoRA Finetune + Eval Lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/03_qlora_finetune_eval/03_qlora_finetune_eval.ipynb)

## Objective
Fine-tune an LLM on a free GPU with **QLoRA** (4-bit frozen base + tiny **LoRA** adapters), then
**evaluate it properly** — because "it trained" is not "it got better."

By the end you can:
- load a model in **4-bit** and attach **LoRA** adapters (train <1% of params)
- run a short **SFT** finetune on an instruction dataset
- **evaluate base vs tuned** with **perplexity** *and* a behavior check — and explain why
  perplexity alone isn't enough
- see how to **scale to FSDP** across 2 GPUs for a full finetune

## Where to run
- **Google Colab / Kaggle**, single **T4** → sections 1–5.
- **Kaggle 2× T4** → section 6 (FSDP full-finetune, via the `06_finetuning` launch script).

## How to run
1. Click **Open in Colab** above.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Run all cells** (~10–15 min — the training run is intentionally short so it completes).

## Maps to
Slides: *Evaluation metrics*, *Software stack (NeMo / PEFT)*, *Multi-GPU (FSDP)*.
Exam domain: **Essential AI Knowledge**.

## What's inside
| § | You run | Lesson |
|---|---------|--------|
| 1–2 | 4-bit base + LoRA adapters | QLoRA = fit + cheap to train |
| 3–4 | short SFT on Alpaca | supervised fine-tuning |
| 5 | perplexity before/after + a generation | evaluation done honestly |
| 6 | FSDP-on-2×T4 pointer | split the memory across GPUs |

> **Scale it up:** set `BASE = "Qwen/Qwen2.5-7B-Instruct"` for the real 7B QLoRA — same code, slower.
> Builds on [`06_finetuning/`](../../06_finetuning) (the launch-time-parallel SFT script).
