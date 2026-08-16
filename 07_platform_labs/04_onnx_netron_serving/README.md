# 🔍 ONNX + Netron + Serving Lab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexith-AI/tasaru-gpu-labs/blob/main/07_platform_labs/04_onnx_netron_serving/04_onnx_netron_serving.ipynb)

## Objective
Move a model out of PyTorch into a **portable, optimizable form** and prove the export is
faithful. Understand where **ONNX**, **ONNX Runtime / TensorRT**, and **Netron** sit in the stack.

By the end you can:
- **export** a PyTorch model to **ONNX** (🤗 Optimum)
- **visualize** the model graph with **Netron**
- run **ONNX Runtime vs PyTorch** and compare **speed** and **agreement**
- prove the export is lossless with an **accuracy** eval (same predictions, faster engine)

## Where to run
- **Google Colab / Kaggle**. **GPU optional** — this lab even runs on CPU.

## How to run
1. Click **Open in Colab** above.
2. (Optional) enable a GPU; not required.
3. **Run all cells** (~3–5 min).
4. To *see* the graph: download `onnx_model/model.onnx` and drag it onto **https://netron.app**.

## Maps to
Slides: *Essential AI — ONNX / interchange*, *Software stack — TensorRT / optimize-then-serve*.
Exam domain: **Essential AI Knowledge**.

## What's inside
| § | You run | Lesson |
|---|---------|--------|
| 1 | PyTorch → ONNX export | the interchange format |
| 2 | graph summary + Netron | inspect the model |
| 3 | ONNX Runtime vs PyTorch | optimize-then-serve, speed + agreement |
| 4 | accuracy on SST-2 | a faithful export changes the *engine*, not the *model* |
