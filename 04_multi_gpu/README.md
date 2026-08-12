# Multi-GPU Labs — DDP · FSDP · NCCL (2×RTX 5090)

Three runnable labs for the **Scale Out** week. They run on the **2×5090 server** (or any multi-GPU box — RunPod A100/H100 works too).

> **Why scripts, not notebooks?** Distributed training needs **one process per GPU**, launched with `torchrun`. That doesn't work cleanly inside Jupyter — so these are `.py` files, exactly how it's done for real.

**Files:** `model.py` (shared mini-GPT) · `train_ddp.py` · `train_fsdp.py` · `bench_nccl.py`
**Needs:** the Blackwell/cu128 PyTorch build already on the server. Check your GPUs first with `nvidia-smi`.

---

## Lab H — DDP: does 2 GPUs give you 2×?

Same model on every GPU, split the data, all-reduce the gradients. Run it on **1 GPU**, then **2**, keeping the **per-GPU batch fixed**, and compute the scaling efficiency.

```bash
cd Multi_GPU_Labs
torchrun --standalone --nproc_per_node=1 train_ddp.py    # baseline
torchrun --standalone --nproc_per_node=2 train_ddp.py    # scaled
```

**Compute:** `efficiency = tokens/sec(2 GPU) ÷ (2 × tokens/sec(1 GPU))`.

**What to observe / answer**
1. What efficiency did you get? Why is it below 1.0?
2. That gap is the **gradient all-reduce over PCIe**. What would raise it — bigger model (more compute per sync) or bigger batch?
3. Peak VRAM per GPU barely changes vs 1 GPU — why? (DDP copies the *whole* model to each card.)

*Expect ~0.8–0.9 on the 2×5090 — no NVLink, so the sync crosses PCIe.*

---

## Lab I — FSDP: fit a model that's too big for one card

The default config is **~2.7B params** → ~44 GB for full training (16 B/param). That **OOMs a single 32 GB 5090**. FSDP shards params + grads + optimizer across both GPUs so it fits.

```bash
# 1) prove it doesn't fit on one card (expect CUDA OOM):
torchrun --standalone --nproc_per_node=1 train_fsdp.py

# 2) now shard it across both GPUs:
torchrun --standalone --nproc_per_node=2 train_fsdp.py --ckpt
```

**What to observe / answer**
1. Confirm the 1-GPU run **OOMs**, and the 2-GPU run **trains**.
2. What is the **peak VRAM per GPU** with FSDP? Compare it to the ~44 GB the full model would need.
3. What three things did FSDP shard? (params, gradients, optimizer state = ZeRO-3 / `FULL_SHARD`.)
4. Turn `--ckpt` off — does it still fit? (Activation checkpointing buys memory for ~20–30% more compute.)
5. Bump `--dim`/`--layers` until even 2 GPUs OOM — that's the point where you'd need *more* GPUs.

---

## Lab J — measure the real interconnect

An all-reduce micro-benchmark (pure PyTorch) that prints your **bus bandwidth** — the true GPU-to-GPU link speed.

```bash
torchrun --standalone --nproc_per_node=2 bench_nccl.py
```

**What to observe / answer**
1. What **busbw** do you see at 256–512 MB? That's your interconnect ceiling.
2. PCIe 5.0 ×16 peaks at ~64 GB/s; NVLink/NVSwitch would show **hundreds**. How far below the ~900 GB/s of a datacenter GPU are you?
3. Tie it back to Lab H: this ceiling is *why* DDP scaling isn't a clean 2×.

---

## Lab K — Tensor Parallelism from scratch

Split one MLP across 2 GPUs **by hand** (column-parallel first layer + row-parallel second layer) and prove it equals the single-GPU result — the exact Megatron pattern.

```bash
torchrun --standalone --nproc_per_node=2 tp_from_scratch.py
```

**What to observe / answer**
1. Each GPU stores only **half** of W1 and W2 — confirm the printed shapes.
2. The whole MLP needs exactly **one** all-reduce. Why does column-parallel need *no* communication, but row-parallel needs one?
3. The error vs the single-GPU MLP is ~1e-3 (float noise) → the math is identical. Why isn't it exactly 0? (summation order differs.)

## Exercise — Plan the Parallelism (pen & paper)

See **`EXERCISE_plan_the_parallelism.md`** — design the TP/PP/DP plan for a given model + cluster. Worked 70B example + three problems with an answer key. No GPU needed; it drills the sizing rule from the lecture.

## Tuning notes (instructor)

- Sizes are `argparse` flags — adjust `--dim --layers --heads --batch` to your VRAM.
- If **DDP** OOMs on one card, lower `--dim`/`--batch`. If it's too fast to show a sync gap, raise `--batch`.
- If **FSDP** OOMs even on 2 GPUs, add `--ckpt` and/or lower `--batch` to 1; to force the "need more GPUs" lesson, raise `--dim`/`--layers`.
- Data is random tokens — throughput and memory are the lesson here, not the loss value.
- FSDP uses the `FullyShardedDataParallel` wrapper (`FULL_SHARD` = ZeRO-3); the lecture's `fully_shard(...)` one-liner is the newer FSDP2 API for the same idea.
