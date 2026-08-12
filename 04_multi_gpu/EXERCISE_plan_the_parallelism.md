# Exercise — Plan the Parallelism (pen & paper)

Given a model and a cluster, design the **3D-parallel plan**: choose **TP**, **PP**, **DP** and say *where each goes*. No code — just the reasoning from the lecture.

## The method
1. **Total model state** = params × **16 bytes** (mixed-precision Adam: fp16 weights + fp16 grads + fp32 master + Adam m,v).
2. **Per-GPU budget:** leave ~half of each GPU's VRAM for activations → target model state ≤ ~half the card.
3. **TP first**, capped by the NVLink domain (**≤ GPUs per node**, usually 8) — it's the chattiest axis.
4. **PP next**, add stages until each GPU's shard fits the budget.
5. **DP with the rest:** `total GPUs = TP × PP × DP`.
6. **Map it:** TP inside a node (NVLink) · PP across a few nodes · DP outermost.

## Worked example
**70B model · 64 GPUs (8 nodes × 8 · 80 GB each).**
- Model state = 70B × 16 B = **1,120 GB**.
- Budget ≈ 40 GB/GPU → need ≥ 1,120 / 40 = **28-way** sharding.
- TP = 8 (one node) → 1,120 / 8 = 140 GB per position (still too big).
- PP = 4 → 140 / 4 = **35 GB/GPU ✅**. One replica = TP × PP = **32 GPUs** (4 nodes).
- DP = 64 / 32 = **2**.
- **Answer: TP 8 · PP 4 · DP 2.** TP inside each node, PP across the 4 nodes of a replica, DP across the two replicas.

## Your turn (three problems)
1. **13B on 16 GPUs** (2 nodes × 8 · 40 GB).
2. **7B on 8 GPUs** (1 node · 24 GB).
3. **400B on 256 GPUs** (32 nodes × 8 · 80 GB).

For each: total model state, the per-GPU budget, then TP → PP → DP, and where each axis sits. More than one answer can be valid — **justify it** against memory and the NVLink cap.

---

<details>
<summary>Answer key (try first!)</summary>

**1) 13B on 16 GPUs (40 GB).** State = 13B × 16 = **208 GB**. Budget ≈ 20 GB/GPU → need ≥ ~11-way.
- TP = 8 → 208 / 8 = 26 GB/GPU (over 20). PP = 2 → 26 / 2 = **13 GB ✅**. Replica = 16 GPUs → **DP = 1**.
- **TP 8 · PP 2 · DP 1.** (Also valid: TP 8 · PP 1 · DP 2 *with activation checkpointing*, since 26 GB fits under 40 — trades a tighter memory margin for 2× throughput.)

**2) 7B on 8 GPUs (24 GB).** State = 7B × 16 = **112 GB**.
- TP = 8 → 112 / 8 = **14 GB/GPU ✅** — fits with just tensor parallel. PP = 1, DP = 1.
- **TP 8 · PP 1 · DP 1** (one replica fills the node). Simplest plan that fits.

**3) 400B on 256 GPUs (80 GB).** State = 400B × 16 = **6,400 GB**. Minimum sharding = 6,400 / 80 = 80-way just to *hold* it; with activations you need more.
- TP = 8 → 6,400 / 8 = 800 GB per position. PP = 32 → 800 / 32 = **25 GB/GPU ✅**. Replica = 8 × 32 = **256 GPUs** → **DP = 1**.
- **TP 8 · PP 32 · DP 1.** The whole cluster is one replica — a good illustration that at frontier scale you sometimes need *all* the GPUs just to make the model fit, with no room left for data parallel.

</details>
