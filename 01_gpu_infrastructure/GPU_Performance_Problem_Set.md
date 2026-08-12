# 🧠 GPU Performance — Compute-It-Yourself Problem Set

*Use the Formula Sheet equations to compute real numbers about **your** GPU — and each answer hands you an insight you can't get by reading specs. Work them in order; they build.*

**By the end you'll be able to prove, with numbers, that:**
- Decode speed is set by **bandwidth**, not TFLOPs.
- The *same* GPU is memory-bound for decode and compute-bound for prefill.
- You must **batch ~100+ requests** before you actually use the compute you paid for.
- Quantization is a **throughput** lever, not just a memory one.
- On a no-NVLink box, **tensor-parallel communication can dwarf the compute**.

---

## Your constants (fill from `nvidia-smi` + Lab C)

| Symbol | Meaning | Example (RTX 5090) | Yours |
|---|---|---|---|
| **C** | VRAM capacity | 32 GB | `____` |
| **BW** | memory bandwidth | 1790 GB/s | `____` |
| **P_peak** | measured peak FP16 compute | ~210 TFLOP/s | `____` |
| **W_pow** | board power | ~575 W | `____` |

*Answer key uses the example column. Redo each with your own numbers.*

---

## A · Warm-ups — where the bottleneck lives

**Q1 — Decode is bandwidth, not FLOPs.**
A **13B** model in FP16. At batch 1, decode must re-read *all* weights from memory for **every** token.
**Find:** (a) weight bytes/token; (b) max decode `tok/s ≈ BW ÷ weight GB`.
**Insight:** `__________________________________________`

**Q2 — Arithmetic intensity of decode.**
Decode's core op is matrix × vector: `FLOPs = 2n²`, bytes read `≈ 2n²` (FP16 weights).
**Find:** AI = FLOPs ÷ bytes. Does it depend on model size *n*?
**Insight:** `__________________________________________`

**Q3 — Your GPU's ridge point.**
**Find:** `Ridge = P_peak ÷ BW` (FLOP/byte). Then classify: decode (AI≈1) and prefill matmul at n=4096 (AI≈n/2).
**Insight:** `__________________________________________`

---

## B · Core — throughput, batching & fit

**Q4 — The batching sweet spot.**
A batch of **B** decode requests reuses each weight read B times, so its intensity is `AI ≈ B FLOP/byte`.
**Find:** how many concurrent requests **B** are needed to reach your ridge point (become compute-bound)?
**Insight:** `__________________________________________`

**Q5 — The quantization double-win.**
Same 13B model, FP16 vs INT4.
**Find:** (a) weight GB each; (b) decode tok/s each (`BW ÷ weight GB`); (c) the speedup.
**Insight:** `__________________________________________`

**Q6 — KV cache vs weights.**
13B model, 40 layers, hidden 5120, FP16 KV (assume MHA): `KV/token = 2 × layers × hidden × 2 bytes`.
**Find:** KV per token, then total KV at **S = 32,000** tokens (batch 1). Compare to the 26 GB of weights.
**Insight:** `__________________________________________`

**Q7 — Prefill vs decode: which dominates the wait?**
Prompt = 2000 tokens, generate = 500 tokens. Prefill ≈ 8000 tok/s (compute-bound); decode = your Q1 answer.
**Find:** prefill time and decode time. Which is bigger?
**Insight:** `__________________________________________`

**Q8 — Will a 34B fit? (with KV)**
32 GB card, want **34B**, context 8k, batch 4, KV ≈ 0.2 MB/token (GQA).
**Find:** weights at FP16 and INT4; KV for this workload; total in INT4 (+2 GB overhead). Does it fit?
**Insight:** `__________________________________________`

---

## C · Multi-GPU & efficiency

**Q9 — The no-NVLink tax.**
Tensor-parallel across 2 GPUs over PCIe (measured link = **50 GB/s**). Per layer you all-reduce activations
`bytes = B×S×H×D = 1×2048×4096×2`. Model has 32 layers.
**Find:** communication time per layer, then per token (×32). (Decode does almost no compute per token.)
**Insight:** `__________________________________________`

**Q10 — Two 5090s: split it or double it?**
A 13B model fits on **one** 32 GB card (26 GB). You have 2× 5090, **no NVLink**.
**Find:** compare (A) tensor-split (pays the Q9 tax every token) vs (B) data-parallel (full model per card, 2 independent streams). Aggregate tok/s for B?
**Insight:** `__________________________________________`

**Q11 — Tokens per watt (stretch).**
Throughput 69 tok/s at 575 W.
**Find:** tokens/joule; energy for 1M tokens (kWh); cost at $0.05/kWh. What happens to tokens/watt when you batch?
**Insight:** `__________________________________________`

---
---

# ✅ Answer Key (worked)

**Q1.** weights = 13 × 2 = **26 GB**/token · decode ≈ 1790 ÷ 26 = **~69 tok/s**.
→ *Even a card doing hundreds of TFLOPs generates only ~69 tok/s. Decode speed = bandwidth ÷ model size — compute barely matters.*

**Q2.** AI = 2n² ÷ 2n² = **1 FLOP/byte**, independent of n.
→ *Decode is ~1 FLOP/byte for any model size → always deep in memory-bound territory.*

**Q3.** Ridge = 210×10¹² ÷ 1790×10⁹ = **~117 FLOP/byte**. Decode AI 1 ≪ 117 → **memory-bound**; prefill AI ≈ 2048 ≫ 117 → **compute-bound**.
→ *The same GPU is memory-bound for decode and compute-bound for prefill — one chip, two regimes.*

**Q4.** Need AI ≥ Ridge → **B ≥ ~117 requests**.
→ *You must batch ~117 concurrent requests before the compute you paid for is actually used. Below that, the Tensor Cores idle — this is why serving = batching.*

**Q5.** FP16 = 26 GB → 1790/26 = **69 tok/s**. INT4 = 6.5 GB → 1790/6.5 = **~275 tok/s**. Speedup = **~4×**.
→ *INT4 doesn't just fit ~4× smaller — it decodes ~4× faster, because decode reads ~4× fewer bytes/token. Quantization is a throughput lever.*

**Q6.** KV/token = 2×40×5120×2 = 819,200 B ≈ **0.8 MB**. At 32k tokens → 0.8 MB × 32768 ≈ **~26 GB** — *equal to the weights.*
→ *At long context the KV cache can match or exceed the model itself. Capacity planning that ignores KV will OOM. (GQA models cut this several-fold — check kv_heads.)*

**Q7.** Prefill = 2000/8000 = **0.25 s**. Decode = 500/69 = **~7.25 s**. Decode is **29× longer**.
→ *For chat, the wait is generation, not prompt-reading. Optimize decode (bandwidth, batching, quantization) — TTFT is already small.*

**Q8.** FP16 = 68 GB (**no fit**). INT4 = 17 GB. KV = 0.2 MB × 8192 × 4 = **6.6 GB**. Total INT4 = 17 + 6.6 + 2 = **~25.6 GB ≤ 32 → fits**.
→ *A 34B fits on 32 GB only in INT4, and only after budgeting KV for your batch × context — the fit depends on the workload, not just the model.*

**Q9.** bytes = 16.78 MB. Comm/layer = 16.78×10⁶ ÷ 50×10⁹ = **0.336 ms**. Per token ×32 = **~10.7 ms** of pure communication.
→ *~10.7 ms/token just moving data — far more than the tiny per-token compute. On PCIe, tensor-parallel is communication-bound.*

**Q10.** (A) tensor-split: pays ~10.7 ms/token → *slower than one GPU.* (B) data-parallel: two full models, ~0 cross-GPU traffic → **2 × 69 = ~138 tok/s** aggregate.
→ *If a model fits on one card, two cards should double **throughput** (data-parallel), not be glued into one bigger-but-slower model.*

**Q11.** 69 ÷ 575 = **0.12 tok/J**. 1M tokens → 1e6/0.12 ≈ 8.3 MJ = **~2.3 kWh** → **~$0.12** (energy only). Batching raises tokens/watt: more tokens for roughly the same power.
→ *Energy per token is tiny; the real cost is the GPU itself. But tokens/watt still ranks efficiency — and batching is the cheapest way to raise it.*

---
*Companion to the Tasar'u GPU Infrastructure Labs. Swap in your own measured numbers and the insights still hold — only the magnitudes change.*
