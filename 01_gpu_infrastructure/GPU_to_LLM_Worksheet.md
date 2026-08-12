# 🧮 GPU → Best-Fit LLM · One-Page Worksheet

*Plug in three numbers about your GPU, do five short calculations, and walk away knowing the **largest model you can run** and **how fast it will generate**.*

---

## ① Read three numbers off your GPU

| Symbol | What | Where to get it | Your value |
|---|---|---|---|
| **C** | VRAM capacity (GB) | `nvidia-smi` → total memory | `________ GB` |
| **BW** | Memory bandwidth (GB/s) | datasheet, or **Lab C** | `________ GB/s` |
| **S**, **B** | context length, batch | your workload | `S = ______`, `B = ______` |

> *Convert TB/s → GB/s by ×1000 (e.g. 1.79 TB/s = 1790 GB/s).*

---

## ② The five calculations

**Formulas you'll use** (from the Formula Sheet):
`weight memory = P × b` · `decode tok/s ≈ BW ÷ weight GB` · `VRAM ≈ weights + KV + overhead`
where **b** = bytes/parameter → **FP16 = 2 · FP8 = 1 · INT4 = 0.5**

---

**Step 1 — Reserve room for KV cache + activations + overhead.**
Rule of thumb (single user, a few-thousand-token context):

> **R = larger of ( 4 GB , 20% × C )**   →   R = `________ GB`
> *(Long context > 8k, or batching? Reserve more — see the KV note below.)*

**Step 2 — Weight budget = what's left for the model's weights.**

> **W = C − R**   →   W = `________ GB`

**Step 3 — Biggest model that fits, at each precision.** (`P_max = W ÷ b`)

> FP16: W ÷ 2 = `______ B params`
> FP8 : W ÷ 1 = `______ B params`
> INT4: W ÷ 0.5 = `______ B params`

**Step 4 — Decode speed for the model you're eyeing.** (batch 1)
First its weight size: `weight GB = P × b`, then:

> **decode tok/s ≈ BW ÷ weight GB**   →   `________ tok/s`
> *Smaller model or lower precision → fewer bytes per token → faster generation.*

**Step 5 — The decision.**
Pick the **largest** model where **both** are true:
- ✅ weights fit in **W** (Step 3), and
- ✅ decode speed ≥ your comfort threshold (interactive chat ≈ **20–30 tok/s**).

> **My pick:** `______________________`  ·  precision `______`  ·  ~`______ tok/s`

---

## ③ Worked example — RTX 5090 (C = 32 GB, BW = 1790 GB/s)

| Step | Calculation | Result |
|---|---|---|
| 1 · Reserve | max(4, 0.20 × 32) | **6.4 GB** |
| 2 · Weight budget | 32 − 6.4 | **25.6 GB** |
| 3 · Fits (FP16) | 25.6 ÷ 2 | **~12–13 B** |
| 3 · Fits (FP8) | 25.6 ÷ 1 | **~25 B** |
| 3 · Fits (INT4) | 25.6 ÷ 0.5 | **~50 B** |
| 4 · Decode, 13B FP16 | 1790 ÷ 26 GB | **~69 tok/s** |
| 4 · Decode, 7B INT4 | 1790 ÷ 3.5 GB | **~511 tok/s** |

**Read-out:** On one 5090 you'd comfortably run a **~13B in FP16 at ~70 tok/s**, or push to a **~50B in INT4** if quality-at-4-bit holds — both far above the 20–30 tok/s interactive bar. For two 5090s with **no NVLink**, don't tensor-split — run a **full model per card** (data parallel) and double throughput, not model size.

---

## ④ Quick reference — weight memory (GB)

| Model | FP16 (×2) | FP8 (×1) | INT4 (×0.5) |
|---|---|---|---|
| **7B** | 14 | 7 | 3.5 |
| **13B** | 26 | 13 | 6.5 |
| **34B** | 68 | 34 | 17 |
| **70B** | 140 | 70 | 35 |

*Fits on 32 GB? Anything in the table **≤ your W** (Step 2).*

---

## ⑤ Rules of thumb

- **Capacity decides *if* it fits; bandwidth decides *how fast* it decodes.** Two different numbers, two different questions.
- **Quantize to fit *and* to go faster** — INT4 both shrinks the model ~4× and reads ~4× fewer bytes per token.
- **KV cache is the sneaky cost.** Precise reserve: `KV GB ≈ 2 × layers × kv_heads × head_dim × S × B × 2 bytes ÷ 1e9`. It grows with context **and** batch — long-context serving OOMs even when the model fits.
- **Decode is memory-bound; prefill is compute-bound.** Batching speeds up decode (one weight read serves the whole batch) but won't exceed your bandwidth ceiling.
- **No NVLink?** Prefer **data / pipeline** parallelism over tensor parallelism — the PCIe link is ~30–50× slower than on-card memory.

---
*Companion to the Tasar'u GPU Infrastructure Labs · fill it in with your own numbers.*
