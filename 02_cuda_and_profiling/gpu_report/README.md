# GPU Report — characterize your card, then read the data

`gpu_report.py` runs a battery of measurements on **one GPU** and writes a **CSV** (plus a printed summary). Run it on the server; hand the CSV to students to analyze.

## Run it (on the 5090 server, WSL)

```bash
python gpu_report.py 0      # cuda:0   (use 1 for the second card)
```

~1–2 minutes. Produces a printed **INSIGHTS** block and **`gpu_report_<GPU>_<timestamp>.csv`** in this folder. Needs `torch` (the Blackwell/cu128 build) and `nvidia-smi` on PATH.

## What it measures

| Category | Experiment | Key column | Ties to |
|---|---|---|---|
| `compute` | matmul at 2048/4096/8192 × fp32/bf16/fp16 | `GFLOP_s` | Tensor Cores, precision |
| `memory` | bandwidth (copy) | `GB_s` | HBM bandwidth |
| `roofline` | ridge point | `pct_of_peak` (FLOP/byte) | the roofline |
| `roofline` | matrix-vector vs matrix-matrix | `pct_of_peak`, `note` | decode vs prefill |
| `dmon` | **live `nvidia-smi dmon`** during a compute vs a memory workload | `sm_pct`, `mem_pct` | compute- vs memory-bound, live |
| `oom` | **allocate until the VRAM wall** | `pct_of_peak` (GB) | the VRAM wall |
| `usecase` | **LLM serving estimates** (decode tok/s, largest model, batch sweet spot) | `pct_of_peak` | real serving |

The CSV starts with `#` metadata lines (GPU, VRAM, SMs, driver) — skip them: `pd.read_csv(path, comment="#")`.

## The two highlights

- **Live dmon** runs a matmul loop, then a copy loop, and captures `sm%` / `mem%` for each. You'll see the matmul light up **sm%** (compute-bound) and the copy light up **mem%** (memory-bound) — the exact distinction from the profiling lecture, measured on your card.
- **The VRAM wall** allocates 1 GB blocks until it OOMs, and reports how much it fit — the hard ceiling from the memory lab.

## Student analysis prompts (answer from the CSV / report)

1. **Peak compute:** highest fp16 `GFLOP_s` → TFLOP/s. What's the **fp16-vs-fp32 speedup** (Tensor Cores)?
2. **Bandwidth:** what `GB_s`? How close to the datasheet?
3. **Ridge point:** read it off. Above it = compute-bound, below = memory-bound.
4. **Bound test:** matrix-vector's **% of peak** vs matrix-matrix's. Which is decode, which is prefill?
5. **Live dmon:** for the compute workload, was `sm%` or `mem%` higher? For the memory workload? Explain what each proves.
6. **VRAM wall:** how many GB did it allocate before OOM, out of total? Where did the rest go?
7. **Serving:** from `usecase` rows — what decode tok/s for 13B FP16 vs 7B INT4? Largest model that fits at fp16 vs int4? How many concurrent requests to reach the ridge?
8. **The conclusion:** in one sentence, why is **batching** the key to using this GPU's compute during generation?

## Quick load in pandas

```python
import pandas as pd
df = pd.read_csv("gpu_report_<...>.csv", comment="#")
df[df.category == "compute"].pivot(index="n", columns="dtype", values="GFLOP_s")   # precision ladder
df[df.category == "dmon"]        # sm% vs mem% for each workload
df[df.category == "usecase"]     # serving estimates
```

*Companion to the Tasar'u GPU Infrastructure Labs.*
