#!/usr/bin/env python3
"""
gpu_report.py — characterize one GPU and emit a CSV + printed report.

Runs a battery of measurements on cuda:<idx> (default 0):
  1. peak matmul throughput across precisions (fp32 / bf16 / fp16)
  2. memory bandwidth  +  the ridge point
  3. matrix-vector vs matrix-matrix  (memory-bound vs compute-bound)
  4. LIVE nvidia-smi dmon during a compute vs a memory workload
  5. OOM — allocate until it hits the VRAM wall
  6. real LLM serving insights, derived from the measured numbers

Writes  gpu_report_<gpu>_<timestamp>.csv  for students to analyze, and prints
a human-readable summary.

Usage:   python gpu_report.py [gpu_index]      e.g.  python gpu_report.py 0
"""
import sys, os, csv, time, platform, subprocess, threading
from datetime import datetime

import torch

# ---------------------------------------------------------------- setup
IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
assert torch.cuda.is_available(), "No CUDA GPU visible."
torch.cuda.set_device(IDX)
P = torch.cuda.get_device_properties(IDX)
DEV = f"cuda:{IDX}"
CAP_GB = P.total_memory / 1e9

# Disable TF32 so 'fp32' is a fair full-precision number (Ampere+ uses TF32 by default).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sync():
    torch.cuda.synchronize(IDX)


def time_op(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t) / iters      # seconds/iter


rows = []


def add(**kw):
    row = dict(category="", experiment="", dtype="", n="", time_ms="",
               GFLOP_s="", GB_s="", sm_pct="", mem_pct="", pct_of_peak="", note="")
    row.update(kw)
    rows.append(row)


DTYPES = [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]

print(f"\n=== characterizing {P.name}  ({CAP_GB:.1f} GB, sm_{P.major}{P.minor}, "
      f"{P.multi_processor_count} SMs)  on {DEV} ===\n")

# ---------------------------------------------------------------- 1. compute
print("[1/6] matmul throughput across sizes & precisions ...")
for n in (2048, 4096, 8192):
    for dname, dt in DTYPES:
        try:
            a = torch.randn(n, n, device=DEV, dtype=dt)
            b = torch.randn(n, n, device=DEV, dtype=dt)
            s = time_op(lambda: a @ b)
            add(category="compute", experiment="matmul", dtype=dname, n=n,
                time_ms=round(s * 1e3, 3), GFLOP_s=round(2 * n ** 3 / s / 1e9, 1))
            del a, b
        except torch.cuda.OutOfMemoryError:
            add(category="compute", experiment="matmul", dtype=dname, n=n, note="OOM")
        torch.cuda.empty_cache()

peak = {d: max([r["GFLOP_s"] for r in rows if r["dtype"] == d and r["GFLOP_s"] != ""], default=0)
        for d in ("fp32", "bf16", "fp16")}

# ---------------------------------------------------------------- 2. bandwidth + ridge
print("[2/6] memory bandwidth + ridge point ...")
NB = 2_000_000_000
x = torch.empty(NB // 2, dtype=torch.float16, device=DEV)
s = time_op(lambda: x.clone())
BW = 2 * NB / s / 1e9
add(category="memory", experiment="bandwidth_copy", dtype="fp16", GB_s=round(BW, 1), note="read+write")
del x
torch.cuda.empty_cache()
RIDGE = peak["fp16"] / BW if BW else 0
add(category="roofline", experiment="ridge_point", pct_of_peak=round(RIDGE, 1),
    note="FLOP/byte  (AI below = memory-bound, above = compute-bound)")

# ---------------------------------------------------------------- 3. bound test
print("[3/6] matrix-vector vs matrix-matrix (the bound test) ...")
n = 8192
a = torch.randn(n, n, device=DEV, dtype=torch.float16)
xv = torch.randn(n, 1, device=DEV, dtype=torch.float16)
s = time_op(lambda: a @ xv)
gmv = 2 * n * n / s / 1e9
add(category="roofline", experiment="matrix_vector", dtype="fp16", n=n, time_ms=round(s * 1e3, 4),
    GFLOP_s=round(gmv, 1), pct_of_peak=round(100 * gmv / peak["fp16"], 2), note="memory-bound (decode)")
b = torch.randn(n, n, device=DEV, dtype=torch.float16)
s = time_op(lambda: a @ b)
gmm = 2 * n ** 3 / s / 1e9
add(category="roofline", experiment="matrix_matrix", dtype="fp16", n=n, time_ms=round(s * 1e3, 3),
    GFLOP_s=round(gmm, 1), pct_of_peak=round(100 * gmm / peak["fp16"], 1), note="compute-bound (prefill)")
del a, b, xv
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 4. live dmon
print("[4/6] live nvidia-smi dmon during compute vs memory work ...")


def capture_dmon(workload, samples=5):
    """Run `workload` in a background thread while nvidia-smi dmon samples this GPU."""
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            workload()

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    time.sleep(1.5)                                  # let it ramp
    sm_vals, mem_vals = [], []
    try:
        r = subprocess.run(["nvidia-smi", "dmon", "-i", str(IDX), "-s", "u", "-c", str(samples), "-d", "1"],
                           capture_output=True, text=True, timeout=samples + 15)
        for ln in r.stdout.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            try:
                sm_vals.append(int(p[1])); mem_vals.append(int(p[2]))
            except (IndexError, ValueError):
                continue
    except Exception as e:
        stop.set(); th.join(timeout=2)
        return None, None, str(e)
    stop.set(); th.join(timeout=2)
    if not sm_vals:
        return None, None, "no dmon samples parsed"
    return sum(sm_vals) / len(sm_vals), sum(mem_vals) / len(mem_vals), None


try:
    Ac = torch.randn(8192, 8192, device=DEV, dtype=torch.float16)
    Bc = torch.randn(8192, 8192, device=DEV, dtype=torch.float16)
    Xm = torch.empty(1_000_000_000 // 2, dtype=torch.float16, device=DEV)

    def compute_load():
        for _ in range(15):
            _ = Ac @ Bc
        sync()

    def memory_load():
        for _ in range(60):
            _ = Xm.clone()
        sync()

    sm_c, mem_c, err_c = capture_dmon(compute_load)
    sm_m, mem_m, err_m = capture_dmon(memory_load)
    del Ac, Bc, Xm
    torch.cuda.empty_cache()

    if err_c or err_m:
        DMON_OK = False
        add(category="dmon", experiment="live", note=f"skipped: {err_c or err_m}")
    else:
        DMON_OK = True
        add(category="dmon", experiment="compute_workload", sm_pct=round(sm_c), mem_pct=round(mem_c),
            note="matmul loop -> SMs busy")
        add(category="dmon", experiment="memory_workload", sm_pct=round(sm_m), mem_pct=round(mem_m),
            note="copy loop -> memory controller busy")
except Exception as e:
    DMON_OK = False
    add(category="dmon", experiment="live", note=f"skipped: {e}")
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- 5. OOM: the VRAM wall
print("[5/6] finding the VRAM wall (OOM) ...")
torch.cuda.empty_cache()
blocks, wall = [], 0.0
try:
    while True:
        blocks.append(torch.empty(1024 * 1024 * 1024 // 2, dtype=torch.float16, device=DEV))  # 1 GB each
        wall = torch.cuda.memory_allocated(IDX) / 1e9
except torch.cuda.OutOfMemoryError:
    pass
finally:
    del blocks
    torch.cuda.empty_cache()
add(category="oom", experiment="vram_wall", GB_s="", pct_of_peak=round(wall, 1),
    note=f"allocatable before OOM, of {CAP_GB:.0f} GB total")

# ---------------------------------------------------------------- 6. LLM serving insights
reserve = max(4.0, 0.2 * CAP_GB)
usable = CAP_GB - reserve


def decode_toks(weight_gb):
    return BW / weight_gb if weight_gb else 0


cases = [
    ("13B FP16", 13 * 2), ("13B INT4", 13 * 0.5),
    ("7B FP16", 7 * 2), ("7B INT4", 7 * 0.5),
]
for name, wgb in cases:
    add(category="usecase", experiment="decode_tok_s", note=name,
        pct_of_peak=round(decode_toks(wgb)), GB_s="")
add(category="usecase", experiment="max_model_fp16_B", pct_of_peak=round(usable / 2, 1),
    note=f"largest fp16 model (params, B) in {usable:.0f} GB usable")
add(category="usecase", experiment="max_model_int4_B", pct_of_peak=round(usable / 0.5, 1),
    note="largest int4 model (params, B)")
add(category="usecase", experiment="batch_sweet_spot", pct_of_peak=round(RIDGE),
    note="concurrent requests to reach the ridge (compute-bound)")

# ---------------------------------------------------------------- write CSV
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe = P.name.replace(" ", "_").replace("/", "-")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"gpu_report_{safe}_{stamp}.csv")
with open(out, "w", newline="") as f:
    f.write(f"# GPU,{P.name}\n# VRAM_GB,{CAP_GB:.1f}\n# SMs,{P.multi_processor_count}\n")
    f.write(f"# compute_capability,sm_{P.major}{P.minor}\n# torch,{torch.__version__}\n")
    f.write(f"# CUDA,{torch.version.cuda}\n# host,{platform.node()}\n# timestamp,{stamp}\n")
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------- printed insights
sp = peak["fp16"] / peak["fp32"] if peak["fp32"] else 0
line = "=" * 64
print("\n" + line)
print(f"  INSIGHTS — {P.name}")
print(line)
print(f"  Peak matmul   FP32 {peak['fp32']/1000:6.1f} TFLOP/s")
print(f"                BF16 {peak['bf16']/1000:6.1f} TFLOP/s")
print(f"                FP16 {peak['fp16']/1000:6.1f} TFLOP/s   ({sp:.1f}x over FP32 — Tensor Cores)")
print(f"  Memory bandwidth   {BW:6.0f} GB/s")
print(f"  Ridge point        {RIDGE:6.0f} FLOP/byte")
print(f"  matrix-vector      {100*gmv/peak['fp16']:5.1f}% of peak   -> MEMORY-bound  (decode)")
print(f"  matrix-matrix      {100*gmm/peak['fp16']:5.1f}% of peak   -> COMPUTE-bound (prefill)")
if DMON_OK:
    print("-" * 64)
    print("  LIVE dmon — same GPU, two workloads:")
    print(f"    compute (matmul) :  sm {round(sm_c):3d}%   mem {round(mem_c):3d}%   <- cores busy")
    print(f"    memory  (copy)   :  sm {round(sm_m):3d}%   mem {round(mem_m):3d}%   <- bandwidth busy")
    print("    (this is 'compute-bound vs memory-bound', seen live)")
print("-" * 64)
print(f"  VRAM wall          {wall:5.1f} GB allocatable before OOM (of {CAP_GB:.0f} GB)")
print("-" * 64)
print("  Serving an LLM on this card (from the measured numbers):")
print(f"    decode 13B FP16  ~{decode_toks(26):4.0f} tok/s    |  13B INT4 ~{decode_toks(6.5):4.0f} tok/s")
print(f"    decode  7B FP16  ~{decode_toks(14):4.0f} tok/s    |   7B INT4 ~{decode_toks(3.5):4.0f} tok/s")
print(f"    fits (fp16) up to ~{usable/2:.0f}B params  |  (int4) up to ~{usable/0.5:.0f}B params")
print(f"    batch ~{round(RIDGE)} requests to reach the ridge (use the compute you paid for)")
print(line)
print(f"\nCSV written:\n  {out}\n")
print("Hand the CSV to students. Analysis prompts are in the README.\n")
