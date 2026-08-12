# Lab A — Introspection: *Read Your GPU*
---

## Goal

In the lecture you learned to read a GPU by four numbers: **capacity · bandwidth · precision floor · link**.

This lab makes those numbers *real* — you'll pull them straight off your own hardware with the terminal, and confirm the two things that shape every multi-GPU decision on this box:

1. Each 5090 has **32 GB** of memory and a **PCIe 5.0 x16** link.
2. Communication:


> ℹ️ The datacenter cards from the lecture (A100/H100/H200) use **HBM**. 
> The consumer 5090 uses **GDDR7**. The *technology* differs, but the *two numbers that matter are identical*: **how much** (capacity) and **how fast** (bandwidth).
---

## 0. Sanity check — does WSL even see the GPUs?

```bash
nvidia-smi
```

- **What it does:** NVIDIA System Management Interface. One-shot snapshot of every GPU — model, driver, CUDA version, memory used/total, utilization, temperature, power, and any running processes.

- **Why it matters:** this is your proof that the WSL2 → Windows → driver passthrough works and that **both** cards are visible to Linux.

- **Two GPU rows** (`0` and `1`) → both 5090s are up.
- **`CUDA Version`** (top-right) → you'll need this to match toolkits.
- **`Memory-Usage`** → should read roughly `0MiB / 32768MiB` when idle. That `32768MiB` **is your capacity number.**
- **`Driver Version`**.

---

## 1. List the GPUs and their unique IDs

```bash
nvidia-smi -L
```

- **What it does:** prints one clean line per GPU with its exact product name and a unique **UUID**.
- **Why it matters:** confirms the *count* and *exact model*, and the UUIDs are how you'll pin a job to a specific card later (`CUDA_VISIBLE_DEVICES`).
- **Focus on:** two lines, both `NVIDIA GeForce RTX 5090`. Note that GPU `0` and GPU `1` are distinct UUIDs — same model, two physical chips.

---

## 2. Read every column at once:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap,pcie.link.gen.max,pcie.link.width.max,power.max_limit --format=csv
```

- **What it does:** pulls only the fields you care about, as a tidy CSV.
- **Why it matters:** this *is* the "read every column" slide, generated from your own silicon — capacity, compute capability (architecture), PCIe link, and power ceiling in one row per GPU.
- **Focus on:**
  - **`memory.total`** ≈ `32768 MiB` → **capacity**.
  - **`compute_cap`** = `12.0` → Blackwell (`sm_120`). This is the architecture tag your CUDA code compiles against in Week 3.
  - **`pcie.link.gen.max`** = `5`, **`pcie.link.width.max`** = `16` → **PCIe 5.0 x16**, the fastest lane this card supports.
  - **`power.max_limit`** → the card's power ceiling in watts (ties to the Week 2 power/cooling topic).

---

## 3. Full hardware dump for one GPU

```bash
nvidia-smi -q -i 0
```

- **What it does:** dumps *everything* NVML knows about GPU `0` — memory, clocks, power limits, PCIe generation/width, ECC state, temperatures.
- **Why it matters:** the deep reference when you need a detail the summary view hides.
- **Focus on** these sub-sections:
  - **`FB Memory Usage → Total`** = `32768 MiB` (FB = "frame buffer" = your VRAM).
  - **`GPU Link Info → PCIe Generation → Max`** = `5` and **`Link Width → Max`** = `16x`.
  - **`Max Clocks → Memory`** — the memory clock; combined with the bus width it sets bandwidth (we measure real bandwidth in Lab C).
  - **`Power Readings → Max Power Limit`**.

Want just one slice instead of the whole dump? Use `-d`:

```bash
nvidia-smi -q -i 0 -d MEMORY      # memory only
nvidia-smi -q -i 0 -d CLOCK       # clocks only
nvidia-smi -q -i 0 -d POWER       # power only
nvidia-smi -q -i 0 -d PCIE        # PCIe link only
```

---

## 4. Topology — how do the two GPUs actually talk? ⭐

```bash
nvidia-smi topo -m
```

- **What it does:** prints a matrix of how every GPU connects to every other GPU (and to the CPU/NICs).
- **Why it matters:** **this is the most important command in the lab.** It answers the question the whole "one GPU → many" topic hinges on: are your two 5090s joined by fast **NVLink**, or only by **PCIe**? That single fact decides how costly it is to split a model across both cards.
- **Focus on:** the cell where **`GPU0` meets `GPU1`**. Decode it with the legend printed underneath:

  | Symbol | Meaning | Speed |
  |---|---|---|
  | `NV#` | NVLink (# = number of links) | 🟢 fastest |
  | `PIX` | one PCIe bridge | |
  | `PXB` | multiple PCIe bridges | |
  | `PHB` | PCIe host bridge (through the CPU) | |
  | `NODE` | same NUMA node, across host bridges | |
  | `SYS` | across CPU sockets / NUMA nodes | 🔴 slowest |

- **What you should see on this box:** something like **`PHB`**, **`NODE`**, or **`SYS`** — **not** `NV#`. That confirms **no NVLink**: the 5090s exchange data over PCIe/through the CPU. Consequence: any tensor- or pipeline-parallel workload pays a real communication tax, and the interconnect — not the cores — becomes the bottleneck. (You'll benchmark exactly how much in Lab C.)

---

## 5. Is the PCIe link running at full speed *right now*?

```bash
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
```

- **What it does:** shows the **current** vs **maximum** PCIe generation and width.
- **Why it matters:** cards downshift the PCIe link to save power when idle, and a bad slot/riser can silently cap you below x16. Current-vs-max tells you whether you're actually getting PCIe 5.0 x16 or being throttled.
- **Focus on:** `gen.current` may read `1` or `2` at idle — that's **normal power saving**. Re-run it while a GPU job is active and it should ramp to `gen.current = 5`, `width.current = 16`. If it never reaches max under load, you have a hardware/slot problem.

---

## 6. Watch the GPU live (you'll use this in every later lab)

```bash
nvidia-smi dmon
```
*(press `Ctrl-C` to stop)*

- **What it does:** streams a new line **every second** with per-GPU utilization, memory-controller activity, power, clocks, and temperature.
- **Why it matters:** this is your live dashboard for the VRAM wall (Lab B) and for *seeing* compute-bound vs memory-bound behavior (Lab C).
- **Focus on:**
  - **`sm`** = % of time the compute cores (SMs) are busy → high during **prefill / compute-bound** work.
  - **`mem`** = % memory-**controller** (bandwidth) utilization → high during **decode / memory-bound** work. (Note: this is *bandwidth* busy-ness, **not** capacity.)
  - **`pwr`** and **`mclk`/`sm clk`** → power and clocks rising under load.

Prefer the classic view refreshing in place:

```bash
watch -n 1 nvidia-smi
```

---

## 7. Who is holding the memory? (debugging OOM later)

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

- **What it does:** lists every process currently using GPU memory and how much.
- **Why it matters:** in Lab B you'll hit out-of-memory errors on purpose. This is how you find a leftover process still squatting on VRAM (a very common "why am I OOM at idle?" cause).
- **Focus on:** at idle this should be **empty**. If it isn't, something is holding memory — kill it before the memory labs.

---

## 8. (Optional) CUDA toolkit check — for Week 3

```bash
nvcc --version
```

- **What it does:** reports the CUDA **compiler** version (separate from the driver's CUDA version in `nvidia-smi`).
- **Why it matters:** Week 3 you *write* CUDA. The toolkit (`nvcc`) must be installed and its version compatible with the driver. If this says "command not found," you'll install the toolkit before Week 3.
- **Focus on:** the `release` line (e.g. `release 12.x`). It should be ≤ the CUDA version shown by `nvidia-smi`.

---

## 📝 Fill this in — your GPU, decoded

After running the commands, complete the table from your **own** output. This is your deliverable for Lab A:

| Field | Command | Your value |
|---|---|---|
| # of GPUs | `nvidia-smi -L` | |
| Model | `nvidia-smi -L` | |
| Architecture / compute cap | query `compute_cap` | |
| **Capacity** (VRAM per GPU) | query `memory.total` | |
| Memory type | (from lecture) | GDDR7 |
| **Bandwidth** (spec) | 5090 datasheet | ~1.79 TB/s |
| Precision floor | (from lecture) | FP4 |
| **Link** (GPU0 ↔ GPU1) | `nvidia-smi topo -m` | |
| PCIe gen × width (max) | query `pcie.link.*` | |
| Driver / CUDA version | `nvidia-smi` | |
| Max power / GPU | query `power.max_limit` | |

**Two questions to answer in one line each:**
1. If a model needs 40 GB of memory, will it fit on **one** 5090? On **two**? What has to change? *(preview of Lab B)*
2. Based on the `topo -m` result, do you expect splitting a model across both 5090s to be cheap or expensive — and why? *(preview of Lab C)*

---

### ⚠️ WSL quirks to know
- `lspci | grep -i nvidia` usually shows **nothing** in WSL2 — the GPU is paravirtualized through `/dev/dxg`, not a normal PCI device. Use `nvidia-smi -q -d PCIE` for PCIe info instead.
- Some NVML features (fan control, persistence mode, MIG) are **unavailable or read-only** under WSL. That's expected — everything in this lab works.
- `nvidia-smi topo -m` works in WSL, but if a field shows `N/A`, run the same commands from **native Windows PowerShell** (`nvidia-smi.exe topo -m`) to cross-check the physical topology.
```
