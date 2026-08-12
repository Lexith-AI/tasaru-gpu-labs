# The 3D Parallelism Arena — Challenge

A live, **auto-scored leaderboard** where each student designs a parallelism plan and competes on efficiency — no GPU required.

## Run it
```bash
pip install gradio
python arena_app.py            # http://localhost:7860
python arena_app.py --share    # public link students can reach from their laptops
```

## The game
- **Scenario:** a **120B dense model** on **128 GPUs** (16 nodes × 8 · 80 GB each · NVLink in-node, InfiniBand across).
- **You submit:** `TP · PP · DP` + precision (BF16/FP8) + activation checkpointing.
- The app checks it's **valid** (`TP×PP×DP = 128`, `TP ≤ 8`) and **fits** (≤ 80 GB/GPU), then scores its **MFU** (model-FLOPs utilization) = `1 / (1 + TP-overhead + pipeline-bubble + DP-overhead)`.
- **Individual · 3 submissions each.** Ranked by **best MFU**, ties broken by **fewest attempts** — so *calculate*, don't just click.

The winning instinct (which students discover by playing): the *smallest* TP + PP that still fits, with everything else in DP. Cross the NVLink node boundary (TP > 8) and the score craters; under-shard and it OOMs.

> The score is a **teaching simulator** — a simplified model of the real tradeoffs (memory sharding, TP all-reduce cost, pipeline bubble). It rewards the right decisions, not exact wall-clock numbers.

## Files
- `arena_app.py` — the Gradio leaderboard.
- `Assignment_Scale_Out_Multi_Node.pdf` — the full *Scale Out* assignment this challenge belongs to (Part 5).
