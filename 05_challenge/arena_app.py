# 3D Parallelism Arena — a live, auto-scored leaderboard (Gradio).
#
#   pip install gradio
#   python arena_app.py         # opens on http://localhost:7860
#   python arena_app.py --share # public link students can reach
#
# Teams submit a parallelism plan (TP · PP · DP + precision + checkpointing) for a
# FIXED model+cluster. The app checks it fits + is valid, scores its MFU (model-FLOPs
# utilization), and ranks everyone live. Highest MFU wins.
#
# NOTE: the score is a *teaching simulator* — a simplified model of the real tradeoffs
# (memory sharding, TP all-reduce cost, pipeline bubble). It rewards the right decisions,
# not exact wall-clock numbers.
import json, os, math, threading, argparse
import gradio as gr

# ── the scenario everyone competes on ──────────────────────────────
S = {
    "name": "Frontier-120B on 128 GPUs",
    "params": 120e9, "layers": 88,
    "gpus": 128, "node_size": 8, "vram_gb": 80,
    "intra_bw": 900, "inter_bw": 50,     # GB/s (NVLink vs InfiniBand)
}
LB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_leaderboard.json")
_lock = threading.Lock()
MAX_TRIES = 3     # individual challenge: everyone gets 3 submissions


def score_plan(tp, pp, dp, precision, ckpt):
    """Return (score, mem_per_gpu_GB, feedback_markdown). score=0 means invalid/OOM."""
    # 1 · validity — must use all GPUs
    if tp * pp * dp != S["gpus"]:
        return 0.0, 0.0, f"❌ **Invalid** — TP×PP×DP = {tp*pp*dp}, must equal **{S['gpus']}** GPUs (use them all)."

    # 2 · memory fit
    bpp = 16.0 if precision == "bf16" else 12.0        # bytes/param (fp8 lighter)
    model_state = S["params"] * bpp / 1e9              # GB, whole model
    model_per_gpu = model_state / (tp * pp)            # TP & PP shard the model
    act = 0.8 * (S["layers"] / pp) * (0.35 if ckpt else 1.0)   # activation memory (per stage)
    mem = model_per_gpu + act
    if mem > S["vram_gb"]:
        return 0.0, mem, (f"❌ **OOM** — {mem:.0f} GB/GPU > {S['vram_gb']} GB. "
                          f"Shard more (↑ TP or ↑ PP) or enable **checkpointing**.")

    # 3 · overheads → MFU
    crosses_node = tp > S["node_size"]
    link_pen = (S["intra_bw"] / S["inter_bw"]) if crosses_node else 1.0
    tp_over = 0.03 * (tp - 1) * link_pen               # TP all-reduces EVERY layer
    bubble = (pp - 1) / ((pp - 1) + 16)                # pipeline bubble, shrinks with micro-batches
    dp_over = 0.01 * math.log2(dp) if dp > 1 else 0.0  # DP: one gradient all-reduce per step (cheap)
    mfu = 1.0 / (1.0 + tp_over + bubble + dp_over)
    score = round(mfu * 100, 1)

    warn = "  ·  ⚠️ **TP crosses NVLink** — comm penalty ×18!" if crosses_node else ""
    fb = (f"✅ **Fits** — {mem:.0f} GB/GPU  ·  **MFU = {score}%**{warn}\n\n"
          f"breakdown → TP overhead `{tp_over:.2f}` · bubble `{bubble:.2f}` · DP `{dp_over:.2f}`  \n"
          f"_lower overheads = higher MFU. Enough TP+PP to fit, the rest in DP._")
    return score, mem, fb


# ── leaderboard persistence (best score per team) ──────────────────
def _load():
    if os.path.exists(LB_FILE):
        try:
            return json.load(open(LB_FILE))
        except Exception:
            return []
    return []


def _table():
    # rank by best score, then FEWEST attempts (rewards working it out over brute-force)
    rows = sorted(_load(), key=lambda r: (-r["score"], r.get("attempts", 99)))
    medal = ["🥇", "🥈", "🥉"]
    return [[medal[i] if i < 3 else i + 1, r["name"], f"{r['tp']}·{r['pp']}·{r['dp']}",
             r["precision"], "✓" if r["ckpt"] else "—", f"{r['mem']:.0f}", r["score"], r.get("attempts", 1)]
            for i, r in enumerate(rows)]


def submit(name, tp, pp, dp, precision, ckpt):
    name = (name or "").strip()
    if not name:
        return "⚠️ Enter **your name** first.", _table()
    tp, pp, dp = int(tp), int(pp), int(dp)
    with _lock:
        rows = _load()
        prev = next((r for r in rows if r["name"] == name), None)
        used = prev["attempts"] if prev else 0
        if used >= MAX_TRIES:
            return (f"🔒 **No attempts left** ({MAX_TRIES}/{MAX_TRIES} used). Your best stands: "
                    f"**{prev['score']}% MFU** with TP{prev['tp']}·PP{prev['pp']}·DP{prev['dp']}.", _table())
        score, mem, fb = score_plan(tp, pp, dp, precision, ckpt)
        tries = used + 1
        cur = {"name": name, "tp": tp, "pp": pp, "dp": dp, "precision": precision,
               "ckpt": bool(ckpt), "mem": float(mem), "score": float(score), "attempts": tries}
        if prev is None:
            rows.append(cur)
        elif score >= prev["score"]:
            rows[rows.index(prev)] = cur                 # new best: keep this plan + updated count
        else:
            prev["attempts"] = tries                     # worse: keep old best, just count the try
        json.dump(rows, open(LB_FILE, "w"), indent=2)
    left = MAX_TRIES - tries
    head = (f"**Attempt {tries} of {MAX_TRIES}** · {left} left.\n\n" if left > 0
            else f"**Final attempt used ({MAX_TRIES}/{MAX_TRIES}).** Your best is locked in.\n\n")
    return head + fb, _table()


# ── UI ─────────────────────────────────────────────────────────────
HEAD = f"""# 🏟️ 3D Parallelism Arena
**Scenario — {S['name']}:** a **{int(S['params']/1e9)}B dense** model on **{S['gpus']} GPUs**
({S['gpus']//S['node_size']} nodes × {S['node_size']} · {S['vram_gb']} GB each · NVLink in-node, InfiniBand across).

Submit a plan → it's scored instantly. **Valid** (TP×PP×DP = {S['gpus']}, TP ≤ {S['node_size']}) + **fits** (≤ {S['vram_gb']} GB/GPU) → you get an **MFU %**. **Individual — you get 3 submissions.** Rank = **best MFU**, ties broken by **fewest attempts**, so *calculate*, don't just click."""

with gr.Blocks(title="3D Parallelism Arena", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown(HEAD)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Your plan")
            name = gr.Textbox(label="Your name", placeholder="e.g. Sara Al-Otaibi")
            with gr.Row():
                tp = gr.Dropdown([1, 2, 4, 8, 16], value=8, label="TP (tensor)")
                pp = gr.Dropdown([1, 2, 4, 8, 16], value=8, label="PP (pipeline)")
                dp = gr.Dropdown([1, 2, 4, 8, 16, 32, 64, 128], value=2, label="DP (data)")
            precision = gr.Radio(["bf16", "fp8"], value="bf16", label="Precision")
            ckpt = gr.Checkbox(label="Activation checkpointing")
            btn = gr.Button("🚀 Submit plan", variant="primary")
            result = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("### 🏆 Leaderboard")
            lb = gr.Dataframe(headers=["#", "Name", "TP·PP·DP", "prec", "ckpt", "GB/GPU", "MFU %", "Tries"],
                              datatype=["str", "str", "str", "str", "str", "str", "number", "number"],
                              interactive=False, wrap=True)
            refresh = gr.Button("↻ Refresh leaderboard")
    with gr.Accordion("How the score works (read me)", open=False):
        gr.Markdown(
            "- **Memory/GPU** = model_state ÷ (TP×PP) + activations. FP8 & checkpointing cut it.\n"
            "- **MFU** = 1 ÷ (1 + TP-overhead + pipeline-bubble + DP-overhead).\n"
            "- **TP** all-reduces on *every layer* → costly, and **catastrophic if TP > node size** (crosses NVLink).\n"
            "- **PP** adds a **bubble** ((PP−1)/(PP−1+16)) → don't over-use it.\n"
            "- **DP** is nearly free → it's where the leftover GPUs should go.\n"
            "- **The sweet spot:** the *smallest* TP + PP that still fits, everything else in DP.\n"
            "- **You get 3 tries.** Rank is your **best** score; ties go to **fewer attempts** — so work it out, then submit.")

    btn.click(submit, [name, tp, pp, dp, precision, ckpt], [result, lb])
    refresh.click(lambda: _table(), None, lb)
    demo.load(lambda: _table(), None, lb)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port)
