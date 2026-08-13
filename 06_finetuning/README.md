# Fine-Tuning Lab — SFT that scales with 🤗 Accelerate

A **real** supervised fine-tuning (SFT) script. The *same* `finetune.py` runs on **1 GPU, multi-GPU DDP, or FSDP** — the parallelism is chosen by how you **launch** it, not by the code. Uses `transformers` + `accelerate` + `peft` (LoRA).

```bash
pip install -r requirements.txt
```

## Run it

**1 GPU (LoRA):**
```bash
accelerate launch finetune.py
```

**Multi-GPU DDP (LoRA) — e.g. 2 GPUs:**
```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 finetune.py
```
> On free **Kaggle 2× T4**, use `--mixed_precision fp16` (T4 has no BF16).

**FSDP full fine-tune (shards the whole model across GPUs):**
```bash
accelerate launch --config_file fsdp_config.yaml finetune.py --full_finetune
```

Or run `accelerate config` once to set your defaults interactively, then just `accelerate launch finetune.py`.

## What the parallelism looks like
| Launch | Strategy | What's split | Use for |
|---|---|---|---|
| `accelerate launch` | single GPU | nothing | quick tests |
| `--multi_gpu` | **DDP** | the data (batch); gradients all-reduced | **LoRA** (small trainable set, base replicated) |
| `--config_file fsdp_config.yaml` | **FSDP** | params + grads + optimizer sharded | **full** fine-tune of a model too big for one card |

The script handles the rest: **gradient accumulation**, **mixed precision**, **gradient checkpointing**, cosine LR schedule, and saving (LoRA adapter, or a gathered full checkpoint under FSDP).

## Key knobs
`--model` (default `Qwen/Qwen2.5-0.5B-Instruct`, open/no-gating — bump to a 7B with LoRA/FSDP) · `--dataset` (default `tatsu-lab/alpaca`) · `--batch` (per-GPU) · `--grad_accum` (effective batch = `batch × grad_accum × #GPUs`) · `--seq_len` · `--lora_r` · `--full_finetune`.

## Try the result
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained("out-sft/checkpoint-final")
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "out-sft/checkpoint-final")   # load the LoRA adapter
msgs = [{"role": "user", "content": "Give me three tips for saving GPU memory when training."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=200)[0]))
```
(For a full fine-tune, load `out-sft/checkpoint-final` directly with `from_pretrained` — no PEFT step.)

## Notes
- **LoRA → DDP, full → FSDP.** LoRA's trainable set is tiny, so DDP is simplest; FSDP is for full fine-tuning of large models.
- For **FSDP**, set `fsdp_transformer_layer_cls_to_wrap` in `fsdp_config.yaml` to your model's decoder block (e.g. `LlamaDecoderLayer`).
- Labels are the full sequence (standard causal-LM SFT). Prompt-token masking is a straightforward extension if you want to train only on the responses.
