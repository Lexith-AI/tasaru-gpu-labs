# finetune.py — real SFT fine-tuning that scales with 🤗 Accelerate.
#
# The SAME script runs on 1 GPU, multi-GPU DDP, or FSDP — the parallelism is chosen
# by how you LAUNCH it, not by the code:
#
#   # 1 GPU
#   accelerate launch finetune.py
#   # multi-GPU DDP (LoRA) — e.g. 2 GPUs
#   accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 finetune.py
#   # FSDP full fine-tune (shards the model across GPUs)
#   accelerate launch --config_file fsdp_config.yaml finetune.py --full_finetune
#
# LoRA (default) → use DDP.  Full fine-tune (--full_finetune) → use FSDP.
import argparse, math, os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup)
from accelerate import Accelerator
from accelerate.utils import set_seed


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")   # open, no gating; bump to 7B with LoRA/FSDP
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--output_dir", default="out-sft")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)                    # per-GPU micro-batch
    p.add_argument("--grad_accum", type=int, default=8)               # effective batch = batch × grad_accum × #GPUs
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=2000)           # subset for a fast demo; 0 = all
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=0)               # 0 = save only at the end
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--full_finetune", action="store_true", help="no LoRA — full fine-tune (use with FSDP)")
    return p.parse_args()


def build_dataset(args, tok):
    ds = load_dataset(args.dataset, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    def to_text(ex):
        instr = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
        msgs = [{"role": "user", "content": instr},
                {"role": "assistant", "content": ex["output"]}]
        # the model's own chat template → works for any instruct model
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=args.seq_len)

    ds = ds.map(to_text, remove_columns=ds.column_names)
    ds = ds.map(tok_fn, remove_columns=["text"])
    return ds


def save(accelerator, model, tok, out, tag, is_lora):
    accelerator.wait_for_everyone()
    save_dir = os.path.join(out, f"checkpoint-{tag}")
    unwrapped = accelerator.unwrap_model(model)
    if is_lora:
        # LoRA path (DDP): the small adapter is identical on every rank → main saves it
        if accelerator.is_main_process:
            unwrapped.save_pretrained(save_dir)          # writes only the adapter
    else:
        # full fine-tune (FSDP): gather the FULL_STATE_DICT and write the whole model
        unwrapped.save_pretrained(save_dir, is_main_process=accelerator.is_main_process,
                                  save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    if accelerator.is_main_process:
        tok.save_pretrained(save_dir)
    accelerator.print(f"✔ saved {save_dir}")


def main():
    args = parse()
    set_seed(42)

    # ── Accelerator handles ALL the parallelism (DDP or FSDP), mixed precision, grad-accum ──
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    accelerator.print(f"parallelism: {accelerator.distributed_type} · processes: {accelerator.num_processes} "
                      f"· mixed precision: {accelerator.mixed_precision}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = build_dataset(args, tok)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)        # labels = input_ids (causal LM), dynamic padding
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collator)

    # ── model ──
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()                            # trade compute for memory
    model.enable_input_require_grads()                              # needed for grad-checkpointing (+ frozen base under LoRA)

    is_lora = not args.full_finetune
    if is_lora:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                          target_modules="all-linear", task_type="CAUSAL_LM")   # robust across architectures
        model = get_peft_model(model, lora)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    # ── hand everything to Accelerate: this is where DDP/FSDP wrapping happens ──
    model, opt, loader = accelerator.prepare(model, opt, loader)

    updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, total_steps)   # per-process; steps identically → in sync

    # ── train ──
    model.train()
    done = 0
    for epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):         # gates grad-sync + optimizer step across micro-batches
                loss = model(**batch).loss
                accelerator.backward(loss)              # DDP all-reduce / FSDP reduce-scatter happens here
                opt.step()
                if accelerator.sync_gradients:          # a real optimizer update just happened
                    sched.step()
                opt.zero_grad()
            if accelerator.sync_gradients:
                done += 1
                if done % 10 == 0:
                    accelerator.print(f"step {done}/{total_steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")
                if args.save_steps and done % args.save_steps == 0:
                    save(accelerator, model, tok, args.output_dir, done, is_lora)

    save(accelerator, model, tok, args.output_dir, "final", is_lora)
    accelerator.print("done.")


if __name__ == "__main__":
    main()
