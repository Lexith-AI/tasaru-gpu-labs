# Lab I — FSDP: shard a model that does NOT fit on one 5090.
# The default config (~2.7B params) needs ~44 GB for full training (16 B/param) — it OOMs on
# a single 32 GB card. FSDP shards params + grads + optimizer across both GPUs so it fits.
#
#   # this OOMs (model too big for one card):
#   torchrun --standalone --nproc_per_node=1 train_fsdp.py
#   # this works — sharded across both GPUs:
#   torchrun --standalone --nproc_per_node=2 train_fsdp.py --ckpt
#
# The lecture showed the FSDP2 one-liner `fully_shard(...)`. Here we use the well-established
# FullyShardedDataParallel wrapper — same idea (FULL_SHARD = ZeRO-3), robust across setups.
import os, time, argparse, functools, torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import MiniGPT, GPTConfig, Block, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2560)      # ~2.7B params → ~44 GB full training
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--heads", type=int, default=16)   # 16 divides 2560/2048/1024 → safe for every preset
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--ckpt", action="store_true", help="activation checkpointing (use if you OOM)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(1234 + rank)

    cfg = GPTConfig(dim=args.dim, n_layer=args.layers, n_head=args.heads, seq=args.seq, use_ckpt=args.ckpt)
    if rank == 0:
        p = count_params(cfg)
        print(f"[FSDP] world_size={world}  params={p/1e9:.2f}B  full training ≈ {p*16/1e9:.0f} GB (16 B/param)")
        print(f"       sharded across {world} GPUs → roughly {p*16/1e9/world:.0f} GB/GPU before activations")

    model = MiniGPT(cfg)                                   # build on CPU; FSDP shards onto the GPUs
    bf16 = torch.cuda.get_device_capability(local_rank)[0] >= 8
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        buffer_dtype=torch.bfloat16) if bf16 else None
    # wrap EACH transformer block as its own shard unit → gather one block at a time
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    model = FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,    # ZeRO-3: shard params, grads AND optimizer
        device_id=local_rank,
        use_orig_params=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device("cuda", local_rank)

    def get_batch():
        d = torch.randint(0, cfg.vocab, (args.batch, cfg.seq + 1), device=device)
        return d[:, :-1], d[:, 1:]

    def step():
        x, y = get_batch()
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)              # FSDP casts to bf16 + all-gathers each block on demand
        loss.backward()                    # reduce-scatter grads back to shards
        opt.step()
        return loss

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize(); dist.barrier()
    torch.cuda.reset_peak_memory_stats(local_rank)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    torch.cuda.synchronize(); dist.barrier()
    dt = time.perf_counter() - t0

    tokens = world * args.steps * args.batch * args.seq
    peak = torch.cuda.max_memory_allocated(local_rank) / 1e9
    print(f"[rank {rank}] peak VRAM {peak:.1f} GB")
    if rank == 0:
        print(f"[FSDP] {world} GPU(s) | {tokens/dt:,.0f} tokens/sec | {dt/args.steps*1e3:.1f} ms/step | loss {loss.item():.2f}")
        print(f"       compare peak VRAM/GPU here to the ~44 GB the full model would need on one card")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
