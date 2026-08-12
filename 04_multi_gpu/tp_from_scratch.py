# Lab K — Tensor Parallelism FROM SCRATCH (2 GPUs).
# Split one MLP across 2 GPUs by hand and prove the result matches a single GPU.
# This is the exact column/row-parallel pattern Megatron-LM uses.
#
#   torchrun --standalone --nproc_per_node=2 tp_from_scratch.py
#
import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "run with --nproc_per_node=2"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    ## same seed on BOTH ranks → both build the identical "true" full weights
    torch.manual_seed(0)
    D = 4096
    W1 = torch.randn(D, 4 * D, device=dev) / D ** 0.5          # first layer  (D → 4D)
    W2 = torch.randn(4 * D, D, device=dev) / (4 * D) ** 0.5    # second layer (4D → D)
    x = torch.randn(8, D, device=dev)                         # same input on both ranks

    ## ── reference: the whole MLP on ONE GPU (both ranks compute it identically) ──
    ref = F.gelu(x @ W1) @ W2

    ## ── the tensor-parallel version, split across the 2 GPUs ──
    half = (4 * D) // world                                    # split the hidden dim in two

    ## 1) COLUMN-parallel first layer: each rank owns HALF the columns of W1.
    ##    No communication — each rank just makes its half of the hidden activations.
    W1_shard = W1[:, rank * half:(rank + 1) * half]           # (D, 2D)
    h_shard = F.gelu(x @ W1_shard)                            # (8, 2D)  — my half of the hidden

    ## 2) ROW-parallel second layer: split W2's ROWS to match the hidden split.
    ##    Each rank makes a PARTIAL output; all-reduce sums them into the full output.
    W2_shard = W2[rank * half:(rank + 1) * half, :]           # (2D, D)
    y_partial = h_shard @ W2_shard                           # (8, D)  — partial sum
    dist.all_reduce(y_partial)                               # ← the ONE collective for the whole MLP
    tp = y_partial

    ## ── verify the split version equals the single-GPU version ──
    err = (tp - ref).abs().max().item()
    scale = ref.abs().mean().item()
    if rank == 0:
        print(f"each GPU stored HALF of W1 ({tuple(W1_shard.shape)}) and HALF of W2 ({tuple(W2_shard.shape)})")
        print(f"max abs error = {err:.2e}   (relative {err/scale:.1e})")
        print("MATCH ✅ — same result, half the weights per GPU." if err / scale < 1e-3
              else "MISMATCH ❌ — check the split.")
        print("Note: the whole MLP needed exactly ONE all-reduce. That's the Megatron trick.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
