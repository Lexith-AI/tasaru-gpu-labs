# Lab J — measure the real GPU-to-GPU link with an all-reduce micro-benchmark.
# Pure PyTorch (no nccl-tests build). Shows the PCIe ceiling on the 2×5090.
#
#   torchrun --standalone --nproc_per_node=2 bench_nccl.py
#
# busbw (bus bandwidth) is the fair number to compare across GPU counts:
#   for ring all-reduce, busbw = algbw × 2·(N−1)/N
# On PCIe 5.0 (no NVLink) expect tens of GB/s; NVLink/NVSwitch would show hundreds.
import os, time, torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"NCCL all-reduce · world_size={world}")
        print(f"{'size':>8} | {'time':>9} | {'algbw':>10} | {'busbw':>10}")
        print("-" * 46)

    for mb in [1, 4, 16, 64, 256, 512]:
        n = mb * 1024 * 1024 // 4                     # float32 elements
        t = torch.ones(n, device="cuda")
        for _ in range(5):                            # warmup
            dist.all_reduce(t)
        torch.cuda.synchronize(); dist.barrier()

        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        nbytes = n * 4
        algbw = nbytes / dt / 1e9                     # GB/s moved (as seen by the caller)
        busbw = algbw * 2 * (world - 1) / world       # ring all-reduce bus bandwidth
        if rank == 0:
            print(f"{mb:6d}MB | {dt*1e3:7.2f}ms | {algbw:7.1f}GB/s | {busbw:7.1f}GB/s")

    if rank == 0:
        print("\nThe busbw at the large sizes is your interconnect ceiling.")
        print("PCIe 5.0 ×16 ≈ 64 GB/s peak → expect well under that in practice.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
