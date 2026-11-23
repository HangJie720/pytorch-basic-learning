"""
AllGather 验证：
  k=8 个 rank，每个 rank 输入长度 N 的向量；
  完成后每个 rank 得到长度 k*N 的相同输出：
    out[Y * N + i] = inY[i]   (Y = 0..k-1)

  ReduceScatter + AllGather ≈ AllReduce。

用法:
  torchrun --nproc_per_node=8 test_allgather.py
  torchrun --nproc_per_node=8 test_allgather.py --N 1024
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="AllGather smoke test")
    parser.add_argument("--N", type=int, default=1024, help="per-rank vector length")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="process group backend, e.g. nccl / gloo",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    assert world_size == 8, f"本 case 固定 k=8，当前 WORLD_SIZE={world_size}"

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)

    # in_rank[i] = rank + 1
    inp = torch.full((args.N,), float(rank + 1), dtype=dtype, device=device)

    # 每个 rank 一块输出缓冲，all_gather 后按 rank 顺序拼接
    gather_list = [
        torch.empty(args.N, dtype=dtype, device=device) for _ in range(world_size)
    ]
    dist.all_gather(gather_list, inp)
    out = torch.cat(gather_list, dim=0)

    # expected: out[Y*N:(Y+1)*N] = Y+1
    expected_chunks = [
        torch.full((args.N,), float(y + 1), dtype=dtype, device=device)
        for y in range(world_size)
    ]
    expected = torch.cat(expected_chunks, dim=0)

    ok = torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
    max_abs = (out.float() - expected.float()).abs().max().item()

    # 抽查几个边界位置：rank0 头、本 rank 段头、最后一段头
    sample_idxs = [0, rank * args.N, (world_size - 1) * args.N]
    samples = ", ".join(
        f"out[{i}]={out[i].item()}" for i in sample_idxs if i < out.numel()
    )
    print(
        f"[rank {rank}] N={args.N} out_len={out.numel()} dtype={args.dtype} "
        f"{samples} max_abs={max_abs:.6g} ok={ok}",
        flush=True,
    )

    fail = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(fail, op=dist.ReduceOp.SUM)

    dist.barrier()
    dist.destroy_process_group()

    if fail.item() != 0:
        raise SystemExit(f"AllGather check failed on {fail.item()} rank(s)")

    if rank == 0:
        print(
            "PASS: all ranks agree on out[Y*N+i] = inY[i]",
            flush=True,
        )


if __name__ == "__main__":
    main()
