"""
Scatter 验证：
  k=8 个 rank；root 持有长度 k*N 的输入，按 rank 切成 k 段；
  分散后每个 rank 得到长度 N 的输出：
    outX[i] = in_root[X*N + i]

  注意：root 是 rank id，不是物理 device id。

用法:
  torchrun --nproc_per_node=8 test_scatter.py
  torchrun --nproc_per_node=8 test_scatter.py --N 1024 --root 2
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="Scatter smoke test")
    parser.add_argument(
        "--N",
        type=int,
        default=1024,
        help="per-rank output length (root input length = k*N)",
    )
    parser.add_argument(
        "--root",
        type=int,
        default=2,
        help="scatter source rank id (not device id)",
    )
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
    assert 0 <= args.root < world_size, f"root={args.root} 非法，需在 [0, {world_size})"

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)

    # root 上：第 X 段填 X+1，即 in_root[X*N:(X+1)*N] = X+1
    # 则 rank X 期望：outX[i] = X+1
    scatter_list = None
    if rank == args.root:
        scatter_list = [
            torch.full((args.N,), float(dst + 1), dtype=dtype, device=device)
            for dst in range(world_size)
        ]

    out = torch.empty(args.N, dtype=dtype, device=device)
    dist.scatter(out, scatter_list, src=args.root)

    expected_val = float(rank + 1)
    expected = torch.full((args.N,), expected_val, dtype=dtype, device=device)
    ok = torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
    max_abs = (out.float() - expected.float()).abs().max().item()

    print(
        f"[rank {rank}] N={args.N} root={args.root} dtype={args.dtype} "
        f"out0={out[0].item()} expected={expected_val} "
        f"max_abs={max_abs:.6g} ok={ok}",
        flush=True,
    )

    fail = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(fail, op=dist.ReduceOp.SUM)

    dist.barrier()
    dist.destroy_process_group()

    if fail.item() != 0:
        raise SystemExit(f"Scatter check failed on {fail.item()} rank(s)")

    if rank == 0:
        print(
            f"PASS: each rank X got outX[i] = in_root[X*N+i] from root={args.root}",
            flush=True,
        )


if __name__ == "__main__":
    main()
