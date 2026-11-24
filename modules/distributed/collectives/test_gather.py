"""
Gather 验证：
  k=8 个 rank，每个 rank 输入长度 N 的向量；
  收集后仅 root rank 得到长度 k*N 的输出：
    out[Y*N + i] = inY[i]
  其它 rank 不保证持有完整结果。

  注意：root 是 rank id，不是物理 device id。

用法:
  torchrun --nproc_per_node=8 test_gather.py
  torchrun --nproc_per_node=8 test_gather.py --N 1024 --root 2
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="Gather smoke test")
    parser.add_argument("--N", type=int, default=1024, help="per-rank vector length")
    parser.add_argument(
        "--root",
        type=int,
        default=2,
        help="gather destination rank id (not device id)",
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

    # inY[i] = Y + 1
    inp = torch.full((args.N,), float(rank + 1), dtype=dtype, device=device)

    gather_list = None
    if rank == args.root:
        gather_list = [
            torch.empty(args.N, dtype=dtype, device=device) for _ in range(world_size)
        ]

    dist.gather(inp, gather_list, dst=args.root)

    if rank == args.root:
        out = torch.cat(gather_list, dim=0)
        expected = torch.cat(
            [
                torch.full((args.N,), float(y + 1), dtype=dtype, device=device)
                for y in range(world_size)
            ],
            dim=0,
        )
        ok = torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
        max_abs = (out.float() - expected.float()).abs().max().item()
        print(
            f"[rank {rank}/root] N={args.N} out_len={out.numel()} dtype={args.dtype} "
            f"out0={out[0].item()} expected0={expected[0].item()} "
            f"max_abs={max_abs:.6g} ok={ok}",
            flush=True,
        )
    else:
        ok = True
        print(
            f"[rank {rank}] N={args.N} root={args.root} "
            f"in0={inp[0].item()} (non-root, result not checked) ok={ok}",
            flush=True,
        )

    fail = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(fail, op=dist.ReduceOp.SUM)

    dist.barrier()
    dist.destroy_process_group()

    if fail.item() != 0:
        raise SystemExit(f"Gather check failed on {fail.item()} rank(s)")

    if rank == 0:
        print(
            f"PASS: root={args.root} got out[Y*N+i] = inY[i]",
            flush=True,
        )


if __name__ == "__main__":
    main()
