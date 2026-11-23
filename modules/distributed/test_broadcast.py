"""
Broadcast 验证：
  k=8 个 rank；指定 root rank 持有长度 N 的输入缓冲，
  广播后每个 rank 得到相同输出：
    out[i] = in_root[i]

  注意：root 是 rank id，不是物理 device id。

用法:
  torchrun --nproc_per_node=8 test_broadcast.py
  torchrun --nproc_per_node=8 test_broadcast.py --N 1024 --root 2
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="Broadcast smoke test")
    parser.add_argument("--N", type=int, default=1024, help="vector length")
    parser.add_argument(
        "--root",
        type=int,
        default=2,
        help="broadcast source rank id (not device id)",
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

    # root 填入可核对的数据；其它 rank 先置 0，广播后应被覆盖
    # in_root[i] = i + root + 1
    if rank == args.root:
        buf = torch.arange(args.N, dtype=dtype, device=device) + float(args.root + 1)
    else:
        buf = torch.zeros(args.N, dtype=dtype, device=device)

    before0 = buf[0].item()
    dist.broadcast(buf, src=args.root)

    expected = torch.arange(args.N, dtype=dtype, device=device) + float(args.root + 1)
    ok = torch.allclose(buf, expected, rtol=1e-3, atol=1e-3)
    max_abs = (buf.float() - expected.float()).abs().max().item()

    print(
        f"[rank {rank}] N={args.N} root={args.root} dtype={args.dtype} "
        f"before0={before0} after0={buf[0].item()} "
        f"expected0={expected[0].item()} max_abs={max_abs:.6g} ok={ok}",
        flush=True,
    )

    fail = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(fail, op=dist.ReduceOp.SUM)

    dist.barrier()
    dist.destroy_process_group()

    if fail.item() != 0:
        raise SystemExit(f"Broadcast check failed on {fail.item()} rank(s)")

    if rank == 0:
        print(
            f"PASS: all ranks received out[i] = in_root[i] from root={args.root}",
            flush=True,
        )


if __name__ == "__main__":
    main()
