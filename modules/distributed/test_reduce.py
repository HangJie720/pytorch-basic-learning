"""
Reduce(sum) 验证：
  k=8 个 rank，每个 rank 输入长度 N 的向量；
  归约后仅 root rank 得到：
    out[i] = in0[i] + in1[i] + ... + in(k-1)[i]
  其它 rank 的缓冲内容不做结果校验（Reduce 不保证写回非 root）。

  注意：root 是 rank id，不是物理 device id。
  Reduce + Broadcast ≈ AllReduce。

用法:
  torchrun --nproc_per_node=8 test_reduce.py
  torchrun --nproc_per_node=8 test_reduce.py --N 1024 --root 2
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="Reduce sum smoke test")
    parser.add_argument("--N", type=int, default=1024, help="vector length")
    parser.add_argument(
        "--root",
        type=int,
        default=2,
        help="reduce destination rank id (not device id)",
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

    # 每个 rank：in_rank[i] = rank + 1
    # root 期望：out[i] = 1+2+...+8 = 36
    inp = torch.full((args.N,), float(rank + 1), dtype=dtype, device=device)
    buf = inp.clone()

    dist.reduce(buf, dst=args.root, op=dist.ReduceOp.SUM)

    expected_val = float(sum(range(1, world_size + 1)))  # 36
    if rank == args.root:
        expected = torch.full((args.N,), expected_val, dtype=dtype, device=device)
        ok = torch.allclose(buf, expected, rtol=1e-3, atol=1e-3)
        max_abs = (buf.float() - expected.float()).abs().max().item()
        print(
            f"[rank {rank}/root] N={args.N} dtype={args.dtype} "
            f"in0={inp[0].item()} out0={buf[0].item()} "
            f"expected={expected_val} max_abs={max_abs:.6g} ok={ok}",
            flush=True,
        )
    else:
        # 非 root 不保证结果，仅记录输入；ok 记为成功
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
        raise SystemExit(f"Reduce check failed on {fail.item()} rank(s)")

    if rank == 0:
        print(
            f"PASS: root={args.root} got out[i] = sum_r in_r[i]",
            flush=True,
        )


if __name__ == "__main__":
    main()
