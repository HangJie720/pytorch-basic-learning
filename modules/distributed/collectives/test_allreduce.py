"""
AllReduce(sum) 验证：
  k=8 个 rank，每个 rank 输入长度 N 的向量；
  完成后每个 rank 得到相同输出：
    out[i] = in0[i] + in1[i] + ... + in(k-1)[i]

用法:
  torchrun --nproc_per_node=8 test_allreduce.py
  torchrun --nproc_per_node=8 test_allreduce.py --N 1024
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="AllReduce sum smoke test")
    parser.add_argument("--N", type=int, default=1024, help="vector length")
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

    # torchrun 注入的环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    assert world_size == 8, f"本 case 固定 k=8，当前 WORLD_SIZE={world_size}"

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)

    # 每个 rank 的输入：in_rank[i] = rank + 1  （常量向量，便于核对）
    # 期望输出：out[i] = sum_{r=0..7} (r+1) = 1+2+...+8 = 36
    inp = torch.full((args.N,), float(rank + 1), dtype=dtype, device=device)
    out = inp.clone()

    dist.all_reduce(out, op=dist.ReduceOp.SUM)

    expected_val = float(sum(range(1, world_size + 1)))  # 36
    expected = torch.full((args.N,), expected_val, dtype=dtype, device=device)

    ok = torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
    max_abs = (out.float() - expected.float()).abs().max().item()

    # 各 rank 打印少量诊断信息
    print(
        f"[rank {rank}] N={args.N} dtype={args.dtype} "
        f"in0={inp[0].item()} out0={out[0].item()} "
        f"expected={expected_val} max_abs={max_abs:.6g} ok={ok}",
        flush=True,
    )

    # 用 all_reduce 汇总失败数，保证任意 rank 失败都会退出非 0
    fail = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(fail, op=dist.ReduceOp.SUM)

    dist.barrier()
    dist.destroy_process_group()

    if fail.item() != 0:
        raise SystemExit(f"AllReduce check failed on {fail.item()} rank(s)")

    if rank == 0:
        print("PASS: all ranks agree on out[i] = sum_r in_r[i]", flush=True)


if __name__ == "__main__":
    main()
