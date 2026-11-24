"""
ReduceScatter(sum) 验证：
  k=8 个 rank，每个 rank 输入长度 k*N 的向量（按 rank 切成 k 段）；
  归约后 rank Y 只拿到第 Y 段，长度 N：
    outY[i] = in0[Y*N+i] + in1[Y*N+i] + ... + in(k-1)[Y*N+i]

  ReduceScatter + AllGather ≈ AllReduce。

用法:
  torchrun --nproc_per_node=8 test_reducescatter.py
  torchrun --nproc_per_node=8 test_reducescatter.py --N 1024
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="ReduceScatter sum smoke test")
    parser.add_argument(
        "--N",
        type=int,
        default=1024,
        help="per-rank output chunk length (input length = k*N)",
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

    dtype = getattr(torch, args.dtype)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)

    # 所有 rank 使用相同布局，便于核对 scatter 是否分到正确段：
    # in[Y*N:(Y+1)*N] = (Y+1)
    # 则 rank Y 期望：outY[i] = k * (Y+1)
    chunks = [
        torch.full((args.N,), float(y + 1), dtype=dtype, device=device)
        for y in range(world_size)
    ]
    inp = torch.cat(chunks, dim=0)
    out = torch.empty(args.N, dtype=dtype, device=device)

    # reduce_scatter: input_list 按 chunk 传入，output 为本 rank 对应段
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM)

    expected_val = float(world_size * (rank + 1))
    expected = torch.full((args.N,), expected_val, dtype=dtype, device=device)
    ok = torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
    max_abs = (out.float() - expected.float()).abs().max().item()

    print(
        f"[rank {rank}] N={args.N} in_len={inp.numel()} out_len={out.numel()} "
        f"dtype={args.dtype} out0={out[0].item()} expected={expected_val} "
        f"max_abs={max_abs:.6g} ok={ok}",
        flush=True,
    )

    fail = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    dist.all_reduce(fail, op=dist.ReduceOp.SUM)

    dist.barrier()
    dist.destroy_process_group()

    if fail.item() != 0:
        raise SystemExit(f"ReduceScatter check failed on {fail.item()} rank(s)")

    if rank == 0:
        print(
            "PASS: rank Y got outY[i] = sum_X inX[Y*N+i]",
            flush=True,
        )


if __name__ == "__main__":
    main()
