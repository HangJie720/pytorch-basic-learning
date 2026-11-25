"""
AlltoAll 验证：
  k=8 个 rank，每个 rank 输入长度 k*N（按目标 rank 切成 k 段）；
  rank Y 的第 X 段发给 rank X；
  完成后 rank X 输出长度仍为 k*N：
    outX[Y*N + i] = inY[X*N + i]

用法:
  torchrun --nproc_per_node=8 test_alltoall.py
  torchrun --nproc_per_node=8 test_alltoall.py --N 1024
"""

import argparse
import os

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="AlltoAll smoke test")
    parser.add_argument(
        "--N",
        type=int,
        default=1024,
        help="per-segment length (input/output length = k*N)",
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

    # 本 rank = Y：第 X 段填 (Y+1)*100 + (X+1)，便于核对来源/去向
    # 即 inY[X*N:(X+1)*N] = (Y+1)*100 + (X+1)
    input_list = [
        torch.full(
            (args.N,),
            float((rank + 1) * 100 + (dst + 1)),
            dtype=dtype,
            device=device,
        )
        for dst in range(world_size)
    ]
    output_list = [
        torch.empty(args.N, dtype=dtype, device=device) for _ in range(world_size)
    ]

    dist.all_to_all(output_list, input_list)
    out = torch.cat(output_list, dim=0)

    # rank X 收到来自 Y 的段：应等于 inY 发给 X 的值 = (Y+1)*100 + (X+1)
    expected_chunks = [
        torch.full(
            (args.N,),
            float((src + 1) * 100 + (rank + 1)),
            dtype=dtype,
            device=device,
        )
        for src in range(world_size)
    ]
    expected = torch.cat(expected_chunks, dim=0)

    ok = torch.allclose(out, expected, rtol=1e-3, atol=1e-3)
    max_abs = (out.float() - expected.float()).abs().max().item()

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
        raise SystemExit(f"AlltoAll check failed on {fail.item()} rank(s)")

    if rank == 0:
        print(
            "PASS: all ranks agree on outX[Y*N+i] = inY[X*N+i]",
            flush=True,
        )


if __name__ == "__main__":
    main()
