"""Symmetric memory put/get + signal smoke test (2 ranks).

Rank 1 writes into rank 0's symmetric buffer via get_buffer(0), then put_signal.
Rank 0 wait_signal and checks the value.

Requires: >=2 GPUs with CUDA P2P (typically NVLink).

Usage:
  torchrun --nproc_per_node=2 test_put_get.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("SKIP: need >=2 CUDA devices with P2P for symmetric memory")
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    assert world_size == 2, f"this case expects world_size=2, got {world_size}"

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Peer access is required for the default CUDA P2P symmetric backend.
    peer = 1 - local_rank
    if not torch.cuda.can_device_access_peer(local_rank, peer):
        print(f"SKIP: cuda:{local_rank} cannot P2P access cuda:{peer}")
        dist.destroy_process_group()
        return

    group = dist.group.WORLD
    symm_mem.enable_symm_mem_for_group(group.group_name)

    t = symm_mem.empty(1024, dtype=torch.float32, device=device)
    hdl = symm_mem.rendezvous(t, group=group)

    # Local view of *this* rank's buffer.
    local = hdl.get_buffer(rank, (1024,), torch.float32)
    local.zero_()
    hdl.barrier()

    if rank == 1:
        # One-sided write into rank 0's memory through the peer mapping.
        remote0 = hdl.get_buffer(0, (1024,), torch.float32)
        remote0.fill_(42)
        hdl.put_signal(dst_rank=0)
    else:
        hdl.wait_signal(src_rank=1)
        ok = bool(local.eq(42).all().item())
        print(f"[rank0] local[0]={local[0].item()} ok={ok}", flush=True)
        if not ok:
            dist.destroy_process_group()
            raise SystemExit("symmetric put/get check failed")

    hdl.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("PASS: rank1 put into rank0 symmetric buffer; signal observed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # Keep educational: missing APIs / backends should SKIP not crash CI-less learning.
        print(f"SKIP/ERR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
