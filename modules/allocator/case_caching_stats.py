"""Smoke: CUDACachingAllocator allocated vs reserved + empty_cache."""
from __future__ import annotations

import torch


def _mb(n: int) -> float:
    return n / (1024 * 1024)


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    before = torch.cuda.memory_stats(device)
    alloc0 = before.get("allocated_bytes.all.current", 0)
    reserved0 = before.get("reserved_bytes.all.current", 0)

    # Force a large pool allocation.
    x = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32, device=device)  # 64 MiB
    mid = torch.cuda.memory_stats(device)
    alloc1 = mid.get("allocated_bytes.all.current", 0)
    reserved1 = mid.get("reserved_bytes.all.current", 0)

    del x
    torch.cuda.synchronize(device)
    after_free = torch.cuda.memory_stats(device)
    alloc2 = after_free.get("allocated_bytes.all.current", 0)
    reserved2 = after_free.get("reserved_bytes.all.current", 0)

    torch.cuda.empty_cache()
    after_empty = torch.cuda.memory_stats(device)
    reserved3 = after_empty.get("reserved_bytes.all.current", 0)

    print("PASS: caching allocator stats smoke")
    print(f"  before     alloc={_mb(alloc0):.1f} MiB  reserved={_mb(reserved0):.1f} MiB")
    print(f"  after 64MiB alloc={_mb(alloc1):.1f} MiB  reserved={_mb(reserved1):.1f} MiB")
    print(f"  after free  alloc={_mb(alloc2):.1f} MiB  reserved={_mb(reserved2):.1f} MiB  (cache keeps reserved)")
    print(f"  empty_cache reserved={_mb(reserved3):.1f} MiB")

    if alloc1 <= alloc0:
        raise SystemExit("FAIL: allocated did not grow after torch.empty")
    if reserved1 < alloc1:
        raise SystemExit("FAIL: reserved should be >= allocated after alloc")
    if alloc2 >= alloc1:
        raise SystemExit("FAIL: allocated should drop after free (tensor deleted)")


if __name__ == "__main__":
    main()
