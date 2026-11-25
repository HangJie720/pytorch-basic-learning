"""Smoke: enable expandable_segments and exercise allocator stats."""
from __future__ import annotations

import os

import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    # Prefer env (must be set before first CUDA context in some builds).
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in conf.replace(" ", ""):
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            print("set expandable_segments:True via _set_allocator_settings")
        except Exception as exc:  # noqa: BLE001
            print(
                "SKIP: cannot enable expandable_segments "
                f"({type(exc).__name__}: {exc}); "
                "retry with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            )
            return

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Slightly varying sizes to stress segment growth / reuse.
    tensors = []
    for i in range(8):
        n = (8 + i) * 1024 * 1024 // 4  # ~8–15 MiB float32
        tensors.append(torch.empty(n, dtype=torch.float32, device=device))

    stats = torch.cuda.memory_stats(device)
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    # Key present on builds that track expandable segments.
    expandable_keys = [k for k in stats if "expandable" in k.lower()]

    print("PASS: expandable_segments alloc smoke")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r}")
    print(f"  allocated={allocated / (1024 ** 2):.1f} MiB  reserved={reserved / (1024 ** 2):.1f} MiB")
    print(f"  expandable-related stats keys: {expandable_keys or '(none exported)'}")
    print(f"  num live tensors={len(tensors)}")

    if allocated <= 0 or reserved < allocated:
        raise SystemExit("FAIL: unexpected memory_stats after alloc pattern")

    del tensors
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
