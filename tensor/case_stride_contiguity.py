"""Stride / contiguity after transpose."""
from __future__ import annotations

import torch


def main() -> None:
    x = torch.arange(12).reshape(3, 4)
    y = x.t()
    assert x.is_contiguous()
    assert not y.is_contiguous()
    z = y.contiguous()
    assert z.is_contiguous()
    assert tuple(y.stride()) != tuple(z.stride())
    print("PASS: transpose breaks contiguity; contiguous() reallocates")
    print(f"  x.stride={tuple(x.stride())} y.stride={tuple(y.stride())} z.stride={tuple(z.stride())}")


if __name__ == "__main__":
    main()
