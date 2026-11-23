"""View vs copy: shared Storage vs new allocation."""
from __future__ import annotations

import torch


def main() -> None:
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    viewed = base.view(4, 3)
    cloned = base.clone()

    assert base.untyped_storage().data_ptr() == viewed.untyped_storage().data_ptr()
    assert base.untyped_storage().data_ptr() != cloned.untyped_storage().data_ptr()

    base[0, 0] = -1
    assert viewed.reshape(3, 4)[0, 0].item() == -1
    assert cloned[0, 0].item() == 0

    print("PASS: view shares Storage; clone allocates")
    print(f"  base.stride={tuple(base.stride())} viewed.stride={tuple(viewed.stride())}")


if __name__ == "__main__":
    main()
