# nn

## What to learn

`nn.Module`, parameter registration, `forward`, functional API.

## Upstream map

| Topic | Path |
| --- | --- |
| Module | `torch/nn/modules/module.py` |
| Functional | `torch/nn/functional.py` |
| Linear / Conv kernels | `aten/src/ATen/native/` (+ `cuda/`, `cudnn/`) |

## Call stack

```text
module(x)  # __call__
  → forward hooks
  → forward() → F.linear / F.conv2d
  → Dispatcher → ATen native / cuDNN / CUDABlas
```

## Cases

- `case_linear_forward.py` — one Linear vs `F.linear` numerics.
