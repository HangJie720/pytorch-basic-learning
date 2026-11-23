# tensor

## What to learn

Storage, stride, dtype, device, **view vs copy**.

## Upstream map

| Topic | Path |
| --- | --- |
| Python Tensor | `torch/_tensor.py` |
| TensorImpl / Storage | `c10/core/TensorImpl.h`, `StorageImpl.h` |
| View ops | `aten/src/ATen/native/TensorShape.cpp` |

## Call stack (view)

```text
t.view(...) / t.reshape(...)
  → binding in torch/csrc
  → at::view / at::reshape
  → Dispatcher → native view kernel
  → new TensorImpl sharing Storage (view) or allocating (copy path)
```

## Cases

- `case_view_copy.py` — prove view shares storage; `clone` does not.
