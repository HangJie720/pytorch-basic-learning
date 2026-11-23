# Dispatcher overview

Source anchors (upstream PyTorch tree):

| Piece | Path |
| --- | --- |
| Dispatch keys | `c10/core/DispatchKey.h` |
| Dispatcher | `aten/src/ATen/core/dispatch/Dispatcher.h` |
| Operator entries | `aten/src/ATen/core/dispatch/OperatorEntry.*` |
| Python dump helpers | `torch/_dispatch/`, `torch.library` |

## Boxed vs unboxed

- **Unboxed** `Dispatcher::call`: generated / C++ API, typed args, fast path.
- **Boxed** `Dispatcher::callBoxed`: stack of IValues (JIT, some Python paths).

## Typical redispatch

Autograd / Functionalize / Python keys often **redispatch** after peeling their key so the math kernel (CPU/CUDA/…) runs underneath.

```text
call(op, tensors)
  → AutogradKernel
       → redispatch(op, tensorsWithoutAutogradKey)
            → CUDAKernel
```

## Debug tips

```python
import torch
print(torch._C._dispatch_dump("aten::add.Tensor")[:2000])
# or explore
# torch.library / torch._dispatch
```

See the main [README](../../README.md) §3 for full eager / compile / distributed stacks.
