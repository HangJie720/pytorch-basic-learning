# Dispatcher overview

- Mechanism: [`../figures/dispatcher_mechanism.jpg`](../figures/dispatcher_mechanism.jpg)
- Backend routing: [`../figures/dispatcher_backend_routing.jpg`](../figures/dispatcher_backend_routing.jpg)
- Eager stack: [`../figures/dispatcher_eager_stack.jpg`](../figures/dispatcher_eager_stack.jpg)
- Architecture: [`../figures/pytorch_architecture.jpg`](../figures/pytorch_architecture.jpg)

## Source anchors

| Piece | Path |
| --- | --- |
| Dispatch keys | `c10/core/DispatchKey.h` |
| Dispatcher | `aten/src/ATen/core/dispatch/Dispatcher.h` |
| Operator entries | `aten/src/ATen/core/dispatch/OperatorEntry.*` |
| Python helpers | `torch/_dispatch/`, `torch.library` |

## Boxed vs unboxed

- **Unboxed** `Dispatcher::call` — typed C++ / generated fast path
- **Boxed** `Dispatcher::callBoxed` — IValue stack (JIT / some Python)

## Redispatch

Wrap keys (Autograd, Functionalize, Python, …) run first, then **redispatch** without that key so the backend kernel (CPU/CUDA/…) executes.

```python
import torch
print(torch._C._dispatch_dump("aten::add.Tensor")[:2000])
```
