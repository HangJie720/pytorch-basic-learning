# Dispatcher overview

- Mechanism: [`../../modules/c10_aten/images/dispatcher_mechanism.jpg`](../../modules/c10_aten/images/dispatcher_mechanism.jpg)
- Backend routing: [`../../modules/c10_aten/images/dispatcher_backend_routing.jpg`](../../modules/c10_aten/images/dispatcher_backend_routing.jpg)
- Eager stack: [`../../modules/c10_aten/images/dispatcher_eager_stack.jpg`](../../modules/c10_aten/images/dispatcher_eager_stack.jpg)
- Architecture: [`../../modules/architecture/images/pytorch_architecture.jpg`](../../modules/architecture/images/pytorch_architecture.jpg)

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
