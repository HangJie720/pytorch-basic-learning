# autograd

## What to learn

Forward graph recording, `grad_fn`, custom `Function`, backward engine.

## Upstream map

| Topic | Path |
| --- | --- |
| Python API | `torch/autograd/__init__.py`, `function.py` |
| Engine | `torch/csrc/autograd/engine.cpp` |
| Variable / Node | `torch/csrc/autograd/` |

## Call stack

```text
y = f(x); y.backward()
  → Autograd meta on tensors (grad_fn chain)
  → Engine::execute
  → Node::apply (saved tensors)
  → accumulated .grad on leaves
```

During forward, Dispatcher hits `Autograd*` keys before math kernels (see repo README §3.3).

## Cases

- `case_custom_function.py` — minimal `torch.autograd.Function`.
