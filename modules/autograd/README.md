# autograd

## Diagrams

<img src="images/autograd_call_stack.jpg" width="640" alt="Autograd call stack" />

<img src="images/backward_stack.png" width="640" alt="backward stack" />

<img src="images/custom_function.png" width="640" alt="custom Function" />

## Cases

| Case | Idea |
| --- | --- |
| `case_custom_function.py` | minimal `torch.autograd.Function` |
| `case_grad_fn_chain.py` | print `grad_fn` / `next_functions` |

```bash
cd modules/autograd
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_custom_function.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_grad_fn_chain.py
```

## Upstream

| Topic | Path |
| --- | --- |
| Python API | `torch/autograd/` |
| Engine | `torch/csrc/autograd/engine.cpp` |
