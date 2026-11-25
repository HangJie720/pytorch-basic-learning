# nn

## Diagrams

<img src="images/nn_call_stack.jpg" width="640" alt="nn.Module call stack" />

<img src="images/module_forward.png" width="640" alt="module forward" />

<img src="images/param_register.png" width="640" alt="parameter registration" />

## Cases

| Case | Idea |
| --- | --- |
| `case_linear_forward.py` | `nn.Linear` vs `F.linear` |
| `case_state_dict.py` | `state_dict` round-trip |

```bash
cd modules/nn
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_linear_forward.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_state_dict.py
```

## Upstream

| Topic | Path |
| --- | --- |
| Module | `torch/nn/modules/module.py` |
| Functional | `torch/nn/functional.py` |
