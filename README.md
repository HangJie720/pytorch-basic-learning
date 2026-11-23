# pytorch-basic-learning

Map PyTorch **source layout**, **dispatcher / backends**, and **module call stacks** with small cases under `modules/`.

Figures: [`docs/figures/`](docs/figures/). Upstream paths mean a normal PyTorch tree (`torch/`, `aten/`, `c10/`).

---

## 1. Architecture

![PyTorch composition](docs/figures/pytorch_architecture.jpg)

| Layer | Path |
| --- | --- |
| Python | `torch/**` |
| Compile | `_dynamo` → `fx` → `_inductor` |
| Bridge | `torch/_C` ← `torch/csrc` |
| Libs | `ATen` (ops + Dispatcher) · `c10` (Tensor/Storage/Device/allocator) |
| Runtime | CUDA / NCCL / drivers |

![torch.compile](docs/figures/torch_compile_flow.jpg)

---

## 2. Source map (short)

| Tree | Role |
| --- | --- |
| `torch/` | Python API + compile (`_dynamo` / `_inductor` / `fx`) |
| `torch/csrc/` | C++ bindings, autograd, c10d, JIT, CUDA helpers |
| `aten/` | Op schemas, Dispatcher, `native` / CUDA / cuDNN kernels |
| `c10/` | `TensorImpl`, `DispatchKey`, CUDA allocator / stream / guard |
| `torchgen/` | Codegen from YAML |

<details>
<summary>Directory cheat-sheet</summary>

**Python:** `nn/`, `autograd/`, `cuda/`, `distributed/`, `optim/`, `jit/`, `export/`, `fx/`, `_dynamo/`, `_inductor/`, `_dispatch/`  
**csrc:** `autograd/`, `cuda/`, `distributed/c10d/`, `jit/`, `dynamo/`, `inductor/`  
**ATen/c10:** `ATen/core/dispatch/`, `ATen/native(/cuda|/cudnn|/transformers)/`, `c10/core/`, `c10/cuda/`

</details>

---

## 3. Dispatcher & backends

Eager ops go through the Dispatcher. Kernels are selected by **DispatchKey** / **DispatchKeySet** (`c10/core/DispatchKey.h`).

![Dispatcher mechanism](docs/figures/dispatcher_mechanism.jpg)

![Backend routing](docs/figures/dispatcher_backend_routing.jpg)

| Idea | Detail |
| --- | --- |
| Register | each `aten::op` has a table of kernels keyed by DispatchKey |
| Select | compute key set from args; pick highest-priority key |
| Redispatch | wrap keys (Autograd / Functionalize / …) peel themselves and call again |
| Backend | final key → CPU / CUDA / MPS / XPU / PrivateUse1 / … implementation |

![Eager stack (`torch.add`)](docs/figures/dispatcher_eager_stack.jpg)

More notes: [`docs/dispatcher/overview.md`](docs/dispatcher/overview.md).

---

## 4. Module call stacks

### Tensor

![Tensor](docs/figures/tensor_call_stack.jpg)

Case: [`modules/tensor`](modules/tensor/)

### Autograd

![Autograd](docs/figures/autograd_call_stack.jpg)

Case: [`modules/autograd`](modules/autograd/)

### NN

![NN](docs/figures/nn_call_stack.jpg)

Case: [`modules/nn`](modules/nn/)

### CUDA

![CUDA](docs/figures/cuda_call_stack.jpg)

Case: [`modules/cuda`](modules/cuda/)

### Distributed

![Distributed](docs/figures/distributed_call_stack.jpg)

Cases + NCCL diagrams: [`modules/distributed`](modules/distributed/)

### Compile / Export

Compile flowchart is in §1. Export sketch: [`modules/export`](modules/export/).  
c10/ATen reading index: [`modules/c10_aten`](modules/c10_aten/).

---

## 5. Cases

| Module | Entry |
| --- | --- |
| `modules/tensor` | `python case_view_copy.py` |
| `modules/autograd` | `python case_custom_function.py` |
| `modules/nn` | `python case_linear_forward.py` |
| `modules/cuda` | `python case_stream_mem.py` |
| `modules/distributed` | `./run_allreduce.sh` (8 GPUs) |
| `modules/compile` | `python case_compile_basic.py` |
| `modules/export` | `python case_export_basic.py` |

```bash
git clone git@github.com:HangJie720/pytorch-basic-learning.git
cd pytorch-basic-learning/modules/tensor && python case_view_copy.py
```
