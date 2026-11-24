# pytorch-basic-learning

This repo is a **source-oriented** guide to how PyTorch is put together and how a single op actually runs — from Python API down through bindings, ATen / c10, the Dispatcher, and into CUDA / NCCL backends.

It is meant for people who already use `torch` day to day and want a clearer mental model of the tree (`torch/`, `torch/csrc/`, `aten/`, `c10/`), not another high-level tutorial. Each topic pairs a short call-stack narrative with a small runnable case under `modules/`, so you can jump from diagram → code path → experiment.

Focus areas:

- **Architecture** — layered map of Python, compile (`_dynamo` / `fx` / `_inductor`), C++ bridge, ATen / c10, and native runtime
- **Dispatcher & backends** — how `DispatchKey` / `DispatchKeySet` select and redispatch kernels (CPU / CUDA / PrivateUse1 / …)
- **Module call stacks** — Tensor, Autograd, `nn.Module`, CUDA runtime, and distributed (collectives + symmetric memory, plus notes on FSDP / mesh / pipeline / RPC / checkpoint)
- **Cases** — minimal scripts and `torchrun` smoke tests you can run against a normal PyTorch checkout

Path names in the notes refer to a standard upstream PyTorch source tree.

## Contents

- [1. Architecture](#1-architecture)
- [2. Source map](#2-source-map-short)
- [3. Dispatcher & backends](#3-dispatcher--backends)
- [4. Module call stacks](#4-module-call-stacks)
  - [tensor](modules/tensor/)
  - [autograd](modules/autograd/)
  - [nn](modules/nn/)
  - [cuda](modules/cuda/)
  - [c10_aten](modules/c10_aten/)
  - [distributed](modules/distributed/)
    - [collectives](modules/distributed/collectives/)
    - [symmetric](modules/distributed/symmetric/)
  - [compile](modules/compile/)
  - [export](modules/export/)
- [5. Cases](#5-cases)

---

## 1. Architecture

<img src="docs/figures/pytorch_architecture.jpg" width="640" alt="PyTorch composition" />

| Layer | Path |
| --- | --- |
| Python | `torch/**` |
| Compile | `_dynamo` → `fx` → `_inductor` |
| Bridge | `torch/_C` ← `torch/csrc` |
| Libs | `ATen` (ops + Dispatcher) · `c10` (Tensor/Storage/Device/allocator) |
| Runtime | CUDA / NCCL / drivers |

<img src="docs/figures/torch_compile_flow.jpg" width="640" alt="torch.compile" />

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

<img src="docs/figures/dispatcher_mechanism.jpg" width="640" alt="Dispatcher mechanism" />

<img src="docs/figures/dispatcher_backend_routing.jpg" width="640" alt="Backend routing" />

| Idea | Detail |
| --- | --- |
| Register | each `aten::op` has a table of kernels keyed by DispatchKey |
| Select | compute key set from args; pick highest-priority key |
| Redispatch | wrap keys (Autograd / Functionalize / …) peel themselves and call again |
| Backend | final key → CPU / CUDA / MPS / XPU / PrivateUse1 / … implementation |

<img src="docs/figures/dispatcher_eager_stack.jpg" width="640" alt="Eager stack torch.add" />

More notes: [`docs/dispatcher/overview.md`](docs/dispatcher/overview.md).

---

## 4. Module call stacks

| Tensor · [`modules/tensor`](modules/tensor/) | Autograd · [`modules/autograd`](modules/autograd/) |
| --- | --- |
| <img src="docs/figures/tensor_call_stack.jpg" width="310" alt="Tensor" /> | <img src="docs/figures/autograd_call_stack.jpg" width="310" alt="Autograd" /> |

| NN · [`modules/nn`](modules/nn/) | CUDA · [`modules/cuda`](modules/cuda/) |
| --- | --- |
| <img src="docs/figures/nn_call_stack.jpg" width="310" alt="NN" /> | <img src="docs/figures/cuda_call_stack.jpg" width="310" alt="CUDA" /> |

**Distributed** · [`modules/distributed`](modules/distributed/) — `collectives/`, `symmetric/`, plus FSDP / mesh / pipeline / RPC / checkpoint notes

<img src="docs/figures/distributed_call_stack.jpg" width="640" alt="Distributed" />

**Compile / Export / c10_aten:** §1 compile figure; [`modules/export`](modules/export/); [`modules/c10_aten`](modules/c10_aten/) (DispatchKey dump case).

---

## 5. Cases

| Module | Entry |
| --- | --- |
| `modules/tensor` | `case_view_copy.py`, `case_stride_contiguity.py` |
| `modules/autograd` | `case_custom_function.py`, `case_grad_fn_chain.py` |
| `modules/nn` | `case_linear_forward.py`, `case_state_dict.py` |
| `modules/cuda` | `case_stream_mem.py`, `case_peer_access.py` |
| `modules/c10_aten` | `case_dispatch_dump.py` |
| `modules/distributed/collectives` | `./run_allreduce.sh` (8 GPUs) |
| `modules/distributed/symmetric` | `./run_put_get.sh` (2 GPUs, P2P) |
| `modules/compile` | `case_compile_basic.py` |
| `modules/export` | `case_export_basic.py` |

```bash
git clone git@github.com:HangJie720/pytorch-basic-learning.git
cd pytorch-basic-learning/modules/tensor && python case_view_copy.py
```
