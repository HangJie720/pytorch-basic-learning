# Figures

Display convention: **width 640px** in Markdown (`<img width="640" …>`; **310** when two share a row). Tall call-stack figures use left-to-right layout; keep pastel documentation box style (not chip strips).

| File | Topic |
| --- | --- |
| `pytorch_architecture.jpg` / `.svg` | Layered composition + `torch.compile` |
| `torch_compile_flow.jpg` / `.svg` | `_dynamo` → `fx` → `_inductor` |
| `dispatcher_mechanism.jpg` | OperatorHandle, key set, redispatch loop |
| `dispatcher_backend_routing.jpg` | DispatchKey → Autograd / CUDA / CPU / PrivateUse1 |
| `dispatcher_eager_stack.jpg` / `.svg` | Eager `torch.add` stack |
| `tensor_call_stack.jpg` | view / Storage |
| `autograd_call_stack.jpg` | `backward` → Engine → Node |
| `nn_call_stack.jpg` | `Module` → functional → Dispatcher |
| `cuda_call_stack.jpg` | stream / allocator / driver |
| `distributed_call_stack.jpg` / `.svg` | c10d → NCCL |
