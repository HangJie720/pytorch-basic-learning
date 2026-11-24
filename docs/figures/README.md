# Figures

Display width **640px** in Markdown (full width; avoid squeezing trees into half-width columns).

**Style canon:** match `dispatcher_backend_routing.jpg` (and `dispatcher_mechanism.jpg`) —
top-down **tree** layout, pastel fills with matching borders, rounded boxes, internal title/subtitle dividers, dark-blue orthogonal arrows, flat white background. Spine nodes then branch to leaves where the path forks.

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
