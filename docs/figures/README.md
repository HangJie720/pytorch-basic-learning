# Figures

Display width **640px** in Markdown (full width; avoid squeezing trees into half-width columns).

**Style canon:** match `dispatcher_backend_routing.jpg` (and `dispatcher_mechanism.jpg`) —
top-down **tree** layout, pastel fills with matching borders, rounded boxes, internal title/subtitle dividers, dark-blue orthogonal arrows, flat white background. Spine nodes then branch to leaves where the path forks.

| File | Topic |
| --- | --- |
| `pytorch_architecture.jpg` | Full tree: Python → Compile → Bridge → Libs → Runtime |
| `torch_compile_flow.jpg` | Compile subtree (`_dynamo` → `fx` → `_inductor`) |
| `dispatcher_mechanism.jpg` | OperatorHandle, key set, redispatch loop |
| `dispatcher_backend_routing.jpg` | DispatchKey → Autograd / CUDA / CPU / PrivateUse1 |
| `dispatcher_eager_stack.jpg` | Eager `torch.add` stack |
| `cuda_graph_architecture.jpg` | CUDAGraph capture → replay / Inductor cudagraphs |
| `distributed_architecture.jpg` | ProcessGroup backends + FSDP / mesh / pipeline / RPC |
| `distributed_call_stack.jpg` | Collective path: c10d → NCCL |
| `tensor_call_stack.jpg` | view / Storage (module detail) |
| `autograd_call_stack.jpg` | `backward` → Engine → Node (module detail) |
| `nn_call_stack.jpg` | `Module` → functional → Dispatcher (module detail) |
| `cuda_call_stack.jpg` | stream / allocator / driver (module detail) |
