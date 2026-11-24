# Figures

Display width **640px** in Markdown (`310` when two share a row).

**Style canon:** match `dispatcher_mechanism.jpg` / `dispatcher_backend_routing.jpg` — pastel fills with matching colored borders, rounded boxes, dark-blue arrows, numbered steps, bold titles + short subtitles, flat light background. Prefer left-to-right for long call stacks.

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
