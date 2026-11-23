# Figures

Flowcharts used by the root README and module notes. Prefer the `.jpg` / `.png` assets in Markdown previews; `.svg` sources are kept for editing.

| File | Topic |
| --- | --- |
| `pytorch_architecture.jpg` / `.svg` | Layered PyTorch composition (`torch/**` → `_C` → `csrc` → c10/ATen → drivers), with `torch.compile` |
| `torch_compile_flow.jpg` / `.svg` | `_dynamo` → `fx` → `_inductor` → kernels / ATen fallback |
| `dispatcher_eager_stack.jpg` / `.svg` | Eager op path through `Dispatcher` (e.g. `torch.add`) |
| `distributed_call_stack.jpg` / `.svg` | `torch.distributed` → ProcessGroupNCCL → NCCL |

Edit the SVGs when labels need a precise tweak; regenerate bitmaps if you want matching PNGs/JPGs.
