# tensor

## Diagrams

<img src="images/tensor_call_stack.jpg" width="640" alt="Tensor view path" />

<img src="images/view_vs_clone.png" width="640" alt="view vs clone" />

<img src="images/dispatch_add.png" width="640" alt="add dispatch" />

## Cases

| Case | Idea |
| --- | --- |
| `case_view_copy.py` | view shares Storage; clone does not |
| `case_stride_contiguity.py` | transpose strides / contiguous() |

```bash
cd modules/tensor
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_view_copy.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python case_stride_contiguity.py
```

## Upstream

| Topic | Path |
| --- | --- |
| TensorImpl | `c10/core/TensorImpl.h` |
| Shape ops | `aten/src/ATen/native/TensorShape.cpp` |
