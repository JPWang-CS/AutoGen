# Vector Add (NPU专用)

TileLang-Ascend 实现的向量加法算子示例，运行在华为昇腾NPU上。

## 算子描述

计算两个tensor的逐元素加法：`C = A + B`

支持 1D/2D/3D tensor，所有计算在 NPU 上执行，精度对比在 CPU 上进行。

## NPU 编程规范

本算子遵循 TileLang-Ascend NPU 编程规范：

| 特性 | NPU 语法 |
|------|---------|
| Kernel 启动 | `T.Kernel(..., is_npu=True) as (cid, vid)` |
| 内存分配 | `T.alloc_ub((shape), dtype)` |
| 向量计算 | `T.tile.add(c, a, b)` |
| 核间同步 | `T.barrier_all()` |
| 作用域 | `with T.Scope("V"):` |

## 接口说明

### vector_add_1d

1D Vector 加法。

```python
@tilelang.jit(out_idx=[-1])
def vector_add_1d(N, block_N=256, dtype="float16"):
    ...
```

### vector_add_2d

2D Tensor 加法。

```python
@tilelang.jit(out_idx=[-1])
def vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16"):
    ...
```

### vector_add_3d

3D Tensor 加法。

```python
@tilelang.jit(out_idx=[-1])
def vector_add_3d(D, M, N, block_M=16, block_N=16, dtype="float16"):
    ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| N | int | - | 向量长度 / 矩阵列数 |
| M | int | - | 矩阵行数 |
| D | int | - | 3D tensor深度 |
| block_N | int | 256/32 | N维度的block大小 |
| block_M | int | 32 | M维度的block大小 |
| dtype | str | "float16" | 数据类型 |

## 使用示例

```python
import torch
import torch_npu
from vector_add import vector_add_1d, vector_add_2d, vector_add_3d

torch.npu.set_device(0)

# 1D
N = 1024 * 1024
kernel_1d = vector_add_1d(N, block_N=256, dtype="float16")
a = torch.randn(N, device="npu", dtype=torch.float16)
b = torch.randn(N, device="npu", dtype=torch.float16)
c = kernel_1d(a, b)

# 2D
M, N = 1024, 1024
kernel_2d = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")
a = torch.randn(M, N, device="npu", dtype=torch.float16)
b = torch.randn(M, N, device="npu", dtype=torch.float16)
c = kernel_2d(a, b)

# 3D
D, M, N = 64, 128, 128
kernel_3d = vector_add_3d(D, M, N, block_M=16, block_N=16, dtype="float16")
a = torch.randn(D, M, N, device="npu", dtype=torch.float16)
b = torch.randn(D, M, N, device="npu", dtype=torch.float16)
c = kernel_3d(a, b)

# 精度对比在CPU上进行
ref_c = a.cpu() + b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
```

## 文件结构

```
vector_add/
├── vector_add.py           # 核心kernel实现 (NPU专用)
├── test_vector_add.py      # 单元测试
├── benchmark_vector_add.py # 性能基准测试
└── README.md               # 本文档
```

## 运行测试

```bash
# 运行单元测试
python test_vector_add.py

# 使用pytest运行
pytest test_vector_add.py -v

# 运行性能测试
python benchmark_vector_add.py

# 指定数据类型
python benchmark_vector_add.py --dtype float32
```

## 性能参考 (910B NPU)

| 维度 | 问题规模 | 延迟 | 带宽 |
|------|---------|------|------|
| 1D | 1M elements | ~0.4 ms | ~15 GB/s |
| 2D | 1024x1024 | ~0.45 ms | ~14 GB/s |
| 3D | 64x128x128 | ~0.4 ms | ~15 GB/s |

## NPU 硬件约束

| 参数 | 约束 |
|------|------|
| UB 容量 | block_M * block_N * sizeof(dtype) <= UB容量 (~2MB) |
| 数据对齐 | 32字节对齐 (cacheline) |
| VEC_NUM | 2 (每个block由2个vector core并行处理) |

## 参考

- [TileLang-Ascend GitHub](https://github.com/tile-ai/tilelang-ascend)
- [TileLang Documentation](https://tilelang.com/)
