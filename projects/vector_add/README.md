# Vector Add (NPU专用)

TileLang-Ascend 实现的向量加法算子示例，运行在华为昇腾NPU上。

## 算子描述

计算两个向量的逐元素加法：`C = A + B`

所有计算在 NPU 上执行，精度对比在 CPU 上进行。

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

### vector_add_2d

2D Tensor 加法版本。

```python
@tilelang.jit(out_idx=[-1])
def vector_add_2d(
    M: int,
    N: int,
    block_M: int = 32,
    block_N: int = 32,
    dtype: str = "float16",
):
    ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| M | int | - | 矩阵行数 |
| N | int | - | 矩阵列数 |
| block_M | int | 32 | M维度的block大小 |
| block_N | int | 32 | N维度的block大小 |
| dtype | str | "float16" | 数据类型 |

### 输入/输出

**输入:**
- `A`: Tensor of shape (M, N), dtype
- `B`: Tensor of shape (M, N), dtype

**输出:**
- `C`: Tensor of shape (M, N), dtype

## 使用示例

```python
import torch
import torch_npu
from vector_add import vector_add_2d

torch.npu.set_device(0)

# 定义问题规模
M, N = 1024, 1024

# 编译kernel
kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

# 准备输入数据 (NPU)
a = torch.randn(M, N, device="npu", dtype=torch.float16)
b = torch.randn(M, N, device="npu", dtype=torch.float16)

# 调用kernel
c = kernel(a, b)

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
python benchmark_vector_add.py --m 4096 --n 4096
```

## NPU 硬件约束

算子的 block 参数受 NPU 硬件约束限制：

| 参数 | 约束 |
|------|------|
| UB 容量 | block_M * block_N * sizeof(dtype) <= UB容量 (~2MB) |
| 数据对齐 | block_M, block_N 需满足32字节对齐 |

推荐配置：
- **910B**: block_M=32, block_N=32
- **310P**: block_M=16, block_N=16

## 性能指标

性能主要由 NPU HBM 带宽决定：

- **理论带宽**：取决于NPU型号（如910B约400GB/s）
- **实际带宽**：约理论带宽的80-90%

计算公式：
```
带宽 = (3 * M * N * sizeof(dtype)) / 延迟
```
其中3表示读2个矩阵 + 写1个矩阵。
