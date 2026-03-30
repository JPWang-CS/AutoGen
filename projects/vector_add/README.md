# Vector Add (NPU专用)

TileLang-Ascend 实现的向量加法算子示例，运行在华为昇腾NPU上。

## 算子描述

计算两个向量的逐元素加法：`C = A + B`

所有计算在 NPU 上执行，精度对比在 CPU 上进行。

## 接口说明

### vector_add

标准版本，支持尾部元素处理。

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add(
    N: int,
    block_N: int = 256,
    dtype: str = "float16",
):
    ...
```

### vector_add_simple

简化版本，要求 N 是 block_N 的整数倍。

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add_simple(
    N: int,
    block_N: int = 256,
    dtype: str = "float16",
):
    ...
```

### vector_add_2d

2D Tensor 加法版本，支持矩阵的逐元素加法。

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add_2d(
    M: int,
    N: int,
    block_M: int = 16,
    block_N: int = 16,
    dtype: str = "float16",
):
    ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| N | int | - | 向量长度 |
| block_N | int | 256 | 每个核处理的元素数量 |
| dtype | str | "float16" | 数据类型 ("float16", "float32", "bfloat16") |

### 输入/输出

**输入:**
- `A`: Tensor of shape (N,), dtype
- `B`: Tensor of shape (N,), dtype

**输出:**
- `C`: Tensor of shape (N,), dtype

## 使用示例

```python
import torch
import torch_npu
from vector_add import vector_add

torch.npu.set_device(0)

# 定义向量长度
N = 1024 * 1024  # 1M elements

# 编译kernel
kernel = vector_add(N, block_N=256, dtype="float16")

# 准备输入数据 (NPU)
a = torch.randn(N, device="npu", dtype=torch.float16)
b = torch.randn(N, device="npu", dtype=torch.float16)

# 调用kernel (NPU版本需要传入shape参数)
c = kernel(a, b, torch.tensor(N, dtype=torch.int32))

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
python benchmark_vector_add.py --N 1048576

# 使用简化版本
python benchmark_vector_add.py --N 1048576 --simple
```

## NPU 硬件约束

算子的 block_N 参数受 NPU 硬件约束限制：

| 参数 | 约束 |
|------|------|
| UB 容量 | block_N * sizeof(dtype) * 3 <= UB容量 (~2MB) |
| 数据对齐 | block_N 需满足32字节对齐 |

推荐配置：
- **910B**: block_N=256 或 512
- **310P**: block_N=128 或 256

## 性能指标

性能主要由 NPU HBM 带宽决定：

- **理论带宽**：取决于NPU型号（如910B约400GB/s）
- **实际带宽**：约理论带宽的80-90%

计算公式：
```
带宽 = (3 * N * sizeof(dtype)) / 延迟
```
其中3表示读2个向量 + 写1个向量。
