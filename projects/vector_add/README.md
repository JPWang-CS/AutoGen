# Vector Add

TileLang实现的向量加法算子示例。

## 算子描述

计算两个向量的逐元素加法：`C = A + B`

这是一个简单的示例算子，展示了TileLang的基本使用方法。

## 接口说明

### vector_add

```python
@tilelang.jit(out_idx=[-1])
def vector_add(
    N: int,
    block_size: int = 256,
    threads: int = 256,
    dtype: T.dtype = T.float16,
):
    ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| N | int | - | 向量长度 |
| block_size | int | 256 | 每个block处理的元素数量 |
| threads | int | 256 | 每个block的线程数 |
| dtype | T.dtype | T.float16 | 数据类型 |

### 输入/输出

**输入:**
- `A`: Tensor of shape (N,), dtype
- `B`: Tensor of shape (N,), dtype

**输出:**
- `C`: Tensor of shape (N,), dtype

## 使用示例

```python
import torch
from vector_add import vector_add

# 定义向量长度
N = 1024 * 1024  # 1M elements

# 编译kernel
kernel = vector_add(N)

# 准备输入数据
a = torch.randn(N, device="cuda", dtype=torch.float16)
b = torch.randn(N, device="cuda", dtype=torch.float16)

# 调用kernel
c = kernel(a, b)

# 验证
ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
```

## 文件结构

```
vector_add/
├── vector_add.py         # 核心kernel实现
├── test_vector_add.py    # 单元测试
├── benchmark_vector_add.py # 性能基准测试
└── README.md             # 本文档
```

## 运行测试

```bash
# 运行单元测试
python test_vector_add.py

# 使用pytest运行
pytest test_vector_add.py -v

# 运行性能测试
python benchmark_vector_add.py --N 1048576

# 使用简单版本
python benchmark_vector_add.py --N 1048576 --simple
```

## 可用版本

### vector_add
标准版本，使用block tiling优化。

### vector_add_simple
简化版本，每个线程处理一个元素，适合理解基本概念。

### vector_add_2d
2D tensor加法版本，支持矩阵/tensor的逐元素加法。

## 性能指标

性能主要由内存带宽决定：

- **理论带宽**：取决于GPU型号（如A100约1.5TB/s，H100约3.3TB/s）
- **实际带宽**：约理论带宽的80-90%

计算公式：
```
带宽 = (3 * N * sizeof(dtype)) / 延迟
```
其中3表示读2个向量 + 写1个向量。
