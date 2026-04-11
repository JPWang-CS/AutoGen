# your_op_name (NPU专用)

TileLang-Ascend 实现的 your_op_name 算子，运行在华为昇腾NPU上。

## 算子描述

请在此处添加算子的详细描述。

所有计算在 NPU 上执行，精度对比在 CPU 上进行。

## NPU 编程规范

| 特性 | NPU 语法 |
|------|---------|
| Kernel 启动 | `T.Kernel(..., is_npu=True) as (cid, vid)` |
| 内存分配 | `T.alloc_ub((shape), dtype)` |
| 向量计算 | `T.tile.add/mul/...` |
| 矩阵计算 | `T.gemm(A, B, C)` |
| 核间同步 | `T.barrier_all()` |
| 作用域 | `with T.Scope("V"):` |

## 接口说明

### 函数签名

```python
@tilelang.jit(out_idx=[-1])
def your_op_name(
    M: int,
    N: int,
    K: int,
    block_M: int = 32,
    block_N: int = 32,
    block_K: int = 32,
    dtype: str = "float16",
):
    ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| M | int | - | 矩阵A的行数 |
| N | int | - | 矩阵B的列数 |
| K | int | - | 矩阵A的列数/矩阵B的行数 |
| block_M | int | 32 | M维度的tiling大小 |
| block_N | int | 32 | N维度的tiling大小 |
| block_K | int | 32 | K维度的tiling大小 |
| dtype | str | "float16" | 输入/输出数据类型 |

### 输入/输出

**输入:**
- `A`: Tensor of shape (M, K), dtype
- `B`: Tensor of shape (K, N)
 dtype

**输出:**
- `C`: Tensor of shape (M, N)
 dtype

## 使用示例

```python
import torch
import torch_npu
from your_op_name import your_op_name

torch.npu.set_device(0)

# 定义问题规模
M, N, K = 1024, 1024, 1024

# 编译kernel
kernel = your_op_name(M, N, K)

# 准备输入数据 (NPU)
a = torch.randn(M, K, device="npu", dtype=torch.float16)
b = torch.randn(K, N, device="npu", dtype=torch.float16)

# 调用kernel
c = kernel(a, b)

# 精度对比在CPU上进行
ref_c = a.cpu() @ b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
```

## 文件结构

```
your_op_name/
├── your_op_name.py           # 核心kernel实现 (NPU专用)
├── test_your_op_name.py      # 单元测试
├── benchmark_your_op_name.py # 性能基准测试
└── README.md                 # 本文档
```

## 运行测试

```bash
# 运行单元测试
python test_your_op_name.py

# 使用pytest运行
pytest test_your_op_name.py -v

# 运行性能测试
python benchmark_your_op_name.py --m 4096 --n 4096 --k 4096
```

## NPU 硬件约束

| 参数 | 约束 |
|------|------|
| UB 容量 | block_M * block_N * sizeof(dtype) <= UB容量 (~2MB) |
| 数据对齐 | 32字节对齐 (cacheline) |
| VEC_NUM | 2 (每个block由2个vector core并行处理) |

## 参考

- [TileLang-Ascend GitHub](https://github.com/tile-ai/tilelang-ascend)
- [TileLang Documentation](https://tilelang.com/)
