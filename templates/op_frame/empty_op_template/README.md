# your_op_name

TileLang实现的your_op_name算子。

## 算子描述

请在此处添加算子的详细描述，包括：
- 算子的数学定义
- 算子的应用场景
- 算子的输入输出规格

## 接口说明

### 函数签名

```python
@tilelang.jit(out_idx=[-1])
def your_op_name(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    ...
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| M | int | - | 矩阵A的行数 |
| N | int | - | 矩阵B的列数 |
| K | int | - | 矩阵A的列数/矩阵B的行数 |
| block_M | int | 64 | M维度的tiling大小 |
| block_N | int | 64 | N维度的tiling大小 |
| block_K | int | 32 | K维度的tiling大小 |
| num_stages | int | 2 | 流水线深度 |
| threads | int | 128 | 每个block的线程数 |
| dtype | T.dtype | T.float16 | 输入/输出数据类型 |
| accum_dtype | T.dtype | T.float32 | 累加数据类型 |

### 输入/输出

**输入:**
- `A`: Tensor of shape (M, K), dtype
- `B`: Tensor of shape (K, N), dtype

**输出:**
- `C`: Tensor of shape (M, N), dtype

## 使用示例

### 基础用法

```python
import torch
import tilelang
from your_op_name import your_op_name

# 定义问题规模
M, N, K = 1024, 1024, 1024

# 编译kernel
kernel = your_op_name(M, N, K)

# 准备输入数据
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

# 调用kernel
c = kernel(a, b)

# 参考实现验证
ref_c = a @ b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
```

### 性能测试

```python
# 获取profiler
profiler = kernel.get_profiler()

# 测量延迟
latency = profiler.do_bench()
print(f"Latency: {latency:.3f} ms")

# 计算TFlops
flops = 2 * M * N * K
tflops = flops / latency * 1e-9
print(f"TFlops: {tflops:.2f}")
```

## 文件结构

```
your_op_name/
├── your_op_name.py         # 核心kernel实现
├── test_your_op_name.py    # 单元测试
├── benchmark_your_op_name.py # 性能基准测试
└── README.md               # 本文档
```

## 运行测试

```bash
# 运行单元测试
python test_your_op_name.py

# 使用pytest运行
pytest test_your_op_name.py -v

# 运行性能测试
python benchmark_your_op_name.py --M 4096 --N 4096 --K 4096

# 运行自动调优
python benchmark_your_op_name.py --M 4096 --N 4096 --K 4096 --autotune
```

## 性能参考

以下是在不同GPU上的性能参考数据：

| GPU | M=N=K | Latency (ms) | TFlops | Speedup vs PyTorch |
|-----|-------|--------------|--------|-------------------|
| A100 | 4096 | TBD | TBD | TBD |
| H100 | 4096 | TBD | TBD | TBD |

## 注意事项

1. 确保输入数据在CUDA设备上
2. 数据类型需要与kernel定义的dtype一致
3. 问题规模需要能被block size整除以获得最佳性能

## 参考

- [TileLang Documentation](https://tilelang.com/)
- [TileLang GitHub](https://github.com/tile-ai/tilelang)
