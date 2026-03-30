# your_op_name (NPU专用)

TileLang-Ascend 实现的 your_op_name 算子，运行在华为昇腾NPU上。

## 算子描述

请在此处添加算子的详细描述，包括：
- 算子的数学定义
- 算子的应用场景
- 算子的输入输出规格

所有计算在 NPU 上执行，精度对比在 CPU 上进行。

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
    dtype: str = "float16",
    accum_dtype: str = "float32",
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
| dtype | str | "float16" | 输入/输出数据类型 |
| accum_dtype | str | "float32" | 累加数据类型 |

### 输入/输出

**输入:**
- `A`: Tensor of shape (M, K), dtype
- `B`: Tensor of shape (K, N), dtype

**输出:**
- `C`: Tensor of shape (M, N), dtype

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
python benchmark_your_op_name.py --M 4096 --N 4096 --K 4096

# 运行自动调优
python benchmark_your_op_name.py --M 4096 --N 4096 --K 4096 --autotune
```

## NPU 硬件约束

Tiling 参数受 NPU 硬件约束限制：

| 参数 | 约束 |
|------|------|
| UB 容量 | block_M * block_K * sizeof(dtype) <= UB容量 (~2MB) |
| 分形对齐 | block_M, block_N, block_K 应为分形大小(16x16x16)的整数倍 |
| 数据对齐 | 32字节对齐 (cacheline) |

推荐配置：
- **910B**: block_M=128, block_N=128, block_K=32
- **310P**: block_M=64, block_N=64, block_K=32

## 注意事项

1. 确保输入数据在 NPU 设备上
2. 数据类型需要与kernel定义的dtype一致
3. 问题规模需要能被block size整除以获得最佳性能
4. 精度对比始终在CPU上进行

## 参考

- [TileLang-Ascend GitHub](https://github.com/tile-ai/tilelang-ascend)
- [TileLang Documentation](https://tilelang.com/)
- [TileLang-Ascend 开发指南](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md)
