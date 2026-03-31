# GEMM参考实现 (NPU专用)

这是一个完整的TileLang-Ascend GEMM（矩阵乘法）NPU参考实现，可作为开发其他算子的模板。

所有计算在NPU上执行，精度对比在CPU上进行。

## NPU 编程规范

本算子遵循 TileLang-Ascend NPU 编程规范：

| 特性 | NPU 语法 |
|------|---------|
| Kernel 启动 | `T.Kernel(..., is_npu=True) as (cid, vid)` |
| 内存分配 | `T.alloc_ub((shape), dtype)` |
| 矩阵计算 | `T.gemm(A, B, C)` |
| 核间同步 | `T.barrier_all()` |
| 作用域 | `with T.Scope("V"):` |

## 功能特性

- 基础GEMM实现 (`gemm`)
- 自动调优支持
- 完整的单元测试

## 文件结构

```
gemm_reference/
├── gemm.py        # 核心kernel实现 (NPU专用)
├── test_gemm.py   # 单元测试
└── README.md      # 本文档
```

## 使用示例

### 基础GEMM

```python
import torch
import torch_npu
from gemm import gemm

torch.npu.set_device(0)

M, N, K = 1024, 1024, 1024
kernel = gemm(M, N, K)

a = torch.randn(M, K, device="npu", dtype=torch.float16)
b = torch.randn(K, N, device="npu", dtype=torch.float16)
c = kernel(a, b)

# 精度对比在CPU上进行
ref_c = a.cpu() @ b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
```

## 运行测试

```bash
# 运行所有测试
python test_gemm.py

# 使用pytest
pytest test_gemm.py -v
```

## 性能测试

```bash
# 基础benchmark
python gemm.py --M 4096 --N 4096 --K 4096

# 自定义block size
python gemm.py --M 4096 --N 4096 --K 4096 --block_M 64 --block_N 64 --block_K 32
```

## NPU 硬件约束

GEMM的tiling参数受NPU硬件约束限制：

| 参数 | 约束 |
|------|------|
| UB 容量 | block_M * block_K * sizeof(dtype) <= UB容量 (~2MB) |
| 分形对齐 | block_M, block_N, block_K 应为16的整数倍 |
| 数据对齐 | 32字节对齐 (cacheline) |

推荐配置：
- **910B**: block_M=32, block_N=32, block_K=32
- **310P**: block_M=16, block_N=16, block_K=16

## 参考

- [TileLang-Ascend GitHub](https://github.com/tile-ai/tilelang-ascend)
- [TileLang Documentation](https://tilelang.com/)
- [TileLang-Ascend 开发指南](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md)
