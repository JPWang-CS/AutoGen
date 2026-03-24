# GEMM参考实现

这是一个完整的TileLang GEMM（矩阵乘法）参考实现，可作为开发其他TileLang算子的模板。

## 功能特性

- 基础GEMM实现 (`gemm`)
- 带L2 Cache优化的GEMM (`gemm_with_swizzle`)
- 支持B矩阵转置的GEMM (`gemm_transposed_b`)
- 自动调优支持
- 完整的单元测试

## 文件结构

```
gemm_reference/
├── gemm.py        # 核心kernel实现
├── test_gemm.py   # 单元测试
└── README.md      # 本文档
```

## 使用示例

### 基础GEMM

```python
from gemm import gemm

M, N, K = 1024, 1024, 1024
kernel = gemm(M, N, K)

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = kernel(a, b)
```

### 带优化的GEMM

```python
from gemm import gemm_with_swizzle

kernel = gemm_with_swizzle(M, N, K, enable_swizzle=True)
c = kernel(a, b)
```

### 转置B矩阵

```python
from gemm import gemm_transposed_b

# 计算 C = A @ B^T
kernel = gemm_transposed_b(M, N, K)

a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(N, K, device="cuda", dtype=torch.float16)  # 注意shape
c = kernel(a, b)
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

# 自动调优
python gemm.py --M 4096 --N 4096 --K 4096 --autotune
```
