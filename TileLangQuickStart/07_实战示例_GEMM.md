# 07 实战示例：GEMM 矩阵乘法

本章通过一个完整的 GEMM（通用矩阵乘法）示例，展示如何从零开始使用 TileLang 编写高性能 GPU 内核。

---

## 一、问题定义

计算 `C = A @ B`，其中：
- `A`：形状 `(M, K)` 的 float16 矩阵
- `B`：形状 `(K, N)` 的 float16 矩阵
- `C`：形状 `(M, N)` 的 float16 矩阵（使用 float32 累加）

---

## 二、分块策略

GEMM 的核心思想是将大矩阵分块（Tiling），每个 Thread Block 负责计算输出矩阵的一个子块：

```
输出矩阵 C (M × N) 被划分为 (M/BM) × (N/BN) 个 Block
每个 Block 大小为 BM × BN
每个 Block 需要从 A 读取 BM × K 的数据，从 B 读取 K × BN 的数据

进一步将 K 维度分块为 K/BK 段，每次处理 BM × BK 和 BK × BN 的数据块
```

```
A (M × K)          B (K × N)           C (M × N)
┌───────┐         ┌───────────┐       ┌───────────┐
│       │         │           │       │  C[bx,by]  │
│ BM×BK │ ×       │ BK×BN     │  =    │  BM × BN   │
│       │         │           │       │            │
└───────┘         └───────────┘       └───────────┘
```

---

## 三、基础版本（无流水线）

```python
import tilelang
import tilelang.language as T
import torch

# 分块参数
BM = 128  # Block 的 M 维度大小
BN = 128  # Block 的 N 维度大小
BK = 32   # Block 的 K 维度大小

@tilelang.jit(out_idx=[2])
def matmul_basic(M: int, N: int, K: int, dtype: str = "float16"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 启动 (M/BM) × (N/BN) 个 Thread Block，每 Block 128 线程
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            # 分配共享内存缓冲区
            A_shared = T.alloc_shared((BM, BK), dtype)   # A 的子块
            B_shared = T.alloc_shared((BK, BN), dtype)   # B 的子块

            # 分配 Fragment（寄存器）用于累加
            C_frag = T.alloc_fragment((BM, BN), "float32")

            # 清零累加器
            T.clear(C_frag)

            # 沿 K 维度循环，每次处理 BK 列
            for ko in T.serial(T.ceildiv(K, BK)):
                # 步骤 1：从全局内存加载到共享内存
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(B[ko * BK, bx * BN], B_shared)

                # 步骤 2：在共享内存上执行矩阵乘法
                T.gemm(A_shared, B_shared, C_frag)

            # 步骤 3：将结果写回全局内存
            T.copy(C_frag, C[by * BM, bx * BN])

    return kernel
```

### 代码详解

| 步骤 | 指令 | 源 → 目标 | 说明 |
|------|------|----------|------|
| 1a | `T.copy(A[...], A_shared)` | Global → Shared | 加载 A 的子块 |
| 1b | `T.copy(B[...], B_shared)` | Global → Shared | 加载 B 的子块 |
| 2 | `T.gemm(A_shared, B_shared, C_frag)` | Shared → Fragment | 执行矩阵乘累加 |
| 3 | `T.copy(C_frag, C[...])` | Fragment → Global | 写回结果 |

---

## 四、优化版本（软件流水线）

通过 `T.Pipelined` 实现数据加载与计算的重叠：

```python
@tilelang.jit(out_idx=[2])
def matmul_pipelined(M: int, N: int, K: int,
                     block_M: int = 128, block_N: int = 128, block_K: int = 32,
                     num_stages: int = 3, num_threads: int = 128,
                     dtype: str = "float16"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_frag)

            # 使用软件流水线，num_stages=3 表示 3 级流水线
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_frag)

            T.copy(C_frag, C[by * block_M, bx * block_N])

    return kernel
```

### 流水线效果

```
无流水线：                有流水线（3 级）：
Load 0 | Comp 0 |        Load 0 |
Load 1 | Comp 1 |        Load 1 | Comp 0 |
Load 2 | Comp 2 |        Load 2 | Comp 1 |
...                      Comp 2 | ...
                          → 加载和计算重叠，隐藏延迟
```

---

## 五、带偏置的 GEMM

```python
@tilelang.jit(out_idx=[2])
def matmul_bias(
    M: int, N: int, K: int,
    block_M: int = 128, block_N: int = 128, block_K: int = 32,
    num_threads: int = 128, dtype: str = "float16"
):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), "float32")
            bias_frag = T.alloc_fragment((block_N,), dtype)

            T.clear(C_frag)

            # 加载偏置
            T.copy(B_global[bx * block_N], bias_frag)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_frag)

            # 加上偏置
            for i, j in T.grid(block_M, block_N):
                C_frag[i, j] = C_frag[i, j] + T.cast(bias_frag[j], "float32")

            T.copy(C_frag, C[by * block_M, bx * block_N])

    return kernel
```

---

## 六、混合精度 GEMM

```python
@tilelang.jit(out_idx=[2])
def matmul_mixed_precision(
    M: int, N: int, K: int,
    block_M: int = 128, block_N: int = 128, block_K: int = 32,
    num_threads: int = 128,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)

            # 使用高精度累加
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_frag)

            T.copy(C_frag, C[by * block_M, bx * block_N])

    return kernel

# 使用示例：FP16 输入，FP32 累加，FP16 输出
kernel = matmul_mixed_precision(
    M=1024, N=1024, K=1024,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float32",
)
A = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
B = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
C = kernel(A, B)
```

---

## 七、完整测试与验证

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[2])
def matmul(M: int, N: int, K: int, dtype: str = "float16"):
    BM, BN, BK = 128, 128, 32
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), dtype)
            B_s = T.alloc_shared((BK, BN), dtype)
            C_f = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * BM, bx * BN])
    return kernel

# 测试
M, N, K = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

# TileLang 计算
C = matmul(A, B)

# PyTorch 参考
C_ref = A @ B

# 验证精度
max_error = (C - C_ref).abs().max().item()
print(f"最大误差: {max_error:.6f}")
assert max_error < 0.1, f"误差过大: {max_error}"

print("GEMM 验证通过！")

# 性能测试
import time
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    C = matmul(A, B)
torch.cuda.synchronize()
elapsed = time.time() - start

flops = 2 * M * N * K * 100 / elapsed / 1e12
print(f"性能: {flops:.2f} TFLOPS")
```

---

## 八、性能优化清单

| 优化技术 | 对应 TileLang 特性 | 效果 |
|---------|-------------------|------|
| 分块计算 | `T.Kernel` + `T.alloc_shared` | 提高数据复用 |
| Tensor Core 加速 | `T.gemm` | 利用硬件矩阵引擎 |
| 混合精度 | `float16` 输入 + `float32` 累加 | 平衡精度与速度 |
| 软件流水线 | `T.Pipelined` | 隐藏内存延迟 |
| 自动调优 | `@tilelang.autotune` | 搜索最优配置 |
| Shared Memory 优化 | 合理设置 block_M/block_K | 减少 bank conflict |

## 下一步

阅读 [08 实战示例：注意力机制](08_实战示例_注意力机制.md) 学习 Flash Attention 的 TileLang 实现。
