# 07 实战示例：GEMM 矩阵乘法

本章通过一个完整的 GEMM（通用矩阵乘法）示例，展示如何从零开始使用 TileLang-Ascend 编写高性能NPU内核。

---

## 一、问题定义

计算 `C = A @ B`，其中：
- `A`：形状 `(M, K)` 的 float16 矩阵
- `B`：形状 `(K, N)` 的 float16 矩阵
- `C`：形状 `(M, N)` 的 float16 矩阵（使用 float32 累加）

---

## 二、分块策略

GEMM 的核心思想是将大矩阵分块（Tiling），每个AI Core负责计算输出矩阵的一个子块：

```
输出矩阵 C (M x N) 被划分为 (M/block_M) x (N/block_N) 个 Block
每个 Block 大小为 block_M x block_N
每个 Block 需要从 A 读取 block_M x K 的数据，从 B 读取 K x block_N 的数据

进一步将 K 维度分块为 K/K_L1 段，每次处理 block_M x K_L1 和 K_L1 x block_N 的数据块
```

```
A (M x K)           B (K x N)            C (M x N)
+----------+       +------------+       +------------+
|          |       |            |       |  C[bx,by]  |
| block_M  | x     | K_L1 x     |  =    |  block_M   |
| x K_L1   |       | block_N    |       |  x block_N |
|          |       |            |       |            |
+----------+       +------------+       +------------+
```

---

## 三、Expert模式 GEMM（基础版本）

```python
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            # Expert模式：显式分配L1和L0C
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    # GM -> L1：加载矩阵块
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    # L1矩阵乘法，结果累加到L0C
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                # L0C -> GM：写回结果
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# 运行
M, N, K = 1024, 1024, 1024
func = matmul(M, N, K, 128, 256, 64)

torch.manual_seed(0)
a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()
print("init successful!")

c = func(a, b)
ref_c = a @ b

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

### 代码详解

| 步骤 | 指令 | 源 -> 目标 | 说明 |
|------|------|-----------|------|
| 1a | `T.copy(A[...], A_L1)` | GM -> L1 | 加载A的子块到Cube缓存 |
| 1b | `T.copy(B[...], B_L1)` | GM -> L1 | 加载B的子块到Cube缓存 |
| 2 | `T.gemm_v0(A_L1, B_L1, C_L0, init=...)` | L1 -> L0C | 执行矩阵乘累加 |
| 3 | `T.copy(C_L0, C[...])` | L0C -> GM | 写回结果到全局内存 |

### 关键点

1. **`T.Kernel(m_num * n_num, is_npu=True)`**：启动 m_num * n_num 个并发任务
2. **`T.alloc_L0C()`**：分配Cube Core的矩阵输出寄存器，用于累加
3. **`init=(k == 0)`**：首次迭代清零累加器，后续迭代累加
4. **`T.barrier_all()`**：确保数据搬运完成后再开始计算

---

## 四、Developer模式 GEMM（自动CV分离）

使用 `pass_configs` 开启自动优化，代码更简洁：

```python
import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_developer(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            # Developer模式：使用alloc_shared/alloc_fragment
            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 无需显式T.Scope("C")和T.barrier_all()
            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# 运行
func = matmul_developer(1024, 1024, 1024, 128, 256, 64)
a = torch.randn(1024, 1024).half().npu()
b = torch.randn(1024, 1024).half().npu()
c = func(a, b)
torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

### Developer模式 vs Expert模式对比

| 特性 | Developer模式 | Expert模式 |
|------|-------------|-----------|
| 内存分配 | `alloc_shared` / `alloc_fragment` | `alloc_L1` / `alloc_ub` / `alloc_L0C` 等 |
| 作用域 | 无需显式 `T.Scope` | 需要显式 `T.Scope("C")` / `T.Scope("V")` |
| 同步 | 自动插入 `barrier_all` | 手动插入 `barrier_all` |
| CV分离 | 自动 | 手动 |
| 代码复杂度 | 简单 | 复杂 |
| 性能控制 | 编译器优化 | 开发者完全控制 |

---

## 五、Persistent模式 GEMM（多核调度）

当数据块数量远大于AI Core数量时，使用 `T.Persistent` 进行缓存友好的调度：

```python
@tilelang.jit(out_idx=[-1])
def matmul_persistent(M, N, K, block_M, block_N, K_L1,
                      core_num=24, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                # T.Persistent：将数据块分配给AI Core
                for bx, by in T.Persistent(
                    [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                    core_num, cid
                ):
                    loop_k = T.ceildiv(K, K_L1)
                    for k in T.serial(loop_k):
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                        T.copy(B[k * K_L1, by * block_N], B_L1)
                        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


# 运行：core_num=24个AI Core处理所有数据块
func = matmul_persistent(8192, 1024, 8192, 128, 256, 64, core_num=24)
```

---

## 六、Pipelined模式 GEMM（计算搬运重叠）

使用 `T.Pipelined` 实现数据加载和计算的重叠：

```python
@tilelang.jit(out_idx=[-1])
def matmul_pipelined(M, N, K, block_M, block_N, K_L1,
                     dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                # 流水线：num_stages=2 实现预取+计算重叠
                for k in T.Pipelined(T.ceildiv(K, K_L1), num_stages=2):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

---

## 七、Expert模式 GEMM（带L0A/L0B流水线）

使用 `T.mma` 和 `T.alloc_L0A`/`T.alloc_L0B` 进行精细的流水线控制：

```python
@tilelang.jit(out_idx=[-1])
def matmul_intrinsic(M, N, K, block_M, block_N, block_K, K_L1, S1, S2,
                     dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    core_num = 20

    @T.macro
    def init_flag():
        T.set_flag("mte1", "mte2", 0)
        T.set_flag("mte1", "mte2", 1)
        T.set_flag("m", "mte1", 0)
        T.set_flag("m", "mte1", 1)
        T.set_flag("fix", "m", 0)

    @T.macro
    def clear_flag():
        T.wait_flag("mte1", "mte2", 0)
        T.wait_flag("mte1", "mte2", 1)
        T.wait_flag("m", "mte1", 0)
        T.wait_flag("m", "mte1", 1)
        T.wait_flag("fix", "m", 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            # L1带流水线缓冲
            A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)

            # L0A/L0B带流水线缓冲
            A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)

            # L0C输出
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()

                for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
                    cid = T.use_swizzle(i * core_num + cid, M, N, K, block_M, block_N, off=3)
                    if cid < m_num * n_num:
                        bx = cid // n_num
                        by = cid % n_num

                        loop_k = T.ceildiv(K, K_L1)

                        # ... 使用set_flag/wait_flag精细控制流水线
                        for k in T.serial(loop_k):
                            # L1 -> L0A/L0B 搬运
                            loop_kk = T.ceildiv(K_L1, block_K)
                            for kk in T.serial(loop_kk):
                                T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
                                T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])

                                # mma矩阵乘法
                                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0,
                                      init=T.And(k == 0, kk == 0))

                        T.copy(C_L0, C[bx * block_M, by * block_N])

                clear_flag()

    return main


# 运行：S1=2, S2=2 为L1和L0A/L0B的流水线缓冲深度
func = matmul_intrinsic(8192, 1024, 8192, 128, 256, 64, 256, 2, 2)
```

---

## 八、处理尾块（Tail Block）

当矩阵尺寸不是block size的整数倍时，需要处理尾块：

```python
@tilelang.jit(out_idx=[-1])
def gemm_tail_block(M, N, K, block_M, block_N, block_K,
                    dtype="float16", accum_type="float32"):
    m_num = M // block_M
    n_num = N // block_N
    k_num = K // block_K

    m_tail = M - m_num * block_M
    n_tail = N - n_num * block_N
    k_tail = K - k_num * block_K

    total_m_blocks = m_num + (1 if m_tail > 0 else 0)
    total_n_blocks = n_num + (1 if n_tail > 0 else 0)
    total_blocks = total_m_blocks * total_n_blocks

    @T.prim_func
    def main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(total_blocks, is_npu=True) as (cid, _):
            bx = cid // total_n_blocks
            by = cid % total_n_blocks
            with T.Scope("C"):
                # 正常块
                if bx < m_num and by < n_num:
                    a_1 = T.alloc_L1([block_M, block_K], dtype)
                    b_1 = T.alloc_L1([block_K, block_N], dtype)
                    c_1 = T.alloc_L0C([block_M, block_N], accum_type)

                    for k in T.serial(k_num):
                        T.copy(A[bx * block_M, k * block_K], a_1)
                        T.copy(B[k * block_K, by * block_N], b_1)
                        T.gemm_v0(a_1, b_1, c_1, init=(k == 0))

                    if k_tail > 0:
                        a_k = T.alloc_L1([block_M, k_tail], dtype)
                        b_k = T.alloc_L1([k_tail, block_N], dtype)
                        T.copy(A[bx * block_M, k_num * block_K], a_k)
                        T.copy(B[k_num * block_K, by * block_N], b_k)
                        T.gemm_v0(a_k, b_k, c_1, init=False)

                    T.copy(c_1, C[bx * block_M, by * block_N])

                # 其他尾块情况类似处理...

    return main
```

---

## 九、性能测试

```python
from tilelang.profiler import do_bench

func = matmul(8192, 1024, 8192, 128, 256, 64)

a = torch.randn(8192, 8192).half().npu()
b = torch.randn(8192, 1024).half().npu()

c = func(a, b)
ref_c = a @ b

torch.npu.synchronize()

tilelang_time = do_bench(lambda: func(a, b))
torch_time = do_bench(lambda: a @ b)

print(f"TileLang time: {tilelang_time} ms")
print(f"Torch time: {torch_time} ms")

torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

---

## 十、性能优化清单

| 优化技术 | 对应 TileLang 特性 | 效果 |
|---------|-------------------|------|
| 分块计算 | `T.Kernel` + `T.alloc_L1` | 提高数据复用 |
| Cube Core加速 | `T.gemm_v0` / `T.mma` | 利用硬件矩阵引擎 |
| 混合精度 | `float16` 输入 + `float32` 累加 | 平衡精度与速度 |
| 软件流水线 | `T.Pipelined` | 隐藏内存延迟 |
| 持久化调度 | `T.Persistent` | 提高缓存命中率 |
| 自动CV分离 | `TL_ASCEND_AUTO_CV_COMBINE` | 简化融合算子开发 |
| 自动同步 | `TL_ASCEND_AUTO_SYNC` | 减少同步遗漏 |
| 内存规划 | `TL_ASCEND_MEMORY_PLANNING` | 减少片上存储占用 |
| L0A/L0B流水线 | `T.alloc_L0A/L0B` + `T.mma` | 精细流水线控制 |

## 下一步

阅读 [08 实战示例：注意力机制](08_实战示例_注意力机制.md) 学习 Flash Attention 的 TileLang-Ascend 实现。
