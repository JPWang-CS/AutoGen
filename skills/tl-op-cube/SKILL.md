---
name: tl-op-cube
description: 生成纯Cube Core算子（矩阵乘法 GEMM、Batch GEMM、Grouped GEMM、GEMV、Conv2D）。基于 tilelang-ascend/examples/gemm/ 实际示例。使用 T.alloc_L1 + T.alloc_L0C + T.gemm_v0 Expert模式，或 T.alloc_shared + T.alloc_fragment + T.gemm_v0 Developer模式。支持流水线T.Pipelined、底层intrinsic flag流水线、持久化T.Persistent、swizzle调度、转置、尾部块处理、自动调优等高级模式。
---

# 纯矩阵乘法算子技能 (Cube Core Only)

基于 `tilelang-ascend/examples/gemm/` 实际示例和 `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/` API参考。

## 两种编程模式

| 特性 | Expert 模式 | Developer 模式 |
|------|------------|---------------|
| 内存分配 | `T.alloc_L1()`, `T.alloc_L0C()` | `T.alloc_shared()`, `T.alloc_fragment()` |
| 同步 | 手动 `T.barrier_all()` | 编译器自动插入（pass_configs） |
| 作用域 | 必须 `T.Scope("C")` | 无需（编译器推断） |
| 适用场景 | 精细控制、流水线优化 | 快速开发、简洁代码 |

## 核心API

### GEMM函数

| API | 说明 | 参数 |
|-----|------|------|
| `T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)` | 块级矩阵乘 C += op(A) x op(B) | A,B在L1/shared; C在L0C/fragment |
| `T.mma(A, B, C, init=False)` | 底层矩阵乘累加 | A在L0A; B在L0B; C在L0C; 不支持transpose |

**init参数说明**:
- `init=True`: 计算前将C清零（首次迭代使用）
- `init=False`: 在已有C上累加（后续迭代使用）
- 典型写法: `init=(k == 0)` 自动判断
- **嵌套循环**: `init=T.And(k == 0, kk == 0)` 用于L1+L0两级分块

### 辅助函数

| API | 说明 |
|-----|------|
| `T.use_swizzle(idx, M, N, K, block_M, block_N, off=3)` | Block调度swizzle优化，提高L2 cache命中率 |
| `T.Persistent([m_num, n_num], core_num, cid)` | 持久化迭代器，自动分配多轮block到固定core |
| `T.And(cond1, cond2)` | 编译期逻辑与，用于init条件 |
| `T.ceildiv(a, b)` | 编译期向上取整除法 |
| `T.macro` | 宏定义，用于代码复用(如flag初始化/清理) |
| `T.func_attr({"enable_auto_sync": True})` | 函数级属性，开启自动同步 |

### 内存分配

| Expert 模式 | Developer 模式 | 说明 |
|------------|---------------|------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` | 数据中转 (L1 Buffer) |
| `T.alloc_L0A(shape, dtype)` | - | Cube左矩阵输入 (仅mma用) |
| `T.alloc_L0B(shape, dtype)` | - | Cube右矩阵输入 (仅mma用) |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` | Cube输出/累加 (L0C) |

### 作用域

| Scope | 说明 | 包含操作 |
|-------|------|---------|
| `T.Scope("C")` | Cube Core域 | T.copy (GM↔L1), T.gemm_v0, T.mma, T.barrier_all |

### Kernel启动

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num
    by = cid % n_num
```

**注意**: Cube操作不使用VEC_NUM，vid用`_`忽略。每个core处理完整block。

## Expert模式 GEMM模板 (推荐)

来源: `tilelang-ascend/examples/gemm/example_gemm.py`

```python
import tilelang
import tilelang.language as T

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

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

## Developer模式 GEMM模板

来源: `tilelang-ascend/examples/gemm/example_gemm_pto_developer.py` 和 `example_gemm_infer_scope.py`

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
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

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)

                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

## 更多GEMM模板

### Persistent模式 (多Block循环，数据量 > 核数)

来源: `tilelang-ascend/examples/gemm/example_gemm_persistent.py`

```python
@tilelang.jit(out_idx=[-1])
def matmul_persistent(M, N, K, block_M, block_N, K_L1,
                      dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    core_num = 24

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                for bx, by in T.Persistent(
                    [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                    core_num, cid):
                    loop_k = T.ceildiv(K, K_L1)
                    for k in T.serial(loop_k):
                        T.copy(A[bx * block_M, k * K_L1], A_L1)
                        T.copy(B[k * K_L1, by * block_N], B_L1)
                        T.barrier_all()
                        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                        T.barrier_all()
                    T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

### 带转置的GEMM (C = A @ B^T)

来源: `tilelang-ascend/examples/gemm/example_gemm_transpose_l1.py`

```python
@tilelang.jit(out_idx=[-1])
def matmul_transB(M, N, K, block_M, block_N, K_L1,
                  dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),   # 注意: B shape是 (N, K)
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((block_N, K_L1), dtype)

            # 使用L0A/L0B + mma方式支持转置
            A_L0 = T.alloc_L0A((block_M, K_L1), dtype)
            B_L0 = T.alloc_L0B((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M:(bx+1)*block_M, k*K_L1:(k+1)*K_L1], A_L1)
                    T.copy(B[by * block_N:(by+1)*block_N, k*K_L1:(k+1)*K_L1], B_L1)
                    T.barrier_all()

                    T.copy(A_L1, A_L0)
                    T.copy(B_L1, B_L0, transpose=True)
                    T.barrier_all()
                    T.mma(A_L0, B_L0, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[bx*block_M:(bx+1)*block_M, by*block_N:(by+1)*block_N])

    return main
```

### 底层流水线GEMM (intrinsic, 高性能)

来源: `tilelang-ascend/examples/gemm/example_gemm_intrinsic.py`

使用L0A/L0B分配、多级L1缓冲(S1=2)和L0缓冲(S2=2)，配合 `T.set_flag`/`T.wait_flag` 实现MTE2-MTE1-M-FIX四级流水线:

```python
@tilelang.jit(out_idx=[-1])
def matmul_intrinsic(M, N, K, block_M=128, block_N=256, block_K=64,
                     K_L1=256, S1=2, S2=2, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    core_num = 20  # AI Core数量

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
        A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(core_num, is_npu=True) as (cid, _):
            # 多级缓冲: S1级L1, S2级L0
            A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)
            A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)
            B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                init_flag()
                for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
                    cid = T.use_swizzle(i * core_num + cid, M, N, K, block_M, block_N, off=3)
                    if cid < m_num * n_num:
                        bx = cid // n_num
                        by = cid % n_num
                        loop_k = T.ceildiv(K, K_L1)

                        # 预取第一块K
                        T.wait_flag("mte1", "mte2", 0)
                        T.copy(A[bx*block_M, 0], A_L1[0, :, :])
                        T.copy(B[0, by*block_N], B_L1[0, :, :])
                        T.set_flag("mte2", "mte1", 0)
                        T.wait_flag("fix", "m", 0)

                        for k in T.serial(loop_k):
                            # 预取下一块K到L1
                            if k < loop_k - 1:
                                T.wait_flag("mte1", "mte2", (k+1) % S1)
                                T.copy(A[bx*block_M, (k+1)*K_L1], A_L1[(k+1)%S1, :, :])
                                T.copy(B[(k+1)*K_L1, by*block_N], B_L1[(k+1)%S1, :, :])
                                T.set_flag("mte2", "mte1", (k+1) % S1)

                            # L1 -> L0A/L0B -> mma
                            loop_kk = T.ceildiv(K_L1, block_K)
                            for kk in T.serial(loop_kk):
                                if kk == 0:
                                    T.wait_flag("mte2", "mte1", k % S1)
                                T.wait_flag("m", "mte1", kk % S2)
                                T.copy(A_L1[k%S1, 0, kk*block_K], A_L0[kk%S2, :, :])
                                T.copy(B_L1[k%S1, kk*block_K, 0], B_L0[kk%S2, :, :])
                                if kk == 3:
                                    T.set_flag("mte1", "mte2", k % S1)
                                T.set_flag("mte1", "m", kk % S2)
                                T.wait_flag("mte1", "m", kk % S2)
                                T.mma(A_L0[kk%S2,:,:], B_L0[kk%S2,:,:], C_L0,
                                      init=T.And(k == 0, kk == 0))
                                T.set_flag("m", "mte1", kk % S2)

                        T.set_flag("m", "fix", 0)
                        T.wait_flag("m", "fix", 0)
                        T.copy(C_L0, C[bx*block_M, by*block_N])
                        T.set_flag("fix", "m", 0)
                clear_flag()
    return main
```

**要点**:
- **四级流水线**: MTE2(GM->L1) -> MTE1(L1->L0A/L0B) -> M(mma计算) -> FIX(后处理+写回)
- **`T.use_swizzle()`**: swizzle调度优化，off=3 表示偏移量，提高L2 cache命中率
- **`T.And(cond1, cond2)`**: 编译期逻辑与，用于 `init=T.And(k==0, kk==0)`
- **L1 双缓冲 (S1=2)**: 当前块计算时预取下一块K到L1
- **L0 双缓冲 (S2=2)**: 当前块mma时加载下一块到L0A/L0B
- **`T.macro`**: 将 flag 初始化/清理定义为宏，避免重复代码

### Persistent模式 (T.Persistent + swizzle)

来源: `tilelang-ascend/examples/gemm/example_gemm_intrinsic_persistent.py`

使用 `T.Persistent` 迭代器替代手动循环 + `T.use_swizzle`:

```python
with T.Scope("C"):
    init_flag()
    for bx, by in T.Persistent(
        [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
        core_num, cid):
        # ... 与intrinsic相同的K循环和流水线逻辑 ...
    clear_flag()
```

**要点**:
- `T.Persistent([m_num, n_num], core_num, cid)` 自动分配 block 到 core
- 当 `m_num * n_num > core_num` 时，每个 core 循环处理多个 block
- C_L0 在不同 block 之间会被覆盖，所以每个 block 结束时必须写回 GM

### Tail Block GEMM (非对齐矩阵)

来源: `tilelang-ascend/examples/gemm/example_gemm_tail_block.py`

当M/N/K不是block大小的整数倍时，需要处理边界block:

```python
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gemm_tail_block(M, N, K, block_M, block_N, block_K, dtype="float16", accum_type="float32"):
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
    def main(A: T.Tensor([M, K], dtype), B: T.Tensor([K, N], dtype), C: T.Tensor([M, N], dtype)):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(total_blocks, is_npu=True) as (cid, _):
            bx = cid // total_n_blocks
            by = cid % total_n_blocks
            with T.Scope("C"):
                # Case 1: 完整 block (无tail)
                if bx < m_num and by < n_num:
                    a_1 = T.alloc_L1([block_M, block_K], dtype)
                    b_1 = T.alloc_L1([block_K, block_N], dtype)
                    c_1 = T.alloc_L0C([block_M, block_N], accum_type)
                    for k in T.serial(k_num):
                        T.copy(A[bx*block_M, k*block_K], a_1)
                        T.copy(B[k*block_K, by*block_N], b_1)
                        T.gemm_v0(a_1, b_1, c_1, init=(k == 0))
                    if k_tail > 0:
                        a_k_1 = T.alloc_L1([block_M, k_tail], dtype)
                        b_k_1 = T.alloc_L1([k_tail, block_N], dtype)
                        T.copy(A[bx*block_M, k_num*block_K], a_k_1)
                        T.copy(B[k_num*block_K, by*block_N], b_k_1)
                        T.gemm_v0(a_k_1, b_k_1, c_1, init=False)
                    T.copy(c_1, C[bx*block_M, by*block_N])

                # Case 2: M tail (底部边缘)
                elif bx == m_num and by < n_num and m_tail > 0:
                    a_m_2 = T.alloc_L1([m_tail, block_K], dtype)
                    b_2 = T.alloc_L1([block_K, block_N], dtype)
                    c_m_2 = T.alloc_L0C([m_tail, block_N], accum_type)
                    # ... 类似但用 m_tail 替代 block_M ...

                # Case 3: N tail (右侧边缘)
                elif bx < m_num and by == n_num and n_tail > 0:
                    # ... 用 n_tail 替代 block_N ...

                # Case 4: M和N都tail (右下角)
                elif bx == m_num and by == n_num and m_tail > 0 and n_tail > 0:
                    # ... 用 m_tail 和 n_tail 替代 block_M 和 block_N ...
    return main
```

**要点**:
- **4种case**: (1) 完整block, (2) M-tail, (3) N-tail, (4) M+N-tail
- **K维tail**: 在完整K循环后额外处理 `k_tail`，使用 `init=False` 累加
- **L1/L0C动态大小**: tail block 使用实际 tail 大小而非 block 大小分配
- **`T.func_attr({"enable_auto_sync": True})`**: 函数级属性开启自动同步
- **`if/elif` 分支**: 在 `T.Scope("C")` 内用条件分支处理不同case

---

## Batch GEMM 模板

来源: `tilelang-ascend/examples/batch_gemm/batch_gemm.py`

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def batch_matmul(B, M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A_mat: T.Tensor((B, M, K), dtype),
            B_mat: T.Tensor((B, K, N), dtype),
            C_mat: T.Tensor((B, M, N), dtype),
    ):
        total = B * m_num * n_num
        with T.Kernel(total, is_npu=True) as (cid, _):
            bid = cid // (m_num * n_num)
            rem = cid % (m_num * n_num)
            bx = rem // n_num
            by = rem % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A_mat[bid, bx * block_M, k * K_L1], A_L1)
                T.copy(B_mat[bid, k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C_mat[bid, bx * block_M, by * block_N])

    return main
```

**要点**:
- **3D坐标**: `bid`(batch), `bx`(M), `by`(N) — `total = B * m_num * n_num`
- **3D Tensor索引**: `A_mat[bid, bx*block_M, k*K_L1]` — 第一维是batch
- **与普通GEMM几乎相同**: 只是多了batch维度的索引
- **Developer模式**: 使用 `T.alloc_shared` + `T.alloc_fragment`

### Batch GEMM Expert模式

来源: `tilelang-ascend/examples/batch_gemm/batch_gemm.py` (改编)

```python
@tilelang.jit(out_idx=[-1])
def batch_matmul_expert(B, M, N, K, block_M, block_N, K_L1,
                        dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
            A_mat: T.Tensor((B, M, K), dtype),
            B_mat: T.Tensor((B, K, N), dtype),
            C_mat: T.Tensor((B, M, N), dtype),
    ):
        total = B * m_num * n_num
        with T.Kernel(total, is_npu=True) as (cid, _):
            bid = cid // (m_num * n_num)
            rem = cid % (m_num * n_num)
            bx = rem // n_num
            by = rem % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A_mat[bid, bx * block_M, k * K_L1], A_L1)
                    T.copy(B_mat[bid, k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C_mat[bid, bx * block_M, by * block_N])

    return main
```

**要点**:
- Expert模式使用 `T.alloc_L1` + `T.alloc_L0C` + `T.Scope("C")` + 手动 `T.barrier_all()`
- Developer模式使用 `T.alloc_shared` + `T.alloc_fragment`，无需 Scope 和 barrier

---

## GEMV 模板 (Cube Core版)

来源: `tilelang-ascend/examples/gemv/example_gemv_c.py`

```python
@tl.jit(out_idx=[-1], pass_configs={...})
def simple_gemv(N, K, block_N, block_K, dtype="float16", accum_dtype="float32"):
    """Cube core GEMV: y = x @ A^T"""
    FRACTAL_SIZE = 16   # Cube分形大小，即使(1,16)也占(16,16)空间

    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)

    @T.prim_func
    def main(x: T.Tensor((K,), dtype), A: T.Tensor((N, K), dtype), y: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            bn = cid % n_num

            A_L1 = T.alloc_L1((block_N, block_K), dtype)
            x_L1 = T.alloc_L1((FRACTAL_SIZE, block_K), dtype)   # 向量x按分形大小分配
            C_L0 = T.alloc_L0C((FRACTAL_SIZE, block_N), accum_dtype)

            for bk in T.serial(k_num):
                T.copy(x[bk * block_K], x_L1)
                T.copy(A[bn * block_N, bk * block_K], A_L1)
                T.gemm_v0(x_L1, A_L1, C_L0, transpose_B=True, init=(bk == 0))

            T.copy(C_L0, y[bn * block_N])

    return main
```

**要点**:
- **Cube GEMV**: 将向量 x 视为 (1, K) 矩阵，用 Cube Core 做 matmul
- **FRACTAL_SIZE = 16**: Cube Core 的最小计算单元是 16x16，x_L1 和 C_L0 第一维必须 >= 16
- **`transpose_B=True`**: 因为 A 的 shape 是 (N, K)，需要转置后与 x 相乘
- **输出**: C_L0 shape 为 (FRACTAL_SIZE, block_N) = (16, block_N)，只有第一行是有效结果

---

## Conv2D 模板 (im2col + GEMM)

来源: `tilelang-ascend/examples/convolution/example_convolution.py`

Conv2D 通过 im2col 转换为 GEMM 实现:

```python
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

tilelang.cache.clear_cache()

B, C, H, W, OC, KH, KW, stride, padding = 2, 2, 15, 15, 128, 8, 8, 1, 0
HO = (H + 2 * padding - KH) // stride + 1
WO = (W + 2 * padding - KW) // stride + 1

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M=128, block_N=256, block_K=64,
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

            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, block_K)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])
    return main

# im2col: 将输入展开为 (C*KH*KW, B*HO*WO) 矩阵
def im2col(input_tensor, KH, KW, stride, padding):
    input_flat = torch.zeros((C * KH * KW, B * HO * WO),
                             dtype=input_tensor.dtype, device=input_tensor.device)
    for n in range(B):
        for i in range(HO):
            for j in range(WO):
                h_start = i * stride - padding
                w_start = j * stride - padding
                col_idx = n * HO * WO + i * WO + j
                row_idx = 0
                for c in range(C):
                    for m in range(KH):
                        for k in range(KW):
                            h = h_start + m
                            w = w_start + k
                            if 0 <= h < H and 0 <= w < W:
                                input_flat[row_idx, col_idx] = input_tensor[n, c, h, w]
                            else:
                                input_flat[row_idx, col_idx] = 0
                            row_idx += 1
    return input_flat

# GEMM维度: M=OC, N=B*HO*WO, K=C*KH*KW
def conv_im2col_gemm(input_tensor, kernel, stride=1, padding=0):
    input_flat = im2col(input_tensor, KH, KW, stride, padding).contiguous()
    kernel_flat = kernel.view(OC, -1).contiguous()

    func = matmul(kernel_flat.shape[0], input_flat.shape[1],
                  kernel_flat.shape[1], 128, 128, 128)
    output = func(kernel_flat, input_flat)
    output = output.view(OC, B, HO, WO).permute(1, 0, 2, 3)
    return output

# 测试
torch.manual_seed(42)
input_torch = torch.randn(B, C, H, W).half().npu()
kernel_torch = torch.randn(OC, C, KH, KW).half().npu()

result_np = conv_im2col_gemm(input_torch, kernel_torch, stride, padding)
result_torch = F.conv2d(input_torch.cpu(), kernel_torch.cpu(),
                        stride=stride, padding=padding).npu()
torch.testing.assert_close(result_np, result_torch, rtol=1e-2, atol=1e-2)
```

**要点**:
- **im2col 在 Python 端完成**: 数据预处理在 kernel 外，kernel 只做标准 GEMM
- **GEMM维度**: M=OC, N=B*HO*WO, K=C*KH*KW
- **kernel复用**: conv2d 的 GEMM kernel 与普通 GEMM 完全相同
- **输出reshape**: `output.view(OC, B, HO, WO).permute(1, 0, 2, 3)` 还原为 (B, OC, HO, WO)
- **纯Cube操作**: im2col+GEMM 方式不涉及 Vector Core

## Grouped GEMM 模板 (变长batch)

来源: `tilelang-ascend/examples/grouped_gemm/example_grouped_gemm_fwd.py`

支持不同 batch 有不同 M 维度的分组矩阵乘法:

```python
@tilelang.jit(out_idx=[2])
def grouped_gemm(batch_sizes_list, K, N, block_M, block_N, block_K,
                 dtype="float16"):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = "float32"
    total_m_blocks = sum(
        (size + block_M - 1) // block_M for size in batch_sizes_list
    )
    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),
        B: T.Tensor([batch_count, K, N], dtype),
        C: T.Tensor([batch_sum, N], dtype),
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),
    ):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            cur_batch_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[m_start:m_start+block_M, k*block_K:(k+1)*block_K], A_L1)
                    T.copy(B[cur_batch_idx, k*block_K:(k+1)*block_K,
                             by*block_N:(by+1)*block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[m_start:m_start+block_M,
                                by*block_N:by*block_N+block_N])

    return kernel
```

**要点**:
- **block_metadata**: int32 Tensor `[total_m_blocks, 3]`，存储 `[batch_idx, m_start, valid_rows]`
- **动态batch**: 每个 batch 可以有不同 M 维度，通过 metadata 表间接索引
- **A shape**: `(batch_sum, K)` — 所有 batch 的 A 拼接在一起
- **B shape**: `(batch_count, K, N)` — 每个 batch 独立的权重矩阵
- **Python端构建metadata**: 通过 `construct_inputs()` 函数生成 metadata 表

## Pipelined GEMM 模板 (T.Pipelined自动流水线)

来源: `tilelang-ascend/examples/pipeline/gemm_v0_pipeline.py`

使用 `T.Pipelined` 替代 `T.serial` 实现核内流水线，重叠 copy 和 gemm:

```python
@tilelang.jit(out_idx=[-2])
def matmul_pipelined(M, N, K, block_M, block_N, block_K,
                     dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    vec_proc = 4

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N // vec_proc), dtype)
            d_ub = T.alloc_ub((block_M // VEC_NUM, block_N // vec_proc), dtype)
            e_ub = T.alloc_ub((block_M // VEC_NUM, block_N // vec_proc), dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.Pipelined(loop_k, num_stages=3):
                    T.barrier_all()
                    T.copy(A[bx * block_M, k * block_K], A_L1)
                    T.copy(B[k * block_K, by * block_N], B_L1)

                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)

                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

**要点**:
- **`T.Pipelined(loop_k, num_stages=3)`**: 自动流水线，3级缓冲重叠 copy+gemm
- **与 `T.serial` 区别**: `T.Pipelined` 自动实现多级缓冲和指令重排
- **`out_idx=[-2]`**: 表示倒数第二个 tensor 是输出
- **注意**: `T.Pipelined` 不支持嵌套使用

## Tail Block GEMM Developer模式 (编译器自动处理tail)

来源: `tilelang-ascend/examples/gemm/example_gemm_tail_block_developer.py`

Developer模式下编译器可自动处理非对齐维度:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul_tail(M, N, K, block_M, block_N, K_L1,
                dtype="float16", accum_dtype="float"):
    m_num = T.ceildiv(M, block_M)   # 非整数除，编译器自动处理tail
    n_num = T.ceildiv(N, block_N)

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
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

**要点**:
- 使用 `T.ceildiv` 替代 `//`，编译器自动处理 tail block
- Developer模式 + AUTO_SYNC + MEMORY_PLANNING 让编译器自动处理边界
- 与 Expert 模式 Tail Block (手动 4-case 分支) 相比更简洁

## 自动调优 (AutoTune)

来源: `tilelang-ascend/examples/autotune/example_gemm_autotune.py`

使用 `@tilelang.autotune` 自动搜索最优 block 配置:

```python
import itertools
import tilelang
import tilelang.language as T
import torch

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

def get_config():
    return [
        {"block_M": 128, "block_N": 128, "K_L1": 64},
        {"block_M": 256, "block_N": 128, "K_L1": 64},
        {"block_M": 128, "block_N": 256, "K_L1": 64},
    ]

def ref_prog(A, B):
    return A @ B

def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(M, K).half().npu(),
        torch.randn(K, N).half().npu()
    ]

@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def matmul(M, N, K, block_M, block_N, K_L1,
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

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            loop_k = T.ceildiv(K, K_L1)
            for k in T.serial(loop_k):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main

# 不传 block_M, block_N, K_L1 触发自动调优
func = matmul(M, N, K)
print("Best Config:", func.get_tuner_result())
```

### Carver模板自动调优

来源: `tilelang-ascend/examples/autotune/example_gemm_carver.py`

使用 `tilelang.carver` 自动生成硬件感知的配置空间:

```python
from tilelang import carver
from tilelang.carver.arch.ascend import Ascend

def get_config() -> list[dict]:
    arch = Ascend()
    carver_template = carver.MatmulTemplate(
        M=M, N=N, K=K,
        in_dtype="float16",
        accum_dtype="float16",
        out_dtype="float16",
    ).with_arch(arch)

    hints = carver_template.recommend_hints(topk=20)
    configs = []
    for hint in hints:
        config = {
            "block_M": hint.block[0],
            "block_N": hint.block[1],
            "K_L1": hint.rstep[0],
        }
        configs.append(config)
    return configs
```

**要点**:
- **`@tilelang.autotune`**: 必须放在 `@tilelang.jit` 之前
- **不传可调参数**: 调用 `matmul(M, N, K)` 不传 block_M 等参数时触发调优
- **传入参数**: 调用 `matmul(M, N, K, 128, 256, 64)` 时跳过调优，直接使用指定值
- **`func.get_tuner_result()`**: 获取最优配置
- **carver**: 基于硬件约束自动生成候选配置，比手动枚举更高效

---

## 性能优化策略

### 优化层次架构

| 层级 | 优化手段 | 适用范围 |
|------|---------|---------|
| **Block配置** | block_M/N/K 大小选择 | 所有GEMM |
| **核内流水线** | T.Pipelined / T.set_flag+T.wait_flag | 大K场景 |
| **持久化** | T.Persistent | m_num*n_num > core_num |
| **Swizzle调度** | T.use_swizzle | 多block并行 |
| **双缓冲** | L1 S1=2, L0 S2=2 | intrinsic模式 |

### Block大小选择指南

**来源**: `tilelang-ascend/examples/` 各示例中的典型配置

| 场景 | block_M | block_N | block_K (K_L1) | 说明 |
|------|---------|---------|-----------------|------|
| **标准GEMM** | 128 | 256 | 64 | 大多数场景的默认值 |
| **小矩阵** | 64 | 128 | 64 | M/N较小时减少浪费 |
| **Pipelined** | 128 | 256 | 64 | num_stages=3 时效果好 |
| **Intrinsic** | 128 | 256 | 64 (L0), 256 (L1) | L1和L0分块不同 |
| **Conv2D** | 128 | 128 | 128 | im2col展开后的GEMM |
| **Batch GEMM** | 128 | 256 | 64 | 与标准GEMM相同 |

**选择原则**:
1. **block_M * block_K <= L1容量 (1MB)**: `(128*64*2B = 16KB)`, 安全
2. **block_M * block_N <= L0C容量**: `(128*256*4B = 128KB)`, 安全
3. **block大小为16或32的倍数**: Cube Core分形对齐要求
4. **K_L1不宜过大**: K_L1越大，每次加载的数据量越大，但循环次数越少
5. **推荐起始值**: block_M=128, block_N=256, K_L1=64

### 流水线策略对比

| 策略 | API | 复杂度 | 性能 | 适用场景 |
|------|-----|--------|------|---------|
| **无流水线** | `T.serial` | 低 | 基线 | 快速验证 |
| **T.Pipelined** | `T.Pipelined(..., num_stages=N)` | 低 | 中高 | Developer模式推荐 |
| **手动flag流水线** | `T.set_flag`/`T.wait_flag` | 高 | 最高 | Expert模式极限优化 |

**T.Pipelined vs T.serial 时延对比** (概念性, 实际数据取决于硬件和矩阵大小):

```
T.serial (无流水线):
  Block0: [copy][gemm][copy_out]
  Block1:               [copy][gemm][copy_out]
  总时间: N * (copy + gemm + copy_out)

T.Pipelined (num_stages=2):
  Block0: [copy][gemm][copy_out]
  Block1:    [copy][gemm][copy_out]
  总时间: copy + N * max(gemm, copy) + copy_out

T.Pipelined (num_stages=3):
  Block0: [copy0][gemm0][copy_out0]
  Block1:    [copy1][gemm1][copy_out1]
  Block2:       [copy2][gemm2][copy_out2]
  总时间: 进一步减少bubble
```

**num_stages选择**:
- `num_stages=2`: 最小双缓冲，适合 copy 和 gemm 时间接近
- `num_stages=3`: 三级缓冲，copy 和 gemm 时间差异较大时更好
- 更大的 num_stages 增加内存占用但不一定提升性能

### Persistent vs 非持久化

| 模式 | API | 启动核心数 | 适用场景 |
|------|-----|-----------|---------|
| **非持久化** | `T.Kernel(m_num * n_num, ...)` | m_num*n_num | block数 <= 核心数 |
| **持久化** | `T.Persistent([...], core_num, cid)` | core_num (如20/24) | block数 > 核心数 |

**Persistent模式优势**:
- 固定核心数 (如20或24)，避免核心空闲
- 每个核心循环处理多个block
- L1/L0C缓冲跨block复用，减少总分配

### Intrinsic四级流水线 (最高性能)

来源: `tilelang-ascend/examples/gemm/example_gemm_intrinsic.py`

四级流水线: MTE2(GM->L1) -> MTE1(L1->L0) -> M(mma计算) -> FIX(写回)

```
Timeline:
MTE2: [load_k0]  [load_k1]  [load_k2]  ...
MTE1:            [L1->L0_0] [L1->L0_1] ...
M:               [===mma0==][===mma1==] ...
FIX:                                    [fix_out] ...
```

**关键参数**:
- `S1=2`: L1双缓冲级数
- `S2=2`: L0双缓冲级数
- `core_num=20`: AI Core数量 (910B典型值)
- `T.use_swizzle(..., off=3)`: block调度优化

### L1驻留策略

当L1空间充足时，将高复用数据(如Q矩阵)驻留在L1:

```python
# Q驻留: 跨多个K block不释放
T.copy(Q[bz, by, bx*block_M:(bx+1)*block_M, :], q_l1)
for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=num_stages):
    T.copy(K[bz, by, k*block_N:(k+1)*block_N, :], k_l1)
    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
```

### 性能分析工具

| 工具 | 用途 | 命令 |
|------|------|------|
| `do_bench` | Python端计时 | `from tilelang.profiler import do_bench` |
| `msprof op` | 硬件级性能数据 | `msprof op --kernel-name="main_kernel" python3 xxx.py` |
| `msprof simulator` | 流水线可视化 | `msprof op simulator --soc-version=Ascend910B4 python3 xxx.py` |
| `func.get_kernel_source()` | 查看生成的AscendC代码 | Python端调用 |

### 优化检查清单

- [ ] 选择合适的 block_M/N/K (推荐 128/256/64 起步)
- [ ] 考虑 T.Pipelined 替代 T.serial (num_stages=2或3)
- [ ] block数 > 核心数时使用 T.Persistent
- [ ] 使用 carver 或 autotune 自动搜索最优配置
- [ ] 检查 L1/L0C 容量是否足够 (L1<=1MB, L0C不溢出)
- [ ] 使用 do_bench 或 msprof 对比不同配置时延
- [ ] block大小为16或32的整数倍 (分形对齐)

---

## 数据搬运路径

```
GM --(T.copy)--> L1 --(T.copy)--> L0A/L0B --(T.mma)--> L0C --(T.copy)--> GM
                                    或
GM --(T.copy)--> L1 --(T.gemm_v0)--> L0C --(T.copy)--> GM
```

- `T.gemm_v0`: 接受L1输入，自动完成L1→L0A/L0B搬运 + mma计算
- `T.mma`: 接受L0A/L0B输入，需要手动搬运L1→L0A/L0B（支持transpose）

## 同步原语

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| `T.barrier_all()` | 全局屏障 | Expert模式，每步手动同步 |
| pass_configs `TL_ASCEND_AUTO_SYNC` | 编译器自动同步 | Developer模式 |
| `T.set_flag` / `T.wait_flag` | 核内流水线同步 | 细粒度流水线优化 |
| `T.Pipelined(..., num_stages=N)` | 自动流水线 | L1多级缓冲 |

## 数据创建与验证

```python
import torch

torch.manual_seed(0)

a = torch.randn(M, K).half().npu()   # float16输入
b = torch.randn(K, N).half().npu()
c = torch.empty(M, N).half().npu()   # float16输出

func = matmul(M, N, K, 128, 256, 64)  # block_M=128, block_N=256, K_L1=64
c = func(a, b)

ref_c = a @ b  # CPU精度对比
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
```

## 约束

- **GEMM API用 `T.gemm_v0()`**: 不是 `T.gemm()`。init参数控制清零: `init=(k == 0)`
- **C必须用 `T.alloc_L0C()` (Expert) 或 `T.alloc_fragment()` (Developer)**: 这是Cube Core的L0C输出
- **A/B用 `T.alloc_L1()` (Expert) 或 `T.alloc_shared()` (Developer)**: L1 Buffer数据中转
- **dtype用字符串**: `"float16"`, `"float32"`, `"float"`, `"bfloat16"` -- 禁止 `T.float16`, `T.bfloat16`
- **Cube操作不使用VEC_NUM**: 每个core处理完整block，vid用`_`忽略
- **Expert模式必须用 `T.Scope("C")`**: Developer模式由编译器推断
- **禁止CUDA语法**: `T.Parallel`, `alloc_ub`(用于Cube输入), `T.clear()`, `T.fill()`, `clear_accum`
- **硬件约束**: 参考 `tl-op-hardware-constraints`

## 参考

- GEMM示例: `tilelang-ascend/examples/gemm/example_gemm.py`
- Developer模式: `tilelang-ascend/examples/gemm/example_gemm_pto_developer.py`
- Infer Scope: `tilelang-ascend/examples/gemm/example_gemm_infer_scope.py`
- Persistent: `tilelang-ascend/examples/gemm/example_gemm_persistent.py`
- Persistent (intrinsic): `tilelang-ascend/examples/gemm/example_gemm_intrinsic_persistent.py`
- 转置: `tilelang-ascend/examples/gemm/example_gemm_transpose_l1.py`
- 流水线 (intrinsic): `tilelang-ascend/examples/gemm/example_gemm_intrinsic.py`
- Tail Block (Expert): `tilelang-ascend/examples/gemm/example_gemm_tail_block.py`
- Tail Block (Developer): `tilelang-ascend/examples/gemm/example_gemm_tail_block_developer.py`
- Pipelined GEMM: `tilelang-ascend/examples/pipeline/gemm_v0_pipeline.py`
- Batch GEMM: `tilelang-ascend/examples/batch_gemm/batch_gemm.py`
- Grouped GEMM: `tilelang-ascend/examples/grouped_gemm/example_grouped_gemm_fwd.py`
- GEMV (Cube): `tilelang-ascend/examples/gemv/example_gemv_c.py`
- Conv2D: `tilelang-ascend/examples/convolution/example_convolution.py`
- AutoTune: `tilelang-ascend/examples/autotune/example_gemm_autotune.py`
- Carver: `tilelang-ascend/examples/autotune/example_gemm_carver.py`
- FA性能优化: `tilelang-ascend/examples/flash_attention/fa_opt/flash_attention_performance_optimization.md`
- API参考: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/`
- 硬件约束: `skills/tl-op-hardware-constraints/SKILL.md`
