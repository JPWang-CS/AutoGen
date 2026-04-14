---
name: tl-op-cube
description: 生成纯Cube Core算子（矩阵乘法 GEMM、Batch GEMM、GEMV、Conv2D）。基于 tilelang-ascend/examples/gemm/ 实际示例。使用 T.alloc_L1 + T.alloc_L0C + T.gemm_v0 Expert模式，或 T.alloc_shared + T.alloc_fragment + T.gemm_v0 Developer模式。支持流水线intrinsic、持久化T.Persistent、转置、尾部块处理等高级模式。
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
# im2col: 将输入展开为 (C*KH*KW, B*HO*WO) 矩阵
# 权重展开为 (OC, C*KH*KW) 矩阵
# GEMM: output = kernel_flat @ input_flat

# kernel参数
B, C, H, W, OC, KH, KW, stride, padding = 2, 2, 15, 15, 128, 8, 8, 1, 0
HO = (H + 2 * padding - KH) // stride + 1
WO = (W + 2 * padding - KW) // stride + 1

# im2col转换 (Python层，非kernel内)
def im2col(input_tensor, KH, KW, stride, padding):
    input_flat = torch.zeros((C * KH * KW, B * HO * WO), ...)
    # ... 展开循环 ...
    return input_flat

# 使用GEMM kernel计算
func = matmul(OC, B*HO*WO, C*KH*KW, 128, 128, 128)
output = func(kernel_flat, input_flat)
output = output.view(OC, B, HO, WO).permute(1, 0, 2, 3)
```

**要点**:
- **im2col 在 Python 端完成**: 数据预处理在 kernel 外，kernel 只做 GEMM
- **GEMM维度**: M=OC, N=B*HO*WO, K=C*KH*KW
- **与普通GEMM相同**: conv2d kernel 复用标准 GEMM 实现

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
- Persistent: `tilelang-ascend/examples/gemm/example_gemm_persistent.py`
- Persistent (intrinsic): `tilelang-ascend/examples/gemm/example_gemm_intrinsic_persistent.py`
- 转置: `tilelang-ascend/examples/gemm/example_gemm_transpose_l1.py`
- 流水线 (intrinsic): `tilelang-ascend/examples/gemm/example_gemm_intrinsic.py`
- Tail Block: `tilelang-ascend/examples/gemm/example_gemm_tail_block.py`
- Batch GEMM: `tilelang-ascend/examples/batch_gemm/batch_gemm.py`
- GEMV (Cube): `tilelang-ascend/examples/gemv/example_gemv_c.py`
- Conv2D: `tilelang-ascend/examples/convolution/example_convolution.py`
- API参考: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/`
- 硬件约束: `skills/tl-op-hardware-constraints/SKILL.md`
