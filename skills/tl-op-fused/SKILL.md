---
name: tl-op-fused
description: 生成融合算子（Cube矩阵乘法 + Vector向量后处理），如 Matmul+Bias+ReLU、Flash Attention、GEMM+Softmax、Matmul+GELU、Matmul+SiLU、Quantized Matmul。基于 tilelang-ascend/examples/flash_attention/flash_attn_bhsd.py 和 pipeline/matmul_add_pipeline.py 实际示例。需要同时使用 Cube Core 和 Vector Core，通过workspace+cross_flag或pass_configs自动CV分离实现数据传递。支持T.Pipelined核间流水线优化。
---

# 融合算子技能 (Cube + Vector)

基于 `tilelang-ascend/examples/flash_attention/flash_attn_bhsd.py` 实际示例和 `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/` API参考。

当用户需要生成**融合算子**时使用本技能。此类算子结合 Cube Core 矩阵乘法和 Vector Core 向量后处理，在一个 kernel 中完成多步计算。

## 融合算子架构

```
GM --(MTE2)--> L1 --(Cube: T.gemm_v0)--> L0C --(T.copy)--> workspace/GM
                                                              |
                                       GM --(MTE2)--> UB <----+ (Vector读取)
                                                         |
                                                    T.tile.* (后处理)
                                                         |
                                       GM <--(MTE3)-- UB <----+
                                                              |
GM <--(T.copy)-- L0C <--(Cube: T.gemm_v0) <-- L1 <--(GM) <---+ (下一轮GEMM输入)
```

**Cube → Vector数据传递方式** (两种):

1. **Workspace + Cross Flag** (Expert模式): L0C → GM(workspace) → UB, 使用`T.set_cross_flag`/`T.wait_cross_flag`同步
2. **PassConfigs自动CV分离** (Developer模式): 开启`TL_ASCEND_AUTO_CV_COMBINE`，编译器自动处理

## 核心API

### 跨域同步 (Cube ↔ Vector)

| API | 说明 |
|-----|------|
| `T.set_cross_flag(pipe, flag)` | 设置核间同步标志 (Cube完成后通知Vector) |
| `T.wait_cross_flag(flag)` | 等待核间同步标志 (Vector等待Cube完成) |

**管道名称**: `"FIX"`, `"MTE3"`, `"V"` 等
**flag编号**: 必须成对使用，Cube端set的flag编号对应Vector端wait的编号

### Workspace (Expert模式)

融合算子需要workspace在Cube和Vector之间传递数据:

```python
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6])
def flash_attention_fwd(...):
    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),  # Cube→Vector: attention scores
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),         # Vector→Cube: softmax result
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),       # Cube→Vector: acc output
    ):
        ...
```

### 地址注解 (Expert模式)

手动规划UB和L1/L0C地址避免冲突:

```python
T.annotate_address({
    # L1 address
    q_l1: 0,
    k_l1: block_M * dim * DataType(dtype).bits // 8,
    # L0C address
    acc_s_l0c: 0,
    acc_o_l0c: 0,
    # UB address
    acc_o: 0,
    sumexp: 65536,
    m_i: 65664,
    ...
})
```

## Flash Attention模板 (Expert模式, 完整参考实现)

来源: `tilelang-ascend/examples/flash_attention/flash_attn_bhsd.py`

这是一个完整的Cube+Vector融合示例，展示了:
- Cube域和Vector域的分离 (`T.Scope("C")` 和 `T.Scope("V")`)
- Workspace作为Cube↔Vector数据传递通道
- Cross flag同步机制
- 在线softmax算法 (max/sum维护)
- VEC_NUM=2的UB分配

```python
import tilelang
from tilelang import DataType, language as T

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6])
def flash_attention_fwd(batch, seq_len, heads, dim):
    block_M, block_N = 64, 64
    dtype = "float16"
    accum_dtype = "float"
    sm_scale = (1.0 / dim)**0.5
    shape = [batch, heads, seq_len, dim]
    block_num = seq_len // block_M * heads * batch

    @T.prim_func
    def main(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        Output: T.Tensor(shape, dtype),
        workspace_1: T.Tensor([block_num, block_M, block_N], accum_dtype),
        workspace_2: T.Tensor([block_num, block_M, block_N], dtype),
        workspace_3: T.Tensor([block_num, block_M, dim], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len // block_M)
            by = cid // (seq_len // block_M) % heads
            bz = cid // (seq_len // block_M) // heads % batch

            # Cube buffers (L1 + L0C)
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            acc_s_l1 = T.alloc_L1([block_M, block_N], dtype)
            acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)
            acc_o_l0c = T.alloc_L0C([block_M, dim], accum_dtype)

            # Vector buffers (UB, divided by VEC_NUM=2)
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)
            sumexp = T.alloc_ub([block_M // 2], accum_dtype)
            m_i = T.alloc_ub([block_M // 2], accum_dtype)
            acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)
            # ... more UB buffers ...

            T.annotate_address({ ... })  # 手动地址规划

            # === Cube域: GEMM计算 ===
            with T.Scope("C"):
                T.copy(Q[bz, by, bx*block_M:(bx+1)*block_M, :], q_l1)
                T.barrier_all()

                for k in T.serial(T.ceildiv(seq_len, block_N)):
                    T.copy(K[bz, by, k*block_N:(k+1)*block_N, :], k_l1)
                    T.barrier_all()

                    # S = Q @ K^T
                    T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
                    T.barrier_all()

                    # L0C → workspace → Vector处理
                    T.copy(acc_s_l0c, workspace_1[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 0)   # 通知Vector

                    T.wait_cross_flag(1)          # 等待Vector完成
                    T.barrier_all()

                    # Vector写回的softmax结果 → L1
                    T.copy(workspace_2[cid, :, :], acc_s_l1)
                    T.copy(V[bz, by, k*block_N:(k+1)*block_N, :], v_l1)
                    T.barrier_all()

                    # O = softmax(S) @ V
                    T.gemm_v0(acc_s_l1, v_l1, acc_o_l0c, init=True)
                    T.barrier_all()

                    T.copy(acc_o_l0c, workspace_3[cid, :, :])
                    T.barrier_all()
                    T.set_cross_flag("FIX", 2)
                    T.wait_cross_flag(3)

            # === Vector域: Softmax后处理 ===
            with T.Scope("V"):
                T.tile.fill(acc_o, 0.0)
                T.tile.fill(sumexp, 0.0)
                T.tile.fill(m_i, -2**30)
                T.barrier_all()

                for _k in T.serial(T.ceildiv(seq_len, block_N)):
                    T.tile.fill(acc_s_ub, 0.0)
                    T.barrier_all()

                    # 等待Cube写入workspace
                    T.wait_cross_flag(0)
                    T.copy(workspace_1[cid, vid*block_M//2:vid*block_M//2+block_M//2, :], acc_s_ub_)
                    T.barrier_all()

                    # 在线softmax: scale, reduce_max, exp, reduce_sum
                    T.tile.add(acc_s_ub, acc_s_ub, acc_s_ub_)
                    T.tile.mul(acc_s_ub, acc_s_ub, sm_scale)
                    T.reduce_max(acc_s_ub, m_i, dim=-1)
                    # ... softmax后续步骤 ...

                    # 写回softmax结果 → workspace → Cube使用
                    T.copy(acc_s_half, workspace_2[cid, vid*block_M//2:...])
                    T.set_cross_flag("MTE3", 1)   # 通知Cube

                    T.wait_cross_flag(2)          # 等待Cube
                    T.copy(workspace_3[cid, vid*block_M//2:...], acc_o_ub)
                    T.tile.add(acc_o, acc_o, acc_o_ub)
                    T.set_cross_flag("V", 3)      # 通知Cube

                # 最终除以sumexp + 写出
                for h_i in range(block_M // 2):
                    T.tile.div(acc_o[h_i, :], acc_o[h_i, :], sumexp[h_i])

                T.copy(acc_o, acc_o_half)
                T.copy(acc_o_half, Output[bz, by, bx*block_M + vid*block_M//2:...])

    return main
```

## Matmul + Bias/Add 模板 (Expert模式, 简单融合)

来源: `tilelang-ascend/examples/simple_fusion/matmul_add.py`

C = A @ B + D 的融合实现，展示 Cube + Vector 最简单的融合模式:

```python
@tilelang.jit(out_idx=[-2])   # -2 表示倒数第二个参数是输出 (C)
def matmul_add(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),   # 输出 (out_idx=-2)
            D: T.Tensor((M, N), dtype),   # bias/add 输入
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # Cube buffers
            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            # Vector buffers
            d_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # === Cube域: GEMM ===
            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    T.copy(A[bx*block_M, k*block_K], A_L1)
                    T.copy(B[k*block_K, by*block_N], B_L1)
                    T.barrier_all()
                    if k == 0:
                        T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                    else:
                        T.gemm_v0(A_L1, B_L1, C_L0)
                    T.barrier_all()

                # 写回GEMM结果到GM
                T.copy(C_L0, C[bx*block_M, by*block_N])
                T.set_cross_flag("FIX", 0)   # 通知Vector: GEMM完成

            # === Vector域: C + D ===
            with T.Scope("V"):
                T.wait_cross_flag(0)          # 等待Cube完成

                # 从GM读取GEMM结果 (L0C -> GM -> UB)
                T.copy(C[bx*block_M + vid*block_M//VEC_NUM, by*block_N], c_ub)
                T.copy(D[bx*block_M + vid*block_M//VEC_NUM, by*block_N], d_ub)
                T.barrier_all()
                T.tile.add(c_ub, c_ub, d_ub)  # C = C + D
                T.barrier_all()
                T.copy(c_ub, C[bx*block_M + vid*block_M//VEC_NUM, by*block_N])

    return main
```

**要点**:
- **数据流**: L0C -> GM(写回) -> UB(重新读取) — 简单融合通过GM中转
- **out_idx=-2**: 倒数第二个 tensor (C) 是输出，D 是输入参数
- **Cube先写GM，Vector再从GM读**: cross_flag 保证时序正确
- **无workspace**: 简单融合直接复用输出tensor C 作为中转
- **Expert模式**: 明确的 `T.Scope("C")` 和 `T.Scope("V")` 分离

## Matmul + Add Pipelined (Developer模式, 流水线融合)

来源: `tilelang-ascend/examples/pipeline/matmul_add_pipeline.py`

使用 `T.Pipelined` 实现核间流水线，重叠 Cube GEMM 和 Vector Add:

```python
@tilelang.jit(
    out_idx=[2],
    workspace_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def matmul_add_pipeline(M, N, K, block_M, block_N, block_K,
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
        D: T.Tensor((M, N), dtype),
        workspace_1: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            A_L1 = T.alloc_shared((block_M, block_K), dtype)
            B_L1 = T.alloc_shared((block_K, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)
            d_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)
            e_ub = T.alloc_shared((block_M // VEC_NUM, block_N // vec_proc), dtype)

            # Cube域: Pipelined GEMM
            loop_k = T.ceildiv(K, block_K)
            for k in T.Pipelined(loop_k, num_stages=3):
                T.copy(A[bx * block_M, k * block_K], A_L1)
                T.copy(B[k * block_K, by * block_N], B_L1)

                if k == 0:
                    T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                else:
                    T.gemm_v0(A_L1, B_L1, C_L0)

            T.copy(C_L0, workspace_1[bx * block_M, by * block_N])

            # Vector域: 逐块Add (VEC_NUM和vec_proc分割)
            for i in T.Pipelined(vec_proc, num_stages=2):
                T.copy(workspace_1[bx*block_M + vid*block_M//VEC_NUM,
                                   by*block_N + i*block_N//vec_proc], c_ub)
                T.copy(D[bx*block_M + vid*block_M//VEC_NUM,
                         by*block_N + i*block_N//vec_proc], d_ub)

                for j, k in T.Parallel(block_M // VEC_NUM, block_N // vec_proc):
                    e_ub[j, k] = c_ub[j, k] + d_ub[j, k]

                T.copy(e_ub, C[bx*block_M + vid*block_M//VEC_NUM,
                               by*block_N + i*block_N//vec_proc])

    return main
```

**要点**:
- **workspace_idx=[-1]**: 最后一个参数是自动分配的workspace
- **Cube用 `T.Pipelined(loop_k, num_stages=3)`**: 核内流水线重叠copy+gemm
- **Vector用 `T.Pipelined(vec_proc, num_stages=2)`**: 核内流水线重叠load+compute+store
- **vec_proc=4**: Vector端每个block再切4个子块，每个AIV处理更细粒度
- **`T.Parallel` 用于Developer模式**: 在Vector域用标量语法做elementwise add
- **`out_idx=[2]`**: 第3个参数(C)是输出，D是bias输入

## Quantized Batch Matmul (量化矩阵乘法融合)

来源: `tilelang-ascend/examples/quant_batch_matmul/example_quant_batch_matmul.py`

int8输入 -> int32累加 -> float32缩放 -> float16/bfloat16输出:

```python
@tilelang.jit(
    out_idx=[3],
    workspace_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    }
)
def quant_batch_matmul(Batch, M, N, K, scale_size,  # "1" 或 "N"
                       block_M, block_N, block_K,
                       in_dtype="int8", out_dtype="float16",
                       accum_dtype="int32", scale_dtype="float32"):
    VEC_NUM = 2
    CAST_MODE = "CAST_RINT"
    N_scale = N if scale_size == "N" else 1
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)
    block_M_2 = T.ceildiv(block_M, VEC_NUM)

    @T.prim_func
    def main(
        A: T.Tensor([Batch, M, K], in_dtype),
        B: T.Tensor([Batch, K, N], in_dtype),
        scale: T.Tensor([N_scale], scale_dtype),
        C: T.Tensor([Batch, M, N], out_dtype),
        workspace_1: T.Tensor([Batch, M, N], accum_dtype),
    ):
        with T.Kernel(Batch * m_num * n_num, is_npu=True) as (cid, vid):
            bb = cid // (m_num * n_num)
            bm = (cid % (m_num * n_num)) // n_num
            bn = (cid % (m_num * n_num)) % n_num

            A_L1 = T.alloc_L1([block_M, block_K], in_dtype)
            B_L1 = T.alloc_L1([block_K, block_N], in_dtype)
            C_L0 = T.alloc_L0C([block_M, block_N], accum_dtype)

            c_ub = T.alloc_ub([block_M_2, block_N], accum_dtype)
            c_scale = T.alloc_ub([block_M_2, block_N], scale_dtype)
            c_out = T.alloc_ub([block_M_2, block_N], out_dtype)
            scale_ub = T.alloc_ub([block_N], scale_dtype)

            # Cube域: int8 GEMM -> int32累加
            for bk in T.serial(k_num):
                T.copy(A[bb, bm*block_M, bk*block_K], A_L1)
                T.copy(B[bb, bk*block_K, bn*block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(bk == 0))

            T.copy(C_L0, workspace_1[bb, bm*block_M, bn*block_N])

            # Vector域: int32 -> scale -> output_dtype
            T.copy(workspace_1[bb, bm*block_M + vid*block_M_2,
                               bn*block_N], c_ub)
            if scale_size == "N":
                T.copy(scale[bn * block_N], scale_ub)
            else:
                T.copy(scale[0], scale_ub)
                T.tile.fill(scale_ub, scale_ub[0])

            if accum_dtype != scale_dtype:
                T.tile.cast(c_scale, c_ub, mode=CAST_MODE,
                           count=block_M_2 * block_N)
            else:
                T.copy(c_ub, c_scale)

            for bm_v, bn_v in T.Parallel(block_M_2, block_N):
                c_scale[bm_v, bn_v] *= scale_ub[bn_v]

            if out_dtype != scale_dtype:
                T.tile.cast(c_out, c_scale, mode=CAST_MODE,
                           count=block_M_2 * block_N)
            else:
                T.copy(c_scale, c_out)

            T.copy(c_out, C[bb, bm*block_M + vid*block_M_2, bn*block_N])

    return main
```

**要点**:
- **Cube域**: int8 GEMM累加到int32 (accum_dtype="int32")
- **Vector域**: int32 -> cast -> scale乘法 -> cast -> float16/bfloat16
- **scale_size**: "1" 表示per-tensor量化, "N" 表示per-channel量化
- **workspace**: int32中间结果暂存 (workspace_idx=[4])
- **`T.tile.cast`**: 精度转换, mode="CAST_RINT" 为四舍五入
- **Developer模式**: AUTO_CV_COMBINE自动分离Cube和Vector操作

## Developer模式自动CV分离

使用pass_configs自动处理Cube↔Vector同步和数据传递:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动CV分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,       # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,          # 自动同步
}
```

**Developer模式特点**:
- 无需手动 `T.Scope("C")` / `T.Scope("V")`
- 无需手动 `T.set_cross_flag` / `T.wait_cross_flag`
- 编译器自动分离Cube和Vector操作
- 适合快速原型开发

**Expert模式特点** (全部关闭):
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
```

## 适用场景

- 矩阵乘法 + 偏置: C = A @ B + bias
- 矩阵乘法 + 激活: C = relu(A @ B), C = gelu(A @ B)
- 矩阵乘法 + 偏置 + 激活: C = relu(A @ B + bias)
- Flash Attention: Q @ K^T → softmax → @ V
- GEMM + Softmax: 用于注意力机制
- GEMM + LayerNorm + 残差连接

## 不适用场景

- **纯矩阵乘法**: 使用 `tl-op-cube` 技能
- **纯向量运算**: 使用 `tl-op-vector` 技能

## Cube → Vector数据传递总结

| 方式 | Cube端 | Vector端 | 同步机制 |
|------|--------|---------|---------|
| Expert: workspace + cross_flag | `T.copy(L0C, workspace)` + `T.set_cross_flag("FIX", N)` | `T.wait_cross_flag(N)` + `T.copy(workspace, UB)` | 手动cross flag |
| Developer: 自动CV分离 | 直接写代码，编译器分离 | 直接写代码，编译器分离 | pass_configs自动 |

## 关键约束 (综合Cube + Vector)

- **GEMM API用 `T.gemm_v0()`**: 不是 `T.gemm()`。init参数: `init=(k == 0)` 或 `init=True/False`
- **GEMM输出用 `T.alloc_L0C()` (Expert) 或 `T.alloc_fragment()` (Developer)**: L0C Buffer
- **GEMM输入用 `T.alloc_L1()` (Expert) 或 `T.alloc_shared()` (Developer)**: L1 Buffer
- **Vector后处理buffer用 `T.alloc_ub()` (Expert) 或 `T.alloc_shared()` (Developer)**: UB
- **Vector部分使用VEC_NUM=2**: UB按 `block_size // VEC_NUM` 分配
- **Cube部分不使用VEC_NUM**: L1/L0C使用完整block大小
- **T.tile.fill在NPU上可用**: 可以在Vector域使用 `T.tile.fill(buffer, value)`
- **T.tile.*和T.copy在 `T.Scope("V")` 内** (Expert模式): Vector操作
- **T.gemm_v0在 `T.Scope("C")` 内** (Expert模式): Cube操作
- **dtype用字符串**: `"float16"`, `"float32"`, `"float"` -- 禁止 `T.float16`
- **禁止CUDA语法**: `T.clear()`, `T.fill()`(顶层), `clear_accum`, `T.Parallel`(Expert模式), `T.Pipelined`(Expert模式嵌套)
- **Buffer容量**: UB总量 <= 2MB (910B), L1总量 <= 1MB (910B)
- **核内流水线和核间流水线不能嵌套**: 使用flat pattern (T.Pipelined for inter-core, manual for intra-core)
- **调试工具**: `T.printf()`, `T.dump_tensor()`, `func.get_kernel_source()` 可用于验证
- **硬件约束**: 参考 `tl-op-hardware-constraints`

## 融合算子设计原则

### Flash Attention 性能优化参考数据

来源: `tilelang-ascend/examples/flash_attention/fa_opt/bench_mark.md`

FA算子 (B=1, Q_N=12, D=128, block_size=128) 优化过程及性能:

| 优化步骤 | 相对原生AscendC性能 | 说明 |
|----------|:-------------------:|------|
| L1驻留 (Q矩阵常驻L1) | - | 减少GM访问 |
| 指令向量化 (标量→tile) | 36% | 消除循环开销 |
| 多缓冲 (Double Buffer) | - | 隐藏数据传输延迟 |
| 核内冗余同步消除 | 50% | 减少不必要的barrier |
| T.Pipelined核间流水线 (num_stages=8) | 60% | Cube/Vector重叠执行 |
| 优化核间同步频率 (每2次同步1次) | 72% | 减少scalar bound |
| 指令合并 (axpy替代mul+sub) | 80% | 减少指令派发数 |

**Expert模式**: 80% 原生性能 | **Developer模式 (auto_pipeline)**: 60% 原生性能

**不同序列长度对比**:

| S (序列长度) | AscendC | TileLang | 比例 |
|-------------|---------|----------|------|
| 32K | 37555us | 46643us | 80.52% |
| 64K | 149578us | 185188us | 80.77% |
| 128K | 600018us | 741211us | 80.95% |

### num_stages选择指南

来源: `tilelang-ascend/examples/flash_attention/fa_opt/flash_attention_performance_optimization.md`

```
num_stages = 2:
  C-core: [C0]--------[C1]--------[C2]
  V-core:      [V0]--------[V1]--------[V2]
               ↑ C1需等V0完成才能开始

num_stages = 3:
  C-core: [C0][C1]----[C2]----[C3]
  V-core:     [V0][V1]----[V2]----[V3]
               ↑ bubble减少

num_stages = 8 (FA最优):
  更多的缓冲级进一步减少等待
```

**调优建议**:
- 从 `num_stages=2` 开始，逐步增加
- 观察C/V core时间比例，选择最小化bubble的值
- 过大的num_stages增加内存占用但不一定提升性能
- FA场景中 `num_stages=8` 配合同步频率优化效果最好

### 核间同步频率优化

减少cross_flag同步次数可以提升性能:

```python
# 差: 每次都同步
for k in range(loop_k):
    cube_process(k)
    sync()    # 每次迭代同步
    vector_process(k)
    sync()

# 好: 每N次同步一次
for k in range(loop_k):
    cube_process(k)
    vector_process(k)
    if k % 2 == 1:  # 每2次同步一次
        sync()
```

**性能影响**: FA优化中从每次同步改为每2次同步，性能从60%提升到72%。

### 优化检查清单 (融合算子)

- [ ] 使用 workspace 传递 Cube->Vector 数据 (workspace_idx)
- [ ] 开启 AUTO_CV_COMBINE + AUTO_CV_SYNC (Developer模式)
- [ ] 考虑 T.Pipelined 实现核间流水线 (从num_stages=2开始)
- [ ] Cube域使用 alloc_L1/alloc_shared + alloc_L0C/alloc_fragment
- [ ] Vector域使用 alloc_ub/alloc_shared + VEC_NUM=2 分割
- [ ] 检查UB总量 <= 2MB, L1总量 <= 1MB
- [ ] 标量操作向量化 (用T.tile.*替代循环)
- [ ] 减少不必要的核间同步
- [ ] 使用 do_bench 或 msprof 对比时延

### 1. 最小化GM访问次数

```python
# 好的融合: 一次GM读取，一次GM写入
# Q,K,V从GM读取 -> 全部在片上完成 -> O写回GM

# 差的设计: 多次GM读写
# Q,K从GM读 -> S写回GM -> S从GM读 -> softmax -> P写回GM -> ...
```

### 2. 正确的跨域同步

Expert模式:
```python
# Cube完成 → set_cross_flag → Vector开始
with T.Scope("C"):
    T.gemm_v0(A_L1, B_L1, C_L0, init=True)
    T.copy(C_L0, workspace[cid, :, :])
    T.set_cross_flag("FIX", 0)   # Cube通知Vector

with T.Scope("V"):
    T.wait_cross_flag(0)          # Vector等待Cube
    T.copy(workspace[cid, ...], UB)
    T.tile.mul(UB, UB, scale)     # 后处理
```

### 3. Buffer容量规划

```python
# Cube buffers (不计入UB容量)
q_l1 = T.alloc_L1([block_M, dim], dtype)         # L1
k_l1 = T.alloc_L1([block_N, dim], dtype)          # L1
acc_s_l0c = T.alloc_L0C([block_M, block_N], accum_dtype)  # L0C

# Vector buffers (计入UB容量 <= 2MB)
acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)  # UB
acc_s_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # UB
# UB总计 = sum(all UB buffers)
```

## 约束

- **GEMM API用 `T.gemm_v0()`**: 使用 `init=` 参数控制清零
- **输出C用 `T.alloc_L0C()` (Expert) 或 `T.alloc_fragment()` (Developer)**: Cube Core的L0C
- **向量后处理buffer用 `T.alloc_ub()` (Expert) 或 `T.alloc_shared()` (Developer)**: Vector Core操作
- **跨域同步**: Expert用cross_flag，Developer用pass_configs
- **dtype用字符串**: `"float16"`, `"float32"` (禁止 `T.float16`)
- **Buffer容量**: 所有UB总量 <= 2MB, 所有L1总量 <= 1MB (L0C不计入UB/L1)
- **硬件约束**: 参考 `tl-op-hardware-constraints`

## 参考

- Flash Attention示例: `tilelang-ascend/examples/flash_attention/flash_attn_bhsd.py`
- Flash Attention (cc_sync): `tilelang-ascend/examples/flash_attention/flash_attn_bhsd_cc_sync.py`
- Paged Flash Attention: `tilelang-ascend/examples/flash_attention/paged_flash_attn_bhsd.py`
- FA优化版: `tilelang-ascend/examples/flash_attention/fa_opt/`
- FA性能数据: `tilelang-ascend/examples/flash_attention/fa_opt/bench_mark.md`
- FA性能优化指南: `tilelang-ascend/examples/flash_attention/fa_opt/flash_attention_performance_optimization.md`
- **Matmul+Add融合**: `tilelang-ascend/examples/simple_fusion/matmul_add.py`
- **Matmul+Add Pipelined**: `tilelang-ascend/examples/pipeline/matmul_add_pipeline.py`
- Matmul+Add (developer): `tilelang-ascend/examples/developer_mode/matmul_add_developer.py`
- Matmul+Add (infer_scope): `tilelang-ascend/examples/simple_fusion/matmul_add_infer_scope.py`
- **Quantized Batch Matmul**: `tilelang-ascend/examples/quant_batch_matmul/example_quant_batch_matmul.py`
- API参考: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/`
- Expert↔Developer对比: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/`
- GEMM参考: `skills/tl-op-cube/SKILL.md`
- Vector参考: `skills/tl-op-vector/SKILL.md`
- 硬件约束: `skills/tl-op-hardware-constraints/SKILL.md`
