---
name: tl-op-fused
description: 生成融合算子（Cube矩阵乘法 + Vector向量后处理），如 Matmul+Bias+ReLU、Flash Attention、GEMM+Softmax、Matmul+GELU、Matmul+SiLU。基于 tilelang-ascend/examples/flash_attention/flash_attn_bhsd.py 和 simple_fusion/matmul_add.py 实际示例。需要同时使用 Cube Core 和 Vector Core，通过workspace+cross_flag或pass_configs自动CV分离实现数据传递。
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
- **Matmul+Add融合**: `tilelang-ascend/examples/simple_fusion/matmul_add.py`
- Matmul+Add (infer_scope): `tilelang-ascend/examples/simple_fusion/matmul_add_infer_scope.py`
- API参考: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/`
- Expert↔Developer对比: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/`
- GEMM参考: `skills/tl-op-cube/SKILL.md`
- Vector参考: `skills/tl-op-vector/SKILL.md`
- 硬件约束: `skills/tl-op-hardware-constraints/SKILL.md`
