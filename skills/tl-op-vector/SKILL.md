---
name: tl-op-vector
description: 生成纯Vector Core算子（向量/元素级计算），如 vector_add、relu、sigmoid、silu、gelu、tanh、swi_glu、reduce、rms_norm 等。基于 tilelang-ascend/examples/ 实际示例和 .agents/skills/ API参考。使用 T.alloc_ub + T.tile.* (Expert) 或 T.alloc_shared + T.Parallel (Developer) + VEC_NUM=2。
---

# 纯向量算子技能 (Vector Core Only)

基于 `tilelang-ascend/examples/elementwise/` 实际示例和 `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/` API参考。

## 两种编程模式

| 特性 | Expert 模式 | Developer 模式 |
|------|------------|---------------|
| 内存分配 | `T.alloc_ub()` | `T.alloc_shared()` |
| 计算 | `T.tile.add(c, a, b)` | `T.Parallel` + 标量 `c[i,j] = a[i,j] + b[i,j]` |
| 同步 | 手动 `T.barrier_all()` | pass_configs 自动 |
| 作用域 | 必须 `T.Scope("V")` | 无需（编译器推断） |
| 适用场景 | 精细控制、直接触发硬件指令 | 快速开发、跨平台兼容 |

## 核心API

### 内存分配

| Expert 模式 | Developer 模式 | 说明 |
|------------|---------------|------|
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` | Unified Buffer |

### 作用域

| Scope | 说明 | 包含操作 |
|-------|------|---------|
| `T.Scope("V")` | Vector Core域 | T.copy (GM↔UB), T.tile.*, T.barrier_all |

### Kernel启动

```python
VEC_NUM = 2
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
```

**注意**: Vector操作使用VEC_NUM=2，vid为0或1。UB按`block_size // VEC_NUM`分配。

### Vector Tile指令 (Expert模式 T.tile.*)

#### 基础算术 (3-operand: dst, src0, src1)

| API | 功能 | src1类型 |
|-----|------|---------|
| `T.tile.add(dst, src0, src1)` | dst = src0 + src1 | buffer或scalar |
| `T.tile.sub(dst, src0, src1)` | dst = src0 - src1 | buffer或scalar |
| `T.tile.mul(dst, src0, src1)` | dst = src0 * src1 | buffer或scalar |
| `T.tile.div(dst, src0, src1)` | dst = src0 / src1 | buffer或scalar |
| `T.tile.max(dst, src0, src1)` | dst = max(src0, src1) | buffer或scalar |
| `T.tile.min(dst, src0, src1)` | dst = min(src0, src1) | buffer或scalar |

#### 单目运算 (2-operand: dst, src)

| API | 功能 |
|-----|------|
| `T.tile.exp(dst, src)` | dst = exp(src) |
| `T.tile.ln(dst, src)` | dst = ln(src) |
| `T.tile.abs(dst, src)` | dst = abs(src) |
| `T.tile.sqrt(dst, src)` | dst = sqrt(src) |
| `T.tile.rsqrt(dst, src)` | dst = 1/sqrt(src) |
| `T.tile.relu(dst, src)` | dst = max(0, src) |
| `T.tile.leaky_relu(dst, src, scalar)` | dst = src >= 0 ? src : src * scalar |
| `T.tile.sigmoid(dst, src)` | dst = sigmoid(src) = 1/(1+exp(-src)) |
| `T.tile.reciprocal(dst, src)` | dst = 1/src |
| `T.tile.sin(dst, src)` | dst = sin(src) |
| `T.tile.cos(dst, src)` | dst = cos(src) |

#### 数据操作

| API | 功能 |
|-----|------|
| `T.tile.fill(buffer, value)` | 用value填充buffer |
| `T.tile.clear(buffer)` | 将buffer填充为0 |
| `T.tile.cast(dst, src, mode, count)` | 精度转换，mode如"CAST_NONE", "CAST_RINT", "CAST_FLOOR", "CAST_CEIL", "CAST_ROUND", "CAST_TRUNC", "CAST_ODD" |
| `T.tile.axpy(dst, src, scalar)` | dst += scalar * src (融合乘加) |
| `T.tile.transpose(dst, src)` | 16x16矩阵转置 |
| `T.tile.createvecindex(dst, first_value)` | 创建向量索引 [first_value, first_value+1, ...] |
| `T.tile.arith_progression(buf, first, diff, count)` | 创建等差数列 |

#### 比较与选择

| API | 功能 |
|-----|------|
| `T.tile.compare(dst, src0, src1, mode)` | 逐元素比较，输出uint8 mask，mode: "EQ","NE","GT","GE","LT","LE" |
| `T.tile.select(dst, mask, src0, src1, selMode)` | 条件选择，selMode: "VSEL_CMPMASK_SPR", "VSEL_TENSOR_SCALAR_MODE", "VSEL_TENSOR_TENSOR_MODE" |

#### 位运算

| API | 功能 |
|-----|------|
| `T.tile.bitwise_and(dst, src0, src1)` | dst = src0 & src1 |
| `T.tile.bitwise_or(dst, src0, src1)` | dst = src0 \| src1 |
| `T.tile.bitwise_not(dst, src0)` | dst = ~src0 |
| `T.tile.bitwise_xor(dst, src0, src1)` | dst = src0 ^ src1 |
| `T.tile.bitwise_lshift(dst, src0, scalar)` | dst = src0 << scalar |
| `T.tile.bitwise_rshift(dst, src0, scalar)` | dst = src0 >> scalar |

#### 排序

| API | 功能 |
|-----|------|
| `T.tile.sort(dst, src, actual_num)` | 降序排序，actual_num为有效元素数量 |
| `T.tile.merge_sort(dst, src0, src1[, src2[, src3]])` | 2/3/4路归并排序，value-index pair格式 |
| `T.tile.topk(dst, src, block_size)` | TopK选择 |

#### 数据收集

| API | 功能 |
|-----|------|
| `T.tile.gather(dst, src, src_offset, base)` | 按偏移收集元素 |
| `T.tile.gather_mask(dst, src, pattern)` | 按模式收集: "P0101"(偶数),"P1010"(奇数),"P0001"~"P1000","P1111"(全取),或自定义buffer索引 |

#### 归约操作

| API | 功能 |
|-----|------|
| `T.reduce_sum(buffer, out, dim)` | 按维度求和 |
| `T.reduce_max(buffer, out, dim)` | 按维度求最大值 |
| `T.reduce_min(buffer, out, dim)` | 按维度求最小值 |
| `T.reduce_max(buffer, out, dim, real_shape=[H, W])` | 按维度求最大值(带切片，仅对有效区域归约) |

**real_shape参数**: 当 UB buffer 大于实际有效数据时，用 `real_shape` 指定有效区域大小。例如 buffer 为 (5,8)，但只有前3行有效，则 `real_shape=[3, 8]`。

### Developer模式标量API (T.Parallel内)

在`T.Parallel`循环内使用标准Python运算符:

```python
for i, j in T.Parallel(block_M // VEC_NUM, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]    # 加法
    c_ub[i, j] = T.exp(a_ub[i, j])            # 指数
    c_ub[i, j] = T.max(a_ub[i, j], 0)         # ReLU
```

**支持的运算**: `+`, `-`, `*`, `/`, `T.abs()`, `T.exp()`, `T.log()`, `T.sqrt()`, `T.rsqrt()`, `T.min()`, `T.max()`

**T.Parallel 广播支持**:
- 向量-标量: `c_ub[i, j] = a_ub[i, j] + 1`
- 行广播: `c_ub[i, j] = a_ub[i, j] * b_ub[i]` (a_ub为2D, b_ub为1D)
- 维度不匹配: `c_ub[i, j] = b_ub[j] + 5` (右侧1D自动广播到2D)

**复杂表达式**: 自动分配临时buffer分解，建议开启 MEMORY_PLANNING 减少空间浪费。

## Expert模式 Elementwise Add模板 (推荐)

来源: `tilelang-ascend/examples/elementwise/elementwise_add.py`

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

## Developer模式 Elementwise Add模板

来源: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/references/api-kernel-memory.md`

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

            for i, j in T.Parallel(block_M // VEC_NUM, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

## 更多模板

### 流水线Elementwise Add (高性能)

来源: `tilelang-ascend/examples/elementwise/elementwise_add_pipeline.py`

使用多级UB缓冲 + `T.set_flag`/`T.wait_flag` 实现MTE2-V-MTE3流水线并行:

```python
@tilelang.jit(out_idx=[-1])
def vec_add_pipeline(M, N, block_M, block_N, sub_M, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    stages = 2

    @T.macro
    def init_flag():
        T.set_flag("mte3", "mte2", 0)
        T.set_flag("mte3", "mte2", 1)

    @T.macro
    def clear_flag():
        T.wait_flag("mte3", "mte2", 0)
        T.wait_flag("mte3", "mte2", 1)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            vec_proc = block_M // sub_M

            a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                init_flag()
                T.wait_flag("mte3", "mte2", 0)
                T.copy(A[bx*block_M + vid*sub_M//VEC_NUM + 0*sub_M, by*block_N], a_ub[0])
                T.copy(B[bx*block_M + vid*sub_M//VEC_NUM + 0*sub_M, by*block_N], b_ub[0])
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.wait_flag("mte3", "mte2", nxt)
                        T.copy(A[bx*block_M + vid*sub_M//VEC_NUM + (mm+1)*sub_M, by*block_N], a_ub[nxt])
                        T.copy(B[bx*block_M + vid*sub_M//VEC_NUM + (mm+1)*sub_M, by*block_N], b_ub[nxt])
                        T.set_flag("mte2", "v", nxt)

                    T.wait_flag("mte2", "v", cur)
                    for (i, j) in T.Parallel(sub_M // VEC_NUM, block_N):
                        c_ub[cur, i, j] = a_ub[cur, i, j] + b_ub[cur, i, j]
                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)

                    T.copy(c_ub[cur], C[bx*block_M + vid*sub_M//VEC_NUM + mm*sub_M, by*block_N])
                    T.set_flag("mte3", "mte2", cur)

                clear_flag()

    return main
```

---

## 激活函数模板

### Sigmoid (手动展开版)

来源: `tilelang-ascend/examples/activation/sigmoid.py`

数学公式: `sigmoid(x) = 1 / (1 + exp(-x))`

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoid(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(a_ub, zero_ub, a_ub)   # a_ub = -x
            T.tile.exp(a_ub, a_ub)            # a_ub = exp(-x)
            T.tile.add(a_ub, a_ub, 1.0)       # a_ub = 1 + exp(-x)
            T.tile.reciprocal(b_ub, a_ub)      # b_ub = 1 / (1 + exp(-x))
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

**要点**:
- 使用 `T.tile.reciprocal(dst, src)` 计算 `1/src`，比 `T.tile.div(dst, one, src)` 更高效
- 使用 `T.tile.sub(dst, zero, src)` 实现 `-src` (negate)，因为无直接 negate API
- Developer模式使用 `T.alloc_shared()`，编译器自动映射到UB

### Sigmoid (内置指令版)

来源: `tilelang-ascend/examples/activation/sigmoidv2.py`

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sigmoidv2():
    dtype = "float"

    @T.prim_func
    def main(input: T.Tensor([4, 8], dtype), output: T.Tensor([4, 8], dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            input_shared = T.alloc_ub((4, 8), dtype)
            output_shared = T.alloc_ub((4, 8), dtype)

            T.copy(input, input_shared)
            T.tile.sigmoid(output_shared, input_shared)  # 直接调用内置sigmoid
            T.copy(output_shared, output)

    return main
```

**要点**:
- `T.tile.sigmoid(dst, src)` 是内置的单指令sigmoid，优先使用
- 也支持 slice 操作: `T.tile.sigmoid(output_shared[i, :], input_shared[i, :])`

### SiLU (Swish)

来源: `tilelang-ascend/examples/activation/silu.py`

数学公式: `silu(x) = x / (1 + exp(-x))` = `x * sigmoid(x)`

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def silu(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            denom_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(denom_ub, zero_ub, a_ub)   # denom = -x
            T.tile.exp(denom_ub, denom_ub)         # denom = exp(-x)
            T.tile.add(denom_ub, denom_ub, 1.0)    # denom = 1 + exp(-x)
            T.tile.div(b_ub, a_ub, denom_ub)        # b = x / (1 + exp(-x))
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

**要点**:
- SiLU = `x * sigmoid(x)` = `x / (1 + exp(-x))`
- 需要保存原始输入 `a_ub` 用于最终除法

### Tanh

来源: `tilelang-ascend/examples/activation/tanh.py`

数学公式: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="pto")
def tanh(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            nega_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            denom_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(nega_ub, zero_ub, a_ub)   # nega = -x
            T.tile.exp(a_ub, a_ub)                # a = exp(x)
            T.tile.exp(nega_ub, nega_ub)          # nega = exp(-x)
            T.tile.sub(b_ub, a_ub, nega_ub)       # b = exp(x) - exp(-x)  [numerator]
            T.tile.add(denom_ub, a_ub, nega_ub)   # denom = exp(x) + exp(-x)
            T.tile.div(b_ub, b_ub, denom_ub)      # b = tanh(x)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

**要点**:
- 注意 `T.tile.exp` 会覆盖输入buffer，所以需要先拷贝到 `nega_ub`
- 需要 5 个 UB buffer（输入/负值/分子/分母/零值填充）
- 使用 `target="pto"` 编译目标

### GELU (近似 tanh 展开版)

来源: `tilelang-ascend/examples/activation/gelu_mul.py`

数学公式: `gelu_mul(x1, x2) = GELU(x1) * x2`，其中 `GELU(x) = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3)))`

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def gelu_mul(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N // 2, block_N)   # N//2: x1和x2沿最后一维切分
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N // 2), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a2_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # 左半: x1, 右半: x2
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a1_ub)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N + N // 2], a2_ub)
            # x^2
            T.tile.mul(temp_ub, a1_ub, a1_ub)
            # x^3
            T.tile.mul(temp_ub, a1_ub, temp_ub)
            # 0.044715 * x^3
            T.tile.mul(temp_ub, temp_ub, 0.044715)
            # x + 0.044715 * x^3
            T.tile.add(temp_ub, a1_ub, temp_ub)
            # -sqrt(8/pi)(x + 0.044715 * x^3)
            T.tile.mul(temp_ub, temp_ub, -1.5957691)
            # exp(...)
            T.tile.exp(temp_ub, temp_ub)
            # 1 + exp(...)
            T.tile.add(temp_ub, temp_ub, 1.0)
            # x / (1 + exp(...))
            T.tile.div(temp_ub, a1_ub, temp_ub)
            # GELU(x1) * x2
            T.tile.mul(b_ub, temp_ub, a2_ub)
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

**要点**:
- 输入沿最后一维切分为 `x1` (左半) 和 `x2` (右半)
- `n_num = T.ceildiv(N // 2, block_N)` 因为只对左半维度做block切分
- 临时变量 `temp_ub` 可复用于多步计算
- `T.tile.mul(temp_ub, temp_ub, scalar)` 支持标量乘法

### SwiGLU

来源: `tilelang-ascend/examples/activation/swi_glu.py`

数学公式: `swi_glu(x) = Swish(x1) * x2 = (x1 * sigmoid(x1)) * x2`

```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def swi_glu(M, N, block_M, block_N, split_dim, dtype="float"):
    # split_dim控制切分维度: -1/1 = 最后一维, 0/-2 = 第一维
    m_div = 1
    n_div = 2
    m_offset = 0
    n_offset = N // 2
    if split_dim == 0 or split_dim == -2:
        m_div = 2
        n_div = 1
        m_offset = M // 2
        n_offset = 0
    m_num = T.ceildiv(M // m_div, block_M)
    n_num = T.ceildiv(N // n_div, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M // m_div, N // n_div), dtype)
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            a0_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            a1_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            zero_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            temp_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a0_ub)
            T.copy(A[bx * block_M + vid * block_M // VEC_NUM + m_offset,
                      by * block_N + n_offset], a1_ub)
            T.tile.fill(zero_ub, 0.0)
            T.tile.sub(temp_ub, zero_ub, a0_ub)    # -x
            T.tile.exp(temp_ub, temp_ub)            # exp(-x)
            T.tile.add(temp_ub, temp_ub, 1.0)       # 1 + exp(-x)
            T.tile.div(temp_ub, a0_ub, temp_ub)     # x / (1 + exp(-x)) = sigmoid(x)
            T.tile.mul(b_ub, temp_ub, a1_ub)         # sigmoid(x1) * x2
            T.copy(b_ub, B[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

**要点**:
- 支持灵活的切分维度 (`split_dim=0/-2` 按行切分, `split_dim=1/-1` 按列切分)
- 输出 tensor shape 为 `(M // m_div, N // n_div)`

### GELU 反向传播 (高级: compare + select)

来源: `tilelang-ascend/examples/activation/gelu_grad.py`

```python
@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def gelu_grad(M, N, block_M, block_N, dtype="float"):
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(dy: T.Tensor((M, N), dtype), x: T.Tensor((M, N), dtype),
             grad_input: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            cmp_ub = T.alloc_shared((block_M // VEC_NUM, block_N), "uint8")
            x_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
            # ... more UB buffers ...

            T.copy(x[bx * block_M + vid * block_M // VEC_NUM, by * block_N], x_ub)
            # ... compute gelu backward logic ...
            T.tile.compare(cmp_ub, x_ub, x_ub, "EQ")   # 比较生成 mask
            T.tile.select(xsqr_ub, cmp_ub, x_ub, px_ub, "VSEL_CMPMASK_SPR")  # 条件选择
            # ... final multiply with dy ...
            T.copy(dy[...], res0_ub)
            T.tile.mul(div_ub, res0_ub, x_ub)
            T.copy(div_ub, grad_input[...])

    return main
```

**要点**:
- `T.tile.compare(dst_mask, src0, src1, "EQ")` 输出 uint8 比较结果
- `T.tile.select(dst, mask, src0, src1, "VSEL_CMPMASK_SPR")` 根据 mask 选择 src0 或 src1
- `T.tile.axpy(dst, src, scalar)` 执行 `dst += scalar * src`，用于融合乘加
- Buffer 复用: 注意多个临时变量如何复用同一块 UB (如 `x_ub` 被多次复用)

---

## Reduce 模板

### 行归约 Reduce Min

来源: `tilelang-ascend/examples/reduce/example_reduce_min.py`

```python
@tilelang.jit(out_idx=[1])
def reduce_min(M, N, block_M, dtype="float"):
    m_num = M // block_M
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    @T.prim_func
    def main(A: T.Tensor([M, N], dtype), B: T.Tensor([M], dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((sub_block_M, N), dtype)
            b_ub = T.alloc_ub((sub_block_M), dtype)

            row_base = cid * block_M + vid * sub_block_M
            with T.Scope("V"):
                T.copy(A[row_base : row_base + sub_block_M, :], a_ub)
                T.barrier_all()
                T.reduce_min(a_ub, b_ub, dim=-1)    # 2D -> 1D: (sub_M, N) -> (sub_M,)
                T.barrier_all()
                T.copy(b_ub, B[row_base : row_base + sub_block_M])

    return main
```

**要点**:
- `T.reduce_min(buffer_2d, buffer_1d, dim=-1)` 将 2D buffer 沿最后一维归约为 1D
- 输出 shape: `(sub_M,)` — 每个 sub_M 行各出一个最小值
- 归约操作需要放在 `T.Scope("V")` 内 (Expert模式)

### 带流水线的 Reduce Min Pipeline

来源: `tilelang-ascend/examples/reduce/example_reduce_min_pipeline.py`

使用多级UB缓冲实现 MTE2-V-MTE3 流水线:

```python
@tilelang.jit(out_idx=[1], target="ascendc")
def reduce_min_pipeline(M, N, block_M, block_N, sub_M, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2
    stages = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            vec_proc = block_M // sub_M

            a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((stages, sub_M // VEC_NUM), dtype)

            with T.Scope("V"):
                T.barrier_all()
                T.copy(A[bx*block_M + vid*sub_M//VEC_NUM + 0*sub_M, by*block_N], a_ub[0, :, :])
                T.barrier_all()

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        T.barrier_all()
                        T.copy(A[bx*block_M + vid*sub_M//VEC_NUM + (mm+1)*sub_M, by*block_N],
                               a_ub[nxt, :, :])
                        T.barrier_all()

                    T.barrier_all()
                    T.reduce_min(a_ub[cur, :, :], b_ub[cur, :], dim=-1)
                    T.barrier_all()
                    T.copy(b_ub[cur, :], B[bx*block_M + vid*sub_M//VEC_NUM + mm*sub_M])
                    T.barrier_all()

    return main
```

**要点**:
- `stages=2` 双缓冲流水线: 加载下一块的同时计算当前块
- `(stages, sub_M // VEC_NUM, block_N)` — 第一维是缓冲级数
- 注意列切分: `by * block_N` 意味着对 N 维也做了分块

### 带切片缓冲的 Reduce Max (real_shape)

来源: `tilelang-ascend/examples/reduce/example_col_reduce_max_slice_buffer.py`

```python
@tilelang.jit(out_idx=[1], target="pto", pass_configs=pass_configs)
def reduce_max_slice_buffer():
    dtype = "float"

    @T.prim_func
    def main(Input: T.Tensor([5, 8], dtype), Output: T.Tensor([1, 8], dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            in_shared = T.alloc_ub((5, 8), dtype)
            out_shared = T.alloc_ub((1, 8), dtype=dtype)

            if vid == 0:
                T.copy(Input, in_shared)
                T.reduce_max(in_shared, out_shared, dim=0, real_shape=[3, 8])
                T.copy(out_shared, Output)

    return main
```

**要点**:
- `real_shape=[3, 8]` 参数: 只对 buffer 中实际有效区域进行归约，忽略 padding 部分
- `dim=0` 表示沿第一个维度(列归约)进行 max 归约: (5,8) -> (1,8)
- `dim=-1` 表示沿最后一个维度(行归约)进行 max 归约: (4,8) -> (1,8) 但只对前4列有效
- 使用 `if vid == 0:` 确保只有 Vector Core 0 执行，避免双核重复

---

## Normalization 模板

### RMS Norm

来源: `tilelang-ascend/examples/normalization/rms_norm.py`

数学公式: `rms_norm(x) = x / sqrt(mean(x^2) + eps)`

```python
@tilelang.jit(out_idx=[1])
def rms_norm(M, N, block_M, block_N, eps=1e-5, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            sum_square_i = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
            sum_square_ub = T.alloc_ub([block_M // VEC_NUM], dtype)
            mean_square_ub = T.alloc_ub([block_M // VEC_NUM], dtype)

            with T.Scope("V"):
                T.tile.fill(sum_square_i, 0.0)
                T.tile.fill(sum_square_ub, 0.0)
                T.tile.fill(mean_square_ub, N)
                T.barrier_all()

                # 阶段1: 累加 x^2 (沿N维度分块)
                for by in T.serial(n_num):
                    T.copy(A[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                             by*block_N:(by+1)*block_N], a_ub)
                    T.barrier_all()
                    T.tile.mul(a_ub, a_ub, a_ub)               # x^2
                    T.barrier_all()
                    T.tile.add(sum_square_i, sum_square_i, a_ub)  # 累加
                    T.barrier_all()

                # 阶段2: reduce + 求rms
                T.reduce_sum(sum_square_i, sum_square_ub, dim=-1)  # (sub_M, block_N) -> (sub_M,)
                T.barrier_all()
                T.tile.div(mean_square_ub, sum_square_ub, mean_square_ub)  # mean
                T.barrier_all()
                T.tile.fill(sum_square_ub, eps)
                T.barrier_all()
                T.tile.add(mean_square_ub, mean_square_ub, sum_square_ub)  # mean + eps
                T.barrier_all()
                T.tile.sqrt(mean_square_ub, mean_square_ub)               # sqrt(mean + eps) = rms
                T.barrier_all()

                # 阶段3: 逐行除以rms (重新读取输入)
                for by in T.serial(n_num):
                    T.copy(A[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                             by*block_N:(by+1)*block_N], a_ub)
                    T.barrier_all()
                    for i in range(block_M // VEC_NUM):
                        T.tile.div(a_ub[i, :], a_ub[i, :], mean_square_ub[i])  # x / rms
                        T.barrier_all()
                    T.copy(a_ub, B[bx*block_M+vid*block_M//VEC_NUM:bx*block_M+(vid+1)*block_M//VEC_NUM,
                                   by*block_N:(by+1)*block_N])
                    T.barrier_all()

    return main
```

**要点**:
- **三阶段算法**: (1) 累加 x^2 → (2) reduce_sum + 求均值 + sqrt → (3) 逐行归一化
- **需要重新读输入**: 归一化阶段重新从 GM 读取原始 x (因为 a_ub 在阶段1被 x^2 覆盖)
- **逐行归一化**: `for i in range(block_M // VEC_NUM)` 对每行分别除以对应的 rms 值
- **标量广播**: `T.tile.fill(mean_square_ub, N)` 填充 N 作为除数
- **跨block归约**: N维按 block_N 分块累加，最终 reduce_sum 得到每行总和

---

## GEMV 模板 (Vector Core版)

来源: `tilelang-ascend/examples/gemv/example_gemv_v.py`

```python
@tl.jit(out_idx=[-1], pass_configs={...})
def simple_gemv(N, K, block_N, block_K, dtype="float16", accum_dtype="float32"):
    VEC_NUM = 2
    TEMP_DTYPE = "uint8"
    CAST_MODE = "CAST_NONE"

    def cast_or_copy(dst, src, mode, count):
        if dtype != accum_dtype:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    n_num = T.ceildiv(N, block_N)
    k_num = T.ceildiv(K, block_K)
    kernel_num = T.ceildiv(n_num, VEC_NUM)

    @T.prim_func
    def main(x: T.Tensor((K,), dtype), A: T.Tensor((N, K), dtype), y: T.Tensor((N,), dtype)):
        with T.Kernel(kernel_num, is_npu=True) as (cid, vid):
            bn = (cid * VEC_NUM + vid) % n_num

            x_ub = T.alloc_ub((1, block_K), dtype)
            x_32_ub = T.alloc_ub((1, block_K), accum_dtype)
            A_ub = T.alloc_ub((block_N, block_K), dtype)
            A_32_ub = T.alloc_ub((block_N, block_K), accum_dtype)
            y_single_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_total_32_ub = T.alloc_ub((block_N,), accum_dtype)
            y_ub = T.alloc_ub((block_N,), dtype)

            T.tile.fill(y_total_32_ub, 0.0)

            for bk in T.serial(k_num):
                T.copy(x[bk * block_K], x_ub)
                T.copy(A[bn * block_N, bk * block_K], A_ub)
                cast_or_copy(x_32_ub, x_ub, CAST_MODE, block_K)        # cast fp16 -> fp32
                cast_or_copy(A_32_ub, A_ub, CAST_MODE, block_N * block_K)
                for i in T.serial(block_N):
                    T.tile.mul(A_32_ub[i, :], A_32_ub[i, :], x_32_ub)  # element-wise mul
                T.reduce_sum(A_32_ub, y_single_32_ub, dim=-1)           # sum along K
                T.tile.add(y_total_32_ub, y_total_32_ub, y_single_32_ub)

            cast_or_copy(y_ub, y_total_32_ub, CAST_MODE, block_N)       # cast fp32 -> fp16
            T.copy(y_ub, y[bn * block_N])

    return main
```

**要点**:
- **Vector GEMV**: y = x @ A^T，使用 `T.tile.mul` + `T.reduce_sum` 实现点积
- **精度提升**: 先 cast 到 `accum_dtype` (float32) 计算，再 cast 回 `dtype` (float16)
- **`T.tile.cast(dst, src, mode, count)`**: mode 如 `"CAST_NONE"`, `"CAST_RINT"` 等
- **累加模式**: `T.tile.fill(y_total_32_ub, 0.0)` + 循环中 `T.tile.add` 累加
- **VEC_NUM 分配**: `kernel_num = T.ceildiv(n_num, VEC_NUM)`，每个 AIC 处理两个 AIV

## 调试工具

### T.printf -- 设备端打印

```python
T.printf("value=%d addr=%x\n", val, addr)
# 格式: %d(整数), %f(浮点), %x(十六进制), %s(字符串)
```

### T.dump_tensor -- 张量转储

```python
T.dump_tensor(buf, 111, 64)             # 转储64个元素
T.dump_tensor(buf, 111, 64, (8, 8))     # 按8x8矩阵格式转储
# 支持: ub_buffer, l1_buffer, l0c_buffer, global_buffer
# desc参数: 用户自定义附加信息 (如行号)
```

### 查看生成的代码

```python
func = vec_add(M, N, 128, 256)
print(f"{func.get_kernel_source()}")    # 输出 AscendC 代码
```

## T.alloc_var -- 标量变量

用于条件标志位、循环计数器等:

```python
flag = T.alloc_var("bool", init=False)
counter = T.alloc_var("int32", init=0)
value = T.alloc_var("float32", init=0.0)
# 变量间初始化
a = T.alloc_var("int32", init=1)
b = T.alloc_var("int32", init=a)
```

## 适用场景

- 元素级运算: add, sub, mul, div
- 激活函数: relu, sigmoid, silu, gelu, tanh, swi_glu (通过T.tile.exp/ln等组合或T.tile.sigmoid)
- 归约运算: reduce_sum, reduce_max, reduce_min (支持real_shape切片)
- 类型转换: cast (精度转换, T.tile.cast)
- 逐行/逐列运算: softmax, layernorm, rms_norm
- 数据填充: T.tile.fill
- 向量-矩阵乘: GEMV (Vector Core版, mul+reduce_sum)
- 比较和选择: T.tile.compare, T.tile.select (用于梯度计算等)
- AXPY运算: T.tile.axpy (融合乘加)

## 不适用场景

- **矩阵乘法 (GEMM)**: 使用 `tl-op-cube` 技能
- **矩阵乘法 + 向量后处理**: 使用 `tl-op-fused` 技能

## Vector Core 架构要点

```
GM --(MTE2)--> UB --(V计算)--> UB --(MTE3)--> GM
                 ^                  ^
              alloc_ub           alloc_ub
              T.Scope("V")       T.Scope("V")
```

**关键约束**:
- **只有 Vector Core (V) 才能操作 UB 进行计算**
- `T.tile.*` 指令**必须在** `T.Scope("V")` 内 (Expert模式)
- `T.copy` (GM↔UB) **必须在** `T.Scope("V")` 内 (Expert模式)
- **VEC_NUM = 2**: 每个AIC有2个AIV，UB按 `block_size // VEC_NUM` 分配

## UB分配规则

```python
VEC_NUM = 2
a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)   # 每个VCore处理一半
b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
```

数据加载地址计算:
```python
T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
# vid=0: 加载前半 block_M//2 行
# vid=1: 加载后半 block_M//2 行
```

## 数据类型

dtype参数使用字符串: `"float16"`, `"float32"`, `"float"`, `"bfloat16"`, `"int8"`, `"int32"`

## 数据创建与验证

```python
import torch

torch.manual_seed(0)

a = torch.randn(M, N).npu()   # float32
b = torch.randn(M, N).npu()
func = vec_add(M, N, 128, 256)
c = func(a, b)
ref_c = a + b
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
```

## 坐标计算对照

| 维度 | Block数量 | 坐标计算 |
|------|----------|---------|
| **1D** | `n_num` | 直接使用 `cid` |
| **2D** | `m_num * n_num` | `bx = cid // n_num`, `by = cid % n_num` |
| **3D** | `d_num * m_num * n_num` | `bz = cid // (m_num * n_num)`, `bx = remaining // n_num`, `by = remaining % n_num` |

## 同步原语

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| `T.barrier_all()` | 全局屏障 | Expert模式，每步手动同步 |
| pass_configs `TL_ASCEND_AUTO_SYNC` | 编译器自动同步 | Developer模式 |
| `T.set_flag` / `T.wait_flag` | 核内流水线同步 | 细粒度MTE2-V-MTE3流水线 |

流水线同步模式:
```
MTE2 (加载) --set_flag("mte2", "v", 0)--> V (计算) --set_flag("v", "mte3", 0)--> MTE3 (存储)
                    |                              |                              |
                    v                              v                              v
              wait_flag                     wait_flag                      wait_flag
```

## 约束

- **dtype用字符串**: `"float16"`, `"float32"`, `"float"` (禁止 `T.float16`, `T.bfloat16`)
- **必须使用NPU专用语法**: `is_npu=True`, `T.Scope("V")` (Expert模式)
- **VEC_NUM = 2**: 每个block由2个vector core并行处理
- **禁止CUDA语法**: `alloc_fragment`(用于Vector), `threads`, `T.Pipelined`
- **T.tile.fill 在NPU上可用**: 可以使用 `T.tile.fill(buffer, value)` 填充buffer
- **硬件约束**: 参考 `tl-op-hardware-constraints`

## 参考

- Elementwise Add示例: `tilelang-ascend/examples/elementwise/elementwise_add.py`
- 流水线示例: `tilelang-ascend/examples/elementwise/elementwise_add_pipeline.py`
- SetValue示例: `tilelang-ascend/examples/elementwise/setvalue_example.py`
- **激活函数**: `tilelang-ascend/examples/activation/` (sigmoid, sigmoidv2, silu, tanh, gelu_mul, swi_glu, gelu_grad)
- **Reduce**: `tilelang-ascend/examples/reduce/` (reduce_min, reduce_min_pipeline, reduce_max_slice_buffer)
- **Normalization**: `tilelang-ascend/examples/normalization/rms_norm.py`
- **GEMV (Vector)**: `tilelang-ascend/examples/gemv/example_gemv_v.py`
- API参考: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/`
- 硬件约束: `skills/tl-op-hardware-constraints/SKILL.md`
