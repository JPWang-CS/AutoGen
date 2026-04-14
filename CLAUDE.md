# TileLang-Ascend AutoGen 仓库 (NPU专用)

本仓库提供TileLang-Ascend NPU算子自动生成的skill提示词和相关模板。所有算子运行在华为昇腾NPU上。

## ⚠️ 权威参考源

- **tilelang-ascend 仓库**: `M:\Desktop\tmp\AgentTest\TileLang\tilelang-ascend` — 正确的 NPU API
- **tilelang-ascend 示例**: `tilelang-ascend/examples/` — 可运行的参考代码
- **tilelang-ascend Skills**: `tilelang-ascend/.agents/skills/` — API最佳实践和编程指南
- **⚠️ 旧仓库仅参考**: `M:\Desktop\tmp\AgentTest\TileLang\tilelang` 是上游 GPU TileLang，API与 NPU 版本差异很大，**禁止直接使用其语法**

## ⚠️ NPU 两种编程模式

| 特性 | Expert 模式 | Developer 模式 |
|------|------------|---------------|
| Cube内存 | `T.alloc_L1()`, `T.alloc_L0C()` | `T.alloc_shared()`, `T.alloc_fragment()` |
| Vector内存 | `T.alloc_ub()` | `T.alloc_shared()` |
| GEMM | `T.gemm_v0(A_L1, B_L1, C_L0C, init=...)` | 同左 |
| 向量计算 | `T.tile.add(c, a, b)` | `T.Parallel` + 标量运算 |
| 同步 | 手动 `T.barrier_all()`, `T.set_flag/wait_flag` | pass_configs 自动 |
| 作用域 | `T.Scope("C")` / `T.Scope("V")` | 无需 |
| pass_configs | 不使用或全False | 开启AUTO_SYNC, MEMORY_PLANNING等 |

## ⚠️ NPU 执行引擎架构

### 内存层级 (不可跨级访问)

```
GM（全局内存）
  ↕ T.copy
L1（Cube 核缓存）/ UB（Vector 核缓冲）
  ↕ T.copy
L0A / L0B（矩阵输入寄存器）→ L0C（矩阵输出寄存器）
```

| 引擎 | 功能 | 操作范围 |
|------|------|---------|
| **MTE2** | GM → L1/UB 数据加载 | T.Scope("C")或T.Scope("V") 内 |
| **Cube** | 矩阵乘法 `T.gemm_v0`/`T.mma` | T.Scope("C") 内 |
| **V (Vector)** | 向量计算 `T.tile.*` | T.Scope("V") 内 |
| **MTE3** | L1/UB → GM 数据存储 | T.Scope("C")或T.Scope("V") 内 |

### 内存分配 API

| Expert模式 | Developer模式 | 存储层级 | 用途 |
|-----------|-------------|---------|------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` | L1 Buffer | Cube数据中转 |
| `T.alloc_L0A(shape, dtype)` | - | L0A Buffer | Cube左矩阵输入(仅mma) |
| `T.alloc_L0B(shape, dtype)` | - | L0B Buffer | Cube右矩阵输入(仅mma) |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` | L0C Buffer | Cube输出/累加 |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` | UB | Vector数据/计算 |

## ⚠️ 核心 API (必读)

### GEMM API

| API | 说明 |
|-----|------|
| `T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)` | 块级矩阵乘 (A,B在L1; C在L0C) |
| `T.mma(A, B, C, init=False)` | 底层矩阵乘累加 (A在L0A; B在L0B; C在L0C) |

**init参数**: `init=True` 清零C后计算，`init=False` 累加。典型: `init=(k == 0)`

### Vector Tile 指令

| 类别 | API |
|------|-----|
| 算术 | `T.tile.add/sub/mul/div/max/min(dst, src0, src1)` -- src1可为scalar |
| 单目 | `T.tile.exp/ln/abs/sqrt/rsqrt/relu/sigmoid/sin/cos(dst, src)` |
| 激活 | `T.tile.leaky_relu(dst, src, scalar)`, `T.tile.axpy(dst, src, scalar)` |
| 位运算 | `T.tile.bitwise_and/or/xor/not/lshift/rshift` |
| 比较 | `T.tile.compare(dst, src0, src1, mode)` -- mode: "EQ","NE","GT","GE","LT","LE" |
| 选择 | `T.tile.select(dst, mask, src0, src1, selMode)` |
| 数据 | `T.tile.fill(buffer, value)`, `T.tile.clear(buffer)`, `T.tile.cast(dst, src, mode, count)` |
| 转置 | `T.tile.transpose(dst, src)` -- 仅16x16 |
| 索引 | `T.tile.createvecindex(dst, first_val)`, `T.tile.arith_progression(buf, first, diff, count)` |
| 排序 | `T.tile.sort/merge_sort/topk` |
| 收集 | `T.tile.gather/gather_mask` |
| 标量变量 | `T.alloc_var(dtype, init, scope)` -- 标志位、计数器等 |
| 归约 | `T.reduce_max/reduce_min/reduce_sum(buffer, out, dim)` |

### 同步原语

| API | 说明 |
|-----|------|
| `T.barrier_all()` | 全局屏障 |
| `T.set_flag(src, dst, eventId)` | 核内流水线标志 |
| `T.wait_flag(src, dst, eventId)` | 等待核内流水线标志 |
| `T.set_cross_flag(pipe, flag)` | 核间同步标志 (Cube↔Vector) |
| `T.wait_cross_flag(flag)` | 等待核间同步标志 |

### Kernel 启动

```python
# Cube操作 (不使用VEC_NUM)
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num
    by = cid % n_num

# Vector操作 (VEC_NUM=2)
VEC_NUM = 2
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
```

### T.copy 支持路径

| src | dst | 说明 |
|-----|-----|------|
| GM | L1 | Cube数据加载 |
| L1 | L0A/L0B | Cube矩阵输入 |
| L0C | GM | Cube结果写回 |
| GM | UB | Vector数据加载 |
| UB | GM | Vector结果写回 |
| UB | UB | UB间拷贝 |
| UB | L1 | Vector到Cube数据传递 |

### 调试工具

```python
# 设备端打印 (支持 %d %f %x %s)
T.printf("val=%d\n", my_val)

# 张量转储 (支持 ub/l1/l0c/global)
T.dump_tensor(buf, desc_id, num_elements, shape_info=())

# 查看生成的 AscendC 代码
print(f"{func.get_kernel_source()}")
```

### T.Pipelined 用法

```python
# 核内流水线 (重叠 copy + gemm)
for k in T.Pipelined(loop_k, num_stages=2):
    T.copy(A[...], A_L1)
    T.copy(B[...], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))

# 核间流水线 (重叠 Cube + Vector, 需 AUTO_CV_COMBINE + AUTO_CV_SYNC)
for k in T.Pipelined(num_iters, num_stages=2, cross_interval=1):
    # Cube writes workspace...
    # Vector reads workspace...
```

### Workspace 自动分配

```python
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7])
def fused_op(...):
    @T.prim_func
    def main(
        Input: ...,
        Output: ...,           # out_idx=3
        workspace_1: ...,      # workspace_idx=4, 自动管理
        workspace_2: ...,      # workspace_idx=5, 自动管理
    ):
        ...
# 用户调用: output = fused_op(input)  -- workspace自动分配
```

### pass_configs (Developer模式)

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 融合算子
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 融合算子
}
```

## ⚠️ GEMM 正确模式 (Expert)

来源: `tilelang-ascend/examples/gemm/example_gemm.py`

```python
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
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

## ⚠️ Vector Add 正确模式 (Expert)

来源: `tilelang-ascend/examples/elementwise/elementwise_add.py`

```python
@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, dtype="float"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
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

## ⚠️ 禁止使用的 API (来自上游 GPU TileLang)

| 禁止使用 | 正确替代 |
|---------|---------|
| `T.gemm(A, B, C)` | `T.gemm_v0(A_L1, B_L1, C_L0C, init=...)` |
| `T.gemm(..., clear_accum=True)` | `T.gemm_v0(..., init=True)` 或 `init=(k==0)` |
| `T.clear(buf)` | `T.gemm_v0(..., init=True)` (GEMM) 或 `T.tile.fill(buf, 0.0)` (Vector) |
| `T.fill(buf, val)` (顶层) | `T.tile.fill(buf, val)` (Vector域) |
| `T.alloc_shared` (Expert模式) | `T.alloc_L1` (Cube) 或 `T.alloc_ub` (Vector) |
| `T.alloc_fragment` (Expert模式) | `T.alloc_L0C` (GEMM输出) |
| `T.float16`, `T.bfloat16` | 字符串 `"float16"`, `"bfloat16"` |
| `T.Parallel` (Expert模式) | `T.tile.*` 向量指令 |
| `T.Pipelined` (Expert模式) | `T.serial` + 手动流水线 |
| `threads=N` | `is_npu=True` |

## 数据类型

**dtype参数使用字符串**: `"float16"`, `"float32"`, `"float"`, `"bfloat16"`, `"int8"`, `"int32"`

## 数据创建与验证

```python
import torch

torch.manual_seed(0)

# float16数据
a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()

# float32数据
a = torch.randn(M, N).npu()

# 调用kernel
func = matmul(M, N, K, 128, 256, 64)
c = func(a, b)

# 精度对比
ref_c = a @ b  # NPU上直接对比
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
```

## ⚠️ 防止硬件地址越界

| 存储单元 | 对齐要求 | Buffer容量 |
|---------|---------|-----------|
| **UB** | 32字节 | ≤ 2MB (910B) |
| **L1** | 32字节 | ≤ 1MB (910B) |
| **L0A/L0B** | 512字节 | - |
| **L0C** | 64字节 | - |

- block_M/block_N/block_K 必须是 **分形大小(16或32)** 的整数倍
- `T.set_flag`/`T.wait_flag` 和 `T.set_cross_flag`/`T.wait_cross_flag` **必须成对使用**

**详细约束**: `skills/tl-op-hardware-constraints/SKILL.md`

## 仓库结构

```
AutoGen/
├── skills/                          # Skill提示词目录
│   ├── tl-op-pipeline/              # 算子生成入口 (分发器)
│   ├── tl-op-vector/                # 纯向量算子 (Vector Core Only)
│   ├── tl-op-cube/                  # 纯矩阵算子 (Cube Core Only)
│   ├── tl-op-fused/                 # 融合算子 (Cube + Vector)
│   ├── tl-op-edit/                  # 修改算子技能
│   ├── tl-op-rename/                # 重命名算子技能
│   ├── tl-op-test/                  # 生成测试技能
│   ├── tl-op-benchmark/             # 生成benchmark技能
│   └── tl-op-hardware-constraints/  # NPU硬件约束技能
├── templates/op_frame/              # 模板目录
│   ├── empty_op_template/           # 空白算子模板
│   └── gemm_reference/              # GEMM参考实现
├── projects/                        # 生成的算子项目目录
│   ├── matmul/                      # GEMM矩阵乘法
│   ├── vector_add/                  # 向量加法
│   ├── softmax/                     # 在线Softmax
│   └── layer_norm/                  # Layer Normalization
├── TileLangQuickStart/              # 快速入门文档
├── CLAUDE.md                        # 本文件
└── README.md                        # 仓库说明
```

## 可用技能

### tl-op-pipeline (入口分发器)
新增TileLang算子时的入口技能。根据算子类型自动分发：
- **无矩阵乘法** → `tl-op-vector` (纯向量计算)
- **仅矩阵乘法** → `tl-op-cube` (纯GEMM)
- **矩阵乘法+后处理** → `tl-op-fused` (融合算子)

### tl-op-vector (纯向量算子)
生成仅使用 Vector Core 的算子: add, mul, relu, softmax, layernorm 等。
- Expert: `T.alloc_ub` + `T.tile.*` + `T.Scope("V")`
- Developer: `T.alloc_shared` + `T.Parallel` + pass_configs

### tl-op-cube (纯矩阵乘法)
生成仅使用 Cube Core 的算子: GEMM (C = A @ B), 含转置和持久化变体。
- Expert: `T.alloc_L1` + `T.alloc_L0C` + `T.gemm_v0` + `T.Scope("C")`
- Developer: `T.alloc_shared` + `T.alloc_fragment` + `T.gemm_v0`

### tl-op-fused (融合算子)
生成 Cube + Vector 融合算子: Matmul+Bias, Flash Attention 等。
- Expert: workspace + `T.set_cross_flag`/`T.wait_cross_flag`
- Developer: pass_configs AUTO_CV_COMBINE

## 开发流程

1. 使用 `tl-op-pipeline` 技能生成算子框架
2. 选择编程模式 (Expert / Developer)
3. 检查硬件约束 (`tl-op-hardware-constraints`)
4. 根据需求修改算子实现
5. 使用 `tl-op-test` 生成测试用例
6. 使用 `tl-op-benchmark` 进行性能测试

## 参考资料

- **tilelang-ascend 仓库**: `M:\Desktop\tmp\AgentTest\TileLang\tilelang-ascend` (权威)
- **GEMM示例**: `tilelang-ascend/examples/gemm/`
- **Elementwise示例**: `tilelang-ascend/examples/elementwise/`
- **Softmax示例**: `tilelang-ascend/examples/softmax/`
- **Normalization示例**: `tilelang-ascend/examples/normalization/`
- **Flash Attention示例**: `tilelang-ascend/examples/flash_attention/`
- **API参考**: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/`
- **Expert↔Developer对比**: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/`
- [TileLang-Ascend GitHub](https://github.com/tile-ai/tilelang-ascend)
