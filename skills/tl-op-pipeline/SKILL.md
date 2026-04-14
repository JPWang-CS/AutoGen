---
name: tl-op-pipeline
description: 用户新增TileLang-Ascend算子时的入口技能。根据算子类型（纯向量/纯矩阵/融合）自动分发到对应子技能。基于 tilelang-ascend/examples/ 和 .agents/skills/ 权威参考。支持Expert模式和Developer模式两种编程范式。
---

# 新增 TileLang-Ascend 算子通路技能 (入口)

基于 `tilelang-ascend/examples/` 和 `tilelang-ascend/.agents/skills/` 权威参考。

当用户提出"新增一个TileLang算子/生成TileLang算子通路/按模板生成TileLang算子"时使用本技能。

本技能是**入口分发器**，根据算子类型自动路由到对应的子技能。

## 生成前必读: 硬件约束

生成算子前**必须**参考 `skills/tl-op-hardware-constraints/SKILL.md` 中的硬件约束，确保:

1. **存储对齐**: UB 32B、L0A/L0B 512B、L0C 64B
2. **分形约束**: block_M/N/K 必须是分形大小(16或32)的整数倍
3. **Buffer容量**: UB总使用量 <= 2MB，L1 <= 1MB
4. **同步配对**: `set_flag`/`wait_flag` 和 `set_cross_flag`/`wait_cross_flag` 必须成对

## 算子类型判断与分发

### 判断流程

```
算子是否包含矩阵乘法 (T.gemm_v0 / T.mma)?
|-- 否 --> 纯向量算子
|   └--> skills/tl-op-vector/SKILL.md
|
\-- 是 --> 矩阵乘法后是否有向量后处理?
    |-- 否 (仅 A @ B = C) --> 纯矩阵算子
    |   └--> skills/tl-op-cube/SKILL.md
    |
    \-- 是 (如 +bias, relu, softmax) --> 融合算子
        └--> skills/tl-op-fused/SKILL.md
```

### 分发表

| 算子特征 | 子技能 | 说明 |
|---------|--------|------|
| **纯向量计算** (无矩阵乘法) | `tl-op-vector` | add, mul, relu, softmax, layernorm |
| **纯矩阵乘法** (无后处理) | `tl-op-cube` | GEMM: C = A @ B |
| **矩阵乘法 + 向量后处理** | `tl-op-fused` | Matmul+Bias, Matmul+ReLU, Flash Attention |

### 常见算子分类示例

| 算子 | 类型 | 子技能 |
|------|------|--------|
| vector_add | 纯向量 | `tl-op-vector` |
| element-wise mul/div | 纯向量 | `tl-op-vector` |
| relu / sigmoid / gelu | 纯向量 | `tl-op-vector` |
| softmax | 纯向量 | `tl-op-vector` |
| layernorm | 纯向量 | `tl-op-vector` |
| GEMM (C = A @ B) | 纯矩阵 | `tl-op-cube` |
| GEMM with transpose (C = A @ B^T) | 纯矩阵 | `tl-op-cube` |
| Matmul + Bias (C = A@B + bias) | 融合 | `tl-op-fused` |
| Matmul + ReLU (C = relu(A@B)) | 融合 | `tl-op-fused` |
| Flash Attention | 融合 | `tl-op-fused` |

## NPU核心API速查

### 两种编程模式

| 特性 | Expert 模式 | Developer 模式 |
|------|------------|---------------|
| Cube内存 | `T.alloc_L1()`, `T.alloc_L0C()` | `T.alloc_shared()`, `T.alloc_fragment()` |
| Vector内存 | `T.alloc_ub()` | `T.alloc_shared()` |
| GEMM | `T.gemm_v0(A_L1, B_L1, C_L0C, init=...)` | 同左 |
| 向量计算 | `T.tile.add(c, a, b)` | `T.Parallel` + 标量运算 |
| 同步 | 手动 `T.barrier_all()`, `T.set_flag/wait_flag` | pass_configs 自动 |
| 作用域 | `T.Scope("C")` / `T.Scope("V")` | 无需 |
| pass_configs | 不使用或全False | 开启AUTO_SYNC, MEMORY_PLANNING等 |

### GEMM API

| API | 说明 |
|-----|------|
| `T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)` | 块级矩阵乘 (A,B在L1/shared; C在L0C/fragment) |
| `T.mma(A, B, C, init=False)` | 底层矩阵乘累加 (A在L0A; B在L0B; C在L0C) |

**init参数**: `init=True`清零C后计算，`init=False`累加。典型: `init=(k == 0)`

### 内存分配

| Expert模式 | Developer模式 | 存储层级 | 用途 |
|-----------|-------------|---------|------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` | L1 Buffer | Cube数据中转 |
| `T.alloc_L0A(shape, dtype)` | - | L0A Buffer | Cube左矩阵输入(仅mma) |
| `T.alloc_L0B(shape, dtype)` | - | L0B Buffer | Cube右矩阵输入(仅mma) |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` | L0C Buffer | Cube输出/累加 |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` | UB | Vector数据/计算 |

### Vector Tile指令 (Expert模式)

| 类别 | API |
|------|-----|
| 算术 | `T.tile.add/sub/mul/div/max/min(dst, src0, src1)` -- src1可为scalar |
| 单目 | `T.tile.exp/ln/abs/sqrt/rsqrt/relu/sigmoid/sin/cos(dst, src)` |
| 激活 | `T.tile.leaky_relu(dst, src, scalar)`, `T.tile.axpy(dst, src, scalar)` |
| 位运算 | `T.tile.bitwise_and/or/xor(dst, src0, src1)`, `T.tile.bitwise_not(dst, src)` |
| 移位 | `T.tile.bitwise_lshift/rshift(dst, src0, scalar)` |
| 比较 | `T.tile.compare(dst, src0, src1, mode)` -- mode: "EQ","NE","GT","GE","LT","LE" |
| 选择 | `T.tile.select(dst, mask, src0, src1, selMode)` -- selMode: "VSEL_CMPMASK_SPR"等 |
| 数据填充 | `T.tile.fill(buffer, value)`, `T.tile.clear(buffer)` |
| 精度转换 | `T.tile.cast(dst, src, mode, count)` -- mode: "CAST_NONE","CAST_RINT"等 |
| 转置 | `T.tile.transpose(dst, src)` -- 仅支持16x16 |
| 索引 | `T.tile.createvecindex(dst, first_value)`, `T.tile.arith_progression(buf, first, diff, count)` |
| 排序 | `T.tile.sort(dst, src, actual_num)`, `T.tile.merge_sort(dst, src0, src1[, src2[, src3]])`, `T.tile.topk(dst, src, block_size)` |
| 收集 | `T.tile.gather(dst, src, src_offset, base)`, `T.tile.gather_mask(dst, src, pattern)` |
| 归约 | `T.reduce_max/reduce_min/reduce_sum(buffer, out, dim)` |

### 标量变量分配

```python
flag = T.alloc_var("bool", init=False)         # 布尔标志位
counter = T.alloc_var("int32", init=1)          # 整数计数器
value = T.alloc_var("float32", init=0.0)        # 浮点临时变量
# scope参数可选: scope="local.var" (默认)
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
| UB | L1 | Vector->Cube数据传递 |

### 同步原语

| API | 说明 |
|-----|------|
| `T.barrier_all()` | 全局屏障 |
| `T.pipe_barrier(pipe)` | 特定流水线阶段屏障 |
| `T.sync_all()` | 计算单元内全局同步 |
| `T.set_flag(src, dst, eventId)` | 核内流水线标志 (管线: "mte1","mte2","mte3","m","v","fix") |
| `T.wait_flag(src, dst, eventId)` | 等待核内流水线标志 |
| `T.set_cross_flag(pipe, flag)` | 核间同步标志 (Cube↔Vector) |
| `T.wait_cross_flag(flag)` | 等待核间同步标志 |

### 作用域

| Scope | 说明 |
|-------|------|
| `T.Scope("C")` | Cube Core域: T.copy(GM↔L1), T.gemm_v0, T.mma, T.barrier_all |
| `T.Scope("V")` | Vector Core域: T.copy(GM↔UB), T.tile.*, T.reduce_*, T.barrier_all |

### Kernel启动

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

### pass_configs (Developer模式)

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动CV分离(融合算子)
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动核间同步(融合算子)
}

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
```

### 其他有用API

| API | 说明 |
|-----|------|
| `T.Persistent(domain, wave_size, index)` | 优化多Block调度，提高缓存命中 |
| `T.Pipelined(range, num_stages=N, cross_interval=1)` | 计算搬运流水线并行，cross_interval控制核间同步频率 |
| `T.ceildiv(a, b)` | 向上取整除法 |
| `T.annotate_address({buf: offset})` | 手动地址规划(Expert模式) |
| `T.annotate_layout({buf: layout})` | 手动数据布局(Expert模式, 配合mma) |
| `T.alloc_var(dtype, init, scope)` | 分配标量变量(标志位、计数器等) |
| `T.printf(format_str, *args)` | 设备端调试打印 |
| `T.dump_tensor(tensor, desc, dump_size, shape_info)` | 设备端张量转储 |
| `func.get_kernel_source()` | 查看生成的AscendC代码 |
| `workspace_idx=[4,5,...]` | JIT装饰器参数，自动管理workspace内存 |

### 调试工具

```python
# 设备端打印
T.printf("value=%d addr=%x\n", val, addr)

# 张量转储 (支持ub/l1/l0c/global)
T.dump_tensor(buf, 111, 64)             # 转储64个元素
T.dump_tensor(buf, 111, 64, (8, 8))     # 按8x8矩阵格式转储

# 查看生成的AscendC代码
print(f"{func.get_kernel_source()}")
```

### 完整 pass_configs 参考

```python
# Developer模式 (推荐)
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,         # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # 自动内存规划
}

# 融合算子 (Cube+Vector)
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # 自动CV分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # 自动核间同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

# 其他可用配置
# TL_SIMPLIFY=True               # TileLang简化Pass
# TIR_MERGE_STATIC_SMEM=True      # 合并静态共享内存
# TIR_USE_ASYNC_COPY=True         # 异步拷贝
# TIR_DISABLE_VECTORIZE=False     # 禁用向量化
# TIR_SIMPLIFY=True               # TIR简化
```

### 性能调优

```bash
# 上板验证
msprof op --kernel-name="kernel_name" python script.py

# 仿真验证
msprof op simulator --soc-version=Ascend910B2 --kernel-name="kernel_name" python script.py
```

### aclGraph 入图加速

```python
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    result = tilelang_op(input1, input2)
g.replay()  # 多次重放，减少Host交互开销
```

## 数据类型

**dtype参数使用字符串**: `"float16"`, `"float32"`, `"float"`, `"bfloat16"`, `"int8"`, `"int32"`

禁止使用: `T.float16`, `T.bfloat16`, `torch.float16` (在TileLang代码中)

## 数据创建与验证

```python
import torch

torch.manual_seed(0)

# float16数据
a = torch.randn(M, K).half().npu()
b = torch.randn(K, N).half().npu()

# float32数据
a = torch.randn(M, K).npu()
b = torch.randn(K, N).npu()

# 调用kernel
func = matmul(M, N, K, 128, 256, 64)
c = func(a, b)

# 精度对比在CPU上进行
ref_c = a.cpu() @ b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
```

## 工作流

1. 确认需要生成的算子类型（纯向量/纯矩阵/融合）。
2. **分发到对应子技能**:
   - 纯向量 --> `skills/tl-op-vector/SKILL.md`
   - 纯矩阵 --> `skills/tl-op-cube/SKILL.md`
   - 融合 --> `skills/tl-op-fused/SKILL.md`
3. 确认目标子项目路径（例如 `projects/gemm/...`）。
4. 选择编程模式（Expert / Developer）。
5. **检查硬件约束**（参考 `tl-op-hardware-constraints`）。
6. 根据子技能的模板生成算子完整通路。
7. 检查生成目录及文件的完整性。
8. 输出改动说明与假设。

## 子技能索引

| 子技能 | 路径 | 说明 |
|--------|------|------|
| **tl-op-vector** | `skills/tl-op-vector/SKILL.md` | 纯向量计算 (Vector Core) |
| **tl-op-cube** | `skills/tl-op-cube/SKILL.md` | 纯矩阵乘法 (Cube Core) |
| **tl-op-fused** | `skills/tl-op-fused/SKILL.md` | 融合算子 (Cube + Vector) |
| **tl-op-hardware-constraints** | `skills/tl-op-hardware-constraints/SKILL.md` | 硬件约束检查 |
| **tl-op-edit** | `skills/tl-op-edit/SKILL.md` | 修改算子参数 |
| **tl-op-test** | `skills/tl-op-test/SKILL.md` | 生成测试用例 |
| **tl-op-benchmark** | `skills/tl-op-benchmark/SKILL.md` | 性能测试 |
| **tl-op-rename** | `skills/tl-op-rename/SKILL.md` | 重命名算子 |

## 约束

- **GEMM API用 `T.gemm_v0()`**: 不是 `T.gemm()`。使用 `init=` 参数控制清零
- **dtype用字符串**: `"float16"`, `"float32"` (禁止 `T.float16`, `T.bfloat16`)
- **必须使用NPU专用语法**: `is_npu=True`
- **禁止CUDA语法**: `T.Parallel`(非Developer模式), `threads`, `T.Pipelined`(非Developer模式)
- **禁止旧API**: `T.clear()`, `T.fill()`(顶层), `clear_accum`, `T.gemm()`
- **T.tile.fill 在NPU上可用**: `T.tile.fill(buffer, value)` 可以在Vector域使用
- **硬件约束**: 参考 `tl-op-hardware-constraints`

## 参考

- GEMM示例: `tilelang-ascend/examples/gemm/`
- Elementwise示例: `tilelang-ascend/examples/elementwise/`
- Flash Attention示例: `tilelang-ascend/examples/flash_attention/`
- API参考: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-api-best-practices/`
- Expert/Developer对比: `tilelang-ascend/.agents/skills/tilelang-custom-skill/tilelang-expert-to-developer/`
- 硬件约束: `skills/tl-op-hardware-constraints/SKILL.md`
