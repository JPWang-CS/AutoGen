# TileLang-Ascend AutoGen 仓库 (NPU专用)

本仓库提供TileLang-Ascend NPU算子自动生成的skill提示词和相关模板。所有算子运行在华为昇腾NPU上，精度对比在CPU上进行。

## ⚠️ NPU 编程规范 (重要)

### NPU 执行引擎架构

| 引擎 | 名称 | 功能 | 能否操作 UB |
|------|------|------|------------|
| **MTE2** | Memory Transfer Engine 2 | GM -> UB 数据加载 | ❌ 只搬运，不计算 |
| **V** | Vector Core | 向量计算 | ✅ **只有 V 核才能操作 UB 进行计算** |
| **MTE3** | Memory Transfer Engine 3 | UB -> GM 数据存储 | ❌ 只搬运，不计算 |
| **Cube** | Cube Core | 矩阵乘法 | ❌ **Cube 核没有 UB！** |

### ⚠️ 关键约束

1. **只有 Vector Core (V) 才能操作 UB 进行计算**
   - `T.tile.add`, `T.tile.mul` 等向量指令**必须在** `T.Scope("V")` 内
   - Cube 核用于矩阵乘法 (`T.gemm`)，**没有 UB**

2. **数据搬运必须在对应的作用域内执行**
   - `T.copy(A[...], a_ub)`: GM -> UB，**必须在** `T.Scope("V")` 内
   - `T.copy(c_ub, C[...])`: UB -> GM，**必须在** `T.Scope("V")` 内
   - 搬运到 Cube 的 L2/L0 需要在 Cube 作用域内

3. **流水线同步确保引擎间正确协作**
   - `T.set_flag("mte2", "v", 0)`: MTE2 完成，通知 V
   - `T.wait_flag("mte2", "v", 0)`: V 等待 MTE2
   - `T.set_flag("v", "mte3", 0)`: V 完成，通知 MTE3
   - `T.wait_flag("v", "mte3", 0)`: MTE3 等待 V

### CUDA vs NPU 关键差异

| 特性 | ❌ CUDA (禁用) | ✅ NPU (正确) |
|------|---------------|-------------|
| **Kernel启动** | `T.Kernel(..., threads=N)` | `T.Kernel(..., is_npu=True) as (cid, vid)` |
| **Block索引** | `(bx, by)` 直接2D | `(cid, vid)` 线性索引，手动计算 |
| **内存分配** | `alloc_shared`, `alloc_fragment` | `alloc_ub` (Unified Buffer) |
| **并行循环** | `T.Parallel(M, N)` | `T.serial` 或向量指令 |
| **向量计算** | 标量 `a + b` | `T.tile.add(c, a, b)` **必须在 V 作用域** |
| **数据搬运** | 自动 | `T.copy` **必须在 `T.Scope("V")` 内** |
| **同步** | 自动 | `T.set_flag`/`T.wait_flag` (流水线同步) |
| **作用域** | 无 | `with T.Scope("V"):` **V 核专用** |
| **向量化因子** | 无 | `VEC_NUM = 2` |

### NPU Kernel 基本模板

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def my_op(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2  # 向量化因子

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # NPU kernel: 线性索引 + is_npu=True
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 计算当前block的2D坐标
            bx = cid // n_num
            by = cid % n_num

            # NPU内存分配: alloc_ub
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # 数据搬运: GM -> UB + 向量计算 (V 作用域)
            with T.Scope("V"):
                # 数据搬运: GM -> UB (MTE2)
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # MTE2 完成，设置标志通知 V 可以开始计算
                T.set_flag("mte2", "v", 0)
                # V 等待 MTE2 加载完成
                T.wait_flag("mte2", "v", 0)

                # 计算: 使用向量指令
                T.tile.add(c_ub, a_ub, b_ub)

                # V 计算完成，设置标志通知 MTE3 可以开始存储
                T.set_flag("v", "mte3", 0)
                # MTE3 等待 V 计算完成
                T.wait_flag("v", "mte3", 0)

                # 数据搬运: UB -> GM (MTE3)
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### NPU Multi-Block Kernel 模板

当数据量大于可用核数时，每个核需要循环处理多个数据块：

```python
@tilelang.jit(out_idx=[-1])
def my_op_multiblock(N, block_N, num_cores=24, dtype="float16"):
    total_blocks = N // block_N
    blocks_per_core = (total_blocks + num_cores - 1) // num_cores  # 向上取整
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            b_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            c_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)

            for i in T.serial(blocks_per_core):
                block_idx = cid * blocks_per_core + i
                if block_idx < total_blocks:  # 边界检查
                    with T.Scope("V"):
                        T.copy(A[block_idx * block_N + vid * block_N // VEC_NUM], a_ub)
                        T.copy(B[block_idx * block_N + vid * block_N // VEC_NUM], b_ub)
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.tile.add(c_ub, a_ub, b_ub)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(c_ub, C[block_idx * block_N + vid * block_N // VEC_NUM])
                        T.barrier_all()  # ⚠️ 必须加！防止跨迭代flag冲突

    return main
```

**关键点**：
1. `blocks_per_core` 使用向上取整：`(total_blocks + num_cores - 1) // num_cores`
2. `if block_idx < total_blocks:` 边界检查跳过多余block
3. **`T.barrier_all()` 必须在每次迭代末尾调用**，否则 `set_flag/wait_flag` 的固定 `stage_id=0` 会在跨迭代时产生同步混乱，导致从位置0开始的全量数据错误

```
AutoGen/
├── skills/                          # Skill提示词目录
│   ├── tl-op-pipeline/              # 新增算子通路技能
│   │   └── SKILL.md
│   ├── tl-op-edit/                  # 修改算子技能
│   │   └── SKILL.md
│   ├── tl-op-rename/                # 重命名算子技能
│   │   └── SKILL.md
│   ├── tl-op-test/                  # 生成测试技能
│   │   └── SKILL.md
│   ├── tl-op-benchmark/             # 生成benchmark技能
│   │   └── SKILL.md
│   └── tl-op-hardware-constraints/  # NPU硬件约束技能
│       └── SKILL.md
├── templates/                       # 模板目录
│   └── op_frame/
│       ├── empty_op_template/       # 空白算子模板
│       │   ├── your_op_name.py      # 核心kernel实现
│       │   ├── test_your_op_name.py # 单元测试
│       │   ├── benchmark_your_op_name.py # 性能测试
│       │   └── README.md            # 算子文档
│       └── gemm_reference/          # GEMM参考实现
│           ├── gemm.py
│           ├── test_gemm.py
│           └── README.md
├── projects/                        # 生成的算子项目目录
│   └── vector_add/                  # Vector Add示例
├── CLAUDE.md                        # 本文件
└── README.md                        # 仓库说明
```

## 可用技能

### tl-op-pipeline
新增TileLang算子时使用。提供输入模板，并基于模板目录生成完整的算子实现。

### tl-op-hardware-constraints
检查算子代码是否符合不同NPU（如910B、910A、310P等）的硬件约束。

## NPU 向量指令

| 指令 | 说明 |
|------|------|
| `T.tile.add(C, A, B)` | 向量加法 C = A + B |
| `T.tile.mul(C, A, B)` | 向量乘法 C = A * B |
| `T.tile.sub(C, A, B)` | 向量减法 C = A - B |
| `T.tile.div(C, A, B)` | 向量除法 C = A / B |
| `T.gemm(A, B, C)` | 矩阵乘法 C += A @ B |

## NPU 内存操作

| 函数 | 说明 |
|------|------|
| `T.alloc_ub(shape, dtype)` | 分配 Unified Buffer |
| `T.fill(buffer, value)` | 填充buffer |
| `T.copy(src, dst)` | 数据拷贝 |

## NPU 同步原语

| 函数 | 说明 |
|------|------|
| `T.set_flag("mte2", "v", stage_id)` | MTE2完成，通知V可开始计算 |
| `T.wait_flag("mte2", "v", stage_id)` | V等待MTE2加载完成 |
| `T.set_flag("v", "mte3", stage_id)` | V完成，通知MTE3可开始存储 |
| `T.wait_flag("v", "mte3", stage_id)` | MTE3等待V计算完成 |
| `T.barrier_all()` | 所有核同步 (旧方式，不推荐) |
| `T.barrier(tid)` | 指定核同步 |

### 流水线同步模式 (推荐)

```
MTE2 (加载) --set_flag("mte2", "v", 0)--> V (计算) --set_flag("v", "mte3", 0)--> MTE3 (存储)
                    |                              |                              |
                    v                              v                              v
              wait_flag                     wait_flag                      wait_flag
```

使用 `set_flag/wait_flag` 可以实现更细粒度的流水线同步，相比 `barrier_all` 有更好的并行性。
- 第一个参数: 源引擎 ("mte2", "v", "mte3")
- 第二个参数: 目标引擎 ("mte2", "v", "mte3")
- 第三个参数: stage_id (用于多级流水线，简单场景用 0)

## ⚠️ 防止硬件地址越界（重要）

生成NPU算子时必须遵守以下硬件约束，否则会导致运行时错误：

### 存储单元对齐要求

| 存储单元 | 对齐要求 | 违反后果 |
|---------|---------|---------|
| **Unified Buffer** | 32字节 | 访问异常/性能下降 |
| **L1 Buffer** | 32字节 | 数据搬运失败 |
| **L0A/L0B Buffer** | 512字节 | 硬件执行异常 |
| **L0C Buffer** | 64字节 | 计算结果错误 |

### 数据搬运约束

- 搬运到UB的数据必须按 **DataBlock (32字节)** 对齐
- L1→L0A/L0B必须按 **分形大小** 对齐，剩余空间不足1个分形会导致硬件异常
- block_M/block_N/block_K 必须是 **分形大小(16或32)** 的整数倍

### 同步控制约束

- `T.set_flag` 和 `T.wait_flag` **必须成对使用**
- 参数必须**完全一致**（源引擎、目标引擎、stage_id）
- **禁止连续设置**同一个EventID

### Buffer容量约束

- UB总使用量 ≤ 2MB (910B)
- L1总使用量 ≤ 1MB (910B)
- 流水线深度 × 单stageBuffer ≤ Buffer容量

**详细约束请参考**: `skills/tl-op-hardware-constraints/SKILL.md`

## 开发流程

1. 使用 `tl-op-pipeline` 技能生成算子框架
2. **使用 `tl-op-hardware-constraints` 检查硬件约束** ⬅️ 新增
3. 根据需求修改算子实现 (遵循NPU编程规范)
4. 使用 `tl-op-test` 生成测试用例
5. 使用 `tl-op-benchmark` 进行性能测试
6. 如需修改，使用 `tl-op-edit` 调整参数

## 参考资料

- [TileLang-Ascend GitHub](https://github.com/tile-ai/tilelang-ascend)
- [TileLang Documentation](https://tilelang.com/)
- [TileLang-Ascend 开发指南](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md)
