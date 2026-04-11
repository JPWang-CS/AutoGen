---
name: tl-op-pipeline
description: 用户新增TileLang-Ascend算子时，提供输入模板到 claude.local.md，并基于模板目录生成算子完整通路
---

# 新增TileLang-Ascend算子通路技能

当用户提出"新增一个TileLang算子/生成TileLang算子通路/按模板生成TileLang算子"时使用本技能。

## 支持设备

TileLang-Ascend算子仅支持NPU设备：
- **NPU** - 华为昇腾NPU (需要安装 torch_npu 和 tilelang-ascend)

所有计算在NPU上执行，精度对比在CPU上进行。

## ⚠️ 生成前必读：硬件约束

生成算子前**必须**参考 `skills/tl-op-hardware-constraints/SKILL.md` 中的硬件约束，确保：

1. **存储对齐**：UB 32B、L0A/L0B 512B、L0C 64B
2. **分形约束**：block_M/N/K 必须是分形大小(16或32)的整数倍
3. **Buffer容量**：UB总使用量 ≤ 2MB，L1 ≤ 1MB
4. **同步配对**：`set_flag` 和 `wait_flag` 必须成对且参数一致

**违反上述约束会导致硬件地址越界或运行时错误！**

## 工作流
1. 确认需要生成的算子类型（例如：GEMM、FlashAttention、VectorAdd等）。
2. 确认目标子项目路径（例如 `projects/gemm/...` 或 `projects/attention/...`）。
3. 仅在触发本技能时才生成 `claude.local.md`。
4. 若 `claude.local.md` 不存在，则创建并写入"新增算子输入模板"。
5. 引导用户填写模板中的关键字段。
6. **检查硬件约束**（参考 `tl-op-hardware-constraints`）。
7. 根据模板输入，使用 `templates/op_frame/empty_op_template` 生成算子完整通路：
   - 复制模板目录到目标路径。
   - 全局替换占位符 `your_op_name` 为真实算子名（同时处理文件名与内容）。
   - 按输入信息完善核心kernel代码与示例。
   - 参考 `templates/op_frame/gemm_reference` 完善代码细节。
   - **确保同步使用 `set_flag/wait_flag` 而非 `barrier_all`**。
8. 检查生成目录及文件的完整性。
9. 输出改动说明与假设。

## ⚠️ NPU 架构约束 (重要)

### NPU 执行引擎

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

## CUDA vs NPU 关键差异

| 特性 | ❌ CUDA (禁用) | ✅ NPU (正确) |
|------|---------------|-------------|
| **Kernel启动** | `T.Kernel(..., threads=N)` | `T.Kernel(..., is_npu=True) as (cid, vid)` |
| **Block索引** | `(bx, by)` 直接2D | `(cid, vid)` 线性索引，手动计算 |
| **内存分配** | `alloc_shared`, `alloc_fragment` | `alloc_ub` (Unified Buffer) |
| **并行循环** | `T.Parallel(M, N)` | `T.serial` 或向量指令 |
| **向量计算** | 标量 `a + b` | `T.tile.add(c, a, b)` **必须在 V 作用域** |
| **数据搬运** | 自动 | `T.copy` **必须在 `T.Scope("V")` 内** |
| **流水线同步** | 自动 | `T.set_flag` / `T.wait_flag` (推荐) |
| **作用域** | 无 | `with T.Scope("V"):` **V 核专用** |
| **向量化因子** | 无 | `VEC_NUM = 2` |

### 流水线同步模式 (推荐)

使用 `set_flag/wait_flag` 实现细粒度流水线同步：

```
MTE2 (加载) --set_flag("mte2", "v", 0)--> V (计算) --set_flag("v", "mte3", 0)--> MTE3 (存储)
                    |                              |                              |
                    v                              v                              v
              wait_flag                     wait_flag                      wait_flag
```

**同步规则（来自官方文档）：**
- `set_flag` 和 `wait_flag` **必须成对使用**
- 参数必须**完全一致**（源引擎、目标引擎、stage_id）
- **禁止连续设置**同一个 stage_id

### NPU Kernel 通用模板

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def my_op(shape_params, block_params, dtype="float16"):
    """
    NPU编程要点:
    - 使用线性block索引 (cid, vid)
    - 使用 alloc_ub 分配 Unified Buffer
    - 使用 T.tile.* 进行向量计算
    - 使用 set_flag/wait_flag 进行流水线同步（推荐）
    - 使用 T.Scope("V") 指定 Vector Core 作用域
    """
    VEC_NUM = 2  # 向量化因子

    @T.prim_func
    def main(
        A: T.Tensor((shape), dtype),
        B: T.Tensor((shape), dtype),
        C: T.Tensor((shape), dtype),
    ):
        # NPU kernel: 线性索引 + is_npu=True
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            # NPU内存分配: alloc_ub
            a_ub = T.alloc_ub((block_shape // VEC_NUM), dtype)
            b_ub = T.alloc_ub((block_shape // VEC_NUM), dtype)
            c_ub = T.alloc_ub((block_shape // VEC_NUM), dtype)

            # 数据搬运: GM -> UB + 向量计算 (V 作用域)
            with T.Scope("V"):
                # 数据搬运: GM -> UB (MTE2)
                T.copy(A[...], a_ub)
                T.copy(B[...], b_ub)

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
                T.copy(c_ub, C[...])

    return main
```

## NPU 示例: 1D/2D/3D Vector Add

### 1D Vector Add

```python
@tilelang.jit(out_idx=[-1])
def vector_add_1d(N, block_N, dtype="float16"):
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            b_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            c_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)

            # 数据搬运: GM -> UB + 向量计算 (V 作用域)
            with T.Scope("V"):
                # 数据搬运: GM -> UB (MTE2)
                T.copy(A[cid * block_N + vid * block_N // VEC_NUM], a_ub)
                T.copy(B[cid * block_N + vid * block_N // VEC_NUM], b_ub)

                # MTE2 完成，通知 V 可以开始计算
                T.set_flag("mte2", "v", 0)
                # V 等待 MTE2 加载完成
                T.wait_flag("mte2", "v", 0)

                # 向量计算
                T.tile.add(c_ub, a_ub, b_ub)

                # V 计算完成，通知 MTE3 可以开始存储
                T.set_flag("v", "mte3", 0)
                # MTE3 等待 V 计算完成
                T.wait_flag("v", "mte3", 0)

                # 数据搬运: UB -> GM (MTE3)
                T.copy(c_ub, C[cid * block_N + vid * block_N // VEC_NUM])

    return main
```

### 2D Vector Add

```python
@tilelang.jit(out_idx=[-1])
def vector_add_2d(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 计算当前block的2D坐标
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # 数据搬运: GM -> UB + 向量计算 (V 作用域)
            with T.Scope("V"):
                # 数据搬运: GM -> UB (MTE2)
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # MTE2 完成，通知 V 可以开始计算
                T.set_flag("mte2", "v", 0)
                # V 等待 MTE2 加载完成
                T.wait_flag("mte2", "v", 0)

                # 向量计算
                T.tile.add(c_ub, a_ub, b_ub)

                # V 计算完成，通知 MTE3 可以开始存储
                T.set_flag("v", "mte3", 0)
                # MTE3 等待 V 计算完成
                T.wait_flag("v", "mte3", 0)

                # 数据搬运: UB -> GM (MTE3)
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### 3D Vector Add

```python
@tilelang.jit(out_idx=[-1])
def vector_add_3d(D, M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(A: T.Tensor((D, M, N), dtype), B: T.Tensor((D, M, N), dtype), C: T.Tensor((D, M, N), dtype)):
        with T.Kernel(D * m_num * n_num, is_npu=True) as (cid, vid):
            # 计算当前block的3D坐标
            bz = cid // (m_num * n_num)
            remaining = cid % (m_num * n_num)
            bx = remaining // n_num
            by = remaining % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # 数据搬运: GM -> UB + 向量计算 (V 作用域)
            with T.Scope("V"):
                # 数据搬运: GM -> UB (MTE2)
                T.copy(A[bz, bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bz, bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # MTE2 完成，通知 V 可以开始计算
                T.set_flag("mte2", "v", 0)
                # V 等待 MTE2 加载完成
                T.wait_flag("mte2", "v", 0)

                # 向量计算
                T.tile.add(c_ub, a_ub, b_ub)

                # V 计算完成，通知 MTE3 可以开始存储
                T.set_flag("v", "mte3", 0)
                # MTE3 等待 V 计算完成
                T.wait_flag("v", "mte3", 0)

                # 数据搬运: UB -> GM (MTE3)
                T.copy(c_ub, C[bz, bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

## 1D/2D/3D 坐标计算对照

| 维度 | Block数量 | 坐标计算 |
|------|----------|---------|
| **1D** | `n_num` | 直接使用 `cid` |
| **2D** | `m_num * n_num` | `bx = cid // n_num`<br>`by = cid % n_num` |
| **3D** | `d_num * m_num * n_num` | `bz = cid // (m_num * n_num)`<br>`bx = remaining // n_num`<br>`by = remaining % n_num` |

## NPU 向量指令

| 指令 | 说明 |
|------|------|
| `T.tile.add(C, A, B)` | 向量加法 C = A + B |
| `T.tile.mul(C, A, B)` | 向量乘法 C = A * B |
| `T.tile.sub(C, A, B)` | 向量减法 C = A - B |
| `T.tile.div(C, A, B)` | 向量除法 C = A / B |
| `T.gemm(A, B, C)` | 矩阵乘法 C += A @ B |

## 数据类型

dtype参数使用字符串类型：
- `"float16"`, `"float32"`, `"bfloat16"`
- `"int8"`, `"int32"`

## 设备使用

```python
import torch
import torch_npu

# 设置NPU设备
torch.npu.set_device(0)

# 准备数据 (NPU)
a = torch.randn(M, N, device="npu", dtype=torch.float16)
b = torch.randn(M, N, device="npu", dtype=torch.float16)

# 调用kernel
c = kernel(a, b)

# 精度对比在CPU上进行
ref_c = a.cpu() + b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
```

## 约束
- 默认最小改动，且不引入新依赖。
- **必须使用 NPU 专用语法**: `is_npu=True`, `alloc_ub`, `T.tile.*`, `T.Scope("V")`
- **推荐使用流水线同步**: `T.set_flag` / `T.wait_flag` (细粒度流水线)
- **禁止使用 CUDA 语法**: `T.Parallel`, `alloc_shared`, `alloc_fragment`, `threads` 参数
- **VEC_NUM = 2**: 每个block由2个vector core并行处理
- **⚠️ 硬件约束**: 生成前必须检查 `tl-op-hardware-constraints` 中的约束规则

## 防止地址越界检查清单

生成算子时必须验证：

- [ ] **UB对齐**: block_size * dtype_size 是32字节的倍数
- [ ] **分形对齐**: block_M/N/K 是16（FP16/BF16）或32（INT8）的倍数
- [ ] **Buffer容量**: 总UB使用量 ≤ 2MB
- [ ] **同步配对**: 每个 `set_flag` 都有对应的 `wait_flag`，且参数一致
- [ ] **地址计算**: block索引计算不会导致GM地址越界

## 参考
- 模板目录：`templates/op_frame/empty_op_template`
- 参考实现：`projects/vector_add/vector_add.py`
- 硬件约束：`skills/tl-op-hardware-constraints/SKILL.md`
- TileLang-Ascend GitHub：`https://github.com/tile-ai/tilelang-ascend`
