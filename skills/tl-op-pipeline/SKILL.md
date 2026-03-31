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

## 工作流
1. 确认需要生成的算子类型（例如：GEMM、FlashAttention、VectorAdd等）。
2. 确认目标子项目路径（例如 `projects/gemm/...` 或 `projects/attention/...`）。
3. 仅在触发本技能时才生成 `claude.local.md`。
4. 若 `claude.local.md` 不存在，则创建并写入"新增算子输入模板"。
5. 引导用户填写模板中的关键字段。
6. 根据模板输入，使用 `templates/op_frame/empty_op_template` 生成算子完整通路：
   - 复制模板目录到目标路径。
   - 全局替换占位符 `your_op_name` 为真实算子名（同时处理文件名与内容）。
   - 按输入信息完善核心kernel代码与示例。
   - 参考 `templates/op_frame/gemm_reference` 完善代码细节。
7. 检查生成目录及文件的完整性。
8. 输出改动说明与假设。

## 生成范围（默认）
- 使用 `templates/op_frame/empty_op_template` 作为唯一来源。
- 主要生成文件：
  - `your_op_name.py` - 核心kernel实现
  - `test_your_op_name.py` - 单元测试文件
  - `benchmark_your_op_name.py` - 性能基准测试文件
  - `README.md` - 算子说明文档

## ⚠️ NPU 编程规范 (重要)

### CUDA vs NPU 对比

| 特性 | CUDA (错误) | NPU (正确) |
|------|------------|-----------|
| **Kernel启动** | `T.Kernel(..., threads=N)` | `T.Kernel(..., is_npu=True) as (cid, vid)` |
| **Block索引** | `(bx, by)` 直接2D | `(cid, vid)` 线性索引，手动计算 |
| **内存分配** | `T.alloc_shared`, `T.alloc_fragment` | `T.alloc_ub` (Unified Buffer) |
| **并行循环** | `T.Parallel(M, N)` | `T.serial` 或向量指令 |
| **向量计算** | 标量 `a + b` | `T.tile.add(c, a, b)` |
| **矩阵计算** | 手动循环 | `T.gemm(A, B, C)` |
| **同步** | 自动 | `T.barrier_all()` |
| **作用域** | 无 | `with T.Scope("V"):` |

### NPU Kernel 基本结构

```python
@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2  # 向量化因子

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # NPU kernel: 使用线性索引，is_npu=True
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 计算当前block的2D坐标
            bx = cid // n_num
            by = cid % n_num

            # NPU内存分配: 使用 alloc_ub 分配 Unified Buffer
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # Vector Core 作用域
            with T.Scope("V"):
                # 数据搬运: GM -> UB
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # 核间同步
                T.barrier_all()

                # 计算: 使用向量指令
                T.tile.add(c_ub, a_ub, b_ub)

                # 核间同步
                T.barrier_all()

                # 数据搬运: UB -> GM
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

### NPU 向量指令

| 指令 | 说明 |
|------|------|
| `T.tile.add(C, A, B)` | 向量加法 C = A + B |
| `T.tile.mul(C, A, B)` | 向量乘法 C = A * B |
| `T.tile.sub(C, A, B)` | 向量减法 C = A - B |
| `T.tile.div(C, A, B)` | 向量除法 C = A / B |
| `T.gemm(A, B, C)` | 矩阵乘法 C += A @ B |

### 内存层次 (NPU)

| 函数 | 说明 |
|------|------|
| `T.alloc_ub(shape, dtype)` | 分配 Unified Buffer |
| `T.fill(buffer, value)` | 填充buffer |

### 同步原语

| 函数 | 说明 |
|------|------|
| `T.barrier_all()` | 所有核同步 |
| `T.barrier(tid)` | 指定核同步 |

### 数据类型

dtype参数使用字符串类型：
- `"float16"`, `"float32"`, `"bfloat16"`
- `"int8"`, `"int32"`
- `"float8_e4m3fn"`, `"float8_e5m2"`

## NPU专用示例

### Vector Add (NPU版本)

```python
@tilelang.jit(out_idx=[-1])
def vector_add_2d(M, N, block_M, block_N, dtype="float16"):
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

### GEMM (NPU版本)

```python
@tilelang.jit(out_idx=[-1])
def gemm(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub((block_M // VEC_NUM, block_K), dtype)
            b_ub = T.alloc_ub((block_K, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), accum_dtype)

            T.fill(c_ub, 0)

            for k in T.serial(K // block_K):
                with T.Scope("V"):
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, k * block_K], a_ub)
                    T.copy(B[k * block_K, by * block_N], b_ub)

                    T.barrier_all()
                    T.gemm(a_ub, b_ub, c_ub)
                    T.barrier_all()

            with T.Scope("V"):
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main
```

## 设备检测与使用

生成的代码使用NPU专用初始化：

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

## NPU 硬件约束

> **重要**: 生成的算子代码必须符合目标 SOC 的硬件约束。详细规格请参考 `skills/tl-op-hardware-constraints/Skill.md`。

### 支持的 SOC 版本

| SOC 版本 | 代号 | 定位 | 推荐配置 |
|---------|------|------|---------|
| **Ascend 910B** | DAV_3510 | 训练（主流） | block_M=32, block_N=32, block_K=32 |
| **Ascend 910A** | DAV_2201 | 训练 | block_M=32, block_N=32, block_K=32 |
| **Ascend 310P** | DAV_310P | 推理 | block_M=16, block_N=16, block_K=16 |

### 关键硬件限制
1. **UB 容量限制**: Tile 分块大小不能超过 UB 容量（约 1-2MB）
2. **数据对齐要求**: 32字节对齐（cacheline）
3. **向量化因子**: 通常 VEC_NUM=2

### 硬件约束检查清单
在生成算子时，必须验证：
- [ ] Tile 大小是否在 UB 容量限制内
- [ ] 分块大小是否满足对齐要求
- [ ] 是否使用了正确的 NPU 语法 (alloc_ub, T.tile.*, barrier_all)

## 约束
- 默认最小改动，且不引入新依赖。
- 未经用户明确要求，不改变模板结构。
- 若输入信息不足，必须先列出假设。
- 严格按照输入文档/模板中列出的参数生成代码，不得自行添加或删减任何参数。
- 生成的代码需要遵循TileLang-Ascend的编程规范和最佳实践。
- **必须使用 NPU 专用语法**: `is_npu=True`, `alloc_ub`, `T.tile.*`, `barrier_all`, `T.Scope("V")`
- **禁止使用 CUDA 语法**: `T.Parallel`, `alloc_shared`, `alloc_fragment`, `threads` 参数
- **生成的代码必须符合目标SOC的硬件上限**，参见 `skills/tl-op-hardware-constraints/Skill.md`。

## 参考
- 模板目录：`templates/op_frame/empty_op_template`
- 参考实现：`templates/op_frame/gemm_reference`
- **硬件约束参考**: `skills/tl-op-hardware-constraints/Skill.md`
- TileLang-Ascend GitHub：`https://github.com/tile-ai/tilelang-ascend`
- TileLang-Ascend 开发指南：`https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md`
