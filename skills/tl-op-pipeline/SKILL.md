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
1. 确认需要生成的算子类型（例如：GEMM、FlashAttention、DequantizeGEMM、VectorAdd等）。
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

## TileLang-Ascend核心概念

### 两种编译后端

TileLang-Ascend 支持两种编译后端：
1. **AscendNPU IR (`target="npuir"`)** - 推荐用于大多数场景，Developer Mode
2. **Ascend C & PTO (`target="ascendc"`)** - 用于底层优化，Expert Mode

### 开发模式

- **Developer Mode (默认)**: `TILELANG_ASCEND_MODE=developer` 或不设置
- **Expert Mode**: `TILELANG_ASCEND_MODE=expert`

### 基本结构

TileLang-Ascend使用Python装饰器 `@tilelang.jit` 来定义可JIT编译的kernel：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1], target="npuir")  # target指定为npuir
def my_kernel(M, N, K, block_M, block_N, block_K, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # kernel实现
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            # 使用is_npu=True标识NPU kernel
            pass
    return main
```

### 内存层次 (NPU专用)

TileLang-Ascend 使用以下NPU内存分配原语：

- `T.alloc_ub`: 分配Unified Buffer (UB)，NPU片上高速缓存
- `T.alloc_L1`: 分配L1 buffer
- `T.alloc_L0A`, `T.alloc_L0B`, `T.alloc_L0C`: 分配L0 buffer (用于Cube单元)
- `T.alloc_fragment`: 分配寄存器fragment

### 核心原语

- `T.copy`: 数据拷贝
- `T.gemm`: 矩阵乘法 (NPU Developer Mode)
- `T.gemm_v0`: NPU专用矩阵乘法，支持init参数
- `T.mma`: NPU矩阵乘累加
- `T.clear`: 清空buffer
- `T.fill`: 填充buffer
- `T.reduce_max`, `T.reduce_sum`: 归约操作
- `T.Parallel`: 并行循环
- `T.Pipelined`: 流水线循环（支持Cube/Vector core流水线）
- `T.npuir_add`, `T.npuir_mul`, `T.npuir_sub`: NPU专用算术操作

### Kernel启动 (NPU模式)

```python
# NPU kernel启动模式
with T.Kernel(n_num, is_npu=True) as (cid, _):
    # cid 是核的索引
    # 第二个参数为保留参数，通常用_表示

# 对于2D分块，使用线性索引计算
with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
    by = cid // T.ceildiv(N, block_N)
    bx = cid % T.ceildiv(N, block_N)
```

### 数据类型

dtype参数使用字符串类型：
- `"float16"`, `"float32"`, `"bfloat16"`
- `"int8"`, `"int32"`
- `"float8_e4m3fn"`, `"float8_e5m2"`

## NPU专用示例

### Vector Add (NPU版本)

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1], target="npuir")
def vec_add(N, block_N, dtype="float32"):
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # 分配Unified Buffer
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

            start_idx = cid * block_N
            remaining = shape - start_idx
            tail_size = T.min(block_N, remaining)

            # 拷贝数据到UB
            T.copy(A[start_idx : start_idx + tail_size], A_VEC[0:tail_size])
            T.copy(B[start_idx : start_idx + tail_size], B_VEC[0:tail_size])

            # NPU专用加法
            T.npuir_add(A_VEC, B_VEC, C_VEC)

            # 拷贝结果回global memory
            T.copy(C_VEC[0:tail_size], C[start_idx : start_idx + tail_size])

    return main
```

### GEMM (NPU Developer Mode)

```python
@tilelang.jit(out_idx=[-1], target="npuir")
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            # 使用Unified Buffer (NPU专用)
            A_ub = T.alloc_ub((block_M, block_K), dtype)
            B_ub = T.alloc_ub((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_ub)
                T.copy(B[k * block_K, bx * block_N], B_ub)
                # 使用initC参数初始化累加器
                T.gemm(A_ub, B_ub, C_local, initC=(k == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm
```

## 性能优化选项

### 流水线
使用 `T.Pipelined` 实现Cube/Vector core流水线，`num_stages` 控制流水线深度：

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    # 加载和计算可以重叠
```

### 自动同步
启用自动同步功能：

```python
import tilelang
pass_configs = {tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}
compiled_kernel = tilelang.compile(func, target="npuir", pass_configs=pass_configs)
```

### 手动同步
NPU专用同步原语：
- `T.barrier_all()`: 全核同步
- `T.set_flag`: 设置同步标志
- `T.wait_flag`: 等待同步标志

### 自动调优
使用 `@autotune` 装饰器自动搜索最优配置：

```python
from tilelang.autotuner import *

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[-1], target="npuir")
def my_kernel(...):
    ...
```

## 设备检测

生成的代码使用NPU专用初始化：

```python
import torch
import torch_npu

# 设置NPU设备
torch.npu.set_device(0)
```

## NPU使用说明

### 安装依赖

```bash
pip install torch-npu
pip install git+https://github.com/tile-ai/tilelang-ascend
```

### 编译和运行

```python
import torch
import torch_npu
import tilelang

# 设置NPU设备
torch.npu.set_device(0)

# 编译kernel
compiled_kernel = tilelang.compile(func, target="npuir")

# 准备数据 (NPU)
a = torch.randn(M, K, device="npu", dtype=torch.float16)
b = torch.randn(K, N, device="npu", dtype=torch.float16)

# 调用kernel
c = compiled_kernel(a, b)

# 精度对比在CPU上进行
ref_c = a.cpu() @ b.cpu()
torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
```

### 数据准备

```python
# NPU tensor创建
torch.npu.set_device(0)
a = torch.randn(M, K, device="npu", dtype=torch.float16)
b = torch.randn(K, N, device="npu", dtype=torch.float16)
```

## NPU 硬件约束

> **重要**: 生成的算子代码必须符合目标 SOC 的硬件约束。详细规格请参考 `skills/tl-op-hardware-constraints/Skill.md`。

### 支持的 SOC 版本

| SOC 版本 | 代号 | 定位 | 推荐配置 |
|---------|------|------|---------|
| **Ascend 910B** | DAV_3510 | 训练（主流） | block_M=128, block_N=128, block_K=64 |
| **Ascend 910A** | DAV_2201 | 训练 | block_M=64, block_N=128, block_K=64 |
| **Ascend 310P** | DAV_310P | 推理 | block_M=64, block_N=64, block_K=32 |

### 关键硬件限制
1. **UB 容量限制**: Tile 分块大小不能超过 UB 容量（约 1-2MB）
2. **L1/L0 Buffer 限制**: 矩阵分块需满足分形大小对齐（512B 对齐）
3. **数据对齐要求**:
   - GM → UB: 32 字节对齐（cacheline）
   - L1 → L0A/L0B: 512 字节对齐（分形大小）
4. **Cube/Vector 资源**: 避免同一 Bank Group 冲突
5. **流水线深度**: 根据 Buffer 占用合理设置 `num_stages`

### 硬件约束检查清单
在生成算子时，必须验证：
- [ ] Tile 大小是否在 UB 容量限制内
- [ ] 分块大小是否满足对齐要求
- [ ] Cube/Vector 操作是否使用不同 Bank Group
- [ ] 流水线深度是否合理

## 约束
- 默认最小改动，且不引入新依赖。
- 未经用户明确要求，不改变模板结构。
- 若输入信息不足，必须先列出假设。
- 严格按照输入文档/模板中列出的参数生成代码，不得自行添加或删减任何参数。
- 生成的代码需要遵循TileLang-Ascend的编程规范和最佳实践。
- 默认使用 `target="npuir"` 和 `is_npu=True` 以确保NPU兼容性。
- **生成的代码必须符合目标SOC的硬件上限**，参见 `skills/tl-op-hardware-constraints/Skill.md`。

- **算子搬运和计算必须符合不同NPU（如910B等）的硬件限制**：
  - UB容量限制： Tile大小不能超过UB容量
  - L1/L0 Buffer限制: 需满足分形大小对齐要求
  - 数据对齐: 32字节对齐（cacheline）/ 512字节对齐（分形）
  - Cube/Vector资源: 避免同一Bank Group冲突
  - 流水线深度: 根据Buffer占用合理设置num_stages

## 参考
- 模板目录：`templates/op_frame/empty_op_template`
- 参考实现：`templates/op_frame/gemm_reference`
- **硬件约束参考**: `skills/tl-op-hardware-constraints/Skill.md`
- TileLang-Ascend GitHub：`https://github.com/tile-ai/tilelang-ascend`
- TileLang-Ascend 开发指南：`https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md`
- TileLang官方示例：`https://github.com/tile-ai/tilelang/tree/main/examples`
- TileLang文档：`https://tilelang.com/`
