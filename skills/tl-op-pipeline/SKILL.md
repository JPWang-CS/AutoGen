---
name: tl-op-pipeline
description: 用户新增TileLang算子时，提供输入模板到 claude.local.md，并基于模板目录生成算子完整通路
---

# 新增TileLang算子通路技能

当用户提出"新增一个TileLang算子/生成TileLang算子通路/按模板生成TileLang算子"时使用本技能。

## 工作流
1. 确认需要生成的算子类型（例如：GEMM、FlashAttention、DequantizeGEMM等）。
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

## TileLang核心概念

### 基本结构
TileLang使用Python装饰器 `@tilelang.jit` 来定义可JIT编译的kernel：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])  # out_idx指定输出tensor的索引
def my_kernel(M, N, K, block_M, block_N, block_K, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # kernel实现
        pass
    return main
```

### 内存层次
- `T.alloc_shared`: 分配shared memory
- `T.alloc_fragment`: 分配寄存器fragment
- `T.alloc_local`: 分配local memory

### 核心原语
- `T.copy`: 数据拷贝
- `T.gemm`: 矩阵乘法
- `T.clear`: 清空buffer
- `T.fill`: 填充buffer
- `T.reduce_max`, `T.reduce_sum`: 归约操作
- `T.Parallel`: 并行循环
- `T.Pipelined`: 流水线循环（支持software pipelining）

### Kernel启动
```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # bx, by 是block索引
    # threads 指定每个block的线程数
```

### 数据类型
- `T.float16`, `T.float32`, `T.bfloat16`
- `T.int8`, `T.int32`
- `T.float8_e4m3fn`, `T.float8_e5m2`

## 性能优化选项

### 流水线
使用 `T.Pipelined` 实现software pipelining，`num_stages` 控制流水线深度：

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    # 加载和计算可以重叠
```

### L2 Cache优化
使用swizzle优化L2 cache局部性：

```python
T.use_swizzle(panel_size=10, enable=True)
```

### 自动调优
使用 `@autotune` 装饰器自动搜索最优配置：

```python
from tilelang.autotuner import *

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[-1])
def my_kernel(...):
    ...
```

## 约束
- 默认最小改动，且不引入新依赖。
- 未经用户明确要求，不改变模板结构。
- 若输入信息不足，必须先列出假设。
- 严格按照输入文档/模板中列出的参数生成代码，不得自行添加或删减任何参数。
- 生成的代码需要遵循TileLang的编程规范和最佳实践。

## 参考
- 模板目录：`templates/op_frame/empty_op_template`
- 参考实现：`templates/op_frame/gemm_reference`
- TileLang官方示例：`https://github.com/tile-ai/tilelang/tree/main/examples`
- TileLang文档：`https://tilelang.com/`
