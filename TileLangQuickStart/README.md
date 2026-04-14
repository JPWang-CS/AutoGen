# TileLang-Ascend 快速入门教程（中文版）

> 本教程基于 [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend) 编写，旨在帮助中文用户快速掌握 TileLang 在华为昇腾NPU上的核心概念与使用方法。

## 什么是 TileLang-Ascend？

TileLang-Ascend 是 TileLang 面向华为昇腾NPU的版本，是一个嵌入在 Python 中的领域特定语言（DSL），用于编写高性能NPU算子内核。它构建在 TVM 之上，提供了一种基于 **Tile** 的编程模型，让开发者可以显式控制内存分配、数据搬运、布局和并行执行。

### 核心特性

- **Python 嵌入式 DSL**：使用 `@T.prim_func` 装饰器在 Python 中定义内核函数
- **昇腾NPU内存层次**：Global Memory -> L1/UB -> L0A/L0B/L0C，精确控制数据搬运
- **双编程模式**：Developer模式（自动优化）和Expert模式（完全控制）
- **双计算引擎**：Cube Core（矩阵乘法）和 Vector Core（向量计算），可独立或融合使用
- **高性能算子原语**：内置 `T.gemm_v0`、`T.mma`、`T.tile.*` 等高效计算原语
- **软件流水线**：通过 `T.Pipelined` 和 `T.Persistent` 实现计算搬运重叠和缓存优化
- **JIT 编译**：使用 `@tilelang.jit` 装饰器即时编译并执行内核，生成AscendC代码

### 昇腾NPU架构

```
AI Core 结构:
+-----------+     +-----------+     +-----------+
|   L0A     |     |   L0B     |     |   L0C     |
| (左矩阵)  | --> | (右矩阵)  | --> | (输出)    |    Cube Core
+-----------+     +-----------+     +-----------+
       ^                ^                  |
       |                |                  v
+-----------------------------------------------+
|              L1 Buffer (Cube缓存)              |
+-----------------------------------------------+
       ^                                  |
       |                                  v
+-----------------------------------------------+
|           Unified Buffer (Vector缓冲)          |    Vector Core
|           T.tile.add/mul/exp/...              |
+-----------------------------------------------+
       ^                                  |
       |                                  v
+-----------------------------------------------+
|              Global Memory (HBM)               |
+-----------------------------------------------+
```

## 教程目录

| 序号 | 章节 | 说明 |
|------|------|------|
| 01 | [安装与环境配置](01_安装与环境配置.md) | 安装 TileLang-Ascend、配置昇腾环境、验证安装 |
| 02 | [语言基础](02_语言基础.md) | 编程模型、Kernel定义、内存层次、Tensor声明、作用域 |
| 03 | [核心指令详解](03_核心指令详解.md) | 内存分配、T.copy、T.gemm_v0、T.mma、T.tile.*、同步指令 |
| 04 | [控制流](04_控制流.md) | T.serial、T.Parallel、T.Pipelined、T.Persistent、T.ceildiv |
| 05 | [类型系统](05_类型系统.md) | dtype字符串、累加器类型、T.tile.cast、T.DataType |
| 06 | [自动调优](06_自动调优.md) | pass_configs、TL_ASCEND_AUTO_SYNC、TL_ASCEND_MEMORY_PLANNING、TL_ASCEND_AUTO_CV_COMBINE |
| 07 | [实战：GEMM 矩阵乘法](07_实战示例_GEMM.md) | 从零编写NPU上的高性能GEMM内核（Expert/Developer/Persistent/Pipelined模式） |
| 08 | [实战：注意力机制](08_实战示例_注意力机制.md) | Flash Attention风格：Cube+Vector融合算子实现 |
| 09 | [高级特性与调试](09_高级特性与调试.md) | T.alloc_var、完整T.tile.*指令、调试工具、PassConfigKey、性能调优 |

## 快速体验

```python
import tilelang
import tilelang.language as T
import torch

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

# 运行
func = vec_add(1024, 1024, 128, 256)
a = torch.randn(1024, 1024).npu()
b = torch.randn(1024, 1024).npu()
c = func(a, b)
torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
print("Kernel Output Match!")
```

## 参考资源

- [TileLang-Ascend GitHub 仓库](https://github.com/tile-ai/tilelang-ascend)
- [TileLang-Ascend 编程指南](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/TileLang-Ascend%20Programming%20Guide.md)
- [TileLang-Ascend 示例代码](https://github.com/tile-ai/tilelang-ascend/tree/npuir/examples)
- [昇腾CANN文档](https://www.hiascend.com/document)
