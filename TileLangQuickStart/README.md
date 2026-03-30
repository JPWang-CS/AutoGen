# TileLang 快速入门教程（中文版）

> 本教程基于 [TileLang 官方文档](https://tilelang.tile-ai.cn/) 编写，旨在帮助中文用户快速掌握 TileLang 的核心概念与使用方法。

## 什么是 TileLang？

TileLang 是一个嵌入在 Python 中的领域特定语言（DSL），用于编写高性能 GPU/CPU 算子内核。它构建在 TVM 之上，提供了一种基于 **Tile** 的编程模型，让开发者可以专注于算法逻辑，而无需直接处理底层硬件细节。

### 核心特性

- **Python 嵌入式 DSL**：使用 `@T.prim_func` 装饰器在 Python 中定义内核函数
- **三级内存层次**：Global Memory → Shared Memory → Fragment/Register，自动管理数据搬运
- **高性能算子原语**：内置 `T.copy`、`T.gemm`、`T.gemm_sp` 等高效计算原语
- **软件流水线**：通过 `T.Pipelined` 实现自动软件流水线，隐藏内存延迟
- **JIT 编译**：使用 `@tilelang.jit` 装饰器即时编译并执行内核
- **自动调优**：内置 `@tilelang.autotune` 支持参数空间搜索
- **多后端支持**：支持 CUDA、HIP（AMD）、Metal 等后端

## 教程目录

| 序号 | 章节 | 说明 |
|------|------|------|
| 01 | [安装与环境配置](01_安装与环境配置.md) | 安装 TileLang、验证环境 |
| 02 | [语言基础](02_语言基础.md) | 编程模型、Kernel 定义、内存层次、Tensor 声明 |
| 03 | [核心指令详解](03_核心指令详解.md) | T.copy、T.gemm、T.alloc_shared/fragment、T.clear、T.fill 等 |
| 04 | [控制流](04_控制流.md) | T.serial、T.unroll、T.Parallel、T.Pipelined、T.Persistent |
| 05 | [类型系统](05_类型系统.md) | 标量类型、向量类型、float8/6/4 等低精度类型 |
| 06 | [自动调优](06_自动调优.md) | 装饰器模式与编程模式的自动调优 |
| 07 | [实战：GEMM 矩阵乘法](07_实战示例_GEMM.md) | 从零编写一个高性能 GEMM 内核 |
| 08 | [实战：注意力机制](08_实战示例_注意力机制.md) | Flash Attention 风格的注意力内核实现 |

## 快速体验

```python
import tilelang
import tilelang.language as T

# 定义一个简单的向量加法内核
@tilelang.jit(out_idx=[2])
def add(M, N, dtype="float16"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(M, N, threads=128) as (bx, by):
            vi = bx * 1  # 简化示例
            vj = by * 1
            C[vi, vj] = A[vi, vj] + B[vi, vj]
    return main

# 运行
import torch
A = torch.randn(128, 128, device="cuda", dtype=torch.float16)
B = torch.randn(128, 128, device="cuda", dtype=torch.float16)
C = add(A, B)
print(C.shape)  # torch.Size([128, 128])
```

## 参考资源

- [TileLang 官方文档](https://tilelang.tile-ai.cn/)
- [TileLang GitHub 仓库](https://github.com/tile-ai/TileLang)
- [TileLang 示例代码](https://github.com/tile-ai/TileLang/tree/main/tilelang/examples)
