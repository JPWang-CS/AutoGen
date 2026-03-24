# TileLang-Ascend AutoGen

TileLang算子自动生成工具集，提供基于Claude Code的skill提示词和相关模板。

## 概述

本仓库提供了一套完整的skill提示词，用于自动生成TileLang算子的完整实现，包括：
- 核心kernel实现
- 单元测试
- 性能基准测试
- 文档

## 快速开始

### 1. 新增算子

使用 `tl-op-pipeline` skill生成新的算子框架：

```
请使用tl-op-pipeline生成一个名为my_gemv的GEMV算子
```

### 2. 生成测试

使用 `tl-op-test` skill生成测试用例：

```
请为my_gemv生成单元测试
```

### 3. 性能测试

使用 `tl-op-benchmark` skill生成benchmark：

```
请为my_gemv生成性能基准测试
```

## 目录结构

```
AutoGen/
├── skills/                    # Skill提示词
│   ├── tl-op-pipeline/        # 新增算子
│   ├── tl-op-edit/            # 修改算子
│   ├── tl-op-rename/          # 重命名算子
│   ├── tl-op-test/            # 生成测试
│   └── tl-op-benchmark/       # 性能测试
├── templates/                 # 模板文件
│   └── op_frame/
│       ├── empty_op_template/ # 空白模板
│       └── gemm_reference/    # GEMM参考实现
└── projects/                  # 生成的项目
```

## 可用技能

| 技能 | 描述 | 触发条件 |
|------|------|----------|
| tl-op-pipeline | 生成完整算子实现 | "新增算子/生成算子通路" |
| tl-op-edit | 修改现有算子 | "修改算子/调整参数" |
| tl-op-rename | 重命名算子 | "重命名算子/修改算子名" |
| tl-op-test | 生成测试用例 | "生成测试/生成UT" |
| tl-op-benchmark | 生成性能测试 | "生成benchmark/性能测试" |

## TileLang简介

TileLang是一个高性能GPU kernel开发DSL，具有以下特点：

- **Pythonic语法**：使用Python语法定义kernel
- **自动优化**：内置流水线、tiling等优化
- **多后端支持**：支持CUDA、HIP、Metal等
- **AscendNPU支持**：支持华为Ascend芯片

### 核心特性

```python
@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M=64, block_N=64):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16),
             B: T.Tensor((K, N), T.float16),
             C: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            # Kernel实现
            ...
    return main
```

## 参考资料

- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang Documentation](https://tilelang.com/)
- [TileLang-Ascend](https://github.com/tile-ai/tilelang-ascend)

## 许可证

MIT License
