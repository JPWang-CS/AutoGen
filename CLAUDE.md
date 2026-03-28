# TileLang-Ascend AutoGen 仓库

本仓库提供TileLang算子自动生成的skill提示词和相关模板。

## 仓库结构

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
│   └── tl-op-benchmark/             # 生成benchmark技能
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
│   └── ...
├── CLAUDE.md                        # 本文件
└── README.md                        # 仓库说明
```

## 可用技能

### tl-op-pipeline
新增TileLang算子时使用。提供输入模板，并基于模板目录生成完整的算子实现。

**触发条件**：用户提出"新增一个TileLang算子/生成TileLang算子通路/按模板生成TileLang算子"

### tl-op-edit
修改现有TileLang算子时使用。支持新增/删除/修改参数，调整tiling参数等。

**触发条件**：用户提出"修改TileLang算子/调整参数/变更 shape 或 dtype/优化block size"

### tl-op-rename
重命名TileLang算子时使用。对算子名字做全量的修改。

**触发条件**：用户提出"修改TileLang算子名称/重命名TileLang算子"

### tl-op-test
为TileLang算子生成单元测试文件与用例。

**触发条件**：用户提出"生成测试/生成UT/补充测试用例"

### tl-op-benchmark
为TileLang算子生成性能基准测试文件。

**触发条件**：用户提出"生成benchmark/性能测试/性能对比"

### tl-op-hardware-constraints
检查算子代码是否符合不同NPU（如910B、910A、310P等）的硬件约束。

**触发条件**：用户提出"检查硬件约束/验证NPU限制/SOC硬件上限"

## TileLang简介

Tile Language (tile-lang) 是一个简洁的领域特定语言，旨在简化高性能GPU/CPU kernel（如GEMM、Dequant GEMM、FlashAttention、LinearAttention）的开发。通过采用Python语法并基于TVM构建编译器基础设施，tile-lang允许开发者专注于生产力，同时不牺牲低级优化。

### 核心概念

1. **JIT编译**：使用 `@tilelang.jit` 装饰器定义可JIT编译的kernel
2. **内存层次**：`T.alloc_shared` (shared memory), `T.alloc_fragment` (寄存器)
3. **核心原语**：`T.copy`, `T.gemm`, `T.clear`, `T.reduce_max`, `T.reduce_sum`
4. **并行编程**：`T.Kernel`, `T.Parallel`, `T.Pipelined`

### 示例代码

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M=64, block_N=64, block_K=32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm

# 使用
kernel = matmul(1024, 1024, 1024)
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = kernel(a, b)
```

## 开发流程

1. 使用 `tl-op-pipeline` 技能生成算子框架
2. 根据需求修改算子实现
3. 使用 `tl-op-test` 生成测试用例
4. 使用 `tl-op-benchmark` 进行性能测试
5. 如需修改，使用 `tl-op-edit` 调整参数

## 参考资料

- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang Documentation](https://tilelang.com/)
- [TileLang Examples](https://github.com/tile-ai/tilelang/tree/main/examples)
- [TileLang-Ascend Backend](https://github.com/tile-ai/tilelang-ascend)
