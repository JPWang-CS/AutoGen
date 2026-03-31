# TileLang-Ascend AutoGen 仓库 (NPU专用)

本仓库提供TileLang-Ascend NPU算子自动生成的skill提示词和相关模板。所有算子运行在华为昇腾NPU上，精度对比在CPU上进行。

## ⚠️ NPU 编程规范 (重要)

### CUDA vs NPU 关键差异

| 特性 | ❌ CUDA (禁用) | ✅ NPU (正确) |
|------|---------------|-------------|
| **Kernel启动** | `T.Kernel(..., threads=N)` | `T.Kernel(..., is_npu=True) as (cid, vid)` |
| **Block索引** | `(bx, by)` 直接2D | `(cid, vid)` 线性索引，手动计算 |
| **内存分配** | `alloc_shared`, `alloc_fragment` | `alloc_ub` (Unified Buffer) |
| **并行循环** | `T.Parallel(M, N)` | `T.serial` 或向量指令 |
| **向量计算** | 标量 `a + b` | `T.tile.add(c, a, b)` |
| **同步** | 自动 | `T.barrier_all()` |
| **作用域** | 无 | `with T.Scope("V"):` |

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
| `T.barrier_all()` | 所有核同步 |
| `T.barrier(tid)` | 指定核同步 |

## 开发流程

1. 使用 `tl-op-pipeline` 技能生成算子框架
2. 根据需求修改算子实现 (遵循NPU编程规范)
3. 使用 `tl-op-test` 生成测试用例
4. 使用 `tl-op-benchmark` 进行性能测试
5. 如需修改，使用 `tl-op-edit` 调整参数

## 参考资料

- [TileLang-Ascend GitHub](https://github.com/tile-ai/tilelang-ascend)
- [TileLang Documentation](https://tilelang.com/)
- [TileLang-Ascend 开发指南](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md)
