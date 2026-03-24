---
name: tl-op-edit
description: 修改现有TileLang算子时，生成 claude.local.md 输入模板并据此新增/删除/修改参数（shape/dtype/block_size等）
---

# 修改TileLang算子通路技能

当用户提出"修改TileLang算子/调整参数/变更 shape 或 dtype/优化block size"时使用本技能。

## 工作流
1. 确认目标算子路径与算子名称。
2. 仅在触发本技能时才生成 `claude.local.md`。
3. 若 `claude.local.md` 不存在，则创建并写入"简化的修改输入模板"。
4. 支持用户用一句话描述变更，必要时再追问补充。
5. 根据输入执行修改：
   - 新增参数（输入/输出/属性）
   - 删除参数（输入/输出/属性）
   - 修改参数的 shape / dtype 约束
   - 调整 tiling 参数（block_M, block_N, block_K等）
   - 修改流水线参数（num_stages）
   - 添加/移除优化选项（swizzle, autotune等）
6. 输出改动说明与假设。

## 修改范围（默认）
- 主要修改：
  - 核心kernel实现文件（`*.py`）
  - 测试文件（`test_*.py`）
  - 基准测试文件（`benchmark_*.py`）
  - README文档（如有参数变更）

## 常见修改场景

### 1. 修改数据类型
```python
# 修改前
dtype = T.float16

# 修改后
dtype = T.bfloat16  # 或 T.float32
```

### 2. 调整tiling参数
```python
# 修改block size
def my_kernel(M, N, K, block_M=128, block_N=128, block_K=32):
    # 原来是 block_M=64, block_N=64, block_K=64
```

### 3. 修改流水线深度
```python
# 修改前
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):

# 修改后
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
```

### 4. 添加新的输入tensor
```python
@T.prim_func
def main(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    Bias: T.Tensor((N,), dtype),  # 新增bias
    C: T.Tensor((M, N), dtype),
):
```

### 5. 启用/禁用优化
```python
# 启用swizzle
T.use_swizzle(panel_size=10, enable=True)

# 禁用swizzle
T.use_swizzle(panel_size=10, enable=False)
```

## 约束
- 默认最小改动，不引入新依赖。
- 未经用户明确要求，不改变目录结构。
- 若输入信息不足，必须先列出假设。
- 严格按照用户明确要求的参数变更进行修改，不得自行添加或删减任何参数。
- 修改后需要确保测试能够通过。

## 生成的 claude.local.md 模板（简化）
```
# 修改算子输入模板（简化）

一句话说明你要改什么（允许最简描述）：
- 例如："给Matmul新增一个bias输入，dtype支持FP16/BF16"
- 例如："将FlashAttention的block_M从64改为128"

必要信息（可选补充）：
- 算子名：
- 算子路径：
- 变更动作（新增/删除/修改）：
- 参数名与类型：
- shape 约束（如有）：
- dtype 约束（如有）：
- 性能优化参数变更（如有）：
```

## 参考
- 模板目录：`templates/op_frame/empty_op_template`
- 参考实现：`templates/op_frame/gemm_reference`
