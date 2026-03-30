---
name: tl-op-hardware-constraints
description: 为TileLang-Ascend算子提供NPU硬件约束检查，确保算子搬运和计算符合不同SOC版本（910B、910A、310P等）的硬件上限
---

# TileLang-Ascend NPU 硬件约束技能

当用户开发 TileLang-Ascend 算子时，本技能提供 NPU 硬件约束检查和优化建议，确保算子搬运和计算符合不同 SOC 版本的硬件上限。

## 触发条件

- 用户提出"检查算子硬件约束/验证算子是否符合NPU限制/SOC硬件上限检查"
- 用户开发算子时需要考虑不同 NPU 型号的适配
- 算子出现硬件资源超限问题

## 支持的 SOC 版本

| SOC 版本 | 代号 | 定位 | 主要用途 |
|---------|------|------|---------|
| **Ascend 910** | DAV_TBE | 训练 | 大模型训练 |
| **Ascend 910A** | DAV_2201 | 训练 | 大模型训练 |
| **Ascend 910B** | DAV_3510 | 训练 | 大模型训练（当前主流） |
| **Ascend 310** | DAV_MINI | 推理 | 边缘推理 |
| **Ascend 310P** | DAV_310P | 推理 | 边缘推理 |

## 硬件规格参考

### 计算能力规格

| SOC 版本 | FP16 算力 | INT8 算力 | BF16 算力 | FP32 算力 |
|---------|----------|----------|----------|----------|
| **910** | 256 TFLOPS | 512 TOPS | - | 64 TFLOPS |
| **910A** | 256 TFLOPS | 512 TOPS | - | 64 TFLOPS |
| **910B** | ~320 TFLOPS | ~640 TOPS | ~320 TFLOPS | ~80 TFLOPS |
| **310** | 8 TFLOPS | 16 TOPS | - | 2 TFLOPS |
| **310P** | 8 TFLOPS | 22 TOPS | - | 2 TFLOPS |

### 存储层次规格

| 组件 | 910/910A | 910B | 310/310P | 用途 |
|------|----------|------|----------|------|
| **HBM/DDR** | 32GB/64GB | 64GB | 8GB/16GB | 全局内存 |
| **L2 Cache** | 共享 | 共享 | 共享 | 芯片级缓存 |
| **L1 Buffer** | ~1MB | ~1MB | ~512KB | 矩阵计算数据缓存 |
| **Unified Buffer (UB)** | ~2MB | ~2MB | ~1MB | 向量/标量计算输入输出 |
| **L0A Buffer** | Cube输入A | Cube输入A | Cube输入A | 矩阵乘输入A |
| **L0B Buffer** | Cube输入B | Cube输入B | Cube输入B | 矩阵乘输入B |
| **L0C Buffer** | Cube输出C | Cube输出C | Cube输出C | 矩阵乘输出 |

### AICore 架构参数

| 参数 | 910/910A | 910B | 310/310P |
|------|----------|------|----------|
| **AICore 数量** | 32 | 32 | 2-4 |
| **Cube Unit** | 有 | 有 | 有 |
| **Vector Unit** | 2个/AICore | 2个/AICore | 2个/AICore |
| **UB Bank 数量** | 32 | 32 | 32 |
| **UB Bank 大小** | 32B | 32B | 32B |

## 核心约束规则

### 1. Buffer 容量约束

算子中的 Tile 分块大小必须满足 Buffer 容量限制：

```python
# UB 容量约束（以910B为例，约2MB）
# 单次分配的UB大小不能超过限制
UB_MAX_SIZE = 2 * 1024 * 1024  # 2MB

# L1 Buffer 容量约束（约1MB）
L1_MAX_SIZE = 1 * 1024 * 1024  # 1MB

# 计算Tile大小时的约束
# 例如：block_M * block_N * sizeof(dtype) <= UB_MAX_SIZE
```

**检查公式**：
```python
def check_ub_constraint(block_size, dtype):
    dtype_size = {"float16": 2, "float32": 4, "bfloat16": 2, "int8": 1}
    required_size = block_size * dtype_size.get(dtype, 2)
    return required_size <= UB_MAX_SIZE
```

### 2. 数据对齐约束

数据搬运需要满足对齐要求：

| 存储类型 | 对齐要求 |
|---------|---------|
| **GM → UB** | 32字节对齐（cacheline大小） |
| **L1 → L0A/L0B** | 512字节对齐（分形大小） |
| **UB 访问** | 32字节对齐（Bank大小） |

**检查规则**：
```python
def check_alignment(size_bytes, alignment=32):
    return size_bytes % alignment == 0
```

### 3. 分形（Fractal）约束

矩阵乘法的分形大小约束：

| 数据类型 | 分形大小 (M×N×K) |
|---------|-----------------|
| **FP16** | 16×16×16 |
| **BF16** | 16×16×16 |
| **INT8** | 32×16×32 |

Tile 分块大小应为分形大小的整数倍。

### 4. Cube/Vector 资源冲突约束

Cube 计算和 Vector 计算不能共享同一 Bank Group：

```python
# 错误示例：Cube和Vector共享Bank Group
# A_VEC 和 A_MAT 使用相同Bank会冲突

# 正确做法：划分不同的Bank Group
A_VEC = T.alloc_shared((block_N,), dtype, bank_group=0)  # Vector使用
A_MAT = T.alloc_shared((block_M, block_K), dtype, bank_group=1)  # Cube使用
```

### 5. 流水线约束

使用 `T.Pipelined` 时，`num_stages` 受限于 Buffer 容量：

```python
# 流水线深度与Buffer的关系
# 总Buffer占用 = num_stages * single_stage_buffer_size
# 总占用不能超过Buffer容量

# 例如：双缓冲
for k in T.Pipelined(K // block_K, num_stages=2):
    # 每个stage需要的Buffer
    # 2 * (A_buffer + B_buffer) <= 可用Buffer
```

## 约束检查工作流

1. **识别目标 SOC 版本**
   - 默认使用 910B（DAV_3510）
   - 根据用户指定或自动检测确定版本

2. **提取算子参数**
   - 从 `@tilelang.jit` 装饰的函数中提取 block_M, block_N, block_K
   - 识别数据类型 dtype, accum_dtype
   - 统计 UB/L1/L0 分配情况

3. **逐项检查约束**
   - Buffer 容量检查
   - 数据对齐检查
   - 分形大小检查
   - 资源冲突检查

4. **输出检查报告**
   - 列出所有约束违反项
   - 提供优化建议
   - 给出推荐参数值

## 检查报告模板

```markdown
## 硬件约束检查报告

### 基本信息
- 算子名称: {op_name}
- 目标SOC: {soc_version}
- 数据类型: {dtype}

### 约束检查结果

| 检查项 | 状态 | 详情 |
|-------|------|------|
| UB容量 | ✅/❌ | 当前使用: {ub_used}, 上限: {ub_max} |
| L1容量 | ✅/❌ | 当前使用: {l1_used}, 上限: {l1_max} |
| 数据对齐 | ✅/❌ | 详情: {alignment_details} |
| 分形约束 | ✅/❌ | Tile大小: {tile_size}, 分形: {fractal_size} |

### 优化建议
1. {suggestion_1}
2. {suggestion_2}
...

### 推荐参数
- block_M: {recommended_block_m}
- block_N: {recommended_block_n}
- block_K: {recommended_block_k}
```

## 优化建议规则

### 1. Buffer 超限优化

当 UB 或 L1 超限时：

```python
# 方案1：减小Tile大小
# 将block_M从128降到64

# 方案2：减少流水线深度
# 将num_stages从3降到2

# 方案3：使用更小的数据类型
# 从float32改为float16
```

### 2. 对齐问题修复

```python
# 将非对齐大小padding到对齐边界
aligned_size = (size + 31) // 32 * 32
```

### 3. 资源冲突解决

```python
# 为Cube和Vector分配不同Bank Group
cube_buffer = T.alloc_shared(shape, dtype, bank_group=0)
vector_buffer = T.alloc_shared(shape, dtype, bank_group=1)
```

## 多 SOC 适配策略

### 条件编译

```python
import os

SOC_VERSION = os.environ.get("SOC_VERSION", "910B")

if SOC_VERSION == "910B":
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
elif SOC_VERSION == "310P":
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
```

### 自动调优适配

```python
from tilelang.autotuner import *

# 针对不同SOC定义不同配置空间
def get_configs_for_soc(soc_version):
    if soc_version == "910B":
        return [
            {"block_M": 128, "block_N": 128, "block_K": 64},
            {"block_M": 64, "block_N": 128, "block_K": 64},
        ]
    elif soc_version == "310P":
        return [
            {"block_M": 64, "block_N": 64, "block_K": 32},
            {"block_M": 32, "block_N": 64, "block_K": 32},
        ]
```

## 与其他 Skill 的协作

- **tl-op-pipeline**: 在生成算子时自动应用硬件约束
- **tl-op-edit**: 修改算子时重新检查约束
- **tl-op-benchmark**: 性能测试时验证约束符合性

## 参考资源

- [TileLang-Ascend 开发指南](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md)
- [昇腾社区 - 硬件架构文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0010.html)
- [Ascend C 编程指南](https://www.hiascend.com/document)
