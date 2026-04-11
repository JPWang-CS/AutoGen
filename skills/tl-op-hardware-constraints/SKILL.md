---
name: tl-op-hardware-constraints
description: 为TileLang-Ascend算子提供NPU硬件约束检查，确保算子搬运和计算符合不同SOC版本（910B、910A、310P等）的硬件上限，防止地址越界和溢出
---

# TileLang-Ascend NPU 硬件约束技能

当用户开发 TileLang-Ascend 算子时，本技能提供 NPU 硬件约束检查和优化建议，确保算子搬运和计算符合不同 SOC 版本的硬件上限，**防止硬件地址越界或溢出**。

## 触发条件

- 用户提出"检查算子硬件约束/验证算子是否符合NPU限制/SOC硬件上限检查"
- 用户开发算子时需要考虑不同 NPU 型号的适配
- 算子出现硬件资源超限问题
- 算子出现地址越界或溢出错误

## ⚠️ NPU 执行引擎架构 (核心约束)

| 引擎 | 名称 | 功能 | 能否操作 UB |
|------|------|------|------------|
| **MTE2** | Memory Transfer Engine 2 | GM -> UB 数据加载 | ❌ 只搬运，不计算 |
| **V** | Vector Core | 向量计算 | ✅ **只有 V 核才能操作 UB 进行计算** |
| **MTE3** | Memory Transfer Engine 3 | UB -> GM 数据存储 | ❌ 只搬运，不计算 |
| **Cube** | Cube Core | 矩阵乘法 | ❌ **Cube 核没有 UB！** |

### 关键约束

1. **只有 Vector Core (V) 才能操作 UB 进行计算**
   - `T.tile.add`, `T.tile.mul` 等向量指令**必须在** `T.Scope("V")` 内
   - Cube 核用于矩阵乘法 (`T.gemm`)，**没有 UB**

2. **数据搬运必须在对应的作用域内执行**
   - `T.copy(A[...], a_ub)`: GM -> UB，**必须在** `T.Scope("V")` 内
   - `T.copy(c_ub, C[...])`: UB -> GM，**必须在** `T.Scope("V")` 内
   - 搬运到 Cube 的 L2/L0 需要在 Cube 作用域内

3. **流水线同步确保引擎间正确协作**
   - `T.set_flag("mte2", "v", 0)`: MTE2 完成，通知 V
   - `T.wait_flag("mte2", "v", 0)`: V 等待 MTE2
   - `T.set_flag("v", "mte3", 0)`: V 完成，通知 MTE3
   - `T.wait_flag("v", "mte3", 0)`: MTE3 等待 V

## NPU 架构版本对照

| NPU_ARCH | 架构类型 | 对应产品 | 特点 |
|----------|---------|---------|------|
| **200x** | Cube+Vector同核 | Atlas 推理系列产品 | Cube和Vector共享Scalar单元 |
| **220x** | Cube+Vector分离 | Atlas A2/A3 训练/推理系列 | AIC(矩阵)和AIV(向量)独立，各自有Scalar单元 |

> **重要**: 不同架构版本的存储单元对齐要求和同步机制有差异，生成算子时必须明确目标架构。

## 存储单元对齐要求（官方规范）

### 200x 架构 (Atlas 推理系列)

| 存储单元 | 对齐要求 | 说明 |
|---------|---------|------|
| **Unified Buffer (UB)** | **32字节** | Vector计算数据来源，必须32字节对齐 |
| **L1 Buffer** | **32字节** | 矩阵计算数据缓存 |
| **L0A Buffer** | **512字节** | Cube左矩阵输入，按分形大小对齐 |
| **L0B Buffer** | **512字节** | Cube右矩阵输入，按分形大小对齐 |
| **L0C Buffer** | **64字节** | Cube矩阵乘输出 |

### 220x 架构 (Atlas A2/A3 系列)

| 核 | 存储单元 | 对齐要求 | 说明 |
|----|---------|---------|------|
| **AIV** | Unified Buffer | **32字节** | 向量计算单元数据来源 |
| **AIC** | L1 Buffer | **32字节** | 矩阵计算数据缓存 |
| **AIC** | L0A Buffer | **512字节** | Cube左矩阵输入 |
| **AIC** | L0B Buffer | **512字节** | Cube右矩阵输入 |
| **AIC** | L0C Buffer | **64字节** | Cube矩阵乘输出 |
| **AIC** | BiasTable Buffer | **64字节** | 偏置表缓冲区 |
| **AIC** | Fixpipe Buffer | **64字节** | Fixpipe加速模块缓冲区 |

## 数据搬运约束（防止越界）

### 搬运到 Unified Buffer

```python
# ⚠️ 官方约束：搬运到UB的数据大小必须按DataBlock对齐
# DataBlock大小 = 32字节

# 正确示例：确保搬运大小是32字节的整数倍
data_size_bytes = block_M * block_N * dtype_size
assert data_size_bytes % 32 == 0, f"搬运大小{data_size_bytes}不是32字节对齐！"

# TileLang中通过正确设置block大小来保证
# 例如：block_N=16, dtype=float16 => 16*2=32字节，满足对齐
```

### 搬运到 L0A/L0B Buffer

```python
# ⚠️ 官方约束：L1->L0A/L0B必须按分形大小对齐
# 如果L1 Buffer剩余大小不足1个分形，硬件执行会异常！

# 分形大小（FP16/BF16）：16x16
# 分形大小（INT8）：32x16或16x32

# 正确示例：
# 从L1搬运到L0A时，数据格式从NZ转换为ZZ，大小必须按分形对齐
fractal_size_m = 16  # FP16分形
fractal_size_n = 16

# 检查是否满足分形对齐
assert block_M % fractal_size_m == 0, f"block_M必须是{fractal_size_m}的倍数"
assert block_N % fractal_size_n == 0, f"block_N必须是{fractal_size_n}的倍数"
```

### 搬运时地址计算约束

```python
# ⚠️ 防止地址越界的关键检查

def check_gm_address_overflow(tensor_shape, block_size, dtype_size):
    """检查GM地址是否可能越界"""
    total_elements = 1
    for dim in tensor_shape:
        total_elements *= dim
    total_bytes = total_elements * dtype_size

    # 检查是否超过GM最大寻址范围
    MAX_GM_ADDRESS = 64 * 1024 * 1024 * 1024  # 64GB (910B HBM)
    assert total_bytes <= MAX_GM_ADDRESS, f"GM总大小{total_bytes}超过上限{MAX_GM_ADDRESS}"

    # 检查block索引计算是否溢出
    num_blocks = total_elements // block_size
    max_block_idx = num_blocks - 1
    max_address_offset = max_block_idx * block_size * dtype_size

    assert max_address_offset < MAX_GM_ADDRESS, "Block索引计算可能导致地址越界"

def check_ub_address_overflow(block_shape, dtype_size, num_buffers=3):
    """检查UB地址是否可能越界"""
    block_elements = 1
    for dim in block_shape:
        block_elements *= dim
    single_buffer_bytes = block_elements * dtype_size
    total_ub_bytes = single_buffer_bytes * num_buffers

    # 910B UB约2MB
    MAX_UB_SIZE = 2 * 1024 * 1024
    assert total_ub_bytes <= MAX_UB_SIZE, f"UB使用量{total_ub_bytes}超过上限{MAX_UB_SIZE}"

    return True
```

## 同步控制约束（官方规范）

### 核内同步规则

**⚠️ 关键约束（来自官方文档）：**

1. **SetFlag 和 WaitFlag 必须成对使用**
   ```python
   # 正确：成对使用
   T.set_flag("mte2", "v", 0)   # MTE2完成，通知V
   T.wait_flag("mte2", "v", 0)  # V等待MTE2

   # 错误：未成对使用会导致计算异常或timeout
   T.set_flag("mte2", "v", 0)
   # 缺少对应的wait_flag，可能导致下一个核的算子执行异常
   ```

2. **参数必须完全一致**
   ```python
   # 正确：参数完全一致
   T.set_flag("mte2", "v", 0)
   T.wait_flag("mte2", "v", 0)  # 源、目标、stage_id都相同

   # 错误：参数不匹配视为不同的EventID
   T.set_flag("mte2", "v", 0)
   T.wait_flag("v", "mte3", 0)  # 这不是同一个EventID！
   ```

3. **禁止连续设置同一个EventID**
   ```python
   # 错误：连续设置同一EventID导致状态混乱
   T.set_flag("mte2", "v", 0)
   T.set_flag("mte2", "v", 0)  # ❌ 禁止！

   # 正确：等待后再设置
   T.set_flag("mte2", "v", 0)
   T.wait_flag("mte2", "v", 0)
   T.set_flag("mte2", "v", 0)  # ✓ 允许
   ```

4. **EventID 数量有限**
   - 官方建议：使用后立即释放
   - 禁止手动插入EventID 6和7（系统预留）

### 流水线同步模式

```
MTE2 (加载) --set_flag("mte2", "v", 0)--> V (计算) --set_flag("v", "mte3", 0)--> MTE3 (存储)
                    |                              |                              |
                    v                              v                              v
              wait_flag                     wait_flag                      wait_flag
```

使用 `set_flag/wait_flag` 可以实现细粒度的流水线同步，相比 `barrier_all` 有更好的并行性。

### ⚠️ 多Block循环中的同步陷阱（实测验证）

**问题**：当 `T.Kernel` 启动少量核（如24个），每个核通过 `T.serial` 循环处理多个数据block时，`set_flag/wait_flag` 使用固定 `stage_id=0` 会导致**跨迭代的同步混乱**。

**根因**：第 N 次迭代的 `set_flag("mte2", "v", 0)` 可能与第 N-1 次迭代的 flag 残留状态冲突，导致数据在未完成加载时就开始计算，或在未完成计算时就开始写入，产生全量数据错误（从位置0就开始不匹配）。

**症状**：
- 输出数据从位置0就完全错误（不是部分错误）
- 错误值看起来像垃圾值，不是简单的未初始化（0）
- 不匹配元素数量接近100%

**解决方案**：在每次迭代结束时添加 `T.barrier_all()`，确保当前block的所有引擎操作完成后再进入下一次迭代。

```python
# ❌ 错误：多block循环中不加barrier，set_flag/wait_flag的stage_id冲突
for i in T.serial(blocks_per_core):
    block_idx = cid * blocks_per_core + i
    if block_idx < total_blocks:
        with T.Scope("V"):
            T.copy(A[block_idx * block_N + vid * block_N // VEC_NUM], a_ub)
            T.copy(B[block_idx * block_N + vid * block_N // VEC_NUM], b_ub)
            T.set_flag("mte2", "v", 0)
            T.wait_flag("mte2", "v", 0)
            T.tile.add(c_ub, a_ub, b_ub)
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(c_ub, C[block_idx * block_N + vid * block_N // VEC_NUM])
            # 循环进入下一次迭代时，flag状态混乱！

# ✅ 正确：每次迭代结束后加 T.barrier_all()
for i in T.serial(blocks_per_core):
    block_idx = cid * blocks_per_core + i
    if block_idx < total_blocks:
        with T.Scope("V"):
            T.copy(A[block_idx * block_N + vid * block_N // VEC_NUM], a_ub)
            T.copy(B[block_idx * block_N + vid * block_N // VEC_NUM], b_ub)
            T.set_flag("mte2", "v", 0)
            T.wait_flag("mte2", "v", 0)
            T.tile.add(c_ub, a_ub, b_ub)
            T.set_flag("v", "mte3", 0)
            T.wait_flag("v", "mte3", 0)
            T.copy(c_ub, C[block_idx * block_N + vid * block_N // VEC_NUM])
            T.barrier_all()  # 确保所有引擎完成当前block
```

**规则**：单block处理（`T.Kernel(n_num)` 每核一个block）不需要 `barrier_all`。多block循环（`T.Kernel(num_cores)` 每核多个block via `T.serial`）**必须**在每次迭代末尾加 `T.barrier_all()`。

### 多Block地址分配规则

当 `T.Kernel` 的block数少于总数据块数时，需要正确的地址分配：

```python
total_blocks = N // block_N
blocks_per_core = (total_blocks + num_cores - 1) // num_cores  # 向上取整

# 在kernel中：
for i in T.serial(blocks_per_core):
    block_idx = cid * blocks_per_core + i
    if block_idx < total_blocks:  # 边界检查：跳过超出范围的block
        # ... 处理 block_idx ...
```

### 220x架构核间同步

```python
# 220x架构支持核间同步，200x不支持
# CrossCoreSetFlag 和 CrossCoreWaitFlag 必须成对使用

# 模式0：AI Core核间同步
# 模式1：AIV核之间同步
# 模式2：AIC与AIV之间同步

# flagId取值范围：0-10
# 同一flagId的计数器最多可以设置15次
```

## 分形（Fractal）约束

### 各存储单元推荐数据排布格式

| 存储单元 | 推荐格式 | 说明 |
|---------|---------|------|
| **L0A Buffer** | FRACTAL_ZZ | 矩阵乘输入A优化格式 |
| **L0B Buffer** | FRACTAL_ZN | 矩阵乘输入B优化格式 |
| **L0C Buffer** | FRACTAL_NZ | 矩阵乘输出优化格式 |
| **L1 Buffer** | FRACTAL_NZ | 降低格式转换开销 |
| **Unified Buffer** | 无要求 | 任意格式 |

### 分形大小规格

| 数据类型 | 分形大小 (M×N×K) | 单个分形字节数 |
|---------|-----------------|---------------|
| **FP16** | 16×16×16 | 512字节 |
| **BF16** | 16×16×16 | 512字节 |
| **INT8** | 32×16×32 | 1024字节 |

**⚠️ Tile分块大小必须是分形大小的整数倍！**

```python
# 检查分形约束
def check_fractal_constraint(block_M, block_N, block_K, dtype):
    fractal_size = 16 if dtype in ["float16", "bfloat16"] else 32

    assert block_M % fractal_size == 0, f"block_M={block_M}必须是{fractal_size}的倍数"
    assert block_N % fractal_size == 0, f"block_N={block_N}必须是{fractal_size}的倍数"
    assert block_K % fractal_size == 0, f"block_K={block_K}必须是{fractal_size}的倍数"

    return True
```

## Bank 冲突约束

### Unified Buffer Bank 冲突

当多个操作同时访问UB同一个bank或bank group时，会发生bank冲突（读写/写写/读读冲突），导致访问排队降低性能。

```python
# 避免Bank冲突：为不同计算单元分配不同Bank Group
# 在TileLang中使用T.alloc_ub分配时注意隔离

# Vector计算使用
vec_ub = T.alloc_ub((block_M, block_N), dtype)  # 默认bank group

# 如果同时有Cube计算需要访问UB，使用不同bank
# （具体bank分配策略需要根据实际硬件配置）
```

## 支持的 SOC 版本

| SOC 版本 | 代号 | NPU_ARCH | 定位 | 主要用途 |
|---------|------|----------|------|---------|
| **Ascend 910** | DAV_TBE | 200x | 训练 | 大模型训练 |
| **Ascend 910A** | DAV_2201 | 200x | 训练 | 大模型训练 |
| **Ascend 910B** | DAV_3510 | 220x | 训练 | 大模型训练（当前主流） |
| **Ascend 310** | DAV_MINI | 200x | 推理 | 边缘推理 |
| **Ascend 310P** | DAV_310P | 200x | 推理 | 边缘推理 |

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
| **AIC 数量 (矩阵核)** | 32 | 24 | 2-4 |
| **AIV 数量 (向量核)** | 64 (2个/AIC) | 48 (2个/AIC) | 4-8 (2个/AIC) |
| **Cube Unit** | 有 | 有 | 有 |
| **Vector Unit** | 2个/AIC | 2个/AIC | 2个/AIC |
| **UB Bank 数量** | 32 | 32 | 32 |
| **UB Bank 大小** | 32B | 32B | 32B |

> **重要**: 910B (220x架构) 有 **24个AIC (矩阵核)** 和 **48个AIV (向量核)**。
> - 向量算子: `T.Kernel` 的 block 数量上限为 24 (每个 block 对应一个 AIC，含2个AIV通过 vid=0/1 并行)
> - 矩阵算子: 同样最多 24 个 block (每个 AIC 含1个 Cube Unit)
> - 默认 `num_cores` 应设为 **24** (非32)

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
A_VEC = T.alloc_ub((block_N,), dtype)  # Vector使用
A_MAT = T.alloc_ub((block_M, block_K), dtype)  # Cube使用（注意分离）
```

### 5. 流水线约束

使用流水线时，`num_stages` 受限于 Buffer 容量：

```python
# 流水线深度与Buffer的关系
# 总Buffer占用 = num_stages * single_stage_buffer_size
# 总占用不能超过Buffer容量

# 例如：双缓冲
# 每个stage需要的Buffer: 2 * (A_buffer + B_buffer) <= 可用Buffer
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
   - 数据对齐检查（32B/512B/64B）
   - 分形大小检查
   - 资源冲突检查
   - **同步控制检查（SetFlag/WaitFlag配对）**

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
- NPU架构: {npu_arch} (200x/220x)
- 数据类型: {dtype}

### 约束检查结果

| 检查项 | 状态 | 详情 |
|-------|------|------|
| UB容量 | ✅/❌ | 当前使用: {ub_used}, 上限: {ub_max} |
| L1容量 | ✅/❌ | 当前使用: {l1_used}, 上限: {l1_max} |
| UB对齐 | ✅/❌ | 要求: 32B, 实际: {actual_align} |
| L0A/L0B对齐 | ✅/❌ | 要求: 512B, 实际: {actual_align} |
| L0C对齐 | ✅/❌ | 要求: 64B, 实际: {actual_align} |
| 分形约束 | ✅/❌ | Tile大小: {tile_size}, 分形: {fractal_size} |
| 同步控制 | ✅/❌ | SetFlag/WaitFlag配对: {sync_status} |

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
aligned_size = (size + 31) // 32 * 32  # 32B对齐
aligned_size_512 = (size + 511) // 512 * 512  # 512B对齐
aligned_size_64 = (size + 63) // 64 * 64  # 64B对齐
```

### 3. 资源冲突解决

```python
# 为Cube和Vector分配不同Bank Group
cube_buffer = T.alloc_ub(shape, dtype)  # 注意分离使用
vector_buffer = T.alloc_ub(shape, dtype)  # 不同时区
```

### 4. 同步控制修复

```python
# 确保SetFlag和WaitFlag成对使用
T.set_flag("mte2", "v", stage_id)
# ... Vector计算 ...
T.wait_flag("mte2", "v", stage_id)  # 必须配对

# 确保参数一致
# 错误：T.set_flag("mte2", "v", 0) 和 T.wait_flag("mte2", "v", 1) 不配对！
```

## 多 SOC 适配策略

### 条件编译

```python
import os

SOC_VERSION = os.environ.get("SOC_VERSION", "910B")

if SOC_VERSION == "910B":
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    NPU_ARCH = "220x"
elif SOC_VERSION == "310P":
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    NPU_ARCH = "200x"
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

- [昇腾社区 - NPU架构版本200x](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_0010.html)
- [昇腾社区 - NPU架构版本220x](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
- [TileLang-Ascend 开发指南](https://github.com/tile-ai/tilelang-ascend/blob/npuir/docs/开发指南.md)
- [Ascend C 编程指南](https://www.hiascend.com/document)
