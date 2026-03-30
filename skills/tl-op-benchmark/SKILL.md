---
name: tl-op-benchmark
description: 为TileLang算子生成性能基准测试文件。用户提出"生成benchmark/性能测试/性能对比"时使用本技能。
---

# TileLang算子性能基准测试生成

当用户提出"生成TileLang算子benchmark/性能测试/性能对比"时使用本技能。

## 工作流
1. 确认目标算子路径与算子名称。
2. 读取算子实现文件，理解计算模式和参数。
3. 生成benchmark文件 `benchmark_{op_name}.py`。
4. 包含以下内容：
   - 性能测试（延迟、吞吐量）
   - 与PyTorch/其他框架的性能对比
   - 不同配置的性能测试
   - 结果输出和可视化

## Benchmark文件结构

```python
import argparse
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner

# 导入被测试的kernel
from your_op_name import your_op_name

def ref_program(*args, **kwargs):
    """PyTorch参考实现"""
    pass

def calculate_flops(*args, **kwargs):
    """计算FLOPs"""
    pass

def benchmark_config(M, N, K, config, warmup=10, rep=100):
    """单配置benchmark"""
    kernel = your_op_name(M, N, K, **config)

    # 准备输入数据
    # ...

    profiler = kernel.get_profiler()

    # TileLang性能测试
    tl_latency = profiler.do_bench()

    # 参考实现性能测试
    ref_latency = profiler.do_bench(ref_program)

    # 正确性验证
    profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)

    return tl_latency, ref_latency

def main():
    parser = argparse.ArgumentParser(description="YourOpName Benchmark")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()

    # 默认配置 (NPU)
    default_config = {
        "block_M": 128,
        "block_N": 128,
        "block_K": 32,
        "num_stages": 2,
    }

    if args.autotune:
        # 自动调优搜索最优配置
        best_config = run_autotune(args.M, args.N, args.K)
        config = best_config
    else:
        config = default_config

    tl_latency, ref_latency = benchmark_config(
        args.M, args.N, args.K, config,
        warmup=args.warmup, rep=args.rep
    )

    flops = calculate_flops(args.M, args.N, args.K)

    print(f"Configuration: {config}")
    print(f"TileLang Latency: {tl_latency:.3f} ms")
    print(f"TileLang TFlops: {flops / tl_latency * 1e-9:.2f}")
    print(f"PyTorch Latency: {ref_latency:.3f} ms")
    print(f"PyTorch TFlops: {flops / ref_latency * 1e-9:.2f}")
    print(f"Speedup: {ref_latency / tl_latency:.2f}x")

def run_autotune(M, N, K):
    """运行自动调优"""
    configs = get_tune_configs()

    best_latency = float('inf')
    best_config = None

    for config in configs:
        try:
            latency, _ = benchmark_config(M, N, K, config)
            if latency < best_latency:
                best_latency = latency
                best_config = config
        except Exception as e:
            print(f"Config {config} failed: {e}")
            continue

    return best_config

def get_tune_configs():
    """获取NPU调优配置空间"""
    import itertools

    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [32, 64]
    num_stages = [0, 1, 2]

    configs = []
    for bm, bn, bk, ns in itertools.product(
        block_M, block_N, block_K, num_stages
    ):
        configs.append({
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
            "num_stages": ns,
        })
    return configs

if __name__ == "__main__":
    main()
```

## 性能指标

### 1. 延迟（Latency）
- 单次kernel执行时间（毫秒）
- 使用 `profiler.do_bench()` 测量

### 2. 吞吐量（TFlops）
- 每秒浮点运算次数（万亿次）
- 计算公式：`FLOPs / latency * 1e-9`

### 3. 加速比（Speedup）
- 相对于参考实现的性能提升
- 计算公式：`ref_latency / tl_latency`

## FLOPs计算

### GEMM
```python
flops = 2 * M * N * K  # 乘法和加法
```

### Flash Attention
```python
flops = 2 * batch * heads * seq_q * seq_kv * dim  # QK^T
flops += 2 * batch * heads * seq_q * dim * seq_kv  # AV
```

## 输出要求
- 清晰的性能数据表格
- 与参考实现的对比
- 不同配置的性能差异
- 最优配置推荐

## 约束
- 确保正确性验证通过后再进行性能测试
- 提供充足的warmup轮次
- 多次测量取平均值

## 参考
- 模板目录：`templates/op_frame/empty_op_template/benchmark_your_op_name.py`
- TileLang benchmark示例：`https://github.com/tile-ai/tilelang/tree/main/benchmark`
