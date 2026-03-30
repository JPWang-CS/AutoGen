"""
TileLang-Ascend 算子性能基准测试模板 - your_op_name (NPU专用)

该文件提供了TileLang-Ascend算子性能测试的基础模板。
请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. FLOPs计算逻辑
3. 调优配置空间
4. 需要测试的shape范围

所有计算在NPU上执行，精度对比在CPU上进行。
"""

import argparse
import itertools
import torch
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner

from your_op_name import your_op_name


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现 (CPU上执行)"""
    return A.cpu() @ B.cpu()


def calculate_flops(M: int, N: int, K: int) -> int:
    """
    计算GEMM的FLOPs

    GEMM: C = A @ B
    FLOPs = 2 * M * N * K (每个输出元素需要K次乘法和K-1次加法，约等于2K次浮点运算)
    """
    return 2 * M * N * K


def get_npu_configs():
    """获取NPU调优配置空间"""
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [32, 64]
    num_stages = [0, 1, 2]

    configs = []
    for bm, bn, bk, ns in itertools.product(block_M, block_N, block_K, num_stages):
        configs.append({
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
            "num_stages": ns,
        })
    return configs


def get_heuristic_config(M: int, N: int, K: int) -> dict:
    """
    根据问题规模获取NPU启发式配置

    推荐配置基于910B硬件规格，可根据目标SOC调整
    """
    return {
        "block_M": 128,
        "block_N": 128,
        "block_K": 32,
        "num_stages": 2,
    }


def run_npu_autotune(M: int, N: int, K: int):
    """NPU自动调优"""

    def kernel_fn(block_M=None, block_N=None, block_K=None, num_stages=None):
        return your_op_name(M, N, K, block_M, block_N, block_K, num_stages)

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel_fn, configs=get_npu_configs())
        .set_compile_args(out_idx=[-1])
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
        )
    )

    result = autotuner.run(warmup=3, rep=20)
    return result.config, result.kernel


def benchmark_single(
    M: int, N: int, K: int,
    config: dict,
    warmup: int = 10,
    rep: int = 100,
):
    """单配置benchmark (NPU)"""
    import torch_npu
    torch.npu.set_device(0)

    kernel = your_op_name(
        M, N, K,
        block_M=config.get("block_M", 128),
        block_N=config.get("block_N", 128),
        block_K=config.get("block_K", 32),
        num_stages=config.get("num_stages", 2),
    )

    # 准备输入数据 (NPU)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)

    # 正确性验证 (CPU对比)
    c = kernel(a, b)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    # 性能测试
    profiler = kernel.get_profiler()
    tl_latency = profiler.do_bench(warmup=warmup, rep=rep)

    return tl_latency


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend YourOpName Benchmark (NPU)")
    parser.add_argument("--M", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--N", type=int, default=4096, help="Matrix dimension N")
    parser.add_argument("--K", type=int, default=4096, help="Matrix dimension K")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Repeat iterations")
    parser.add_argument("--autotune", action="store_true", help="Run autotuning")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K

    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: NPU (华为昇腾)")

    if args.autotune:
        print("Running autotuning...")
        best_config, kernel = run_npu_autotune(M, N, K)
        print(f"Best config: {best_config}")
        config = best_config
    else:
        config = get_heuristic_config(M, N, K)
        print(f"Heuristic config: {config}")

    # 运行benchmark
    tl_latency = benchmark_single(M, N, K, config, args.warmup, args.rep)

    # 计算性能指标
    flops = calculate_flops(M, N, K)
    tl_tflops = flops / tl_latency * 1e-9

    # 打印结果
    print("\n" + "=" * 60)
    print("TileLang-Ascend Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Device:            NPU")
    print(f"Configuration:     {config}")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang TFlops:   {tl_tflops:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
