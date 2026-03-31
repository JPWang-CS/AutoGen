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
import torch

from your_op_name import your_op_name


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现 (CPU上执行)"""
    return A.cpu() @ B.cpu()


def calculate_flops(M: int, N: int, K: int) -> int:
    """
    计算GEMM的FLOPs

    GEMM: C = A @ B
    FLOPs = 2 * M * N * K
    """
    return 2 * M * N * K


def get_heuristic_config(M: int, N: int, K: int) -> dict:
    """
    根据问题规模获取NPU启发式配置

    推荐配置基于910B硬件规格，可根据目标SOC调整
    """
    return {
        "block_M": 32,
        "block_N": 32,
        "block_K": 32,
    }


def run_benchmark(M: int, N: int, K: int, config: dict, warmup: int, rep: int):
    """NPU性能测试"""
    torch.npu.set_device(0)

    kernel = your_op_name(
        M, N, K,
        block_M=config.get("block_M", 32),
        block_N=config.get("block_N", 32),
        block_K=config.get("block_K", 32),
    )

    # 准备输入数据 (NPU)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)

    # 正确性验证 (CPU对比)
    c = kernel(a, b)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(warmup=warmup, rep=rep)

    return latency


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend YourOpName Benchmark (NPU)")
    parser.add_argument("--m", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=4096, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=4096, help="Matrix dimension K")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Repeat iterations")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k

    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: NPU (华为昇腾)")

    config = get_heuristic_config(M, N, K)
    print(f"Heuristic config: {config}")

    # 运行benchmark
    latency = run_benchmark(M, N, K, config, args.warmup, args.rep)

    # 计算性能指标
    flops = calculate_flops(M, N, K)
    tflops = flops / latency * 1e-9

    # 打印结果
    print("\n" + "=" * 60)
    print("TileLang-Ascend Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Device:            NPU")
    print(f"Configuration:     {config}")
    print(f"TileLang Latency:  {latency:.3f} ms")
    print(f"TileLang TFlops:   {tflops:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
