"""
TileLang-Ascend算子性能基准测试模板 - your_op_name

该文件提供了TileLang-Ascend算子性能测试的基础模板。
请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. FLOPs计算逻辑
3. 调优配置空间
4. 需要测试的shape范围

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)
"""

import argparse
import itertools
import torch
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner

from your_op_name import your_op_name, your_op_name_cuda, get_device, setup_device


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return A @ B


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


def get_cuda_configs():
    """获取CUDA调优配置空间"""
    block_M = [64, 128, 256]
    block_N = [64, 128, 256]
    block_K = [32, 64]
    num_stages = [0, 1, 2, 3]
    threads = [128, 256]

    configs = []
    for bm, bn, bk, ns, t in itertools.product(
        block_M, block_N, block_K, num_stages, threads
    ):
        configs.append({
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
            "num_stages": ns,
            "threads": t,
        })
    return configs


def get_heuristic_config(M: int, N: int, K: int) -> dict:
    """
    根据设备类型和问题规模获取启发式配置
    支持NPU和CUDA设备
    """
    device_type = get_device()

    if device_type == "npu":
        # 华为昇腾NPU配置
        print(f"NPU device detected")
        return {
            "block_M": 128,
            "block_N": 128,
            "block_K": 32,
            "num_stages": 2,
        }
    elif device_type == "cuda":
        device = torch.cuda.current_device()
        sm_major, sm_minor = torch.cuda.get_device_capability(device)
        sm_version = sm_major * 10 + sm_minor
        print(f"CUDA device capability: {sm_version}")

        # 根据SM版本选择配置
        if sm_version == 80:  # A100
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 32,
                "num_stages": 2,
                "threads": 128,
            }
        elif sm_version == 90:  # H100
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 64,
                "num_stages": 3,
                "threads": 256,
            }
        else:
            return {
                "block_M": 128,
                "block_N": 128,
                "block_K": 32,
                "num_stages": 2,
                "threads": 128,
            }
    else:
        return {
            "block_M": 128,
            "block_N": 128,
            "block_K": 32,
            "num_stages": 2,
            "threads": 128,
        }


def run_npu_autotune(M: int, N: int, K: int, topk: int = 20):
    """NPU自动调优"""

    def kernel_fn(block_M=None, block_N=None, block_K=None, num_stages=None):
        return your_op_name(M, N, K, block_M, block_N, block_K, num_stages)

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel_fn, configs=get_npu_configs())
        .set_compile_args(out_idx=[-1], target="npuir")
        .set_profile_args(
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
        )
    )

    result = autotuner.run(warmup=3, rep=20)
    return result.config, result.kernel


def run_cuda_autotune(M: int, N: int, K: int, topk: int = 20):
    """CUDA自动调优"""

    def kernel_fn(block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
        return your_op_name_cuda(M, N, K, block_M, block_N, block_K, num_stages, threads)

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel_fn, configs=get_cuda_configs())
        .set_compile_args(out_idx=[-1], target="cuda")
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
    """单配置benchmark"""
    device_type = get_device()

    if device_type == "npu":
        kernel = your_op_name(
            M, N, K,
            block_M=config.get("block_M", 128),
            block_N=config.get("block_N", 128),
            block_K=config.get("block_K", 32),
            num_stages=config.get("num_stages", 2),
        )
    else:
        kernel = your_op_name_cuda(
            M, N, K,
            block_M=config.get("block_M", 128),
            block_N=config.get("block_N", 128),
            block_K=config.get("block_K", 32),
            num_stages=config.get("num_stages", 2),
            threads=config.get("threads", 128),
        )

    # 设置设备
    setup_device(device_type)

    # 准备输入数据
    a = torch.randn(M, K, device=device_type, dtype=torch.float16)
    b = torch.randn(K, N, device=device_type, dtype=torch.float16)

    profiler = kernel.get_profiler()

    # 正确性验证
    profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)

    # TileLang性能测试
    tl_latency = profiler.do_bench(warmup=warmup, rep=rep)

    # PyTorch性能测试
    ref_latency = profiler.do_bench(ref_program, warmup=warmup, rep=rep)

    return tl_latency, ref_latency


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend YourOpName Benchmark")
    parser.add_argument("--M", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--N", type=int, default=4096, help="Matrix dimension N")
    parser.add_argument("--K", type=int, default=4096, help="Matrix dimension K")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Repeat iterations")
    parser.add_argument("--autotune", action="store_true", help="Run autotuning")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    device_type = get_device()

    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: {device_type}")

    if args.autotune:
        print("Running autotuning...")
        if device_type == "npu":
            best_config, kernel = run_npu_autotune(M, N, K)
        else:
            best_config, kernel = run_cuda_autotune(M, N, K)
        print(f"Best config: {best_config}")
        config = best_config
    else:
        config = get_heuristic_config(M, N, K)
        print(f"Heuristic config: {config}")

    # 运行benchmark
    tl_latency, ref_latency = benchmark_single(M, N, K, config, args.warmup, args.rep)

    # 计算性能指标
    flops = calculate_flops(M, N, K)
    tl_tflops = flops / tl_latency * 1e-9
    ref_tflops = flops / ref_latency * 1e-9
    speedup = ref_latency / tl_latency

    # 打印结果
    print("\n" + "=" * 60)
    print("TileLang-Ascend Benchmark Results:")
    print("=" * 60)
    print(f"Device:            {device_type}")
    print(f"Configuration:     {config}")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang TFlops:   {tl_tflops:.2f}")
    print(f"PyTorch Latency:   {ref_latency:.3f} ms")
    print(f"PyTorch TFlops:    {ref_tflops:.2f}")
    print(f"Speedup:           {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
