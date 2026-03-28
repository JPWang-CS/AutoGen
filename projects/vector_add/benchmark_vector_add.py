"""
TileLang-Ascend Vector Add性能基准测试

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)
"""

import argparse
import torch
import tilelang
import tilelang.language as T

from vector_add import (
    vector_add_npu, vector_add_cuda,
    vector_add_npu_simple, vector_add_cuda_simple,
    setup_device
)


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return A + B


def calculate_bytes_moved(N: int, dtype_size: int = 2) -> int:
    """
    计算数据移动量（字节）

    读2个向量 + 写1个向量 = 3 * N * sizeof(dtype)
    """
    return 3 * N * dtype_size


def benchmark_npu(N: int, block_N: int, warmup: int, rep: int, simple: bool = False):
    """NPU性能测试"""
    if simple:
        kernel = vector_add_npu_simple(N, dtype="float16")
    else:
        kernel = vector_add_npu(N, block_N=block_N, dtype="float16")

    a = torch.randn(N, device="npu", dtype=torch.float16)
    b = torch.randn(N, device="npu", dtype=torch.float16)

    # 正确性验证
    if simple:
        c = kernel(a, b)
    else:
        c = kernel(a, b, torch.tensor(N, dtype=torch.int32))
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    tl_latency = profiler.do_bench(warmup=warmup, rep=rep)
    ref_latency = profiler.do_bench(ref_program, warmup=warmup, rep=rep)

    return tl_latency, ref_latency


def benchmark_cuda(N: int, block_size: int, warmup: int, rep: int, simple: bool = False):
    """CUDA性能测试"""
    if simple:
        kernel = vector_add_cuda_simple(N)
    else:
        kernel = vector_add_cuda(N, block_size=block_size)

    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)

    # 正确性验证
    c = kernel(a, b)
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    tl_latency = profiler.do_bench(warmup=warmup, rep=rep)
    ref_latency = profiler.do_bench(ref_program, warmup=warmup, rep=rep)

    return tl_latency, ref_latency


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend Vector Add Benchmark")
    parser.add_argument("--N", type=int, default=1048576, help="Vector size (default: 1M)")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--simple", action="store_true", help="Use simple version")
    args = parser.parse_args()

    N = args.N
    print(f"Vector size: {N} ({N / 1024 / 1024:.2f}M elements)")

    # 自动检测设备
    device = setup_device()
    print(f"Using device: {device}")

    # 根据设备运行benchmark
    if device == "npu":
        tl_latency, ref_latency = benchmark_npu(
            N, args.block_size, args.warmup, args.rep, args.simple
        )
        print(f"Using block_size={args.block_size}" if not args.simple else "Using simple version")
    else:
        tl_latency, ref_latency = benchmark_cuda(
            N, args.block_size, args.warmup, args.rep, args.simple
        )
        print(f"Using block_size={args.block_size}" if not args.simple else "Using simple version")

    # 计算性能指标
    bytes_moved = calculate_bytes_moved(N)
    tl_bandwidth = bytes_moved / tl_latency * 1e-6
    ref_bandwidth = bytes_moved / ref_latency * 1e-6
    speedup = ref_latency / tl_latency

    # 打印结果
    print("\n" + "=" * 60)
    print("TileLang-Ascend Vector Add Benchmark Results:")
    print("=" * 60)
    print(f"Device:            {device}")
    print(f"Vector size:       {N} ({N / 1024 / 1024:.2f}M)")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang Bandwidth: {tl_bandwidth:.2f} GB/s")
    print(f"PyTorch Latency:   {ref_latency:.3f} ms")
    print(f"PyTorch Bandwidth: {ref_bandwidth:.2f} GB/s")
    print(f"Speedup:           {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
