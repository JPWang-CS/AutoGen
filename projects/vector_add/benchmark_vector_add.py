"""
TileLang Vector Add性能基准测试
"""

import argparse
import torch
import tilelang
import tilelang.language as T

from vector_add import vector_add, vector_add_simple


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return A + B


def calculate_bytes_moved(N: int, dtype_size: int = 2) -> int:
    """
    计算数据移动量（字节）

    读2个向量 + 写1个向量 = 3 * N * sizeof(dtype)
    """
    return 3 * N * dtype_size


def main():
    parser = argparse.ArgumentParser(description="Vector Add Benchmark")
    parser.add_argument("--N", type=int, default=1048576, help="Vector size (default: 1M)")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--simple", action="store_true", help="Use simple version")
    args = parser.parse_args()

    N = args.N
    print(f"Vector size: {N} ({N / 1024 / 1024:.2f}M elements)")

    # 编译kernel
    if args.simple:
        kernel = vector_add_simple(N)
        print("Using simple version")
    else:
        kernel = vector_add(N, block_size=args.block_size)
        print(f"Using block_size={args.block_size}")

    # 准备输入数据
    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)

    # 调用kernel验证正确性
    c = kernel(a, b)
    ref_c = a + b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()

    tl_latency = profiler.do_bench(warmup=args.warmup, rep=args.rep)
    ref_latency = profiler.do_bench(ref_program, warmup=args.warmup, rep=args.rep)

    # 计算性能指标
    bytes_moved = calculate_bytes_moved(N)
    tl_bandwidth = bytes_moved / tl_latency * 1e-6
    ref_bandwidth = bytes_moved / ref_latency * 1e-6
    speedup = ref_latency / tl_latency

    # 打印结果
    print("\n" + "=" * 50)
    print("Benchmark Results:")
    print("=" * 50)
    print(f"Vector size:       {N} ({N / 1024 / 1024:.2f}M)")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang Bandwidth: {tl_bandwidth:.2f} GB/s")
    print(f"PyTorch Latency:   {ref_latency:.3f} ms")
    print(f"PyTorch Bandwidth: {ref_bandwidth:.2f} GB/s")
    print(f"Speedup:           {speedup:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
