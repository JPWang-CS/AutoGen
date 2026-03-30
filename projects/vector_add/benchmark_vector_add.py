"""
TileLang-Ascend Vector Add 性能基准测试 (NPU专用)

所有计算在NPU上执行，精度对比在CPU上进行。
"""

import argparse
import torch
import tilelang
import tilelang.language as T

from vector_add import vector_add, vector_add_simple


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现 (CPU上执行)"""
    return A.cpu() + B.cpu()


def calculate_bytes_moved(N: int, dtype_size: int = 2) -> int:
    """
    计算数据移动量（字节）

    读2个向量 + 写1个向量 = 3 * N * sizeof(dtype)
    """
    return 3 * N * dtype_size


def benchmark(N: int, block_N: int, warmup: int, rep: int, simple: bool = False):
    """NPU性能测试"""
    import torch_npu
    torch.npu.set_device(0)

    if simple:
        kernel = vector_add_simple(N, block_N=block_N, dtype="float16")
    else:
        kernel = vector_add(N, block_N=block_N, dtype="float16")

    a = torch.randn(N, device="npu", dtype=torch.float16)
    b = torch.randn(N, device="npu", dtype=torch.float16)

    # 正确性验证 (CPU对比)
    c = kernel(a, b)
    ref_c = a.cpu() + b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    tl_latency = profiler.do_bench(warmup=warmup, rep=rep)

    return tl_latency


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend Vector Add Benchmark (NPU)")
    parser.add_argument("--N", type=int, default=1048576, help="Vector size (default: 1M)")
    parser.add_argument("--block_N", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--simple", action="store_true", help="Use simple version")
    args = parser.parse_args()

    N = args.N
    print(f"Vector size: {N} ({N / 1024 / 1024:.2f}M elements)")
    print(f"Device: NPU (华为昇腾)")

    tl_latency = benchmark(N, args.block_N, args.warmup, args.rep, args.simple)

    # 计算性能指标
    bytes_moved = calculate_bytes_moved(N)
    tl_bandwidth = bytes_moved / tl_latency * 1e-6

    # 打印结果
    print("\n" + "=" * 60)
    print("TileLang-Ascend Vector Add Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Device:            NPU")
    print(f"Vector size:       {N} ({N / 1024 / 1024:.2f}M)")
    print(f"Block size:        {args.block_N}")
    version = "simple" if args.simple else "standard"
    print(f"Version:           {version}")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang Bandwidth: {tl_bandwidth:.2f} GB/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
