"""
TileLang Vector Add 性能基准测试 (NPU专用)

所有计算在NPU上执行。
"""

import argparse
import torch

from vector_add import vector_add_2d


def run_benchmark(M, N, block_M, block_N, warmup, rep):
    """NPU性能测试"""
    torch.npu.set_device(0)

    kernel = vector_add_2d(M, N, block_M=block_M, block_N=block_N, dtype="float16")

    a = torch.randn(M, N, device="npu", dtype=torch.float16)
    b = torch.randn(M, N, device="npu", dtype=torch.float16)

    # 正确性验证
    c = kernel(a, b)
    ref_c = a.cpu() + b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(warmup=warmup, rep=rep)

    return latency


def main():
    parser = argparse.ArgumentParser(description="TileLang Vector Add Benchmark (NPU)")
    parser.add_argument("--m", type=int, default=4096, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=4096, help="Matrix dimension N")
    parser.add_argument("--block_m", type=int, default=32, help="Block size M")
    parser.add_argument("--block_n", type=int, default=32, help="Block size N")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Repeat iterations")
    args = parser.parse_args()

    M, N = args.m, args.n
    print(f"Problem size: M={M}, N={N}")
    print(f"Device: NPU")

    latency = run_benchmark(M, N, args.block_m, args.block_n, args.warmup, args.rep)

    # 计算性能指标
    bytes_moved = 3 * M * N * 2  # float16 = 2 bytes, 读2+写1
    bandwidth_gbs = bytes_moved / latency * 1e-6

    # 打印结果
    print("\n" + "=" * 60)
    print("TileLang Vector Add Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Device:            NPU")
    print(f"Problem size:      M={M}, N={N}")
    print(f"Block size:        {args.block_m} x {args.block_n}")
    print(f"Latency:           {latency:.3f} ms")
    print(f"Bandwidth:         {bandwidth_gbs:.2f} GB/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
