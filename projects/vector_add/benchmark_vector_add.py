"""
TileLang Vector Add 性能基准测试 (NPU专用)

测试 1D/2D/3D Vector Add 算子的性能。
所有计算在NPU上执行。
"""

import argparse
import torch

from vector_add import vector_add_1d, vector_add_2d, vector_add_3d, vector_add_1d_multi_block


def run_benchmark(dtype="float16"):
    """NPU性能测试"""
    torch.npu.set_device(0)

    print("=" * 60)
    print("TileLang Vector Add Benchmark (NPU)")
    print("=" * 60)
    print(f"Dtype: {dtype}")
    print()

    from tilelang.profiler import do_bench

    torch_dtype = getattr(torch, dtype)

    # 1D benchmark - 多种block size对比
    N = 1024 * 1024  # 1M elements
    print(f"[1D] N={N}")
    a1d = torch.randn(N, dtype=torch_dtype, device="npu")
    b1d = torch.randn(N, dtype=torch_dtype, device="npu")

    # 测试不同block size
    block_sizes = [256, 512, 1024, 2048]
    best_bw1d = 0
    best_config1d = None

    for block_n in block_sizes:
        if N % block_n != 0:
            continue
        kernel = vector_add_1d(N, block_N=block_n, dtype=dtype)
        c1d = kernel(a1d, b1d)
        ref1d = a1d.cpu() + b1d.cpu()
        torch.testing.assert_close(c1d.cpu(), ref1d, rtol=1e-2, atol=1e-2)

        latency = do_bench(lambda: kernel(a1d, b1d))
        bytes_moved = 3 * N * 2
        bw = bytes_moved / latency * 1e-6
        print(f"  block_N={block_n}: {latency:.3f} ms, {bw:.2f} GB/s")

        if bw > best_bw1d:
            best_bw1d = bw
            best_config1d = (block_n, latency)

    print(f"  Best: block_N={best_config1d[0]}, {best_config1d[1]:.3f} ms, {best_bw1d:.2f} GB/s")
    print()

    # 2D benchmark - 多种block size对比
    M, N2d = 1024, 1024
    print(f"[2D] M={M}, N={N2d}")
    a2d = torch.randn(M, N2d, dtype=torch_dtype, device="npu")
    b2d = torch.randn(M, N2d, dtype=torch_dtype, device="npu")

    block_configs = [(16, 16), (32, 32), (64, 64), (32, 64), (64, 32)]
    best_bw2d = 0
    best_config2d = None

    for bm, bn in block_configs:
        if M % bm != 0 or N2d % bn != 0:
            continue
        kernel = vector_add_2d(M, N2d, block_M=bm, block_N=bn, dtype=dtype)
        c2d = kernel(a2d, b2d)
        ref2d = a2d.cpu() + b2d.cpu()
        torch.testing.assert_close(c2d.cpu(), ref2d, rtol=1e-2, atol=1e-2)

        latency = do_bench(lambda: kernel(a2d, b2d))
        bytes_moved = 3 * M * N2d * 2
        bw = bytes_moved / latency * 1e-6
        print(f"  block=({bm},{bn}): {latency:.3f} ms, {bw:.2f} GB/s")

        if bw > best_bw2d:
            best_bw2d = bw
            best_config2d = (bm, bn, latency)

    print(f"  Best: block=({best_config2d[0]},{best_config2d[1]}), {best_config2d[2]:.3f} ms, {best_bw2d:.2f} GB/s")
    print()

    # 3D benchmark
    D, M3d, N3d = 64, 128, 128
    print(f"[3D] D={D}, M={M3d}, N={N3d}")
    a3d = torch.randn(D, M3d, N3d, dtype=torch_dtype, device="npu")
    b3d = torch.randn(D, M3d, N3d, dtype=torch_dtype, device="npu")
    kernel3d = vector_add_3d(D, M3d, N3d, block_M=16, block_N=16, dtype=dtype)

    c3d = kernel3d(a3d, b3d)
    ref3d = a3d.cpu() + b3d.cpu()
    torch.testing.assert_close(c3d.cpu(), ref3d, rtol=1e-2, atol=1e-2)
    print("  Correctness check passed!")

    latency3d = do_bench(lambda: kernel3d(a3d, b3d))
    bytes3d = 3 * D * M3d * N3d * 2
    bw3d = bytes3d / latency3d * 1e-6
    print(f"  block=(16,16): {latency3d:.3f} ms, {bw3d:.2f} GB/s")
    print()

    print("=" * 60)
    print("Benchmark Summary:")
    print("=" * 60)
    print(f"1D Best: {best_config1d[1]:.3f} ms, {best_bw1d:.2f} GB/s (block_N={best_config1d[0]})")
    print(f"2D Best: {best_config2d[2]:.3f} ms, {best_bw2d:.2f} GB/s (block=({best_config2d[0]},{best_config2d[1]}))")
    print(f"3D: {latency3d:.3f} ms, {bw3d:.2f} GB/s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TileLang Vector Add Benchmark (NPU)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    run_benchmark(args.dtype)


if __name__ == "__main__":
    main()
