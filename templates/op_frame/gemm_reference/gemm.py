"""
TileLang GEMM参考实现

这是一个完整的GEMM（矩阵乘法）实现，可作为其他算子开发的参考。
包含完整的kernel实现、测试和benchmark。
"""

import argparse
import itertools
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner


@tilelang.jit(out_idx=[-1])
def gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """
    高性能GEMM（矩阵乘法）的TileLang实现

    计算: C = A @ B

    参数:
        M: 矩阵A的行数
        N: 矩阵B的列数
        K: 矩阵A的列数（等于矩阵B的行数）
        block_M: M维度的tiling大小
        block_N: N维度的tiling大小
        block_K: K维度的tiling大小
        num_stages: 流水线深度，0表示不使用流水线
        threads: 每个block的线程数
        dtype: 输入/输出数据类型
        accum_dtype: 累加数据类型

    返回:
        编译后的kernel函数
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # 分配shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # 分配fragment用于累加
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 清空累加器
            T.clear(C_local)

            # 使用流水线进行分块计算
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # 从global memory加载到shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # 执行分块矩阵乘法
                T.gemm(A_shared, B_shared, C_local)

            # 将结果写回global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[-1])
def gemm_with_swizzle(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    enable_swizzle: bool = True,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """
    带L2 Cache优化的GEMM实现

    使用swizzle技术优化L2 cache局部性
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # 启用swizzle优化L2 cache
            T.use_swizzle(panel_size=10, enable=enable_swizzle)

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[-1])
def gemm_transposed_b(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """
    支持B矩阵转置的GEMM实现

    计算: C = A @ B^T
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),  # 注意：B的shape是(N, K)
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=True)
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # 使用transpose_B=True进行转置乘法
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def get_configs():
    """获取自动调优配置空间"""
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


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return A @ B


def main():
    parser = argparse.ArgumentParser(description="TileLang GEMM Benchmark")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--block_K", type=int, default=32)
    parser.add_argument("--num_stages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K

    if args.autotune:
        print("Running autotuning...")

        def kernel_fn(block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
            @T.prim_func
            def main(
                A: T.Tensor((M, K), T.float16),
                B: T.Tensor((K, N), T.float16),
                C: T.Tensor((M, N), T.float16),
            ):
                with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                    A_shared = T.alloc_shared((block_M, block_K), T.float16)
                    B_shared = T.alloc_shared((block_K, block_N), T.float16)
                    C_local = T.alloc_fragment((block_M, block_N), T.float32)
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, bx * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, C[by * block_M, bx * block_N])
            return main

        autotuner = (
            AutoTuner.from_kernel(kernel=kernel_fn, configs=get_configs())
            .set_compile_args(out_idx=[-1], target="auto")
            .set_profile_args(
                supply_type=tilelang.TensorSupplyType.Integer,
                ref_prog=ref_program,
                skip_check=False,
            )
        )
        result = autotuner.run(warmup=3, rep=20)
        print(f"Best config: {result.config}")
        kernel = result.kernel
    else:
        kernel = gemm(
            M, N, K,
            block_M=args.block_M,
            block_N=args.block_N,
            block_K=args.block_K,
            num_stages=args.num_stages,
            threads=args.threads,
        )

    # 准备输入数据
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # 调用kernel
    c = kernel(a, b)

    # 正确性验证
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()

    tl_latency = profiler.do_bench(warmup=args.warmup, rep=args.rep)
    ref_latency = profiler.do_bench(ref_program, warmup=args.warmup, rep=args.rep)

    # 计算性能指标
    flops = 2 * M * N * K
    tl_tflops = flops / tl_latency * 1e-9
    ref_tflops = flops / ref_latency * 1e-9
    speedup = ref_latency / tl_latency

    print("\n" + "=" * 50)
    print("Benchmark Results:")
    print("=" * 50)
    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"TileLang Latency: {tl_latency:.3f} ms")
    print(f"TileLang TFlops:  {tl_tflops:.2f}")
    print(f"PyTorch Latency:  {ref_latency:.3f} ms")
    print(f"PyTorch TFlops:   {ref_tflops:.2f}")
    print(f"Speedup:          {speedup:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
