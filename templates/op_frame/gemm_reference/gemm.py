"""
TileLang-Ascend GEMM参考实现 (NPU专用)

这是一个完整的GEMM（矩阵乘法）实现，可作为其他算子开发的参考。
包含完整的kernel实现、测试和benchmark。

所有计算在NPU上执行，精度对比在CPU上进行。

参考: https://github.com/tile-ai/tilelang-ascend
"""

import argparse
import itertools
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner


# ============================================================================
# GEMM 实现
# ============================================================================

@tilelang.jit(out_idx=[-1], target="npuir")
def gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    高性能GEMM（矩阵乘法）的TileLang-Ascend NPU实现

    计算: C = A @ B

    参数:
        M: 矩阵A的行数
        N: 矩阵B的列数
        K: 矩阵A的列数（等于矩阵B的行数）
        block_M: M维度的tiling大小
        block_N: N维度的tiling大小
        block_K: K维度的tiling大小
        num_stages: 流水线深度，0表示不使用流水线
        dtype: 输入/输出数据类型 ("float16", "float32", "bfloat16")
        accum_dtype: 累加数据类型 ("float32")

    返回:
        编译后的kernel函数
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # NPU kernel: 使用线性索引模式
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            # 分配shared memory (npuir后端自动映射到UB)
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # 分配fragment用于累加
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 使用流水线进行分块计算
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, initC=(k == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def gemm_with_ub(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    使用Unified Buffer的NPU GEMM实现

    展示T.alloc_shared的使用方法
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            # 使用Unified Buffer (NPU专用)
            A_ub = T.alloc_shared((block_M, block_K), dtype)
            B_ub = T.alloc_shared((block_K, block_N), dtype)
            C_ub = T.alloc_shared((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_ub)
                T.copy(B[k * block_K, bx * block_N], B_ub)
                T.gemm(A_ub, B_ub, C_ub, initC=(k == 0))

            T.copy(C_ub, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def gemm_transposed_b(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    支持B矩阵转置的NPU GEMM实现

    计算: C = A @ B^T，其中B的shape为(N, K)
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),  # 注意: B是转置的
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            A_ub = T.alloc_shared((block_M, block_K), dtype)
            B_ub = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_ub)
                # B是转置的，需要按列读取
                T.copy(B[bx * block_N, k * block_K], B_ub)
                T.gemm(A_ub, B_ub, C_local, initC=(k == 0))

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ============================================================================
# 自动调优配置
# ============================================================================

def get_configs():
    """获取NPU自动调优配置空间"""
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


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现 (CPU上执行)"""
    return A.cpu() @ B.cpu()


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend GEMM Benchmark (NPU)")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--block_M", type=int, default=128)
    parser.add_argument("--block_N", type=int, default=128)
    parser.add_argument("--block_K", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()

    import torch_npu
    torch.npu.set_device(0)

    M, N, K = args.M, args.N, args.K
    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: NPU")

    if args.autotune:
        print("Running autotuning...")

        def kernel_fn(block_M=None, block_N=None, block_K=None, num_stages=None):
            return gemm(M, N, K, block_M, block_N, block_K, num_stages)

        autotuner = (
            AutoTuner.from_kernel(kernel=kernel_fn, configs=get_configs())
            .set_compile_args(out_idx=[-1], target="npuir")
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
        )

    # 准备输入数据 (NPU)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)

    # 调用kernel
    c = kernel(a, b)

    # 精度对比在CPU上进行
    ref_c = a.cpu() @ b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()

    tl_latency = profiler.do_bench(warmup=args.warmup, rep=args.rep)

    # 计算性能指标
    flops = 2 * M * N * K
    tl_tflops = flops / tl_latency * 1e-9

    print("\n" + "=" * 60)
    print("TileLang-Ascend GEMM Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Device:            NPU")
    print(f"Problem size:      M={M}, N={N}, K={K}")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang TFlops:   {tl_tflops:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
