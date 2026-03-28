"""
TileLang-Ascend GEMM参考实现

这是一个完整的GEMM（矩阵乘法）实现，可作为其他算子开发的参考。
包含完整的kernel实现、测试和benchmark。

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)
"""

import argparse
import itertools
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner


def get_device():
    """
    自动检测可用设备

    返回:
        "npu" - 华为昇腾NPU
        "cuda" - NVIDIA GPU
        "cpu" - CPU（后备）
    """
    try:
        import torch_npu
        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def setup_device(device=None):
    """设置设备环境"""
    if device is None:
        device = get_device()

    if device == "npu":
        torch.npu.set_device(0)
    elif device == "cuda":
        torch.cuda.set_device(0)

    return device


# ============================================================================
# NPU版本GEMM实现
# ============================================================================

@tilelang.jit(out_idx=[-1], target="npuir")
def gemm_npu(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
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
        # NPU kernel: 使用线性索引模式
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            # 计算二维block索引
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            # 分配shared memory
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # 分配fragment用于累加
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 使用流水线进行分块计算
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # 从global memory加载到shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # 执行分块矩阵乘法
                # initC参数: 第一次迭代时初始化累加器为0
                T.gemm(A_shared, B_shared, C_local, initC=(k == 0))

            # 将结果写回global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def gemm_npu_with_ub(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """
    使用Unified Buffer的NPU GEMM实现

    展示T.alloc_ub的使用方法
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
            A_ub = T.alloc_ub((block_M, block_K), dtype)
            B_ub = T.alloc_ub((block_K, block_N), dtype)
            C_ub = T.alloc_ub((block_M, block_N), accum_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_ub)
                T.copy(B[k * block_K, bx * block_N], B_ub)
                T.gemm(A_ub, B_ub, C_ub, initC=(k == 0))

            T.copy(C_ub, C[by * block_M, bx * block_N])

    return main


# ============================================================================
# CUDA版本GEMM实现（后备）
# ============================================================================

@tilelang.jit(out_idx=[-1], target="cuda")
def gemm_cuda(
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
    高性能GEMM的CUDA实现（后备）

    用于NPU不可用时的CUDA后备实现
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
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


@tilelang.jit(out_idx=[-1], target="cuda")
def gemm_cuda_with_swizzle(
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
    带L2 Cache优化的CUDA GEMM实现

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


# ============================================================================
# 统一接口
# ============================================================================

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
    device: str = None,
):
    """
    统一的GEMM接口，自动选择NPU或CUDA实现

    参数:
        M, N, K: 矩阵维度
        block_M, block_N, block_K: tiling参数
        num_stages: 流水线深度
        threads: CUDA线程数（仅CUDA版本使用）
        dtype: 输入/输出数据类型
        accum_dtype: 累加数据类型
        device: 指定设备，None则自动检测

    返回:
        编译后的kernel函数
    """
    if device is None:
        device = get_device()

    if device == "npu":
        return gemm_npu(M, N, K, block_M, block_N, block_K, num_stages, dtype, accum_dtype)
    else:
        return gemm_cuda(M, N, K, block_M, block_N, block_K, num_stages, threads, dtype, accum_dtype)


# ============================================================================
# 自动调优配置
# ============================================================================

def get_npu_configs():
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


def get_cuda_configs():
    """获取CUDA自动调优配置空间"""
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
    parser = argparse.ArgumentParser(description="TileLang-Ascend GEMM Benchmark")
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
    parser.add_argument("--device", type=str, choices=["npu", "cuda", "auto"], default="auto")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    device = args.device if args.device != "auto" else get_device()

    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: {device}")

    # 设置设备
    setup_device(device)

    if args.autotune:
        print("Running autotuning...")

        if device == "npu":

            def kernel_fn(block_M=None, block_N=None, block_K=None, num_stages=None):
                @T.prim_func
                def main(
                    A: T.Tensor((M, K), T.float16),
                    B: T.Tensor((K, N), T.float16),
                    C: T.Tensor((M, N), T.float16),
                ):
                    with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
                        by = cid // T.ceildiv(N, block_N)
                        bx = cid % T.ceildiv(N, block_N)
                        A_shared = T.alloc_shared((block_M, block_K), T.float16)
                        B_shared = T.alloc_shared((block_K, block_N), T.float16)
                        C_local = T.alloc_fragment((block_M, block_N), T.float32)
                        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                            T.copy(A[by * block_M, k * block_K], A_shared)
                            T.copy(B[k * block_K, bx * block_N], B_shared)
                            T.gemm(A_shared, B_shared, C_local, initC=(k == 0))
                        T.copy(C_local, C[by * block_M, bx * block_N])
                return main

            autotuner = (
                AutoTuner.from_kernel(kernel=kernel_fn, configs=get_npu_configs())
                .set_compile_args(out_idx=[-1], target="npuir")
                .set_profile_args(
                    supply_type=tilelang.TensorSupplyType.Integer,
                    ref_prog=ref_program,
                    skip_check=False,
                )
            )

        else:

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
                AutoTuner.from_kernel(kernel=kernel_fn, configs=get_cuda_configs())
                .set_compile_args(out_idx=[-1], target="cuda")
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
        if device == "npu":
            kernel = gemm_npu(
                M, N, K,
                block_M=args.block_M,
                block_N=args.block_N,
                block_K=args.block_K,
                num_stages=args.num_stages,
            )
        else:
            kernel = gemm_cuda(
                M, N, K,
                block_M=args.block_M,
                block_N=args.block_N,
                block_K=args.block_K,
                num_stages=args.num_stages,
                threads=args.threads,
            )

    # 准备输入数据
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)

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

    print("\n" + "=" * 60)
    print("TileLang-Ascend GEMM Benchmark Results:")
    print("=" * 60)
    print(f"Device:            {device}")
    print(f"Problem size:      M={M}, N={N}, K={K}")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang TFlops:   {tl_tflops:.2f}")
    print(f"PyTorch Latency:   {ref_latency:.3f} ms")
    print(f"PyTorch TFlops:    {ref_tflops:.2f}")
    print(f"Speedup:           {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
