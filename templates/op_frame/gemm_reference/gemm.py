"""
TileLang-Ascend GEMM参考实现 (NPU专用)

这是一个完整的GEMM（矩阵乘法）实现，可作为其他算子开发的参考。
包含完整的kernel实现、测试和benchmark。

所有计算在NPU上执行，精度对比在CPU上进行。

NPU编程规范:
- 使用 T.Kernel(..., is_npu=True) 启动NPU kernel
- 使用 T.alloc_ub 分配 Unified Buffer
- 使用 T.gemm 进行矩阵乘法
- 使用 T.barrier_all() 进行同步
- 使用 T.Scope("V") 指定 Vector Core 作用域
"""

import argparse
import itertools
import torch
import tilelang
import tilelang.language as T


# ============================================================================
# GEMM 实现
# ============================================================================

@tilelang.jit(out_idx=[-1])
def gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 32,
    block_N: int = 32,
    block_K: int = 32,
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
        dtype: 输入数据类型 ("float16", "float32")
        accum_dtype: 累加数据类型 ("float32")

    返回:
        编译后的kernel函数
    """
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2  # 向量化因子

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # NPU kernel: 使用线性索引
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 计算当前block的2D坐标
            bx = cid // n_num
            by = cid % n_num

            # NPU内存分配: 使用 alloc_ub 分配 Unified Buffer
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_K), dtype)
            b_ub = T.alloc_ub((block_K, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), accum_dtype)

            # 初始化累加器
            T.fill(c_ub, 0)

            # K维度循环
            for k in T.serial(K // block_K):
                # Vector Core 作用域
                with T.Scope("V"):
                    # 数据搬运: GM -> UB
                    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, k * block_K], a_ub)
                    T.copy(B[k * block_K, by * block_N], b_ub)

                    # 核间同步
                    T.barrier_all()

                    # 矩阵乘法累加
                    T.gemm(a_ub, b_ub, c_ub)

                    # 核间同步
                    T.barrier_all()

            # 数据搬运: UB -> GM
            with T.Scope("V"):
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


# ============================================================================
# 自动调优配置
# ============================================================================

def get_configs():
    """获取NPU自动调优配置空间"""
    block_M = [32, 64, 128]
    block_N = [32, 64, 128]
    block_K = [32, 64]

    configs = []
    for bm, bn, bk in itertools.product(block_M, block_N, block_K):
        configs.append({
            "block_M": bm,
            "block_N": bn,
            "block_K": bk,
        })
    return configs


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现 (CPU上执行)"""
    return A.cpu() @ B.cpu()


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend GEMM Benchmark (NPU)")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--block_M", type=int, default=32)
    parser.add_argument("--block_N", type=int, default=32)
    parser.add_argument("--block_K", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)

    M, N, K = args.M, args.N, args.K
    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: NPU")

    kernel = gemm(
        M, N, K,
        block_M=args.block_M,
        block_N=args.block_N,
        block_K=args.block_K,
        dtype="float16",
        accum_dtype="float32",
    )

    # 准备输入数据 (NPU)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)

    # 调用kernel
    c = kernel(a, b)

    # 精度对比在CPU上进行
    ref_c = ref_program(a, b)
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
    print(f"Block size:        {args.block_M} x {args.block_N} x {args.block_K}")
    print(f"TileLang Latency:  {tl_latency:.3f} ms")
    print(f"TileLang TFlops:   {tl_tflops:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
