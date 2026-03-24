"""
TileLang Vector Add算子实现

这是一个简单的向量加法算子示例，展示了TileLang的基本使用方法。
计算: C = A + B
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def vector_add(
    N: int,
    block_size: int = 256,
    threads: int = 256,
    dtype: T.dtype = T.float16,
):
    """
    Vector Add的TileLang实现

    参数:
        N: 向量长度
        block_size: 每个block处理的元素数量
        threads: 每个block的线程数
        dtype: 数据类型

    返回:
        编译后的kernel函数
    """

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=threads) as (bx,):
            # 分配shared memory（如果需要）
            # 对于简单的向量加法，直接使用线程索引即可

            # 每个线程处理多个元素
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]

    return main


@tilelang.jit(out_idx=[-1])
def vector_add_simple(
    N: int,
    dtype: T.dtype = T.float16,
):
    """
    简化版本的Vector Add

    使用最简单的方式实现向量加法
    """

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        # 使用1D kernel，每个线程处理一个元素
        with T.Kernel(N, threads=256) as (i,):
            C[i] = A[i] + B[i]

    return main


@tilelang.jit(out_idx=[-1])
def vector_add_2d(
    M: int,
    N: int,
    block_M: int = 16,
    block_N: int = 16,
    threads: int = 256,
    dtype: T.dtype = T.float16,
):
    """
    2D Tensor加法的TileLang实现

    计算: C = A + B，其中A, B, C的shape都是(M, N)

    参数:
        M: tensor的行数
        N: tensor的列数
        block_M: 每个block处理的行数
        block_N: 每个block处理的列数
        threads: 每个block的线程数
        dtype: 数据类型
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # 每个block处理一个tile
            for i, j in T.Parallel(block_M, block_N):
                row = by * block_M + i
                col = bx * block_N + j
                if row < M and col < N:
                    C[row, col] = A[row, col] + B[row, col]

    return main


def main():
    """测试入口函数"""
    import torch

    # 测试参数
    N = 1024 * 1024  # 1M elements

    # 编译kernel
    kernel = vector_add(N)
    kernel_simple = vector_add_simple(N)

    # 准备输入数据
    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)

    # 调用kernel
    c = kernel(a, b)
    c_simple = kernel_simple(a, b)

    # 参考实现
    ref_c = a + b

    # 验证结果
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(c_simple, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"Vector Add Latency: {latency:.3f} ms")

    # 计算带宽
    # 读2个向量 + 写1个向量 = 3 * N * sizeof(dtype)
    bytes_moved = 3 * N * 2  # float16 = 2 bytes
    bandwidth_gbs = bytes_moved / latency * 1e-6
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")


if __name__ == "__main__":
    main()
