"""
TileLang Vector Add 算子实现 (NPU)

计算: C = A + B
所有计算在NPU上执行，精度对比在CPU上进行。

参考: tilelang/examples/elementwise/example_elementwise_add.py
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add(
    N: int,
    block_N: int = 256,
    dtype: str = "float16",
):
    """
    Vector Add 的 TileLang NPU 实现

    计算: C = A + B

    参数:
        N: 向量长度
        block_N: 每个核处理的元素数量
        dtype: 数据类型 ("float16", "float32")

    返回:
        编译后的kernel函数

    注意:
        NPU版本需要额外传入shape参数 (torch.tensor(N, dtype=torch.int32))
    """
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        shape: T.int32,
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_local = T.alloc_shared((block_N,), dtype)
            B_local = T.alloc_shared((block_N,), dtype)
            C_local = T.alloc_fragment((block_N,), dtype)

            start_idx = cid * block_N
            remaining = shape - start_idx
            tail_size = T.min(block_N, remaining)

            T.copy(A[start_idx : start_idx + tail_size], A_local[0:tail_size])
            T.copy(B[start_idx : start_idx + tail_size], B_local[0:tail_size])

            for i in T.Parallel(block_N):
                C_local[i] = A_local[i] + B_local[i]

            T.copy(C_local[0:tail_size], C[start_idx : start_idx + tail_size])

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add_simple(
    N: int,
    block_N: int = 256,
    dtype: str = "float16",
):
    """
    简化版 NPU Vector Add

    不处理尾部元素，要求 N 是 block_N 的整数倍
    """
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_local = T.alloc_shared((block_N,), dtype)
            B_local = T.alloc_shared((block_N,), dtype)
            C_local = T.alloc_fragment((block_N,), dtype)

            start_idx = cid * block_N

            T.copy(A[start_idx : start_idx + block_N], A_local)
            T.copy(B[start_idx : start_idx + block_N], B_local)

            for i in T.Parallel(block_N):
                C_local[i] = A_local[i] + B_local[i]

            T.copy(C_local, C[start_idx : start_idx + block_N])

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add_2d(
    M: int,
    N: int,
    block_M: int = 16,
    block_N: int = 16,
    dtype: str = "float16",
):
    """
    2D Tensor 加法的 NPU 实现

    计算: C = A + B，其中 A, B, C 的 shape 都是 (M, N)
    """
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), is_npu=True) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)

            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def main():
    """测试入口函数"""
    import torch
    import torch_npu

    torch.npu.set_device(0)

    # 测试参数
    N = 1024 * 1024  # 1M elements

    # 编译kernel
    kernel = vector_add(N, block_N=256, dtype="float16")
    kernel_simple = vector_add_simple(N, block_N=256, dtype="float16")

    # 准备输入数据 (NPU)
    a = torch.randn(N, device="npu", dtype=torch.float16)
    b = torch.randn(N, device="npu", dtype=torch.float16)

    # 调用kernel
    c = kernel(a, b, torch.tensor(N, dtype=torch.int32))
    c_simple = kernel_simple(a, b)

    # 精度对比在CPU上进行
    ref_c = a.cpu() + b.cpu()

    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(c_simple.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"Vector Add Latency: {latency:.3f} ms")

    # 计算带宽
    bytes_moved = 3 * N * 2  # float16 = 2 bytes, 读2+写1
    bandwidth_gbs = bytes_moved / latency * 1e-6
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")


if __name__ == "__main__":
    main()
