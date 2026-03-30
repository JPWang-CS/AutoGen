"""
TileLang-Ascend Vector Add 算子实现 (NPU专用)

计算: C = A + B
所有计算在NPU上执行，精度对比在CPU上进行。

参考: https://github.com/tile-ai/tilelang-ascend
"""

import tilelang
import tilelang.language as T


# ============================================================================
# NPU版本 Vector Add 实现
# ============================================================================

@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add(
    N: int,
    block_N: int = 256,
    dtype: str = "float16",
):
    """
    Vector Add的TileLang-Ascend NPU实现

    计算: C = A + B

    参数:
        N: 向量长度
        block_N: 每个核处理的元素数量
        dtype: 数据类型 ("float16", "float32", "bfloat16")

    返回:
        编译后的kernel函数
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
            # 分配Unified Buffer (NPU专用)
            A_VEC = T.alloc_ub((block_N,), dtype)
            B_VEC = T.alloc_ub((block_N,), dtype)
            C_VEC = T.alloc_ub((block_N,), dtype)

            # 计算起始索引和尾部大小
            start_idx = cid * block_N
            remaining = shape - start_idx
            tail_size = T.min(block_N, remaining)

            # 从global memory拷贝到Unified Buffer
            T.copy(A[start_idx : start_idx + tail_size], A_VEC[0:tail_size])
            T.copy(B[start_idx : start_idx + tail_size], B_VEC[0:tail_size])

            # NPU专用加法操作
            T.npuir_add(A_VEC, B_VEC, C_VEC)

            # 从Unified Buffer拷贝回global memory
            T.copy(C_VEC[0:tail_size], C[start_idx : start_idx + tail_size])

    return main


@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add_simple(
    N: int,
    block_N: int = 256,
    dtype: str = "float16",
):
    """
    简化版NPU Vector Add

    不处理尾部元素，要求N是block_N的整数倍
    """
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N,), dtype)
            B_VEC = T.alloc_ub((block_N,), dtype)
            C_VEC = T.alloc_ub((block_N,), dtype)

            start_idx = cid * block_N

            T.copy(A[start_idx : start_idx + block_N], A_VEC)
            T.copy(B[start_idx : start_idx + block_N], B_VEC)
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[start_idx : start_idx + block_N])

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
    2D Tensor加法的NPU实现

    计算: C = A + B，其中A, B, C的shape都是(M, N)
    """
    total_blocks = (M // block_M) * (N // block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(total_blocks, is_npu=True) as (cid, _):
            num_blocks_n = N // block_N
            by = cid // num_blocks_n
            bx = cid % num_blocks_n

            # 分配Unified Buffer
            A_TILE = T.alloc_ub((block_M, block_N), dtype)
            B_TILE = T.alloc_ub((block_M, block_N), dtype)
            C_TILE = T.alloc_ub((block_M, block_N), dtype)

            start_row = by * block_M
            start_col = bx * block_N

            # 拷贝数据到UB
            T.copy(A[start_row : start_row + block_M, start_col : start_col + block_N], A_TILE)
            T.copy(B[start_row : start_row + block_M, start_col : start_col + block_N], B_TILE)

            # 执行加法
            T.npuir_add(A_TILE, B_TILE, C_TILE)

            # 写回结果到global memory
            T.copy(C_TILE, C[start_row : start_row + block_M, start_col : start_col + block_N])

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
