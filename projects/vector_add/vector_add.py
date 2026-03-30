"""
TileLang Vector Add 算子实现 (NPU专用)

计算: C = A + B
所有计算在NPU上执行，精度对比在CPU上进行。

参考: tilelang-ascend NPU编程规范
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def vector_add(N: int, block_N: int = 256, dtype: str = "float16"):
    """
    1D Vector Add 的 TileLang NPU实现

    计算: C = A + B
    要求 N 是 block_N 的整数倍。

    NPU约束:
    - block_N * sizeof(dtype) * 3 <= UB容量 (~2MB)
    - block_N 需满足32字节对齐
    """
    # 使用 2D shape (N, 1) 来适配 NPU 的 2D copy
    @T.prim_func
    def main(
        A: T.Tensor((N, 1), dtype),
        B: T.Tensor((N, 1), dtype),
        C: T.Tensor((N, 1), dtype),
    ):
        with T.Kernel(N // block_N, is_npu=True) as (cid, _):
            # NPU内存分配 - 使用 2D shape
            A_shared = T.alloc_shared((block_N, 1), dtype)
            B_shared = T.alloc_shared((block_N, 1), dtype)
            C_local = T.alloc_fragment((block_N, 1), dtype)

            start_idx = cid * block_N

            # 数据搬运: 使用 2D 索引
            T.copy(A[start_idx, 0], A_shared)
            T.copy(B[start_idx, 0], B_shared)

            # 计算: 逐元素加法
            for i in T.Parallel(block_N):
                C_local[i, 0] = A_shared[i, 0] + B_shared[i, 0]

            # 数据搬运
            T.copy(C_local, C[start_idx, 0])

    return main


# vector_add_simple 与 vector_add 相同
vector_add_simple = vector_add


@tilelang.jit(out_idx=[-1])
def vector_add_2d(M: int, N: int, block_M: int = 16, block_N: int = 16, dtype: str = "float16"):
    """
    2D Tensor 加法 (NPU版本)

    计算: C = A + B，其中 A, B, C 的 shape 都是 (M, N)
    要求 M 是 block_M 的整数倍，N 是 block_N 的整数倍。

    NPU约束: 使用线性block索引，手动计算2D坐标
    """
    num_blocks_m = M // block_M
    num_blocks_n = N // block_N
    total_blocks = num_blocks_m * num_blocks_n

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # NPU kernel 使用单维度block
        with T.Kernel(total_blocks, is_npu=True) as (cid, _):
            # 计算当前block的2D坐标
            by = cid // num_blocks_n
            bx = cid % num_blocks_n

            # NPU内存分配
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            # 数据搬运
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)

            # 计算
            for local_y, local_x in T.Parallel(block_M, block_N):
                C_local[local_y, local_x] = A_shared[local_y, local_x] + B_shared[local_y, local_x]

            # 数据搬运: 直接从 local -> Global Memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def main():
    """测试入口函数 (NPU版本)"""
    import torch
    import torch_npu

    # 设置NPU设备
    torch.npu.set_device(0)

    # 测试参数
    N = 1024 * 1024  # 1M elements

    # 编译 1D kernel (内部使用 2D shape)
    kernel = vector_add(N, block_N=256, dtype="float16")

    # 准备输入数据 (NPU) - reshape to (N, 1) for kernel
    a = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)
    b = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)

    # 调用 kernel
    c = kernel(a, b)

    # 精度对比在CPU上进行 - squeeze back to 1D
    ref_c = a.cpu() + b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("1D Vector Add passed!")

    # 2D 测试
    M_2d, N_2d = 128, 128
    kernel_2d = vector_add_2d(M_2d, N_2d, block_M=16, block_N=16, dtype="float16")
    a_2d = torch.randn(M_2d, N_2d, device="npu", dtype=torch.float16)
    b_2d = torch.randn(M_2d, N_2d, device="npu", dtype=torch.float16)
    c_2d = kernel_2d(a_2d, b_2d)
    ref_c_2d = a_2d.cpu() + b_2d.cpu()
    torch.testing.assert_close(c_2d.cpu(), ref_c_2d, rtol=1e-2, atol=1e-2)
    print("2D Vector Add passed!")

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
