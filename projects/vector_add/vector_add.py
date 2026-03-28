"""
TileLang-Ascend Vector Add算子实现

这是一个向量加法算子实现，展示了TileLang-Ascend的基本使用方法。
计算: C = A + B

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)

参考: https://github.com/tile-ai/tilelang-ascend
"""

import tilelang
import tilelang.language as T


def get_device():
    """
    自动检测可用设备

    返回:
        "npu" - 华为昇腾NPU
        "cuda" - NVIDIA GPU
        "cpu" - CPU（后备）
    """
    import torch
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

    import torch
    if device == "npu":
        torch.npu.set_device(0)
    elif device == "cuda":
        torch.cuda.set_device(0)

    return device


# ============================================================================
# NPU版本 Vector Add 实现
# ============================================================================

@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add_npu(
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
        dtype: 数据类型

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
        # NPU kernel: 使用is_npu=True
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # 分配Unified Buffer (NPU专用)
            A_VEC = T.alloc_ub((block_N), dtype)
            B_VEC = T.alloc_ub((block_N), dtype)
            C_VEC = T.alloc_ub((block_N), dtype)

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
def vector_add_npu_simple(
    N: int,
    dtype: str = "float16",
):
    """
    简化版NPU Vector Add

    使用更简单的方式实现，不处理尾部元素
    """

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        # 假设N是block_N的整数倍
        with T.Kernel(N // 256, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((256), dtype)
            B_VEC = T.alloc_ub((256), dtype)
            C_VEC = T.alloc_ub((256), dtype)

            start_idx = cid * 256

            T.copy(A[start_idx : start_idx + 256], A_VEC)
            T.copy(B[start_idx : start_idx + 256], B_VEC)
            T.npuir_add(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[start_idx : start_idx + 256])

    return main


# ============================================================================
# CUDA版本 Vector Add 实现（后备）
# ============================================================================

@tilelang.jit(out_idx=[-1], target="cuda")
def vector_add_cuda(
    N: int,
    block_size: int = 256,
    threads: int = 256,
    dtype: T.dtype = T.float16,
):
    """
    Vector Add的CUDA实现

    用于NPU不可用时的CUDA后备实现
    """

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=threads) as (bx,):
            # 每个线程处理多个元素
            for i in T.Parallel(block_size):
                idx = bx * block_size + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]

    return main


@tilelang.jit(out_idx=[-1], target="cuda")
def vector_add_cuda_simple(
    N: int,
    dtype: T.dtype = T.float16,
):
    """
    简化版CUDA Vector Add

    使用1D kernel，每个线程处理一个元素
    """

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(N, threads=256) as (i,):
            C[i] = A[i] + B[i]

    return main


# ============================================================================
# 2D Tensor 加法实现
# ============================================================================

@tilelang.jit(out_idx=[-1], target="npuir")
def vector_add_2d_npu(
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
            # 计算二维索引
            num_blocks_n = N // block_N
            by = cid // num_blocks_n
            bx = cid % num_blocks_n

            # 分配Unified Buffer
            A_TILE = T.alloc_ub((block_M, block_N), dtype)
            B_TILE = T.alloc_ub((block_M, block_N), dtype)
            C_TILE = T.alloc_ub((block_M, block_N), dtype)

            start_row = by * block_M
            start_col = bx * block_N

            # 拷贝数据
            T.copy(A[start_row : start_row + block_M, start_col : start_col + block_N], A_TILE)
            T.copy(B[start_row : start_row + block_M, start_col : start_col + block_N], B_TILE)

            # 执行加法
            T.npuir_add(A_TILE, B_TILE, C_TILE)

            # 写回结果
            T.copy(C_TILE, C[start_row : start_row + block_M, start_col : start_col + block_N])

    return main


@tilelang.jit(out_idx=[-1], target="cuda")
def vector_add_2d_cuda(
    M: int,
    N: int,
    block_M: int = 16,
    block_N: int = 16,
    threads: int = 256,
    dtype: T.dtype = T.float16,
):
    """
    2D Tensor加法的CUDA实现
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                row = by * block_M + i
                col = bx * block_N + j
                if row < M and col < N:
                    C[row, col] = A[row, col] + B[row, col]

    return main


# ============================================================================
# 统一接口
# ============================================================================

def vector_add(
    N: int,
    block_size: int = 256,
    threads: int = 256,
    dtype: T.dtype = T.float16,
    device: str = None,
):
    """
    统一的Vector Add接口，自动选择NPU或CUDA实现
    """
    if device is None:
        device = get_device()

    if device == "npu":
        return vector_add_npu(N, block_size, "float16")
    else:
        return vector_add_cuda(N, block_size, threads, dtype)


def vector_add_simple(
    N: int,
    dtype: T.dtype = T.float16,
    device: str = None,
):
    """简化版Vector Add接口"""
    if device is None:
        device = get_device()

    if device == "npu":
        return vector_add_npu_simple(N, "float16")
    else:
        return vector_add_cuda_simple(N, dtype)


def vector_add_2d(
    M: int,
    N: int,
    block_M: int = 16,
    block_N: int = 16,
    threads: int = 256,
    dtype: T.dtype = T.float16,
    device: str = None,
):
    """2D Tensor加法接口"""
    if device is None:
        device = get_device()

    if device == "npu":
        return vector_add_2d_npu(M, N, block_M, block_N, "float16")
    else:
        return vector_add_2d_cuda(M, N, block_M, block_N, threads, dtype)


def main():
    """测试入口函数"""
    import torch

    # 测试参数
    N = 1024 * 1024  # 1M elements

    # 自动检测并设置设备
    device = setup_device()
    print(f"Using device: {device}")

    # 编译kernel
    if device == "npu":
        # NPU版本需要传入shape参数
        kernel = vector_add_npu(N, block_N=256, dtype="float16")
        kernel_simple = vector_add_npu_simple(N, dtype="float16")
    else:
        kernel = vector_add_cuda(N)
        kernel_simple = vector_add_cuda_simple(N)

    # 准备输入数据
    a = torch.randn(N, device=device, dtype=torch.float16)
    b = torch.randn(N, device=device, dtype=torch.float16)

    # 调用kernel
    if device == "npu":
        # NPU版本需要传入shape参数
        c = kernel(a, b, torch.tensor(N, dtype=torch.int32))
        c_simple = kernel_simple(a, b)
    else:
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
