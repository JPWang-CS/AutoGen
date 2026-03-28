"""
TileLang-Ascend算子模板 - your_op_name

该文件提供了一个TileLang-Ascend算子的基础模板。
请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. 输入/输出tensor的shape和dtype
3. kernel计算逻辑
4. 性能优化参数

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)
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
    """
    设置设备环境

    参数:
        device: 指定设备，None则自动检测

    返回:
        设备字符串
    """
    if device is None:
        device = get_device()

    import torch
    if device == "npu":
        torch.npu.set_device(0)
    elif device == "cuda":
        torch.cuda.set_device(0)

    return device


@tilelang.jit(out_idx=[-1], target="npuir")  # 默认使用npuir后端
def your_op_name(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """
    your_op_name算子的TileLang-Ascend实现

    参数说明:
        M, N, K: 矩阵维度
        block_M, block_N, block_K: tiling参数
        num_stages: 流水线深度（0表示不使用流水线）
        dtype: 输入/输出数据类型
        accum_dtype: 累加数据类型

    返回:
        编译后的kernel函数
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),      # 输入tensor A
        B: T.Tensor((K, N), dtype),      # 输入tensor B
        C: T.Tensor((M, N), dtype),      # 输出tensor C
    ):
        # NPU kernel启动模式：使用线性索引
        # is_npu=True 标识这是一个NPU kernel
        with T.Kernel(T.ceildiv(N, block_N) * T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            # 计算二维block索引
            by = cid // T.ceildiv(N, block_N)
            bx = cid % T.ceildiv(N, block_N)

            # 分配shared memory用于存储tile数据
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            # 分配fragment（寄存器）用于累加结果
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 使用流水线进行分块计算
            # T.Pipelined实现Cube/Vector core流水线，可以隐藏内存延迟
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # 从global memory拷贝数据到shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)

                # 执行分块矩阵乘法
                # initC参数用于在第一次迭代时初始化累加器
                T.gemm(A_shared, B_shared, C_local, initC=(k == 0))

            # 将结果从fragment拷贝回global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def your_op_name_cuda(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """
    your_op_name算子的CUDA版本（备选）

    用于NPU不可用时的CUDA后备实现
    """

    @tilelang.jit(out_idx=[-1], target="cuda")
    def kernel():
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

    return kernel()


def main():
    """测试入口函数"""
    import torch

    # 设置测试参数
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32

    # 自动检测并设置设备
    device = setup_device()
    print(f"Using device: {device}")

    # 编译kernel
    if device == "npu":
        kernel = your_op_name(M, N, K, block_M, block_N, block_K)
    else:
        kernel = your_op_name_cuda(M, N, K, block_M, block_N, block_K)

    # 准备输入数据
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)

    # 调用kernel
    c = kernel(a, b)

    # 参考实现
    ref_c = a @ b

    # 验证结果
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"TileLang-Ascend Latency: {latency:.3f} ms")

    # 计算TFlops
    flops = 2 * M * N * K
    tflops = flops / latency * 1e-9
    print(f"TileLang-Ascend TFlops: {tflops:.2f}")


if __name__ == "__main__":
    main()
