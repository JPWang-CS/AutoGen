"""
TileLang-Ascend 算子模板 - your_op_name (NPU专用)

该文件提供了TileLang-Ascend算子的基础模板。
请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. 输入/输出tensor的shape和dtype
3. kernel计算逻辑
4. 性能优化参数

所有计算在NPU上执行，精度对比在CPU上进行。

参考: https://github.com/tile-ai/tilelang-ascend
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="npuir")
def your_op_name(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    your_op_name算子的TileLang-Ascend NPU实现

    参数说明:
        M, N, K: 矩阵维度
        block_M, block_N, block_K: tiling参数
        num_stages: 流水线深度（0表示不使用流水线）
        dtype: 输入/输出数据类型 ("float16", "float32", "bfloat16")
        accum_dtype: 累加数据类型 ("float32")

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

            # 分配Unified Buffer用于存储tile数据 (NPU专用)
            A_ub = T.alloc_shared((block_M, block_K), dtype)
            B_ub = T.alloc_shared((block_K, block_N), dtype)

            # 分配fragment用于累加结果
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # 使用流水线进行分块计算
            # T.Pipelined实现Cube/Vector core流水线，可以隐藏内存延迟
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # 从global memory拷贝数据到Unified Buffer
                T.copy(A[by * block_M, k * block_K], A_ub)
                T.copy(B[k * block_K, bx * block_N], B_ub)

                # 执行分块矩阵乘法
                # initC参数用于在第一次迭代时初始化累加器
                T.gemm(A_ub, B_ub, C_local, initC=(k == 0))

            # 将结果从fragment拷贝回global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def main():
    """测试入口函数"""
    import torch
    import torch_npu

    torch.npu.set_device(0)

    # 设置测试参数
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32

    # 编译kernel
    kernel = your_op_name(M, N, K, block_M, block_N, block_K)

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
    latency = profiler.do_bench()
    print(f"TileLang-Ascend Latency: {latency:.3f} ms")

    # 计算TFlops
    flops = 2 * M * N * K
    tflops = flops / latency * 1e-9
    print(f"TileLang-Ascend TFlops: {tflops:.2f}")


if __name__ == "__main__":
    main()
