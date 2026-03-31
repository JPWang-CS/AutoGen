"""
TileLang-Ascend 算子模板 - your_op_name (NPU专用)

该文件提供了TileLang-Ascend算子的基础模板。
请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. 输入/输出tensor的shape和dtype
3. kernel计算逻辑
4. 性能优化参数

所有计算在NPU上执行，精度对比在CPU上进行。

NPU编程规范:
- 使用 T.Kernel(..., is_npu=True) 启动NPU kernel
- 使用 T.alloc_ub 分配 Unified Buffer
- 使用 T.tile.* 等向量指令进行计算
- 使用 T.barrier_all() 进行同步
- 使用 T.Scope("V") 指定 Vector Core 作用域
"""

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def your_op_name(
    M: int,
    N: int,
    K: int,
    block_M: int = 32,
    block_N: int = 32,
    block_K: int = 32,
    dtype: str = "float16",
):
    """
    your_op_name算子的TileLang-Ascend NPU实现

    参数说明:
        M, N, K: 矩阵维度
        block_M, block_N, block_K: tiling参数
        dtype: 数据类型 ("float16", "float32", "bfloat16")

    返回:
        编译后的kernel函数

    NPU编程要点:
    - 使用线性block索引 (cid)，手动计算2D坐标
    - 使用 alloc_ub 分配 Unified Buffer
    - 使用 T.tile.* 进行向量计算
    - 使用 barrier_all 进行核间同步
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
        # NPU kernel: 使用线性索引，is_npu=True
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 计算当前block的2D坐标
            bx = cid // n_num
            by = cid % n_num

            # NPU内存分配: 使用 alloc_ub 分配 Unified Buffer
            # 每个 vector core 处理 block_M // VEC_NUM 行
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_K), dtype)
            b_ub = T.alloc_ub((block_K, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # Vector Core 作用域
            with T.Scope("V"):
                # 数据搬运: Global Memory -> UB
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # 核间同步
                T.barrier_all()

                # ========================================
                # TODO: 在此处添加你的计算逻辑
                # 示例: 使用 T.tile.* 向量指令
                # T.tile.add(c_ub, a_ub, b_ub)  # 加法
                # T.tile.mul(c_ub, a_ub, b_ub)  # 乘法
                # T.gemm(a_ub, b_ub, c_ub)      # 矩阵乘法
                # ========================================

                # 核间同步
                T.barrier_all()

                # 数据搬运: UB -> Global Memory
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def main():
    """测试入口函数"""
    import torch
    import torch_npu

    torch.npu.set_device(0)

    # 设置测试参数
    M, N, K = 1024, 1024, 1024

    # 编译kernel
    kernel = your_op_name(M, N, K, block_M=32, block_N=32, block_K=32)

    # 准备输入数据 (NPU)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)

    # 调用kernel
    c = kernel(a, b)

    # 精度对比在CPU上进行
    # TODO: 替换为你的参考实现
    ref_c = a.cpu() @ b.cpu()  # 示例: GEMM

    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"TileLang-Ascend Latency: {latency:.3f} ms")

    # 计算性能指标 (示例: GEMM TFlops)
    flops = 2 * M * N * K
    tflops = flops / latency * 1e-9
    print(f"TileLang-Ascend TFlops: {tflops:.2f}")


if __name__ == "__main__":
    main()
