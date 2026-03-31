"""
TileLang Vector Add 算子实现 (NPU专用)

计算: C = A + B
所有计算在NPU上执行，精度对比在CPU上进行。

NPU编程规范:
- 使用 T.Kernel(..., is_npu=True) 启动NPU kernel
- 使用 T.alloc_ub 分配 Unified Buffer
- 使用 T.tile.add 等向量指令进行计算
- 使用 T.barrier_all() 进行同步
- 使用 T.Scope("V") 指定 Vector Core 作用域
"""

import argparse
import torch
import tilelang
import tilelang.language as T


def ref_program(x, y):
    """参考实现 (CPU)"""
    return x + y


@tilelang.jit(out_idx=[-1])
def vector_add_2d(M, N, block_M, block_N, dtype="float16"):
    """
    2D Tensor 加法 (NPU版本)

    计算: C = A + B，其中 A, B, C 的 shape 都是 (M, N)
    要求 M 是 block_M 的整数倍，N 是 block_N 的整数倍。

    NPU编程要点:
    - 使用线性block索引 (cid)，手动计算2D坐标
    - 使用 alloc_ub 分配 Unified Buffer
    - 使用 T.tile.add 进行向量加法
    - 使用 barrier_all 进行核间同步
    """
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2  # 向量化因子，每个block由2个vector core并行处理

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # NPU kernel: 使用线性索引，is_npu=True
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 计算当前block的2D坐标
            bx = cid // n_num
            by = cid % n_num

            # NPU内存分配: 使用 alloc_ub 分配 Unified Buffer
            # 注意: 每个 vector core 处理 block_M // VEC_NUM 行
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # Vector Core 作用域
            with T.Scope("V"):
                # 数据搬运: Global Memory -> UB
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                # 核间同步
                T.barrier_all()

                # 计算: 使用向量加法指令
                T.tile.add(c_ub, a_ub, b_ub)

                # 核间同步
                T.barrier_all()

                # 数据搬运: UB -> Global Memory
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def main_func(M=1024, N=1024):
    """测试入口函数"""
    # 设置NPU设备
    torch.npu.set_device(0)

    print(f"Problem size: M={M}, N={N}")
    print(f"Device: NPU")

    a = torch.randn(M, N, dtype=torch.float16, device="npu")
    b = torch.randn(M, N, dtype=torch.float16, device="npu")

    kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

    out = kernel(a, b)

    # 精度对比在CPU上进行
    ref_c = ref_program(a.cpu(), b.cpu())
    torch.testing.assert_close(out.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("2D Vector Add passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"Vector Add Latency: {latency:.3f} ms")

    # 计算带宽
    bytes_moved = 3 * M * N * 2  # float16 = 2 bytes, 读2+写1
    bandwidth_gbs = bytes_moved / latency * 1e-6
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")


def run_benchmark():
    """性能基准测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    args, _ = parser.parse_known_args()
    M, N = args.m, args.n

    torch.npu.set_device(0)

    a = torch.randn(M, N, dtype=torch.float16, device="npu")
    b = torch.randn(M, N, dtype=torch.float16, device="npu")

    kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

    from tilelang.profiler import do_bench
    latency = do_bench(lambda: kernel(a, b))

    print(f"M={M}, N={N}, Latency: {latency:.3f} ms")

    bytes_moved = 3 * M * N * 2
    bandwidth_gbs = bytes_moved / latency * 1e-6
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true")
    args, _ = parser.parse_known_args()

    if args.benchmark:
        run_benchmark()
    else:
        main_func(args.m, args.n)
