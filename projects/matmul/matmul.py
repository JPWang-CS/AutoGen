"""
TileLang-Ascend Matmul 算子实现 (NPU专用)
支持 A @ B = C，A/B 可选转置
"""
import argparse
import torch
import tilelang
import tilelang.language as T


def ref_program(A, B, transpose_A=False, transpose_B=False):
    """PyTorch参考实现 (CPU上执行)"""
    a = A.cpu().T if transpose_A else A.cpu()
    b = B.cpu().T if transpose_B else B.cpu()
    return a @ b


@tilelang.jit(out_idx=[-1])
def matmul(
    M: int,
    N: int,
    K: int,
    block_M: int = 32,
    block_N: int = 32,
    block_K: int = 32,
    transpose_A: bool = False,
    transpose_B: bool = False,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    通用矩阵乘法 (NPU版本)
    C = A @ B，支持A/B转置

    A shape: (M, K) -> transpose_A=False, 或 (K, M) -> transpose_A=True
    B shape: (K, N) -> transpose_B=False, 或 (N, K) -> transpose_B=True
    C shape: (M, N)
    """
    m_num = M // block_M
    n_num = N // block_N
    k_num = K // block_K
    VEC_NUM = 2

    # 根据转置确定GM中A/B的实际shape
    a_shape = (K, M) if transpose_A else (M, K)
    b_shape = (N, K) if transpose_B else (K, N)

    # UB分配shape：需匹配T.gemm对transpose参数的维度推导
    # T.gemm(transpose_A=True) 读 A_shape[-1]=M, A_shape[-2]=K
    # 所以转置时a_ub的shape要反过来
    a_ub_shape = (block_K, block_M // VEC_NUM) if transpose_A else (block_M // VEC_NUM, block_K)
    b_ub_shape = (block_N, block_K) if transpose_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_ub(a_ub_shape, dtype)
            b_ub = T.alloc_ub(b_ub_shape, dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), accum_dtype)

            T.fill(c_ub, 0)

            for k in T.serial(k_num):
                with T.Scope("V"):
                    # 数据搬运：根据转置状态选择不同的地址计算
                    if not transpose_A:
                        # A: (M, K) -> 取行 [bx*BM : bx*BM+BM, k*BK : k*BK+BK]
                        T.copy(A[bx * block_M + vid * block_M // VEC_NUM, k * block_K], a_ub)
                    else:
                        # A: (K, M) -> 取行 [k*BK : k*BK+BK, bx*BM : bx*BM+BM]
                        T.copy(A[k * block_K, bx * block_M + vid * block_M // VEC_NUM], a_ub)

                    if not transpose_B:
                        # B: (K, N) -> 取行 [k*BK : k*BK+BK, by*BN : by*BN+BN]
                        T.copy(B[k * block_K, by * block_N], b_ub)
                    else:
                        # B: (N, K) -> 取行 [by*BN : by*BN+BN, k*BK : k*BK+BK]
                        T.copy(B[by * block_N, k * block_K], b_ub)

                    T.barrier_all()

                    # 矩阵乘法累加
                    T.gemm(a_ub, b_ub, c_ub, transpose_A=transpose_A, transpose_B=transpose_B)

                    T.barrier_all()

            with T.Scope("V"):
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def main_func():
    """测试入口函数"""
    torch.npu.set_device(0)

    print("Device: NPU")
    print("=" * 60)

    # =========================================================================
    # 测试 1: 基础 GEMM (无转置)
    # =========================================================================
    M, N, K = 512, 512, 512
    print(f"\n[Matmul NN] M={M}, N={N}, K={K}")
    kernel_nn = matmul(M, N, K, block_M=32, block_N=32, block_K=32,
                       transpose_A=False, transpose_B=False)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)
    c = kernel_nn(a, b)
    ref_c = a.cpu() @ b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Matmul NN passed!")
    profiler = kernel_nn.get_profiler()
    latency = profiler.do_bench()
    flops = 2 * M * N * K
    tflops = flops / latency * 1e-9
    print(f"Latency: {latency:.3f} ms, TFlops: {tflops:.2f}")

    # =========================================================================
    # 测试 2: A 转置 (TN)
    # =========================================================================
    print(f"\n[Matmul TN] M={M}, N={N}, K={K}")
    kernel_tn = matmul(M, N, K, block_M=32, block_N=32, block_K=32,
                       transpose_A=True, transpose_B=False)
    a_t = torch.randn(K, M, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)
    c = kernel_tn(a_t, b)
    ref_c = a_t.cpu().T @ b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Matmul TN passed!")
    profiler = kernel_tn.get_profiler()
    latency = profiler.do_bench()
    tflops = flops / latency * 1e-9
    print(f"Latency: {latency:.3f} ms, TFlops: {tflops:.2f}")

    # =========================================================================
    # 测试 3: B 转置 (NT)
    # =========================================================================
    print(f"\n[Matmul NT] M={M}, N={N}, K={K}")
    kernel_nt = matmul(M, N, K, block_M=32, block_N=32, block_K=32,
                       transpose_A=False, transpose_B=True)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b_t = torch.randn(N, K, device="npu", dtype=torch.float16)
    c = kernel_nt(a, b_t)
    ref_c = a.cpu() @ b_t.cpu().T
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Matmul NT passed!")
    profiler = kernel_nt.get_profiler()
    latency = profiler.do_bench()
    tflops = flops / latency * 1e-9
    print(f"Latency: {latency:.3f} ms, TFlops: {tflops:.2f}")

    # =========================================================================
    # 测试 4: AB 都转置 (TT)
    # =========================================================================
    print(f"\n[Matmul TT] M={M}, N={N}, K={K}")
    kernel_tt = matmul(M, N, K, block_M=32, block_N=32, block_K=32,
                       transpose_A=True, transpose_B=True)
    a_t = torch.randn(K, M, device="npu", dtype=torch.float16)
    b_t = torch.randn(N, K, device="npu", dtype=torch.float16)
    c = kernel_tt(a_t, b_t)
    ref_c = a_t.cpu().T @ b_t.cpu().T
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Matmul TT passed!")
    profiler = kernel_tt.get_profiler()
    latency = profiler.do_bench()
    tflops = flops / latency * 1e-9
    print(f"Latency: {latency:.3f} ms, TFlops: {tflops:.2f}")

    # =========================================================================
    # 测试 5: 大尺寸 + 不同block配置
    # =========================================================================
    M, N, K = 1024, 1024, 1024
    print(f"\n[Matmul NN Large] M={M}, N={N}, K={K}, block=64x64x32")
    kernel_large = matmul(M, N, K, block_M=64, block_N=64, block_K=32,
                          transpose_A=False, transpose_B=False)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)
    c = kernel_large(a, b)
    ref_c = a.cpu() @ b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Matmul NN Large passed!")
    profiler = kernel_large.get_profiler()
    latency = profiler.do_bench()
    flops = 2 * M * N * K
    tflops = flops / latency * 1e-9
    print(f"Latency: {latency:.3f} ms, TFlops: {tflops:.2f}")

    # =========================================================================
    # 测试 6: 非方阵
    # =========================================================================
    M, N, K = 1024, 512, 256
    print(f"\n[Matmul NN NonSquare] M={M}, N={N}, K={K}")
    kernel_ns = matmul(M, N, K, block_M=32, block_N=32, block_K=32,
                       transpose_A=False, transpose_B=False)
    a = torch.randn(M, K, device="npu", dtype=torch.float16)
    b = torch.randn(K, N, device="npu", dtype=torch.float16)
    c = kernel_ns(a, b)
    ref_c = a.cpu() @ b.cpu()
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Matmul NN NonSquare passed!")
    profiler = kernel_ns.get_profiler()
    latency = profiler.do_bench()
    flops = 2 * M * N * K
    tflops = flops / latency * 1e-9
    print(f"Latency: {latency:.3f} ms, TFlops: {tflops:.2f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main_func()
