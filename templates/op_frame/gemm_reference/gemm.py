"""
TileLang-Ascend GEMM参考实现 (NPU专用)

基于 tilelang-ascend/examples/gemm/example_gemm.py 官方示例。

Expert模式: T.alloc_L1 + T.alloc_L0C + T.gemm_v0 + T.Scope("C")

NPU GEMM 数据流:
  GM --(MTE2)--> L1 --(Cube内部)--> L0A/L0B --(Cube计算)--> L0C --(MTE3)--> GM

关键API:
- T.alloc_L1: Cube数据中转 (L1 Buffer)
- T.alloc_L0C: GEMM输出累加器 (L0C Buffer)
- T.gemm_v0(..., init=True/False): 矩阵乘法 (init控制清零)
- T.Scope("C"): Cube Core作用域
- dtype用字符串: "float16", "float32"
"""

import argparse
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 256,
    K_L1: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    GEMM: C = A @ B

    基于 tilelang-ascend/examples/gemm/example_gemm.py
    """
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def ref_program(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch参考实现"""
    return A @ B


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend GEMM (NPU)")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Device: NPU")

    kernel = gemm(M, N, K, block_M=128, block_N=256, K_L1=64)

    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    c = kernel(a, b)
    ref_c = ref_program(a, b)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    print("\n" + "=" * 60)
    print("TileLang-Ascend GEMM Benchmark Results (NPU)")
    print("=" * 60)


if __name__ == "__main__":
    main()
