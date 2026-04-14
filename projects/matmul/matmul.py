"""
Basic GEMM (Matrix Multiplication) on NPU using TileLang-Ascend.

Uses Cube Core operations:
  - T.alloc_L1()   : L1 buffer for data staging
  - T.alloc_L0C()  : L0C buffer for GEMM accumulator output
  - T.gemm_v0()    : Cube Core matrix multiply
  - T.Scope("C")   : Cube scope

Based on: tilelang-ascend/examples/gemm/example_gemm.py
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    """
    Compute C = A @ B on NPU Cube Core.

    Args:
        M, N, K: Matrix dimensions.
        block_M, block_N: Output tile sizes.
        K_L1: K-dimension tile size for L1 staging buffers.
        dtype: Input data type (e.g. "float16").
        accum_dtype: Accumulator data type (e.g. "float").
    """
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Each Cube Core processes one output tile (block_M x block_N)
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            # L1 staging buffers for A and B tiles
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)

            # L0C accumulator buffer for GEMM output
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            # Cube scope: all ops here run on Cube Core
            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    # Load tiles from global memory to L1
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    T.barrier_all()
                    # GEMM: C_L0 += A_L1 @ B_L1
                    # init=True on first iteration to zero the accumulator
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

                    T.barrier_all()

                # Write result back to global memory
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)

    M, N, K = 1024, 1024, 1024
    block_M, block_N, K_L1 = 128, 256, 64

    # Compile the kernel
    func = matmul(M, N, K, block_M, block_N, K_L1)
    print("Kernel compiled successfully!")

    # Create input tensors on NPU (float16)
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()

    # Run the kernel
    c = func(a, b)

    # Reference result using PyTorch matmul on NPU
    ref_c = a @ b

    # Verify correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Matmul test passed!")
