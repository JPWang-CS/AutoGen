"""
Elementwise Vector Addition on NPU using TileLang-Ascend.

Uses Vector Core operations:
  - T.alloc_ub()  : Unified Buffer allocation
  - T.Scope("V")  : Vector scope
  - T.tile.add()  : Elementwise addition
  - VEC_NUM=2     : Two Vector Engines per core

Based on: tilelang-ascend/examples/elementwise/elementwise_add.py
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, dtype="float"):
    """
    Compute C = A + B on NPU Vector Cores.

    Args:
        M, N: Matrix dimensions.
        block_M, block_N: Tile sizes per core.
        dtype: Data type (e.g. "float", "float16").
    """
    m_num = M // block_M
    n_num = N // block_N

    # Each core has 2 Vector Engines (AIVs)
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            # Each Vector Engine handles half the block rows
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            # Vector scope: all ops here run on Vector Core
            with T.Scope("V"):
                # Load input tiles from global memory to UB
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)

                T.barrier_all()

                # Elementwise addition: c_ub = a_ub + b_ub
                T.tile.add(c_ub, a_ub, b_ub)

                T.barrier_all()

                # Store result back to global memory
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)

    M, N = 1024, 1024
    block_M, block_N = 128, 256

    # Compile the kernel
    func = vec_add(M, N, block_M, block_N)
    print("Kernel compiled successfully!")

    # Create input tensors on NPU (float32)
    a = torch.randn(M, N).npu()
    b = torch.randn(M, N).npu()

    torch.npu.synchronize()

    # Run the kernel
    c = func(a, b)

    # Reference result
    ref_c = a + b

    # Verify correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Vector Add test passed!")
