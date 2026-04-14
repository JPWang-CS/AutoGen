"""
Online Softmax on NPU using TileLang-Ascend.

Implements the numerically-stable online softmax algorithm using Vector Core ops:
  - T.alloc_ub()           : Unified Buffer allocation
  - T.Scope("V")           : Vector scope
  - T.reduce_max()         : Row-wise max reduction
  - T.reduce_sum()         : Row-wise sum reduction
  - T.tile.exp/mul/sub/add/div/broadcast/fill/cast : Elementwise ops

Algorithm (online normalizer):
    m_0 = -inf,  s_0 = 0
    for j in [1, N]:
        m_j = max(m_{j-1}, x_j)
        s_j = s_{j-1} * exp(m_{j-1} - m_j) + exp(x_j - m_j)
    for j in [1, N]:
        y_j = exp(x_j - m_N) / s_N

Based on: tilelang-ascend/examples/softmax/example_online_softmax.py
"""

import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

# Developer mode pass configs: enables auto sync and memory planning
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def online_softmax(M, N, block_M, block_N, dtype="float"):
    """
    Compute softmax(A, dim=-1) on NPU Vector Cores using the online algorithm.

    Supports float, float16, and bfloat16 (with float32 compute cast).

    Args:
        M, N: Input matrix dimensions (M rows, N columns).
        block_M, block_N: Tile sizes.
        dtype: Input/output data type.
    """
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    def cast_or_copy(dst, src, mode, count):
        """Cast to float32 for compute, or plain copy if already float32."""
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        # One core processes one block row
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            # Working buffers (divided by VEC_NUM for dual Vector Engines)
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)

            # Online softmax state: running max and running sum
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)

            # Temporary for correction factor exp(m_{j-1} - m_j) * s_{j-1}
            tmp_exp = T.alloc_ub([sub_block_M, 1], cal_dtype)

            # Initialize running state
            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            # ----------------------------------------------------------
            # Pass 1: Compute online max (m_N) and normalizer (s_N)
            # ----------------------------------------------------------
            for by in T.serial(n_num):
                # Load tile from global memory (pad with -inf for tail blocks)
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        by * block_N : (by + 1) * block_N,
                    ],
                    a,
                    pad_value=-T.infinity(cal_dtype),
                )

                # Cast to compute dtype if needed
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                # Compute tile-wise max and merge with running max
                T.reduce_max(a_cal, tile_max, dim=-1)          # max of current tile
                T.tile.max(tile_max, prev_max, tile_max)        # m_j = max(m_{j-1}, x_j)

                # Correction factor: s_{j-1} * exp(m_{j-1} - m_j)
                T.tile.sub(tmp_exp, prev_max, tile_max)         # m_{j-1} - m_j
                T.tile.exp(tmp_exp, tmp_exp)                    # exp(m_{j-1} - m_j)
                T.tile.mul(tmp_exp, prev_sum, tmp_exp)          # s_{j-1} * exp(...)

                # Compute exp(x_j - m_j) and its sum
                T.tile.broadcast(tile_max_2d, tile_max)         # broadcast for sub
                T.tile.sub(a_cal, a_cal, tile_max_2d)           # x_j - m_j
                T.tile.exp(a_cal, a_cal)                        # exp(x_j - m_j)
                T.reduce_sum(a_cal, tile_sum, dim=-1)           # sum of exp(...)

                # Update running sum: s_j = correction + new_sum
                T.tile.add(prev_sum, tile_sum, tmp_exp)

                # Carry forward current max
                T.copy(tile_max, prev_max)

            # ----------------------------------------------------------
            # Pass 2: Compute final softmax output
            #   y_j = exp(x_j - m_N) / s_N
            # ----------------------------------------------------------
            # After pass 1: prev_max = m_N, prev_sum = s_N
            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)

            for by in T.serial(n_num):
                # Reload input tile
                T.copy(
                    A[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        by * block_N : (by + 1) * block_N,
                    ],
                    a,
                )

                # Cast to compute dtype if needed
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                # Compute softmax values
                T.tile.sub(a_cal, a_cal, prev_max_2d)           # x_j - m_N
                T.tile.exp(a_cal, a_cal)                        # exp(x_j - m_N)
                T.tile.div(a_cal, a_cal, prev_sum_2d)           # / s_N

                # Cast back to original dtype if needed
                cast_or_copy(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)

                # Store output
                T.copy(
                    a,
                    B[
                        bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M,
                        by * block_N : (by + 1) * block_N,
                    ],
                )

    return main


if __name__ == "__main__":
    torch.manual_seed(0)

    test_configs = [
        # (M, N, block_M, block_N, dtype)
        (34, 130, 32, 32, "float"),
        (34, 130, 32, 32, "float16"),
        (34, 130, 32, 32, "bfloat16"),
        (1024, 1024, 128, 128, "float"),
        (1024, 1024, 128, 128, "float16"),
        (1024, 1024, 128, 128, "bfloat16"),
    ]

    for M, N, block_M, block_N, dtype in test_configs:
        print(f"Testing online_softmax: M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")

        # Compile kernel
        func = online_softmax(M, N, block_M, block_N, dtype=dtype)
        print("  Kernel compiled!")

        # Create input on NPU
        torch_dtype = getattr(torch, dtype) if dtype != "float" else torch.float32
        a = torch.randn(M, N, dtype=torch_dtype).npu()

        # Run kernel
        b = func(a)

        # Reference: PyTorch softmax
        ref_b = torch.nn.functional.softmax(a, dim=1)

        # Verify
        rtol = 1e-2 if dtype in ["float16", "bfloat16"] else 1e-4
        atol = 1e-3 if dtype in ["float16", "bfloat16"] else 1e-4
        torch.testing.assert_close(b, ref_b, rtol=rtol, atol=atol)
        print("  Test passed!")

    print("\nAll online softmax tests passed!")
