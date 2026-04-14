"""
Layer Normalization on NPU using TileLang-Ascend.

Uses Vector Core operations:
  - T.alloc_ub()          : Unified Buffer allocation
  - T.Scope("V")          : Vector scope
  - T.reduce_sum()        : Row-wise sum reduction
  - T.tile.fill/add/mul/sub/div/sqrt/broadcast/cast : Elementwise ops

Computes:
    mean_i = (1/N) * sum(x_i)
    var_i  = (1/N) * sum((x_i - mean)^2) - mean^2 + mean^2
    y_i    = (x_i - mean) / sqrt(var + eps)

Based on: tilelang-ascend/examples/normalization/layer_norm.py
"""

import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def layer_norm(M, N, block_M, block_N, eps=1e-5, dtype="float"):
    """
    Compute Layer Normalization on NPU Vector Cores.

    Normalizes each row of the (M, N) input tensor.
    Supports float, float16, and bfloat16 (with float32 compute cast).

    Args:
        M, N: Input matrix dimensions (M rows, N features per row).
        block_M, block_N: Tile sizes.
        eps: Epsilon for numerical stability.
        dtype: Input/output data type.
    """
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    def cast_or_copy(dst, src, mode, count):
        """Cast to float32 for compute, or plain copy if already float32."""
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid

            # Working buffers (divided by VEC_NUM for dual Vector Engines)
            a_ub = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)

            # Accumulators for sum and sum-of-squares
            sum_i = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            sum_square_i = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            sum_ub = T.alloc_ub([sub_block_M], cal_dtype)
            sum_square_ub = T.alloc_ub([sub_block_M], cal_dtype)

            # Mean and variance (scalar per row, stored as [sub_block_M, 1])
            mean_ub = T.alloc_ub([sub_block_M, 1], cal_dtype)
            mean_square_ub = T.alloc_ub([sub_block_M, 1], cal_dtype)

            with T.Scope("V"):
                # Initialize accumulators to zero
                T.tile.fill(sum_i, 0.0)
                T.tile.fill(sum_square_i, 0.0)
                T.tile.fill(sum_ub, 0.0)
                T.tile.fill(sum_square_ub, 0.0)
                # Fill mean buffers with N (row length) for computing mean = sum / N
                T.tile.fill(mean_ub, N)
                T.tile.fill(mean_square_ub, N)

                # ----------------------------------------------------------
                # Pass 1: Accumulate sum(x) and sum(x^2) across all tiles
                # ----------------------------------------------------------
                for by in T.serial(n_num):
                    T.copy(
                        A[
                            bx * block_M + vid * block_M // VEC_NUM : bx * block_M + (vid + 1) * block_M // VEC_NUM,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_ub,
                    )
                    cast_or_copy(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                    # sum += x
                    T.tile.add(sum_i, sum_i, a_cal)
                    # sum_sq += x * x
                    T.tile.mul(a_cal, a_cal, a_cal)
                    T.tile.add(sum_square_i, sum_square_i, a_cal)

                # Reduce across columns: sum_ub[i] = sum over j of sum_i[i, j]
                T.reduce_sum(sum_i, sum_ub, dim=-1)
                T.reduce_sum(sum_square_i, sum_square_ub, dim=-1)

                # ----------------------------------------------------------
                # Compute mean and variance
                #   mean = sum / N
                #   var  = E[x^2] - (E[x])^2 = (sum_sq / N) - mean^2
                #   std  = sqrt(var + eps)
                # ----------------------------------------------------------
                T.tile.div(mean_ub, sum_ub, mean_ub)                 # mean = sum / N
                T.tile.div(mean_square_ub, sum_square_ub, mean_square_ub)  # E[x^2] = sum_sq / N
                T.tile.mul(sum_ub, mean_ub, mean_ub)                 # mean = mean (reinstated into sum_ub for reuse)
                T.tile.sub(mean_square_ub, mean_square_ub, sum_ub)   # var = E[x^2] - mean^2
                T.tile.fill(sum_ub, eps)                              # sum_ub = eps
                T.tile.add(mean_square_ub, mean_square_ub, sum_ub)   # var + eps
                T.tile.sqrt(mean_square_ub, mean_square_ub)          # std = sqrt(var + eps)

                # Broadcast mean and std to 2D for elementwise normalization
                T.tile.broadcast(sum_i, mean_ub)                     # sum_i = mean (broadcast)
                T.tile.broadcast(sum_square_i, mean_square_ub)       # sum_square_i = std (broadcast)

                # ----------------------------------------------------------
                # Pass 2: Normalize each tile: y = (x - mean) / std
                # ----------------------------------------------------------
                for by in T.serial(n_num):
                    T.copy(
                        A[
                            bx * block_M + vid * block_M // VEC_NUM : bx * block_M + (vid + 1) * block_M // VEC_NUM,
                            by * block_N : (by + 1) * block_N,
                        ],
                        a_ub,
                    )
                    cast_or_copy(a_cal, a_ub, CAST_MODE_LOW2HIGH, sub_block_M * block_N)

                    # Normalize: (x - mean) / std
                    T.tile.sub(a_cal, a_cal, sum_i)
                    T.tile.div(a_cal, a_cal, sum_square_i)

                    # Cast back to original dtype and store
                    cast_or_copy(a_ub, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                    T.copy(
                        a_ub,
                        B[
                            bx * block_M + vid * block_M // VEC_NUM : bx * block_M + (vid + 1) * block_M // VEC_NUM,
                            by * block_N : (by + 1) * block_N,
                        ],
                    )

    return main


if __name__ == "__main__":
    torch.manual_seed(0)

    test_configs = [
        # (M, N, block_M, block_N, dtype)
        (34, 34, 32, 32, "float"),
        (34, 34, 32, 32, "float16"),
        (34, 34, 32, 32, "bfloat16"),
        (270, 270, 64, 64, "float"),
        (1024, 1024, 128, 128, "float16"),
    ]

    for M, N, block_M, block_N, dtype in test_configs:
        print(f"Testing layer_norm: M={M}, N={N}, block_M={block_M}, block_N={block_N}, dtype={dtype}")

        # Compile kernel
        func = layer_norm(M, N, block_M, block_N, dtype=dtype)
        print("  Kernel compiled!")

        # Create input on NPU
        torch_dtype = getattr(torch, dtype) if dtype != "float" else torch.float32
        a = torch.randn(M, N, dtype=torch_dtype).npu()

        # Run kernel
        b = func(a)

        # Reference: PyTorch layer_norm
        ref_b = torch.layer_norm(a, normalized_shape=[N])

        # Verify
        torch.testing.assert_close(b.cpu(), ref_b.cpu(), rtol=1e-2, atol=1e-2)
        print("  Test passed!")

    print("\nAll layer norm tests passed!")
