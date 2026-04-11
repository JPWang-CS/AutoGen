"""
TileLang-Ascend Matmul 性能基准测试 (NPU专用)
"""
import argparse
import torch
import tilelang
import tilelang.language as T

from matmul import matmul, ref_program


def get_heuristic_config(M, N, K):
    """根据问题规模推荐NPU配置"""
    if M >= 512 and N >= 512:
        return {"block_M": 64, "block_N": 64, "block_K": 32}
    return {"block_M": 32, "block_N": 32, "block_K": 32}


def main():
    parser = argparse.ArgumentParser(description="TileLang-Ascend Matmul Benchmark (NPU)")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--block_M", type=int, default=None)
    parser.add_argument("--block_N", type=int, default=None)
    parser.add_argument("--block_K", type=int, default=None)
    parser.add_argument("--transpose_A", action="store_true")
    parser.add_argument("--transpose_B", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)

    M, N, K = args.M, args.N, args.K
    config = get_heuristic_config(M, N, K)
    block_M = args.block_M or config["block_M"]
    block_N = args.block_N or config["block_N"]
    block_K = args.block_K or config["block_K"]

    trans_a = args.transpose_A
    trans_b = args.transpose_B
    mode = f"{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}"

    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"Mode: {mode} (transpose_A={trans_a}, transpose_B={trans_b})")
    print(f"Block size: {block_M} x {block_N} x {block_K}")

    kernel = matmul(
        M, N, K,
        block_M=block_M, block_N=block_N, block_K=block_K,
        transpose_A=trans_a, transpose_B=trans_b,
        dtype="float16", accum_dtype="float32",
    )

    # 准备输入数据
    a_shape = (K, M) if trans_a else (M, K)
    b_shape = (N, K) if trans_b else (K, N)
    a = torch.randn(*a_shape, device="npu", dtype=torch.float16)
    b = torch.randn(*b_shape, device="npu", dtype=torch.float16)

    # 正确性验证
    c = kernel(a, b)
    ref_c = ref_program(a, b, transpose_A=trans_a, transpose_B=trans_b)
    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print("Correctness check passed!")

    # 性能测试
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(warmup=args.warmup, rep=args.rep)

    flops = 2 * M * N * K
    tflops = flops / latency * 1e-9

    print("\n" + "=" * 60)
    print(f"TileLang-Ascend Matmul Benchmark Results (NPU):")
    print("=" * 60)
    print(f"Mode:              {mode}")
    print(f"Problem size:      M={M}, N={N}, K={K}")
    print(f"Block size:        {block_M} x {block_N} x {block_K}")
    print(f"Latency:           {latency:.3f} ms")
    print(f"TFlops:            {tflops:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
