"""
TileLang Vector Add 算子实现 (NPU专用)
"""
import argparse
import torch
import tilelang
import tilelang.language as T


def ref_program(a, b):
    """参考实现"""
    return a + b


@tilelang.jit(out_idx=[-1])
def vector_add_1d(N, block_N, dtype="float16"):
    """
    1D Vector 加法 (NPU版本)
    """
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            b_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            c_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)

            with T.Scope("V"):
                T.copy(A[cid * block_N + vid * block_N // VEC_NUM], a_ub)
                T.copy(B[cid * block_N + vid * block_N // VEC_NUM], b_ub)
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)
                T.tile.add(c_ub, a_ub, b_ub)
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(c_ub, C[cid * block_N + vid * block_N // VEC_NUM])

    return main


@tilelang.jit(out_idx=[-1])
def vector_add_1d_multi_block(N, block_N, num_cores=24, dtype="float16"):
    """
    1D Vector 加法 (多block版本 - 正确处理完整数据)
    使用向上取整计算 blocks_per_core，添加边界检查跳过无法完整处理的block
    """
    total_blocks = N // block_N
    blocks_per_core = (total_blocks + num_cores - 1) // num_cores
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            a_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            b_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)
            c_ub = T.alloc_ub((block_N // VEC_NUM,), dtype)

            for i in T.serial(blocks_per_core):
                block_idx = cid * blocks_per_core + i
                if block_idx < total_blocks:
                    with T.Scope("V"):
                        T.copy(A[block_idx * block_N + vid * block_N // VEC_NUM], a_ub)
                        T.copy(B[block_idx * block_N + vid * block_N // VEC_NUM], b_ub)
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.tile.add(c_ub, a_ub, b_ub)
                        T.set_flag("v", "mte3", 0)
                        T.wait_flag("v", "mte3", 0)
                        T.copy(c_ub, C[block_idx * block_N + vid * block_N // VEC_NUM])
                        T.barrier_all()

    return main


@tilelang.jit(out_idx=[-1])
def vector_add_2d(M, N, block_M, block_N, dtype="float16"):
    """
    2D Tensor 加法 (NPU版本)
    """
    m_num = M // block_M
    n_num = N // block_N
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
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)
                T.tile.add(c_ub, a_ub, b_ub)
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


@tilelang.jit(out_idx=[-1])
def vector_add_3d(D, M, N, block_M, block_N, dtype="float16"):
    """
    3D Tensor 加法 (NPU版本)
    """
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((D, M, N), dtype),
        B: T.Tensor((D, M, N), dtype),
        C: T.Tensor((D, M, N), dtype),
    ):
        with T.Kernel(D * m_num * n_num, is_npu=True) as (cid, vid):
            bz = cid // (m_num * n_num)
            remaining = cid % (m_num * n_num)
            bx = remaining // n_num
            by = remaining % n_num
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bz, bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
                T.copy(B[bz, bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
                T.set_flag("mte2", "v", 0)
                T.wait_flag("mte2", "v", 0)
                T.tile.add(c_ub, a_ub, b_ub)
                T.set_flag("v", "mte3", 0)
                T.wait_flag("v", "mte3", 0)
                T.copy(c_ub, C[bz, bx * block_M + vid * block_M // VEC_NUM, by * block_N])

    return main


def main_func(M=1024, N=1024):
    """测试入口函数"""
    torch.npu.set_device(0)

    print(f"Device: NPU")
    print("=" * 60)
    # 1D Vector Add 测试
    N_1d = 1024 * 1024
    print(f"\n[1D Vector Add] N={N_1d}")
    kernel_1d = vector_add_1d(N_1d, block_N=256, dtype="float16")
    a_1d = torch.randn(N_1d, dtype=torch.float16, device="npu")
    b_1d = torch.randn(N_1d, dtype=torch.float16, device="npu")
    out_1d = kernel_1d(a_1d, b_1d)
    ref_c_1d = a_1d.cpu() + b_1d.cpu()
    torch.testing.assert_close(out_1d.cpu(), ref_c_1d, rtol=1e-2, atol=1e-2)
    print("1D Vector Add passed!")
    profiler_1d = kernel_1d.get_profiler()
    latency_1d = profiler_1d.do_bench()
    bytes_moved_1d = 3 * N_1d * 2
    bandwidth_gbs_1d = bytes_moved_1d / latency_1d * 1e-6
    print(f"1D Latency: {latency_1d:.3f} ms, Bandwidth: {bandwidth_gbs_1d:.2f} GB/s")
    # 1D Vector Add - block_N=512
    print(f"\n[1D Vector Add - block_N=512]")
    kernel_1d_opt = vector_add_1d(N_1d, block_N=512, dtype="float16")
    out_1d_opt = kernel_1d_opt(a_1d, b_1d)
    torch.testing.assert_close(out_1d_opt.cpu(), ref_c_1d, rtol=1e-2, atol=1e-2)
    print("1D Optimized passed!")
    profiler_1d_opt = kernel_1d_opt.get_profiler()
    latency_1d_opt = profiler_1d_opt.do_bench()
    bandwidth_gbs_1d_opt = bytes_moved_1d / latency_1d_opt * 1e-6
    print(f"1D Optimized Latency: {latency_1d_opt:.3f} ms, Bandwidth: {bandwidth_gbs_1d_opt:.2f} GB/s")
    # 1D Vector Add Multi-Block
    print(f"\n[1D Vector Add Multi-Block - block_N=256, num_cores=24]")
    num_cores = 24
    block_N = 256
    kernel_1d_mb = vector_add_1d_multi_block(N_1d, block_N=block_N, num_cores=num_cores, dtype="float16")
    out_1d_mb = kernel_1d_mb(a_1d, b_1d)
    # 打印详细信息
    total_blocks = N_1d // block_N
    blocks_per_core = (total_blocks + num_cores - 1) // num_cores
    print(f"  total_blocks = {total_blocks}")
    print(f"  num_cores = {num_cores}, blocks_per_core = {blocks_per_core}")
    print(f"  实际处理: {num_cores} * {blocks_per_core} * {block_N} = {num_cores * blocks_per_core * block_N} elements")

    # 比较前128个元素
    print(f"  前128元素比较:")
    print(f"    ref: {ref_c_1d[:128][:5]}...")
    print(f"    out: {out_1d_mb.cpu()[:128][:5]}...")
    # 检查哪里开始不匹配
    diff = (out_1d_mb.cpu() - ref_c_1d).abs()
    max_diff = diff.max().item()
    mismatch_count = (diff > 1e-2).sum().item()
    print(f"  最大差异: {max_diff}, 不匹配元素数: {mismatch_count}/{N_1d}")
    # 找到第一个不匹配的位置
    if mismatch_count > 0:
        first_mismatch = (diff > 1e-2).nonzero()[0].item()
        print(f"  第一个不匹配位置: {first_mismatch}")
        print(f"    ref[{first_mismatch}] = {ref_c_1d[first_mismatch].item():.6f}")
        print(f"    out[{first_mismatch}] = {out_1d_mb.cpu()[first_mismatch].item():.6f}")
    torch.testing.assert_close(out_1d_mb.cpu(), ref_c_1d, rtol=1e-2, atol=1e-2)
    print("1D Multi-Block passed!")
    profiler_1d_mb = kernel_1d_mb.get_profiler()
    latency_1d_mb = profiler_1d_mb.do_bench()
    bandwidth_gbs_1d_mb = bytes_moved_1d / latency_1d_mb * 1e-6
    print(f"1D Multi-Block Latency: {latency_1d_mb:.3f} ms, Bandwidth: {bandwidth_gbs_1d_mb:.2f} GB/s")
    # 2D Vector Add 测试
    print(f"\n[2D Vector Add] M={M}, N={N}")
    kernel_2d = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")
    a_2d = torch.randn(M, N, dtype=torch.float16, device="npu")
    b_2d = torch.randn(M, N, dtype=torch.float16, device="npu")
    out_2d = kernel_2d(a_2d, b_2d)
    ref_c_2d = ref_program(a_2d.cpu(), b_2d.cpu())
    torch.testing.assert_close(out_2d.cpu(), ref_c_2d, rtol=1e-2, atol=1e-2)
    print("2D Vector Add passed!")
    profiler_2d = kernel_2d.get_profiler()
    latency_2d = profiler_2d.do_bench()
    bytes_moved_2d = 3 * M * N * 2
    bandwidth_gbs_2d = bytes_moved_2d / latency_2d * 1e-6
    print(f"2D Latency: {latency_2d:.3f} ms, Bandwidth: {bandwidth_gbs_2d:.2f} GB/s")
    # 3D Vector Add 测试
    D, M3d, N3d = 64, 128, 128
    print(f"\n[3D Vector Add] D={D}, M={M3d}, N={N3d}")
    kernel_3d = vector_add_3d(D, M3d, N3d, block_M=16, block_N=16, dtype="float16")
    a_3d = torch.randn(D, M3d, N3d, dtype=torch.float16, device="npu")
    b_3d = torch.randn(D, M3d, N3d, dtype=torch.float16, device="npu")
    out_3d = kernel_3d(a_3d, b_3d)
    ref_c_3d = ref_program(a_3d.cpu(), b_3d.cpu())
    torch.testing.assert_close(out_3d.cpu(), ref_c_3d, rtol=1e-2, atol=1e-2)
    print("3D Vector Add passed!")
    profiler_3d = kernel_3d.get_profiler()
    latency_3d = profiler_3d.do_bench()
    bytes_moved_3d = 3 * D * M3d * N3d * 2
    bandwidth_gbs_3d = bytes_moved_3d / latency_3d * 1e-6
    print(f"3D Latency: {latency_3d:.3f} ms, Bandwidth: {bandwidth_gbs_3d:.2f} GB/s")
    # 性能汇总
    print("=" * 60)
    print(f"1D Standard: {latency_1d:.3f} ms, {bandwidth_gbs_1d:.2f} GB/s")
    print(f"1D Optimized (block_N=512): {latency_1d_opt:.3f} ms, {bandwidth_gbs_1d_opt:.2f} GB/s")
    print(f"1D Multi-Block: {latency_1d_mb:.3f} ms, {bandwidth_gbs_1d_mb:.2f} GB/s")
    print(f"2D: {latency_2d:.3f} ms, {bandwidth_gbs_2d:.2f} GB/s")
    print(f"3D: {latency_3d:.3f} ms, {bandwidth_gbs_3d:.2f} GB/s")
    print("=" * 60)


def run_benchmark():
    """性能基准测试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1048576, help="1D vector size")
    parser.add_argument("--m", type=int, default=1024, help="2D matrix M")
    parser.add_argument("--m2", type=int, default=1024, help="2D matrix N")
    parser.add_argument("--d", type=int, default=64, help="3D tensor D")
    args, _ = parser.parse_known_args()
    torch.npu.set_device(0)
    # 1D benchmark
    N = args.n
    print(f"[1D] N={N}")
    a = torch.randn(N, dtype=torch.float16, device="npu")
    b = torch.randn(N, dtype=torch.float16, device="npu")
    kernel = vector_add_1d(N, block_N=256, dtype="float16")
    from tilelang.profiler import do_bench
    latency = do_bench(lambda: kernel(a, b))
    print(f"1D Latency: {latency:.3f} ms")
    bytes_moved = 3 * N * 2
    bandwidth_gbs = bytes_moved / latency * 1e-6
    print(f"1D Bandwidth: {bandwidth_gbs:.2f} GB/s")
    # 2D benchmark
    M, N2d = args.m, args.m2
    print(f"\n[2D] M={M}, N={N2d}")
    a2d = torch.randn(M, N2d, dtype=torch.float16, device="npu")
    b2d = torch.randn(M, N2d, dtype=torch.float16, device="npu")
    kernel_2d = vector_add_2d(M, N2d, block_M=32, block_N=32, dtype="float16")
    latency_2d = do_bench(lambda: kernel_2d(a2d, b2d))
    print(f"2D Latency: {latency_2d:.3f} ms")
    bytes_moved_2d = 3 * M * N2d * 2
    bandwidth_gbs_2d = bytes_moved_2d / latency_2d * 1e-6
    print(f"2D Bandwidth: {bandwidth_gbs_2d:.2f} GB/s")
    # 3D benchmark
    D = args.d
    print(f"\n[3D] D={D}, M={M}, N={N2d}")
    a3d = torch.randn(D, M, N2d, dtype=torch.float16, device="npu")
    b3d = torch.randn(D, M, N2d, dtype=torch.float16, device="npu")
    kernel_3d = vector_add_3d(D, M, N2d, block_M=32, block_N=32, dtype="float16")
    latency_3d = do_bench(lambda: kernel_3d(a3d, b3d))
    print(f"3D Latency: {latency_3d:.3f} ms")
    bytes_moved_3d = 3 * D * M * N2d * 2
    bandwidth_gbs_3d = bytes_moved_3d / latency_3d * 1e-6
    print(f"3D Bandwidth: {bandwidth_gbs_3d:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=64, help="3D tensor D")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true")
    args, _ = parser.parse_known_args()
    if args.benchmark:
        run_benchmark()
    else:
        main_func(args.m, args.n)
