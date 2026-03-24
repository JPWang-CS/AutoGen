"""
TileLang GEMM测试 - 参考实现

完整的GEMM单元测试，可作为其他算子测试的参考。
"""

import torch
import pytest
import tilelang
import tilelang.language as T

from gemm import gemm, gemm_with_swizzle, gemm_transposed_b


class TestGemm:
    """GEMM算子测试类"""

    def test_basic_correctness(self):
        """基础正确性测试"""
        M, N, K = 128, 128, 128
        kernel = gemm(M, N, K)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_float16(self):
        """float16数据类型测试"""
        M, N, K = 256, 256, 256
        kernel = gemm(M, N, K, dtype=T.float16)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_bfloat16(self):
        """bfloat16数据类型测试"""
        M, N, K = 256, 256, 256
        kernel = gemm(M, N, K, dtype=T.bfloat16)

        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ])
    def test_various_shapes(self, M, N, K):
        """参数化测试不同shape"""
        kernel = gemm(M, N, K)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (1024, 512, 256),
        (512, 1024, 256),
        (256, 512, 1024),
        (2048, 128, 512),
    ])
    def test_non_square(self, M, N, K):
        """非方阵测试"""
        kernel = gemm(M, N, K)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_M,block_N,block_K", [
        (64, 64, 32),
        (128, 128, 32),
        (128, 128, 64),
        (256, 128, 32),
        (128, 256, 64),
    ])
    def test_different_tiling(self, block_M, block_N, block_K):
        """测试不同tiling配置"""
        M, N, K = 512, 512, 512

        kernel = gemm(M, N, K, block_M, block_N, block_K)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("num_stages", [0, 1, 2, 3])
    def test_pipeline_stages(self, num_stages):
        """测试不同流水线深度"""
        M, N, K = 512, 512, 512

        kernel = gemm(M, N, K, num_stages=num_stages)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestGemmWithSwizzle:
    """带swizzle优化的GEMM测试"""

    def test_swizzle_enabled(self):
        """启用swizzle测试"""
        M, N, K = 1024, 1024, 1024
        kernel = gemm_with_swizzle(M, N, K, enable_swizzle=True)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_swizzle_disabled(self):
        """禁用swizzle测试"""
        M, N, K = 1024, 1024, 1024
        kernel = gemm_with_swizzle(M, N, K, enable_swizzle=False)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestGemmTransposedB:
    """转置B矩阵的GEMM测试"""

    def test_transposed_b(self):
        """B转置测试"""
        M, N, K = 1024, 1024, 1024
        kernel = gemm_transposed_b(M, N, K)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16)  # 注意shape

        c = kernel(a, b)
        ref_c = a @ b.T

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (512, 512, 512),
        (1024, 512, 256),
        (256, 1024, 512),
    ])
    def test_transposed_b_various_shapes(self, M, N, K):
        """不同shape的B转置测试"""
        kernel = gemm_transposed_b(M, N, K)

        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b.T

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
