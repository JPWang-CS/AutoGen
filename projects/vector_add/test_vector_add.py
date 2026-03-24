"""
TileLang Vector Add算子测试
"""

import torch
import pytest
import tilelang
import tilelang.language as T

from vector_add import vector_add, vector_add_simple, vector_add_2d


class TestVectorAdd:
    """Vector Add算子测试类"""

    def test_basic_correctness(self):
        """基础正确性测试"""
        N = 1024
        kernel = vector_add(N)

        a = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_simple_version(self):
        """简化版本测试"""
        N = 1024
        kernel = vector_add_simple(N)

        a = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("N", [
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,  # 1M
    ])
    def test_various_sizes(self, N):
        """测试不同大小的向量"""
        kernel = vector_add(N)

        a = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_float16(self):
        """float16数据类型测试"""
        N = 4096
        kernel = vector_add(N, dtype=T.float16)

        a = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_float32(self):
        """float32数据类型测试"""
        N = 4096
        kernel = vector_add(N, dtype=T.float32)

        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_size", [128, 256, 512, 1024])
    def test_different_block_sizes(self, block_size):
        """测试不同的block size"""
        N = 65536
        kernel = vector_add(N, block_size=block_size)

        a = torch.randn(N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestVectorAdd2D:
    """2D Tensor加法测试类"""

    def test_basic_2d(self):
        """基础2D测试"""
        M, N = 128, 128
        kernel = vector_add_2d(M, N)

        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(M, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N", [
        (64, 64),
        (128, 256),
        (256, 512),
        (1024, 1024),
    ])
    def test_various_shapes(self, M, N):
        """测试不同shape"""
        kernel = vector_add_2d(M, N)

        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(M, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_non_square(self):
        """非方阵测试"""
        M, N = 1024, 256
        kernel = vector_add_2d(M, N)

        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(M, N, device="cuda", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
