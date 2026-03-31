"""
TileLang Vector Add 算子测试 (NPU专用)

所有计算在NPU上执行，精度对比在CPU上进行。
"""

import torch
import pytest

from vector_add import vector_add_2d


def npu_available():
    """检查NPU是否可用"""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd2D:
    """Vector Add 2D NPU测试"""

    def setup_method(self):
        import torch_npu
        torch.npu.set_device(0)

    def test_basic_correctness(self):
        """基础正确性测试 - FP16"""
        M, N = 128, 128
        kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        # 精度对比在CPU上进行
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    def test_large_shape(self):
        """大shape测试"""
        M, N = 1024, 1024
        kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N", [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ])
    def test_various_shapes(self, M, N):
        """参数化测试不同shape"""
        kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_M,block_N", [
        (16, 16),
        (32, 32),
        (64, 64),
    ])
    def test_different_block_sizes(self, block_M, block_N):
        """测试不同的block size"""
        M, N = 256, 256
        kernel = vector_add_2d(M, N, block_M=block_M, block_N=block_N, dtype="float16")

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
