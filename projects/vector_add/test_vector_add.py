"""
TileLang Vector Add 算子测试 (NPU专用)

测试 1D/2D/3D Vector Add 算子的正确性。
所有计算在NPU上执行，精度对比在CPU上进行。
"""

import torch
import pytest

from vector_add import vector_add_1d, vector_add_2d, vector_add_3d, vector_add_1d_multi_block, vector_add_1d_multi_block


def npu_available():
    """检查NPU是否可用"""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False


def setup_npu():
    """设置NPU设备"""
    import torch_npu
    torch.npu.set_device(0)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd1D:
    """1D Vector Add NPU测试"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        N = 1024
        kernel = vector_add_1d(N, block_N=256, dtype="float16")

        a = torch.randn(N, device="npu", dtype=torch.float16)
        b = torch.randn(N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        # 精度对比在CPU上进行
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("N", [256, 1024, 4096, 65536, 262144, 1048576])
    def test_various_sizes(self, N):
        """测试不同大小"""
        kernel = vector_add_1d(N, block_N=256, dtype="float16")

        a = torch.randn(N, device="npu", dtype=torch.float16)
        b = torch.randn(N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd2D:
    """2D Vector Add NPU测试"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        M, N = 128, 128
        kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N", [(64, 64), (128, 256), (256, 512), (1024, 1024)])
    def test_various_shapes(self, M, N):
        """测试不同shape"""
        kernel = vector_add_2d(M, N, block_M=32, block_N=32, dtype="float16")

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd3D:
    """3D Vector Add NPU测试"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        D, M, N = 32, 64, 64
        kernel = vector_add_3d(D, M, N, block_M=16, block_N=16, dtype="float16")

        a = torch.randn(D, M, N, device="npu", dtype=torch.float16)
        b = torch.randn(D, M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("D,M,N", [(16, 32, 32), (32, 64, 64), (64, 128, 128)])
    def test_various_shapes(self, D, M, N):
        """测试不同shape"""
        kernel = vector_add_3d(D, M, N, block_M=16, block_N=16, dtype="float16")

        a = torch.randn(D, M, N, device="npu", dtype=torch.float16)
        b = torch.randn(D, M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
