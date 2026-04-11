"""
TileLang-Ascend Matmul 单元测试 (NPU专用)
"""
import torch
import pytest

from matmul import matmul, ref_program


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
class TestMatmulNN:
    """Matmul NN (无转置) 测试"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        M, N, K = 128, 128, 128
        kernel = matmul(M, N, K)
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ])
    def test_various_shapes(self, M, N, K):
        kernel = matmul(M, N, K)
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (1024, 512, 256),
        (512, 1024, 256),
        (256, 512, 1024),
        (2048, 128, 512),
    ])
    def test_non_square(self, M, N, K):
        kernel = matmul(M, N, K)
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestMatmulTN:
    """Matmul TN (A转置) 测试"""

    def setup_method(self):
        setup_npu()

    def test_basic(self):
        M, N, K = 128, 128, 128
        kernel = matmul(M, N, K, transpose_A=True, transpose_B=False)
        a = torch.randn(K, M, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b, transpose_A=True, transpose_B=False)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ])
    def test_various_shapes(self, M, N, K):
        kernel = matmul(M, N, K, transpose_A=True)
        a = torch.randn(K, M, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b, transpose_A=True)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestMatmulNT:
    """Matmul NT (B转置) 测试"""

    def setup_method(self):
        setup_npu()

    def test_basic(self):
        M, N, K = 128, 128, 128
        kernel = matmul(M, N, K, transpose_A=False, transpose_B=True)
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(N, K, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b, transpose_B=True)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ])
    def test_various_shapes(self, M, N, K):
        kernel = matmul(M, N, K, transpose_B=True)
        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(N, K, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b, transpose_B=True)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestMatmulTT:
    """Matmul TT (AB都转置) 测试"""

    def setup_method(self):
        setup_npu()

    def test_basic(self):
        M, N, K = 128, 128, 128
        kernel = matmul(M, N, K, transpose_A=True, transpose_B=True)
        a = torch.randn(K, M, device="npu", dtype=torch.float16)
        b = torch.randn(N, K, device="npu", dtype=torch.float16)
        c = kernel(a, b)
        ref_c = ref_program(a, b, transpose_A=True, transpose_B=True)
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
