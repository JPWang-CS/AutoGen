"""
TileLang-Ascend GEMM测试 - 参考实现 (NPU专用)

完整的GEMM单元测试，可作为其他算子测试的参考。

所有计算在NPU上执行，精度对比在CPU上进行。
"""

import torch
import pytest

from gemm import gemm


# ============================================================================
# 测试辅助函数
# ============================================================================

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


# ============================================================================
# 测试类
# ============================================================================

@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestGemm:
    """GEMM算子NPU测试类"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        M, N, K = 128, 128, 128
        kernel = gemm(M, N, K)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        # 精度对比在CPU上进行
        ref_c = a.cpu() @ b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ])
    def test_various_shapes(self, M, N, K):
        """参数化测试不同shape"""
        kernel = gemm(M, N, K)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a.cpu() @ b.cpu()

        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (1024, 512, 256),
        (512, 1024, 256),
        (256, 512, 1024),
        (2048, 128, 512),
    ])
    def test_non_square(self, M, N, K):
        """非方阵测试"""
        kernel = gemm(M, N, K)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a.cpu() @ b.cpu()

        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_M,block_N,block_K", [
        (32, 32, 32),
        (64, 64, 32),
        (32, 64, 32),
    ])
    def test_different_tiling(self, block_M, block_N, block_K):
        """测试不同tiling配置"""
        M, N, K = 256, 256, 256

        kernel = gemm(M, N, K, block_M, block_N, block_K)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a.cpu() @ b.cpu()

        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
