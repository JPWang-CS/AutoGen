"""
TileLang-Ascend Vector Add 算子测试 (NPU专用)

所有计算在NPU上执行，精度对比在CPU上进行。
"""

import torch
import pytest
import tilelang
import tilelang.language as T

from vector_add import vector_add, vector_add_2d


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
# 1D Vector Add 测试
# ============================================================================

@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd:
    """Vector Add NPU测试"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试 - FP16"""
        N = 1024
        kernel = vector_add(N, block_N=256, dtype="float16")

        # 使用 (N, 1) shape 适配 NPU kernel
        a = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)
        b = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)

        c = kernel(a, b)

        # 精度对比在CPU上进行
        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    def test_basic_correctness_fp32(self):
        """基础正确性测试 - FP32"""
        N = 1024
        kernel = vector_add(N, block_N=256, dtype="float32")

        a = torch.randn(N, device="npu", dtype=torch.float32).unsqueeze(1)
        b = torch.randn(N, device="npu", dtype=torch.float32).unsqueeze(1)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-3, atol=1e-3)

    def test_simple_version(self):
        """简化版本测试 (与标准版本相同)"""
        N = 1024
        kernel = vector_add(N, block_N=256, dtype="float16")

        a = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)
        b = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

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
        kernel = vector_add(N, block_N=256, dtype="float16")

        a = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)
        b = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_N", [128, 256, 512, 1024])
    def test_different_block_sizes(self, block_N):
        """测试不同的block size"""
        N = 65536
        kernel = vector_add(N, block_N=block_N, dtype="float16")

        a = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)
        b = torch.randn(N, device="npu", dtype=torch.float16).unsqueeze(1)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


# ============================================================================
# 2D Vector Add 测试
# ============================================================================

@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestVectorAdd2D:
    """2D Tensor加法NPU测试"""

    def setup_method(self):
        setup_npu()

    def test_basic_2d(self):
        """基础2D测试"""
        M, N = 128, 128
        kernel = vector_add_2d(M, N)

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N", [
        (64, 64),
        (128, 256),
        (256, 512),
        (1024, 1024),
    ])
    def test_various_shapes(self, M, N):
        """测试不同shape"""
        kernel = vector_add_2d(M, N)

        a = torch.randn(M, N, device="npu", dtype=torch.float16)
        b = torch.randn(M, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = a.cpu() + b.cpu()
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
