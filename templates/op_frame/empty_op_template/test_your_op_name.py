"""
TileLang-Ascend 算子测试模板 - your_op_name (NPU专用)

该文件提供了TileLang-Ascend算子测试的基础模板。
请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. 参考实现（ref_your_op_name）
3. 测试用例（添加更多边界情况和特殊场景）

所有计算在NPU上执行，精度对比在CPU上进行。
"""

import torch
import pytest

from your_op_name import your_op_name


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


def ref_your_op_name(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    your_op_name的PyTorch参考实现 (CPU上执行)

    TODO: 根据你的算子修改此实现

    参数:
        A: 输入tensor
        B: 输入tensor

    返回:
        输出tensor
    """
    return A @ B  # 示例: GEMM


# ============================================================================
# 测试类
# ============================================================================

@pytest.mark.skipif(not npu_available(), reason="NPU not available")
class TestYourOpName:
    """your_op_name算子NPU测试类"""

    def setup_method(self):
        setup_npu()

    def test_basic_correctness(self):
        """基础正确性测试"""
        M, N, K = 128, 128, 128
        kernel = your_op_name(M, N, K, block_M=32, block_N=32, block_K=32)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        # 精度对比在CPU上进行
        ref_c = ref_your_op_name(a.cpu(), b.cpu())
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    def test_large_shape(self):
        """大shape测试"""
        M, N, K = 1024, 1024, 1024
        kernel = your_op_name(M, N, K, block_M=32, block_N=32, block_K=32)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = ref_your_op_name(a.cpu(), b.cpu())
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    def test_non_square(self):
        """非方阵测试"""
        M, N, K = 512, 256, 128
        kernel = your_op_name(M, N, K)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = ref_your_op_name(a.cpu(), b.cpu())
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ])
    def test_various_shapes(self, M, N, K):
        """参数化测试不同shape"""
        kernel = your_op_name(M, N, K)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = ref_your_op_name(a.cpu(), b.cpu())
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_M,block_N,block_K", [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 32),
    ])
    def test_different_block_sizes(self, block_M, block_N, block_K):
        """测试不同block size配置"""
        M, N, K = 256, 256, 256

        kernel = your_op_name(M, N, K, block_M, block_N, block_K)

        a = torch.randn(M, K, device="npu", dtype=torch.float16)
        b = torch.randn(K, N, device="npu", dtype=torch.float16)

        c = kernel(a, b)

        ref_c = ref_your_op_name(a.cpu(), b.cpu())
        torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
