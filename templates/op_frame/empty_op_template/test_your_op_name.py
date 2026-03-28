"""
TileLang-Ascend算子测试模板 - your_op_name

该文件提供了TileLang-Ascend算子测试的基础模板。
请根据实际算子需求修改以下内容：
1. 算子名称（your_op_name -> 实际算子名）
2. 参考实现（ref_your_op_name）
3. 测试用例（添加更多边界情况和特殊场景）

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)
"""

import torch
import pytest
import tilelang
import tilelang.language as T

from your_op_name import your_op_name, your_op_name_cuda, get_device, setup_device


# 模块级设备检测
DEVICE = get_device()


def ref_your_op_name(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    your_op_name的PyTorch参考实现

    参数:
        A: 输入tensor，shape为(M, K)
        B: 输入tensor，shape为(K, N)

    返回:
        输出tensor，shape为(M, N)
    """
    return A @ B


class TestYourOpName:
    """your_op_name算子测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        setup_device(DEVICE)

    def test_basic_correctness(self):
        """基础正确性测试"""
        M, N, K = 128, 128, 128
        block_M, block_N, block_K = 64, 64, 32

        # 根据设备选择kernel
        if DEVICE == "npu":
            kernel = your_op_name(M, N, K, block_M, block_N, block_K)
        else:
            kernel = your_op_name_cuda(M, N, K, block_M, block_N, block_K)

        # 准备输入数据
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        # 调用kernel
        c = kernel(a, b)

        # 参考实现
        ref_c = ref_your_op_name(a, b)

        # 验证结果
        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_float16(self):
        """float16数据类型测试"""
        M, N, K = 256, 256, 256
        if DEVICE == "npu":
            kernel = your_op_name(M, N, K, dtype=T.float16)
        else:
            kernel = your_op_name_cuda(M, N, K, dtype=T.float16)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = ref_your_op_name(a, b)

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_float32(self):
        """float32数据类型测试"""
        M, N, K = 256, 256, 256
        if DEVICE == "npu":
            kernel = your_op_name(M, N, K, dtype=T.float32, accum_dtype=T.float32)
        else:
            kernel = your_op_name_cuda(M, N, K, dtype=T.float32, accum_dtype=T.float32)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

        c = kernel(a, b)
        ref_c = ref_your_op_name(a, b)

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_large_shape(self):
        """大shape测试"""
        M, N, K = 4096, 4096, 4096
        block_M, block_N, block_K = 128, 128, 32

        if DEVICE == "npu":
            kernel = your_op_name(M, N, K, block_M, block_N, block_K)
        else:
            kernel = your_op_name_cuda(M, N, K, block_M, block_N, block_K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = ref_your_op_name(a, b)

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_non_square(self):
        """非方阵测试"""
        M, N, K = 1024, 512, 256
        if DEVICE == "npu":
            kernel = your_op_name(M, N, K)
        else:
            kernel = your_op_name_cuda(M, N, K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = ref_your_op_name(a, b)

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("M,N,K", [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ])
    def test_various_shapes(self, M, N, K):
        """参数化测试不同shape"""
        if DEVICE == "npu":
            kernel = your_op_name(M, N, K)
        else:
            kernel = your_op_name_cuda(M, N, K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = ref_your_op_name(a, b)

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_M,block_N,block_K", [
        (64, 64, 32),
        (128, 128, 32),
        (128, 128, 64),
        (64, 128, 32),
    ])
    def test_different_tiling(self, block_M, block_N, block_K):
        """测试不同tiling配置"""
        M, N, K = 512, 512, 512

        if DEVICE == "npu":
            kernel = your_op_name(M, N, K, block_M, block_N, block_K)
        else:
            kernel = your_op_name_cuda(M, N, K, block_M, block_N, block_K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = ref_your_op_name(a, b)

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_pipeline_stages(self):
        """测试不同流水线深度"""
        M, N, K = 512, 512, 512

        for num_stages in [0, 1, 2, 3]:
            if DEVICE == "npu":
                kernel = your_op_name(M, N, K, num_stages=num_stages)
            else:
                kernel = your_op_name_cuda(M, N, K, num_stages=num_stages)

            a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
            b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

            c = kernel(a, b)
            ref_c = ref_your_op_name(a, b)

            torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestDeviceDetection:
    """设备检测测试类"""

    def test_get_device(self):
        """测试设备检测功能"""
        device = get_device()
        assert device in ["npu", "cuda", "cpu"]

    def test_setup_device(self):
        """测试设备设置功能"""
        device = setup_device()
        assert device in ["npu", "cuda", "cpu"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
