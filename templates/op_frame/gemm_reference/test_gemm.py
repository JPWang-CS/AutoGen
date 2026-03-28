"""
TileLang-Ascend GEMM测试 - 参考实现

完整的GEMM单元测试，可作为其他算子测试的参考。

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)
"""

import torch
import pytest
import tilelang
import tilelang.language as T

from gemm import gemm, gemm_npu, gemm_cuda, gemm_npu_with_ub, gemm_cuda_with_swizzle, get_device, setup_device


# 模块级设备检测
DEVICE = get_device()


class TestGemm:
    """GEMM算子测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        setup_device(DEVICE)

    def test_basic_correctness(self):
        """基础正确性测试"""
        M, N, K = 128, 128, 128
        kernel = gemm(M, N, K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_float16(self):
        """float16数据类型测试"""
        M, N, K = 256, 256, 256
        kernel = gemm(M, N, K, dtype=T.float16)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_float32(self):
        """float32数据类型测试"""
        M, N, K = 256, 256, 256
        kernel = gemm(M, N, K, dtype=T.float32, accum_dtype=T.float32)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

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

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

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

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

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

        kernel = gemm(M, N, K, block_M, block_N, block_K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("num_stages", [0, 1, 2])
    def test_pipeline_stages(self, num_stages):
        """测试不同流水线深度"""
        M, N, K = 512, 512, 512

        kernel = gemm(M, N, K, num_stages=num_stages)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestGemmNpu:
    """NPU专用GEMM测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        if DEVICE != "npu":
            pytest.skip("NPU not available")
        setup_device(DEVICE)

    def test_npu_basic(self):
        """NPU基础测试"""
        M, N, K = 128, 128, 128
        kernel = gemm_npu(M, N, K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_npu_with_ub(self):
        """NPU Unified Buffer测试"""
        M, N, K = 128, 128, 128
        kernel = gemm_npu_with_ub(M, N, K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestGemmCuda:
    """CUDA专用GEMM测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        if DEVICE != "cuda":
            pytest.skip("CUDA not available")
        setup_device(DEVICE)

    def test_cuda_basic(self):
        """CUDA基础测试"""
        M, N, K = 128, 128, 128
        kernel = gemm_cuda(M, N, K)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_cuda_with_swizzle(self):
        """CUDA swizzle优化测试"""
        M, N, K = 1024, 1024, 1024
        kernel = gemm_cuda_with_swizzle(M, N, K, enable_swizzle=True)

        a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a @ b

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
