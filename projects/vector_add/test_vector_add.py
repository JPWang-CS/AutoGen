"""
TileLang-Ascend Vector Add算子测试

支持设备：NPU (华为昇腾), CUDA (NVIDIA GPU)
"""

import torch
import pytest
import tilelang
import tilelang.language as T

from vector_add import (
    vector_add_npu, vector_add_cuda,
    vector_add_npu_simple, vector_add_cuda_simple,
    vector_add_2d_npu, vector_add_2d_cuda,
    vector_add, vector_add_simple, vector_add_2d,
    get_device, setup_device
)


# 模块级设备检测
DEVICE = get_device()


class TestVectorAddNPU:
    """NPU Vector Add测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        if DEVICE != "npu":
            pytest.skip("NPU not available")
        setup_device(DEVICE)

    def test_basic_correctness(self):
        """基础正确性测试"""
        N = 1024
        kernel = vector_add_npu(N, block_N=256, dtype="float16")

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

        # NPU版本需要传入shape参数
        c = kernel(a, b, torch.tensor(N, dtype=torch.int32))
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_simple_version(self):
        """简化版本测试"""
        N = 1024
        kernel = vector_add_npu_simple(N, dtype="float16")

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

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
        kernel = vector_add_npu(N, block_N=256, dtype="float16")

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b, torch.tensor(N, dtype=torch.int32))
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("block_N", [128, 256, 512, 1024])
    def test_different_block_sizes(self, block_N):
        """测试不同的block size"""
        N = 65536
        kernel = vector_add_npu(N, block_N=block_N, dtype="float16")

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b, torch.tensor(N, dtype=torch.int32))
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestVectorAddCUDA:
    """CUDA Vector Add测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        if DEVICE != "cuda":
            pytest.skip("CUDA not available")
        setup_device(DEVICE)

    def test_basic_correctness(self):
        """基础正确性测试"""
        N = 1024
        kernel = vector_add_cuda(N)

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_simple_version(self):
        """简化版本测试"""
        N = 1024
        kernel = vector_add_cuda_simple(N)

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

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
        kernel = vector_add_cuda(N)

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestVectorAddUnified:
    """统一接口测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        setup_device(DEVICE)

    def test_unified_interface(self):
        """测试统一接口"""
        N = 1024
        kernel = vector_add(N)

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

        if DEVICE == "npu":
            c = kernel(a, b, torch.tensor(N, dtype=torch.int32))
        else:
            c = kernel(a, b)

        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    def test_simple_unified_interface(self):
        """测试简化版统一接口"""
        N = 1024
        kernel = vector_add_simple(N)

        a = torch.randn(N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


class TestVectorAdd2D:
    """2D Tensor加法测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        setup_device(DEVICE)

    def test_basic_2d(self):
        """基础2D测试"""
        M, N = 128, 128
        kernel = vector_add_2d(M, N)

        a = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(M, N, device=DEVICE, dtype=torch.float16)

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

        a = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
        b = torch.randn(M, N, device=DEVICE, dtype=torch.float16)

        c = kernel(a, b)
        ref_c = a + b

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
