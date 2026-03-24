---
name: tl-op-test
description: 为TileLang算子生成单元测试文件与用例。用户提出"生成测试/生成UT/补充测试用例"时使用本技能。
---

# TileLang算子测试生成

当用户提出"生成TileLang算子测试/生成UT/补充测试用例"时使用本技能。

## 工作流
1. 确认目标算子路径与算子名称。
2. 读取算子实现文件，理解输入/输出/参数。
3. 生成测试文件 `test_{op_name}.py`。
4. 包含以下测试类别：
   - 正确性测试（与PyTorch参考实现对比）
   - 边界测试（最小/最大shape）
   - 数据类型测试（不同dtype）
   - 可选参数测试（如有）

## 测试文件结构

```python
import torch
import tilelang
import tilelang.language as T
import pytest

# 导入被测试的kernel
from your_op_name import your_op_name

def ref_your_op_name(*args, **kwargs):
    """PyTorch参考实现"""
    # 实现参考逻辑
    pass

class TestYourOpName:
    """YourOpName算子测试类"""

    def test_basic_correctness(self):
        """基础正确性测试"""
        # 准备输入数据
        # 调用kernel
        # 与参考实现对比
        pass

    def test_different_dtypes(self):
        """不同数据类型测试"""
        for dtype in [torch.float16, torch.bfloat16]:
            # 测试不同dtype
            pass

    def test_edge_cases(self):
        """边界情况测试"""
        # 测试最小shape
        # 测试非对齐shape
        pass

    def test_with_optional_params(self):
        """可选参数测试"""
        # 测试各种参数组合
        pass

    @pytest.mark.parametrize("M,N,K", [
        (128, 128, 128),
        (256, 256, 256),
        (1024, 1024, 1024),
    ])
    def test_various_shapes(self, M, N, K):
        """参数化测试不同shape"""
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## 测试生成规则

### 1. 正确性验证
- 使用 `torch.testing.assert_close` 进行结果对比
- 设置合理的 `rtol` 和 `atol`（通常为 1e-2）

### 2. 参考实现
- 使用PyTorch原生操作作为参考实现
- 对于复杂算子，使用 `torch.nn.functional` 提供的函数

### 3. 输入数据生成
- 使用 `torch.randn` 生成随机数据
- 使用 `.cuda()` 将数据放到GPU上
- 使用 `.half()` 或 `.bfloat16()` 转换数据类型

### 4. 参数化测试
- 使用 `@pytest.mark.parametrize` 覆盖多种场景
- 包含典型的shape组合
- 包含不同的dtype组合

## 输出要求
- 列出生成的文件路径
- 说明已覆盖的测试用例类别
- 提示需要用户补充的特殊测试场景

## 约束
- 仅生成测试代码，不修改算子实现
- 遵循pytest测试框架规范
- 确保测试可以直接运行

## 参考
- 模板目录：`templates/op_frame/empty_op_template/test_your_op_name.py`
- TileLang测试示例：`https://github.com/tile-ai/tilelang/tree/main/examples`
