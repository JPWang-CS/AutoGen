---
name: tl-op-rename
description: 用户需要修改TileLang算子名字时，提供旧的算子名以及新的算子名，对算子名字做全量的修改
---

# 重命名TileLang算子

当用户提出"修改TileLang算子名称/重命名TileLang算子"时使用本技能。

## 工作流

1. 确认算子**旧的名字**和算子**新的名字**。
2. 确认需要修改的算子的路径，在用户确认后再进行修改。
3. 执行重命名：
   - 全局替换**原有算子名**为**新算子名**（同时处理文件名与内容）。
   - 修改文件名（如 `old_name.py` -> `new_name.py`）
   - 修改函数名（如 `def old_name(...)` -> `def new_name(...)`）
   - 修改测试文件中的引用
   - 修改benchmark文件中的引用
   - 修改README中的引用
4. 输出改动说明与假设。

## 重命名范围

### 需要重命名的文件
- `{old_name}.py` -> `{new_name}.py`
- `test_{old_name}.py` -> `test_{new_name}.py`
- `benchmark_{old_name}.py` -> `benchmark_{new_name}.py`

### 需要重命名的内容
- 函数名
- 类名（如果有）
- 文档中的引用
- 测试用例名称

## 约束
- 只涉及到名称变更，不进行其他修改。
- 重命名后需要确保所有引用都已更新。
- 重命名后需要确保测试仍然能够运行。

## 示例

```
用户输入：
- 旧名称：matmul_relu
- 新名称：matmul_gelu
- 路径：projects/gemm/matmul_relu

修改内容：
1. matmul_relu.py -> matmul_gelu.py
2. test_matmul_relu.py -> test_matmul_gelu.py
3. 函数名：matmul_relu -> matmul_gelu
4. 测试类名：TestMatmulRelu -> TestMatmulGelu
```
