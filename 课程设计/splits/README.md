# 数据划分文件

本目录保存可复现的数据划分清单。图像文件仍存放在 `datasets/` 目录中。

重新生成 LOL-v1 固定划分：

```powershell
python scripts/split_lol.py --data-root datasets --seed 42
```

默认划分：

- `train`：LOL-v1 our485 中的 435 对图像。
- `val`：LOL-v1 our485 中的 50 对图像。
- `test`：LOL-v1 eval15 中的 15 对图像。

实验中应使用训练集优化模型，验证集选择 checkpoint，测试集只用于最终评价。
