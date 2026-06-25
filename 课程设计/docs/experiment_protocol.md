# 实验协议

本文档说明本项目采用的实验流程、数据划分、对比方法和评价指标，用于保证课程设计结果可复现、可核对。

## 数据划分

固定划分文件由下面命令生成：

```powershell
python scripts/split_lol.py --data-root datasets --seed 42
```

默认划分如下：

```text
train：LOL-v1 our485，435 对图像
val：LOL-v1 our485，50 对图像
test：LOL-v1 eval15，15 对图像
```

划分规则：

- `train` 只用于模型训练。
- `val` 用于选择最佳 checkpoint 和模型早期比较。
- `test` 只用于最终结果报告。
- 最终测试集不能用于选择模型 checkpoint。

## 消融实验设置

所有模型变体应使用相同的数据划分、训练轮数、优化器、损失权重和评价指标实现。

本项目比较的模型变体包括：

```text
plain_unet
fixed_gamma_unet
learnable_gamma_unet
attention_unet
lpia_lightunet
lpia_former_lite
zero_dce_lite
```

其中 `lpia_former_lite` 是本项目升级后的轻量模型，结合 Retinex/Gamma 先验、轻量 CNN 编码解码器和瓶颈层窗口注意力，面向低算力设备和端侧轻量化场景。

严格消融实验入口：

```powershell
python run_strict_ablation.py
```

## 对比基线

传统增强方法：

```text
Gamma(0.6)
CLAHE
Retinex
```

深度学习基线：

```text
zero_dce_lite
```

`zero_dce_lite` 是一个轻量 Zero-DCE 风格监督复现模型，用于在相同 train/val/test 划分下提供现代曲线估计类增强基线。

## 评价指标

有参考指标：

```text
PSNR
SSIM
```

效率指标：

```text
参数量
模型大小
FLOPs
峰值显存
warmup 后平均推理时间
```

增强行为指标：

```text
平均亮度提升
暗区亮度提升
过曝比例
合理曝光比例
颜色偏移
```

模型复杂度统计入口：

```powershell
python run_benchmark.py --height 256 --width 256 --warmup 10 --repeats 50
```

## 推理时间统计规则

报告推理时间时，不应把 CUDA 初始化时间计入模型推理时间。

推荐统计流程：

```text
1. 加载模型和输入图像。
2. 先运行 warmup 迭代。
3. 再运行多次正式计时迭代。
4. 报告平均值、标准差和中位数。
5. 如果使用 GPU，计时前后需要同步 CUDA。
```

## 可视化证据

代表性图像应尽量导出以下结果：

```text
低光照输入图
参考图
增强图
Gamma Map 热力图
Illumination Map 热力图
Dark Mask
误差图
亮度直方图
局部细节对比图
```

辅助可解释性可视化入口：

```powershell
python run_visualize_aux.py --max-images 5
```

## 泛化测试

无参考泛化测试入口：

```powershell
python run_generalization.py
```

泛化图片存放在：

```text
datasets/generalization/low/
```

由于这些图片没有对应 ground truth，因此不报告 PSNR/SSIM，只报告亮度提升、暗区提升、过曝比例、颜色偏移和视觉对比结果。
