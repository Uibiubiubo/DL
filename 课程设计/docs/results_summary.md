# 实验结果汇总

本文档汇总当前提交版中可直接用于课程报告的主要实验结果。

## 实验设置

- 硬件环境：低算力笔记本 GPU 环境
- 数据集：LOL-v1
- 训练集：`our485` 训练划分，435 对图像
- 验证集：`our485` 验证划分，50 对图像
- 最终测试集：`eval15`，15 对图像
- 固定划分文件：`splits/lol_v1_seed42.json`

## 主要结果文件

- 严格 50 epoch 消融汇总：`outputs/strict_ablation_50epoch_with_zero_dce_summary.csv`
- 严格 50 epoch 测试明细：`outputs/strict_ablation_50epoch_test_eval.csv`
- Zero-DCE-Lite 测试明细：`outputs/strict_ablation_50epoch_zero_dce_lite_test_eval.csv`
- 模型复杂度表：`outputs/benchmarks/model_complexity_256_with_zero_dce.csv`
- LPIA-Former-Lite 辅助可解释性结果：`outputs/results/aux_visualizations_lpia_former_lite_strict50/`
- 无参考泛化指标：`outputs/generalization_eval.csv`
- 泛化对比图：`outputs/results/generalization/`
- Gradio 演示入口：`run_gradio_app.py`

## 传统方法与原始主模型对比

以下结果来自 `outputs/all_eval.csv`，测试集为 LOL-v1 `eval15`。

| 方法 | PSNR | SSIM | 时间/ms | 平均亮度提升 | 暗区亮度提升 | 过曝比例 |
|---|---:|---:|---:|---:|---:|---:|
| Gamma(0.6) | 10.67 | 0.4770 | 9.25 | 0.1074 | 0.1073 | 0.000006 |
| CLAHE | 9.13 | 0.3565 | 15.98 | 0.0567 | 0.0565 | 0.000006 |
| Retinex | 15.49 | 0.4112 | 648.62 | 0.4780 | 0.4780 | 0.000008 |
| LPIA-LightU-Net | 20.39 | 0.8110 | 59.85 | 0.3950 | 0.3947 | 0.000681 |

结论：原始 LPIA-LightU-Net 在有参考质量指标上明显优于传统增强方法。Retinex 的亮度提升较强，但速度慢，SSIM 明显较低。

## 严格 50 Epoch 消融实验

以下结果使用固定 train/val/test 划分。模型 checkpoint 根据验证集选择，最终结果在测试集上报告。

| 模型变体 | Val PSNR | Val SSIM | Val Loss | Test PSNR | Test SSIM | 测试时间/ms | 过曝比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_unet | 21.62 | 0.8501 | 0.1510 | 19.81 | 0.8002 | 22.20 | 0.000568 |
| attention_unet | 21.52 | 0.8552 | 0.1491 | 19.99 | 0.8039 | 23.37 | 0.000738 |
| fixed_gamma_unet | 21.58 | 0.8549 | 0.1501 | 20.62 | 0.8110 | 27.35 | 0.000506 |
| learnable_gamma_unet | 21.49 | 0.8472 | 0.1567 | 20.17 | 0.8001 | 25.80 | 0.000805 |
| lpia_lightunet | 21.53 | 0.8573 | 0.1479 | 20.47 | 0.8082 | 26.99 | 0.000673 |
| lpia_former_lite | 21.67 | 0.8567 | 0.1493 | 20.43 | 0.8096 | 24.01 | 0.000864 |
| zero_dce_lite | 16.80 | 0.5414 | 0.3549 | 18.74 | 0.5557 | 28.54 | 0.000062 |

结论：`fixed_gamma_unet` 在本次最终测试集上取得最高 PSNR 和 SSIM，因此报告中不应宣称 LPIA-Former-Lite 在所有测试指标上第一。更准确的结论是：LPIA-Former-Lite 取得最高验证集 PSNR，在最终测试集上接近最优 PSNR/SSIM，同时模型体积更小，具有更好的轻量化价值。

## 模型复杂度

输入尺寸：`1 x 3 x 256 x 256`。推理时间采用 warmup 后多次重复统计。

| 模型变体 | 参数量/M | 模型大小/MB | GFLOPs | 时间/ms | 时间标准差/ms | 峰值显存/MB |
|---|---:|---:|---:|---:|---:|---:|
| plain_unet | 1.287 | 4.91 | 5.50 | 5.13 | 0.69 | 46.68 |
| attention_unet | 1.288 | 4.91 | 5.50 | 5.30 | 0.68 | 46.68 |
| fixed_gamma_unet | 1.287 | 4.91 | 5.57 | 5.47 | 0.89 | 49.43 |
| learnable_gamma_unet | 1.290 | 4.92 | 5.94 | 5.70 | 0.83 | 50.32 |
| lpia_lightunet | 1.291 | 4.93 | 5.94 | 6.04 | 0.81 | 49.45 |
| lpia_former_lite | 0.620 | 2.36 | 4.92 | 6.50 | 1.01 | 55.64 |
| zero_dce_lite | 0.045 | 0.17 | 5.88 | 5.34 | 1.55 | 22.65 |

结论：相比 LPIA-LightU-Net，LPIA-Former-Lite 参数量由约 1.29M 降至约 0.62M，模型大小由约 4.93MB 降至约 2.36MB。窗口注意力带来一定推理开销，但整体仍处于轻量图像增强模型范围。

## Zero-DCE-Lite 基线

`zero_dce_lite` 是一个轻量监督式 Zero-DCE 风格曲线估计基线，不等同于官方预训练 Zero-DCE 模型。它用于在相同 LOL-v1 划分和训练流程下比较现代曲线估计类方法。

关键结果：

- 测试集 PSNR：18.74
- 测试集 SSIM：0.5557
- 参数量：0.045M
- 模型大小：0.17MB

结论：该曲线估计基线极其轻量，但在本项目的监督 LOL-v1 设置下，SSIM 明显弱于 U-Net 系列和 LPIA-Former-Lite，说明低光照增强仍需要更丰富的空间特征提取和光照建模能力。

## 可解释性输出

LPIA-Former-Lite 已为 5 张测试图导出辅助可解释性结果：

```text
outputs/results/aux_visualizations_lpia_former_lite_strict50/1/
outputs/results/aux_visualizations_lpia_former_lite_strict50/22/
outputs/results/aux_visualizations_lpia_former_lite_strict50/23/
outputs/results/aux_visualizations_lpia_former_lite_strict50/55/
outputs/results/aux_visualizations_lpia_former_lite_strict50/79/
```

每个目录包含：

- 增强结果
- 参考图
- Gamma 热力图
- Illumination 热力图
- Dark Mask
- 误差图
- 亮度直方图
- 辅助图汇总表

这些图说明模型不是单纯的黑箱亮度放大器，而是会预测与光照相关的辅助图，突出暗区，并展示重建误差位置。

## 无参考泛化测试

泛化图片来自 DICM、LIME、MEF 和 NPE 样本，存放于 `datasets/generalization/low/`。这些图片没有正常曝光参考图，因此不报告 PSNR/SSIM。

| 方法 | 平均亮度提升 | 暗区亮度提升 | 过曝比例 |
|---|---:|---:|---:|
| Gamma(0.6) | 0.1171 | 0.1169 | 0.019244 |
| Retinex | 0.2751 | 0.3244 | 0.008445 |
| Zero-DCE-Lite | 0.3602 | 0.3910 | 0.100674 |
| LPIA-LightU-Net | 0.1456 | 0.1548 | 0.002561 |
| LPIA-Former-Lite | 0.1669 | 0.1728 | 0.005028 |

结论：Zero-DCE-Lite 亮度提升最强，但过曝比例最高。LPIA-Former-Lite 相比 LPIA-LightU-Net 亮度提升更明显，同时过曝比例远低于 Zero-DCE-Lite，体现出更保守、稳定的增强行为。

## 推荐最终结论

推荐在报告中使用如下表述：

`LPIA-Former-Lite` 是对原始 LPIA-LightU-Net 的轻量现代化扩展。它结合 Retinex/Gamma 先验和瓶颈层窗口注意力，在严格 50 epoch 实验中取得最高验证集 PSNR，最终测试集 PSNR/SSIM 接近最优模型，同时相较 LPIA-LightU-Net 将参数量减少约一半。

需要避免的表述：

- 不应声称 LPIA-Former-Lite 在所有最终测试指标上都是第一。
- 不应声称 zero_dce_lite 是官方 Zero-DCE 结果。
- 不应使用 eval15 测试集选择 checkpoint。

## 后续可改进方向

- 在依赖环境稳定后加入 LPIPS、NIQE 或 BRISQUE。
- 补充更多真实手机低光照图片用于泛化展示。
- 从上述 CSV 文件中继续整理正式报告表格。
