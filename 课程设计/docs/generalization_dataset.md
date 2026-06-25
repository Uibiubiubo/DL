# 泛化测试数据说明

无参考泛化测试图片存放在：

```text
datasets/generalization/low/
```

这些图片来自 Figshare 公开低光照增强数据集合：

- 数据集名称：`low light image enhancement datasets`
- DOI：`10.6084/m9.figshare.27192921.v8`
- 地址：`https://figshare.com/articles/dataset/_Information_zip/27192921`
- 引用信息：`Yin, Mohan (2024). low light image enhancement datasets. figshare. Dataset. https://doi.org/10.6084/m9.figshare.27192921.v8`
- 许可协议：CC BY 4.0

原始下载压缩包包括：

```text
DICM.rar
LIME.rar
MEF.rar
NPE.rar
```

当前提交版保留的样本包括：

```text
dicm_01.jpg ... dicm_05.jpg
lime_01.bmp ... lime_05.bmp
mef_01.png ... mef_05.png
npe_01.png ... npe_05.png
```

这些图片没有对应的正常曝光参考图，因此只用于视觉泛化比较和无参考行为指标统计，例如亮度提升、暗区提升、过曝比例、颜色偏移和推理时间等。

注意：该泛化集合不能报告 PSNR 或 SSIM，因为它没有 ground truth。
