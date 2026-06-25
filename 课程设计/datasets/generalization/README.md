# 泛化测试图片

本目录保存少量无参考低光照泛化测试图片，用于视觉对比和无参考行为指标统计。

数据来源：

- Figshare 数据集：`low light image enhancement datasets`
- DOI：`10.6084/m9.figshare.27192921.v8`
- 公开地址：`https://figshare.com/articles/dataset/_Information_zip/27192921`
- Figshare 引用信息：`Yin, Mohan (2024). low light image enhancement datasets. figshare. Dataset. https://doi.org/10.6084/m9.figshare.27192921.v8`
- 许可协议：CC BY 4.0

当前样本来自 DICM、LIME、MEF 和 NPE 四类低光照图像集合。

工作目录：

```text
datasets/generalization/low/
```

这些图片只用于泛化展示，不提供正常曝光参考图。因此不要在该集合上报告 PSNR/SSIM，应使用亮度提升、暗区提升、过曝比例、颜色偏移和视觉对比图进行说明。
