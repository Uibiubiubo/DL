# LPIA-LightU-Net 低光照图像增强项目

本项目围绕低光照图像增强任务，构建了传统增强方法、轻量 U-Net、物理先验引导网络、轻量 Transformer 混合模型和现代曲线增强基线的完整实验流程。项目重点关注增强质量、端侧轻量化、泛化稳定性和可解释性分析。

## 项目内容

本项目包含以下核心内容：

- 传统增强方法：Gamma 校正、CLAHE、Retinex。
- 深度学习基线：Plain U-Net、Attention U-Net、Fixed Gamma U-Net、Learnable Gamma U-Net。
- 主方法：LPIA-LightU-Net。
- 升级模型：LPIA-Former-Lite。
- 现代基线：Zero-DCE-Lite。
- 固定 train/val/test 数据划分。
- 严格消融实验、泛化测试和模型复杂度分析。
- Gamma Map、Illumination Map、Dark Mask 等可解释性可视化。
- 研究型分析图和优势指标分析图。

## 推荐查看文件

| 文件 | 作用 |
|---|---|
| `DL课程设计.html` | 已导出的完整 HTML 报告，可直接用浏览器打开查看。 |
| `DL课程设计.ipynb` | 已运行完成的课程设计 Notebook，包含报告内容、表格和图片输出。 |
| `文件结构说明.md` | 说明当前文件夹中各文件和目录的作用。 |
| `docs/results_summary.md` | 汇总主要实验结果和结论。 |
| `docs/experiment_protocol.md` | 说明实验协议、数据划分和评价指标。 |
| `docs/generalization_dataset.md` | 说明泛化测试数据来源。 |

## 环境配置

建议先安装依赖：

```powershell
pip install -r requirements.txt
```

如果本机需要 CUDA 版本 PyTorch，请优先按照 PyTorch 官网命令安装对应版本，再安装其余依赖。

## 运行报告

重新执行课程设计 Notebook：

```powershell
jupyter nbconvert --to notebook --execute "DL课程设计.ipynb" --output "DL课程设计_运行结果.ipynb" --ExecutePreprocessor.timeout=600
```

重新导出 HTML：

```powershell
jupyter nbconvert --to html "DL课程设计.ipynb" --output "DL课程设计.html"
```

## 主要实验脚本

| 脚本 | 作用 |
|---|---|
| `run_lpia_training.py` | 单模型训练入口。 |
| `run_strict_ablation.py` | 严格消融实验入口。 |
| `run_quick_ablation.py` | 快速消融流程验证入口。 |
| `run_generalization.py` | 泛化测试和结果图生成入口。 |
| `run_benchmark.py` | 模型复杂度、FLOPs、推理时间统计入口。 |
| `run_visualize_aux.py` | 辅助可解释性图导出入口。 |
| `run_gradio_app.py` | Gradio 图像增强演示入口。 |

## 数据划分

项目使用固定随机种子划分，划分文件位于：

```text
splits/lol_v1_seed42.json
splits/lol_v1_seed42.csv
```

默认划分：

- `train`：LOL-v1 our485 中的 435 对图像。
- `val`：LOL-v1 our485 中的 50 对图像。
- `test`：LOL-v1 eval15 中的 15 对图像。

生成划分文件的命令：

```powershell
python scripts/split_lol.py --data-root datasets --seed 42
```

## 主要模型

### LPIA-LightU-Net

LPIA-LightU-Net 是原始主方法，将可学习 Gamma 光照先验、光照引导注意力和轻量 U-Net 结合，用于实现空间自适应低光照增强。

### LPIA-Former-Lite

LPIA-Former-Lite 是升级后的轻量模型，代码位于：

```text
lpia_project/models/lpia_former_lite.py
```

它结合 Retinex/Gamma 双先验、轻量 CNN 编码解码结构和瓶颈层窗口注意力，在保持较小模型体积的同时提升建模能力。

### Zero-DCE-Lite

Zero-DCE-Lite 是现代曲线估计类增强基线，代码位于：

```text
lpia_project/models/zero_dce_lite.py
```

它通过预测像素级曲线参数进行多轮增强，用于补充现代低光照增强路线的对比。

## 分析图复现

重新生成研究型模型对比分析图：

```powershell
python scripts/generate_analysis_charts.py
```

重新生成优势指标和优势分析图：

```powershell
python scripts/generate_advantage_metrics.py
```

生成结果位于：

```text
outputs/results/analysis_charts/
outputs/results/advantage_charts/
```

## 主要结论

本项目不声称主方法在所有指标上绝对第一。当前实验结果表明：

- `fixed_gamma_unet` 在 LOL 测试集 PSNR/SSIM 上取得最优结果。
- `LPIA-Former-Lite` 在接近最优重建质量的同时，模型体积更小，更符合端侧轻量化目标。
- 泛化测试中，Zero-DCE-Lite 增亮更激进，但过曝风险更高；LPIA 系列方法整体更保守稳定。
- 项目的核心价值在于形成了物理先验、轻量化结构、严格消融、泛化测试和可解释性分析的完整研究流程。
