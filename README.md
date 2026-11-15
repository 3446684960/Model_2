# 真实 Mosaic 与 PAN 融合对比实验

统一入口位于仓库根目录，当前使用 `real_world` 数据。模型源码集中在 `compare` 下。

## 论文与统一方案

三篇论文均为 2026 年。UVPL 时间最晚，其 PDF 标注于 2026 年 8 月 30 日接收，Crossref 的 DOI 登记日期为 2026 年 9 月 24 日。共同训练设置与指标实现采用 UVPL 的真实数据方案。训练算法由各模型的官方函数执行。[UVPL 论文](https://doi.org/10.1109/TNNLS.2026.3730458)

| 项目 | 统一设置 |
| --- | --- |
| 训练轮数 | 每阶段 10 epochs |
| 每轮训练批数 | 1000 |
| 每阶段训练步数 | 10,000 |
| batch size | 16 |
| 优化器 | Adam |
| 学习率 | 固定 `1e-4` |
| weight decay | 0，采用官方 Adam 默认值 |
| `train_size` | 64 |
| `stride` | 64 |
| 保存与可视化 | 每轮一次 |

对于融合阶段，`train_size=64` 对应 PAN 的 `64 × 64` 图块，以及 Mosaic 的 `32 × 32` 图块。独立去马赛克阶段按其官方数据读取器裁剪 `64 × 64` 的 Mosaic。

统一训练入口在每轮提供 1000 批数据。数据不足时重新遍历训练集，重新遍历沿用官方 DataLoader 的打乱方式。LSAN、SFNet、SpNet 的组合先训练去马赛克阶段，再训练融合阶段，每个阶段分别执行 10,000 步。PPID 采用官方确定性去马赛克方法，随后训练融合阶段。

## 目录

```text
Model_L_2/
├── train.py
├── test.py
├── quality_index.py
├── experiment.py
├── run_realworld.sh
├── README.md
└── compare/
    ├── EFN/ours/
    ├── EBVIF/EBVIF/
    ├── UVPL/UVPL/
    ├── LSAN_plus_PanGAN/
    ├── LSAN_plus_VBPN/
    ├── LSAN_plus_WFANet/
    ├── PPID_plus_PanGAN/
    ├── PPID_plus_VBPN/
    ├── PPID_plus_WFANet/
    ├── SFNet_plus_PanGAN/
    ├── SFNet_plus_VBPN/
    ├── SFNet_plus_WFANet/
    ├── SpNet_plus_PanGAN/
    ├── SpNet_plus_VBPN/
    ├── SpNet_plus_WFANet/
    └── official_sources.json
```

EFN、EBVIF、UVPL 保留原有目录。其余 12 个组合模型直接位于 `compare` 下。重复组合优先保留 UVPL 仓库的版本，其次为 EBVIF、EFN。UVPL 仓库中的 EFN 副本也已去重，采用 EFN 自身官方仓库的模型。

每个保留目录的所有文件及 SHA256、原始路径、来源仓库和提交记录保存在 `compare/official_sources.json`。保留的官方源文件与下载时逐字节一致。统一入口调用官方训练、数据处理和推理函数，并在入口处提供共同训练参数、每轮批数和评价函数。

## 评价设置

根目录 `quality_index.py` 的 12 个函数直接取自 UVPL 官方实现，函数代码保持一致。

| 阶段 | patch size | p | q | alpha | beta | 输入数值尺度 |
| --- | --- | --- | --- | --- | --- | --- |
| 训练期间验证与融合权重选择 | 128 | 0.3 | 1 | 0.8 | 1.2 | 归一化到约 0–1 |
| 最终测试 | 64 | 1 | 1 | 1 | 1 | 官方生成文件的 12 位量化数值 |

两个阶段分别采用 UVPL 对应的官方参数。高斯平滑采用 `7 × 7`、`sigma=2`，作用于重建高光谱和 PAN。训练期间按平均 QNR 选择融合阶段最佳权重。独立去马赛克阶段使用官方训练损失选择最佳权重。

EBVIF 及两个 SpNet 与 PanGAN/VBPN 组合的官方训练器传入 12 位数值，统一入口先除以 4096，再使用共同的训练验证指标。其他训练器已传入归一化数据。

最终测试逐张计算 QNR、D_lambda、D_S，然后对测试图取算术平均。全体模型采用同一根目录评价函数。

## 数据与环境

默认数据根目录为当前账户的 `~/autodl-pub/Masico`。所述 root 账户对应 `/root/autodl-pub/Masico`。

```text
Masico/real_world/
├── origin/
├── train/
│   ├── mosaic/
│   └── pan/
└── test/
    ├── mosaic/
    └── pan/
```

官方 RAW 读取器使用 int16。Mosaic 文件含 `16 × 255 × 276` 个数值，交错恢复为 `1020 × 1104`，PAN 为 `2040 × 2208`，两路除以 4096。

服务器使用已安装 CUDA 版 PyTorch 的环境。EFN 官方完整依赖表位于 `compare/EFN/requirements.txt`。

```bash
python -c 'import torch, numpy, scipy, h5py, cv2, matplotlib, tqdm; print(torch.__version__); print(torch.cuda.is_available())'
```

## 运行

同步整个仓库后，在仓库根目录执行以下命令。

```bash
python train.py --models all --data-root "$HOME/autodl-pub/Masico" --device 0
python test.py --models all --data-root "$HOME/autodl-pub/Masico" --device 0
```

第二条命令依次生成各模型结果并计算共同指标。也可以使用根目录脚本一次执行全流程。

```bash
DATA_ROOT="$HOME/autodl-pub/Masico" GPU=0 bash run_realworld.sh all all
```

以上命令在前台实时显示训练进度。进度条包含模型、阶段、Epoch、批次进度、本轮平均 loss 和当前学习率。loss 取自主优化器更新时使用的损失，PanGAN 对应生成器损失。日志文件按轮记录时间、模型名称、轮次和学习率。记录时间使用 UTC+8，异常信息也会写入日志。

根目录入口负责显示和日志记录。保留的官方模型源文件仍可通过来源清单中的 SHA256 核对。显示更新在同步后的新进程中生效。

单独训练和评估所选模型。

```bash
python train.py --models EFN EBVIF UVPL --device 0
python train.py --models LSAN_plus_VBPN --device 0
python test.py --models LSAN_plus_VBPN --device 0
```

分阶段执行时使用以下入口。

```bash
python train.py --models LSAN_plus_VBPN --stage demosaic --device 0
python train.py --models LSAN_plus_VBPN --stage fusion --device 0
python test.py --models all --stage generate --device 0
python test.py --models all --stage evaluate --device 0
```

查看清单或仅预览调用计划。

```bash
python train.py --list-models
python train.py --models all --dry-run
python test.py --models all --dry-run
```

## 输出与检查

默认输出在根目录 `runs/real_world` 下，每个模型单独保存。

```text
runs/real_world/<模型名>/
├── 1/model/best_*.pth           # 独立去马赛克阶段
├── 2/model/best_*.pth           # 融合阶段
├── train_demosaic.log
├── train_fusion.log
├── generate.log
├── test.log
├── test_per_image.csv
├── test_summary.csv
├── 2/record/
│   ├── record.json
│   ├── validation.log
│   ├── validation_per_image.csv
│   └── validation_summary.csv
└── real_world/1/result/mat/
    ├── mosaic/
    ├── pan/
    └── fused/
```

汇总指标保存为 `runs/real_world/metrics.csv`。推理时自动读取各阶段唯一的 `best_*.pth`。训练入口检查阶段输出中是否已有权重，以避免官方从头训练逻辑混入旧权重。安排新实验时可通过 `--output-root` 指定另一个输出目录，训练与测试使用同一目录。

融合阶段每轮训练后，使用 `real_world/test` 中的数据进行验证。逐图 QNR、D_lambda、D_S 写入 `validation_per_image.csv`，本轮平均指标写入 `validation_summary.csv`，终端和 `validation.log` 同时显示结果。训练验证图像按官方读取顺序编号。

根目录 `test.py` 也会在测试过程中逐图输出指标和进度。逐图结果写入 `test_per_image.csv`，该次测试的平均结果写入 `test_summary.csv`，可读记录写入 `test.log`。独立测试用 MAT 文件名标识图像。CSV 和评价日志以时间记录每次评价，支持保留重复测试的历史。

本地完成源码哈希、语法、调用接口和预览检查。真实数据读取、显卡训练和模型推理的验证由同步到服务器后的运行完成。
