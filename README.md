# Mosaic 与 PAN 融合对比实验

统一训练与测试入口位于仓库根目录，支持 `CAVE`、`ICVL`、`real_world`。15 个模型的源码集中在 `compare` 下。

## 论文与统一方案

三篇论文均为 2026 年。UVPL 时间最晚，其 PDF 标注于 2026 年 8 月 30 日接收，Crossref 的 DOI 登记日期为 2026 年 9 月 24 日。CAVE 和 real_world 的共同设置采用 UVPL 官方配置。UVPL 未提供 ICVL 配置，ICVL 沿用 EBVIF 官方的读取方式与数据集配置。训练算法由各模型的官方函数执行。[UVPL 论文](https://doi.org/10.1109/TNNLS.2026.3730458)

| 项目 | CAVE | ICVL | real_world |
| --- | --- | --- | --- |
| 每阶段训练轮数 | 100 | 10 | 10 |
| 每轮训练批数 | 最多 500，遍历结束即停止 | 最多 500，遍历结束即停止 | 1000，不足时重新遍历 |
| batch size | 16 | 16 | 16 |
| 优化器 | Adam | Adam | Adam |
| 学习率 | 固定 `1e-4` | 固定 `1e-4` | 固定 `1e-4` |
| `train_size` | 64 | 64 | 64 |
| `stride` | 32 | 64 | 64 |
| 仿真噪声标准差范围 | 0.01–0.05 | 0.008–0.01 | 使用实测输入 |

CAVE 的每轮批数沿用 UVPL 官方循环的上限规则，实际批数取决于读取器产生的图块数量。例如 4500 个图块、batch size 16 对应每轮 282 批。ICVL 使用同样的循环规则。real_world 延续每轮 1000 批的统一设置。EBVIF 仿真训练循环的提前退出条件改为统一批数上限，避免在数据遍历结束前额外跳过最后一批。

融合阶段的 `train_size=64` 对应 PAN 的 `64 × 64` 图块，以及 Mosaic 的 `32 × 32` 图块。独立去马赛克阶段沿用各模型官方的训练样本构造。LSAN、SFNet、SpNet 的组合先训练去马赛克阶段，再训练融合阶段。PPID 采用官方确定性去马赛克方法，随后训练融合阶段。

`noise_level` 传入支持噪声的官方读取器。EFN/WFANet 的官方仿真不加噪声，EBVIF/部分 SpNet 读取器包含额外模糊。当前接入保留这些官方预处理差异，统一仿真方式需另行确认。

同一数据集下，各模型使用上表的共同训练参数。可通过 `--epochs`、`--steps-per-epoch`、`--stride`、`--noise-level` 覆盖对应设置。

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

来源仓库、原始路径和提交记录保存在 `compare/official_sources.json`。统一入口调用各模型的官方训练和推理函数，并提供共同训练参数、批数规则与评价函数。缺少 ICVL 分支的读取器与生成脚本已补入 EFN/EBVIF 的读取方式。

## 评价设置

CAVE、ICVL 采用 UVPL 官方最终测试中的 float32 CPU 计算方式，逐张计算 PSNR、SSIM、SAM、ERGAS、Q2n，再对测试图取算术平均。融合阶段每轮验证与最终测试调用同一个 `calc_reference_test`，按平均 PSNR 选择最佳融合权重。独立去马赛克阶段也记录这些指标，评价对象是该阶段官方读取器返回的目标图像。ERGAS 使用 UVPL 测试函数的默认 `scale_factor=4`。

根目录 `quality_index.py` 的 12 个原始函数取自 UVPL 官方实现。验证与测试共同调用根目录新增的 `calc_qnr_test`，采用 UVPL 最终测试配置。

| 阶段 | patch size | p | q | alpha | beta | 输入数值尺度 |
| --- | --- | --- | --- | --- | --- | --- |
| 训练期间验证与融合权重选择 | 64 | 1 | 1 | 1 | 1 | 12 位量化数值 |
| 最终测试 | 64 | 1 | 1 | 1 | 1 | 官方生成文件的 12 位量化数值 |

训练验证与最终测试采用同一套数值尺度、分块大小和指标参数。高斯平滑采用 UVPL 最终测试实现的 `7 × 7`、`sigma=2`，作用于重建高光谱和 PAN。训练期间按这套评价得到的平均 QNR 选择融合阶段最佳权重。独立去马赛克阶段使用官方训练损失选择最佳权重。

EBVIF 及两个 SpNet 与 PanGAN/VBPN 组合的官方训练器已向指标函数传入 12 位数值。其他训练器传入的归一化评价张量在根目录指标适配器中乘以 4096，得到与最终测试一致的数值尺度。

最终测试逐张计算 QNR、D_lambda、D_S，然后对测试图取算术平均。全体模型采用同一根目录评价函数。

## 数据与环境

默认数据根目录为当前账户的 `~/autodl-pub/Masico`。所述 root 账户对应 `/root/autodl-pub/Masico`。

CAVE 的 MAT 文件读取键 `b`，选取 Python 索引 `12:28` 的 16 个波段。ICVL 的 HDF5 MAT 文件读取键 `rad`，按 EFN/EBVIF 的实现调整维度、旋转、逐波段最大值归一化，然后选取 `12:28`。文件结构如下。

```text
Masico/
├── CAVE/
│   ├── train/*.mat
│   └── test/*.mat
├── ICVL/
│   ├── train/*.mat
│   └── test/*.mat
└── real_world/
```

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

同步整个仓库后，在仓库根目录执行以下命令，按 CAVE、ICVL、real_world 的顺序运行全部 15 个模型。每个数据集中的每个模型训练完成后立即生成结果并测试。

```bash
python -u train.py --datasets all --models all --data-root "$HOME/autodl-pub/Masico" --device 0
```

分别运行某个数据集，或为已有权重重新生成结果并测试。

```bash
python -u train.py --datasets CAVE --models all --device 0
python -u train.py --datasets ICVL --models UVPL --device 0
python -u test.py --datasets all --models all --device 0
```

`test.py --stage evaluate` 直接评估已生成的 MAT 文件。独立测试默认选取各数据集、各模型最近的实验编号。训练会为每个数据集分别训练权重。

省略 `--datasets` 时默认运行 real_world，原有命令继续使用。

```bash
python train.py --models all --data-root "$HOME/autodl-pub/Masico" --device 0
```

每个模型完成训练后，立即加载它的最佳权重、生成重建结果并计算最终测试指标，然后处理下一个模型。也可以使用根目录脚本执行这个全流程。

```bash
DATA_ROOT="$HOME/autodl-pub/Masico" GPU=0 bash run_realworld.sh all all
```

指定一个模型即可开始新实验。每次完整训练自动取该模型目录下的最大数字加 1，训练和自动测试共用本次编号。

```bash
bash run_realworld.sh EFN all
bash run_realworld.sh UVPL all
```

EFN 的不同实验分别保存在 `runs/real_world/EFN/1`、`EFN/2` 等目录。每个模型独立递增编号。

以上命令在前台实时显示训练进度。进度条包含模型、阶段、Epoch、批次进度、本轮平均 loss 和当前学习率。loss 取自主优化器更新时使用的损失，PanGAN 对应生成器损失。日志文件按轮记录时间、模型名称、轮次和学习率。记录时间使用 UTC+8，异常信息也会写入日志。

根目录入口负责显示和日志记录。显示更新在同步后的新进程中生效。

单独训练和评估所选模型。

```bash
python train.py --models EFN EBVIF UVPL --device 0
python train.py --models LSAN_plus_VBPN --device 0
```

独立测试已有权重时执行以下命令。

```bash
python test.py --models LSAN_plus_VBPN --device 0
```

独立生成和测试默认使用该模型最近的编号。选择某次已有实验时使用 `--run-id`。

```bash
python test.py --models EFN --run-id 2 --device 0
```

分阶段训练和测试时使用以下入口。

```bash
python train.py --models LSAN_plus_VBPN --stage demosaic --train-only --device 0
python train.py --models LSAN_plus_VBPN --stage fusion --train-only --device 0
python test.py --models all --stage generate --device 0
python test.py --models all --stage evaluate --device 0
```

`--train-only` 仅执行所选训练阶段。脚本 `run_realworld.sh <模型名> train` 使用此模式，`all` 模式执行每个模型的训练和自动测试。分阶段执行两阶段模型时，去马赛克阶段创建新编号，后续融合阶段默认接续最近编号，也可以通过 `--run-id` 选择已有编号。自定义评价分块时，`train.py --patch-size` 的值会同时传给每轮验证和训练后的测试。

查看清单或仅预览调用计划。

```bash
python train.py --list-models
python train.py --models all --dry-run
python test.py --models all --dry-run
```

## 输出与检查

输出按数据集分别保存在 `runs/CAVE`、`runs/ICVL`、`runs/real_world`，每个数据集中的模型独立递增实验编号。每个数据集根目录下的 `metrics.csv` 汇总该数据集所有模型、各次实验的最终指标。每次实验也保存独立的 `metrics.csv`。

单数据集的 `--output-root` 直接指定该数据集的输出根目录。选择多个数据集时，入口会在指定根目录下分别创建各数据集子目录。

下面以 real_world 为例。CAVE、ICVL 使用相同的目录结构，生成结果额外包含 `gt/`。

```text
runs/real_world/<模型名>/<实验编号>/
├── .run.json                   # 实验编号记录
├── metrics.csv                 # 本次实验的最终指标
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

所有实验的汇总指标保存为 `runs/real_world/metrics.csv`，其中 `run_id` 标识模型的实验编号。各次实验也在自己的目录中保存 `metrics.csv`。推理时自动读取所选实验各阶段唯一的 `best_*.pth`。

完整训练会自动创建新编号目录。`--output-root` 可用于选择另一处实验根目录。旧版直接存放在模型目录中的结果按编号 0 读取，指定 `--run-id 0` 即可选择它们。已有旧版数字目录也计入最大编号，例如旧的 `EFN/2/model` 存在时，后续新实验从编号 3 开始。

融合阶段每轮训练后，使用所选数据集的 `test` 中的数据进行验证。CAVE、ICVL 记录 PSNR、SSIM、SAM、ERGAS、Q2n。real_world 的逐图 QNR、D_lambda、D_S 写入 `validation_per_image.csv`，本轮平均指标写入 `validation_summary.csv`，终端和 `validation.log` 同时显示结果。训练验证图像按官方读取顺序编号。

根目录 `test.py` 也会在测试过程中逐图输出指标和进度。逐图结果写入 `test_per_image.csv`，该次测试的平均结果写入 `test_summary.csv`，可读记录写入 `test.log`。独立测试用 MAT 文件名标识图像。CSV 和评价日志以时间记录每次评价，支持保留重复测试的历史。

训练数据读取、显卡训练和模型推理由同步到服务器后的运行完成。
