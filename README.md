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
| 周期权重保存间隔 | 10 轮 | 5 轮 | 1 轮 |
| 仿真噪声标准差范围 | 0.01–0.05 | 0.008–0.01 | 使用实测输入 |

CAVE 的每轮批数沿用 UVPL 官方循环的上限规则，实际批数取决于读取器产生的图块数量。例如 4500 个图块、batch size 16 对应每轮 282 批。ICVL 使用同样的循环规则。real_world 延续每轮 1000 批的统一设置。实际更新由各模型的官方循环执行，EBVIF 仿真训练采用官方跳过最后一批的条件。

融合阶段的 `train_size=64` 对应 PAN 的 `64 × 64` 图块，以及 Mosaic 的 `32 × 32` 图块。独立去马赛克阶段沿用各模型官方的训练样本构造。LSAN、SFNet、SpNet 的组合先训练去马赛克阶段，再训练融合阶段。PPID 采用官方确定性去马赛克方法，随后训练融合阶段。

CAVE、ICVL 的全部 15 个模型共同调用根目录 `simulation.py` 生成相机观测。高光谱经 `2 × 2` 平均池化后，按波段索引 `0–15` 的行优先 `4 × 4` MSFA 排列生成马赛克。PAN 使用 UVPL 的归一化光谱响应向量 `[1,1,2,4,8,9,10,12,16,12,10,9,7,3,2,1]` 加权求和。两路分别从表中范围采样高斯噪声标准差，加入独立高斯噪声后裁剪至 `[0,1]`。

所有模型按相同的文件名排序处理场景。训练观测使用独立 NumPy 随机生成器与种子 42，验证和最终测试观测共同使用种子 22。共同仿真从参考高光谱的平均池化开始执行，因此此前部分读取器的观测预模糊已由 UVPL 方式取代。每个模型随后通过其官方数据读取器构造训练图块、监督目标或自监督退化样本。

仿真缓存文件带有 `uvpl_v1`、数据集、噪声范围、图块、步长和阵列参数，按新协议独立保存。训练通过官方入口的 `idx` 目录写入模型与记录。

同一数据集下，各模型使用上表的共同训练参数。可通过 `--epochs`、`--steps-per-epoch`、`--stride`、`--noise-level` 覆盖对应设置。

## 目录

```text
Model_L_2/
├── train.py
├── test.py
├── quality_index.py
├── simulation.py
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

CAVE、ICVL 的训练验证调用官方 PSNR 函数并选择最佳权重。最终测试调用 UVPL 官方 `test.py`，采用 float32 CPU 计算方式，逐张计算 PSNR、SSIM、SAM、ERGAS、Q2n，再对测试图取算术平均。ERGAS 使用官方测试函数的默认 `scale_factor=4`。

根目录 `quality_index.py` 的 12 个原始函数取自 UVPL 官方实现。训练验证通过 `calc_qnr_test` 使用 UVPL 最终测试配置，最终测试由 UVPL 官方 `test.py` 调用这些指标函数。

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

服务器在仓库根目录执行全流程。

```bash
python -u train.py --datasets all --models all --data-root "$HOME/autodl-pub/Masico" --device 0
```

选择数据集、模型或阶段时执行相应命令。

```bash
python -u train.py --datasets CAVE --models UVPL --device 0
python -u train.py --datasets ICVL --models all --device 0
python -u train.py --models LSAN_plus_VBPN --stage demosaic --train-only --device 0
python -u train.py --models LSAN_plus_VBPN --stage fusion --device 0
python -u test.py --datasets all --models all --device 0
python -u test.py --models UVPL --stage evaluate --device 0
```

real_world 的原有脚本也可调用。

```bash
DATA_ROOT="$HOME/autodl-pub/Masico" GPU=0 bash run_realworld.sh all all
```

`--train-only` 执行训练阶段。`test.py --stage generate` 生成结果，`--stage evaluate` 评估已生成的结果。`--dry-run` 预览命令，`train.py --list-models` 列出 15 个模型。

## 官方输出

每个数据集、模型使用固定工作目录 `runs/<数据集>/<模型>`。官方训练入口使用自身的 `idx` 子目录，EFN、EBVIF、UVPL 对应 `1`，两阶段组合的去马赛克和融合分别对应 `1`、`2`。

```text
runs/<数据集>/<模型>/
├── 1/
│   ├── model/
│   ├── record/record.json
│   └── 官方数据缓存
├── 2/                         # 两阶段组合的融合输出
│   ├── model/
│   ├── record/record.json
│   └── 官方数据缓存
└── <数据集>/1/result/mat/      # 官方 generate.py 的结果
```

训练记录采用官方 `record.json`、缓存和模型权重。周期保存间隔采用上表的官方示例设置，可通过 `--save-freq` 指定。模型在相同工作目录再次训练时，按官方脚本写入记录与权重，自动生成结果时加载该阶段最近保存的 `best_*.pth`。

最终指标由 UVPL 官方测试脚本输出到终端。Git 同步采用官方 `record.json`，运行缓存、日志以及 PNG、MAT、PTH 文件通过 `.gitignore` 过滤。
