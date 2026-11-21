#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_root=${CAVE_DATA_ROOT:-/root/autodl-pub/Masico}
python_bin=${COMPARE_PYTHON:-python}
models=(EFN UVPL EBVIF PPID_plus_PanGAN PPID_plus_VBPN LSAN_plus_PanGAN
        LSAN_plus_VBPN SFNet_plus_PanGAN SFNet_plus_VBPN)

usage() {
    cat <<'EOF'
用法: bash run_compare_cave.sh [--dry-run] [模型名 ...]
省略模型名会依次训练 compare 下全部 9 个 CAVE 模型。
每次训练写到 CAVE/<模型名>/<运行编号>/，编号从 0 开始。

CAVE_DATA_ROOT         数据集根目录，默认 /root/autodl-pub/Masico
COMPARE_GPU            GPU 编号，默认使用当前可见的第一张 GPU
COMPARE_PYTHON         Python 命令或绝对路径，默认 python
COMPARE_EPOCHS         覆盖所有阶段的训练轮数，默认按各方法 README
COMPARE_BATCH_SIZE     覆盖所有阶段的 batch size，默认按各方法 README
COMPARE_NUM_WORKERS    数据加载的 worker 数，默认 1

模型名: EFN UVPL EBVIF PPID_plus_PanGAN PPID_plus_VBPN
        LSAN_plus_PanGAN LSAN_plus_VBPN SFNet_plus_PanGAN SFNet_plus_VBPN
EOF
}

dry_run=false
selected=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true ;;
        EFN|UVPL|EBVIF|PPID_plus_PanGAN|PPID_plus_VBPN|LSAN_plus_PanGAN|LSAN_plus_VBPN|SFNet_plus_PanGAN|SFNet_plus_VBPN)
            selected+=("$arg") ;;
        *) printf '未知参数或模型名: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
    esac
done
if (( ${#selected[@]} )); then
    models=("${selected[@]}")
fi
if [[ ! ${COMPARE_EPOCHS:-1} =~ ^[1-9][0-9]*$ || ! ${COMPARE_BATCH_SIZE:-1} =~ ^[1-9][0-9]*$ ||
      ! ${COMPARE_NUM_WORKERS:-1} =~ ^(0|[1-9][0-9]*)$ ]]; then
    printf '轮数和 batch size 必须为正整数，worker 数必须为非负整数。\n' >&2
    exit 2
fi

data_view=
if ! $dry_run; then
    data_root=$(cd -- "$data_root" && pwd) || exit 1
    splits=()
    for split in train test; do
        split_path=
        for candidate in "$split" "${split^}" "cave_$split"; do
            split_files=("$data_root/CAVE/$candidate"/*.mat)
            if (( ${#split_files[@]} )); then
                split_path="$data_root/CAVE/$candidate"
                break
            fi
        done
        if [[ -z "$split_path" ]]; then
            printf '找不到 CAVE 的 %s 数据，请检查 %s/CAVE 下的 .mat 文件。\n' "$split" "$data_root" >&2
            exit 1
        fi
        splits+=("$split_path")
    done
    # 原训练入口要求小写目录。大小写不匹配时在临时目录映射。
    if [[ "${splits[0]}" != "$data_root/CAVE/train" || "${splits[1]}" != "$data_root/CAVE/test" ]]; then
        data_view=$(mktemp -d)
        trap 'rm -rf -- "$data_view"' EXIT
        mkdir -- "$data_view/CAVE"
        ln -s -- "${splits[0]}" "$data_view/CAVE/train"
        ln -s -- "${splits[1]}" "$data_view/CAVE/test"
        data_root=$data_view
    fi

    export CUDA_VISIBLE_DEVICES=${COMPARE_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
    python_bin=$(command -v "$python_bin") || { printf '找不到 Python 解释器。\n' >&2; exit 1; }
    "$python_bin" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else "当前 Python 环境没有可用的 CUDA GPU")'
fi

run_stage() {
    local run_dir=$1 model=$2 entry=$3 idx=$4 epochs=$5 batch_size=$6 save_freq=$7 stage=$8
    shift 8
    epochs=${COMPARE_EPOCHS:-$epochs}
    if (( save_freq > epochs )); then
        save_freq=$epochs
    fi
    local -a cmd=("$python_bin" -u "$project_root/compare/$model/$entry"
        --idx "$idx" --dataset CAVE --data_path "$data_root"
        --epochs "$epochs" --train_size 64 --stride 32
        --batch_size "${COMPARE_BATCH_SIZE:-$batch_size}" --lr_decay --save_freq "$save_freq"
        --device 0 --num_workers "${COMPARE_NUM_WORKERS:-1}" --noise_level 0.01 0.05 "$@")

    printf '\n[%s / %s] %s\n' "$model" "$stage" "$run_dir"
    printf '  %q' "${cmd[@]}"
    printf '\n'
    if ! $dry_run; then
        (cd -- "$run_dir" && "${cmd[@]}" 2>&1 | tee "$stage.log")
    fi
}

run_model() {
    local model=$1 model_dir="$project_root/CAVE/$1" run_id=0 run_dir
    if ! $dry_run; then
        mkdir -p -- "$model_dir" || return 1
    fi
    while true; do
        run_dir="$model_dir/$run_id"
        if [[ -e "$run_dir" || -L "$run_dir" ]]; then
            run_id=$((run_id + 1))
        elif $dry_run || mkdir -- "$run_dir"; then
            break
        elif [[ ! -e "$run_dir" && ! -L "$run_dir" ]]; then
            return 1
        fi
    done

    case "$model" in
        EFN|UVPL|EBVIF)
            local epochs=100 save_freq=50
            case "$model" in
                EFN) epochs=1000 ;;
                UVPL) save_freq=10 ;;
            esac
            run_stage "$run_dir" "$model" train_simulate.py 1 "$epochs" 16 "$save_freq" train
            ;;
        *)
            local epochs=100 demosaic_batch=16
            local -a demosaic_args=()
            if [[ "$model" == LSAN_plus_VBPN ]]; then
                epochs=20
            fi
            if [[ "$model" == SFNet_* ]]; then
                demosaic_batch=1
            fi
            if [[ "$model" != PPID_* ]]; then
                run_stage "$run_dir" "$model" train_demosaic_simulate.py 1 1000 "$demosaic_batch" 50 demosaic || return
                local -a weights=("$run_dir"/1/model/best_*.pth)
                if $dry_run; then
                    weights=("$run_dir/1/model/best_<epoch>.pth")
                elif (( ${#weights[@]} == 0 )); then
                    # 旧训练脚本只在 PSNR 超过 0 时保存 best，短跑可能只有周期权重。
                    local -a saved=("$run_dir"/1/model/[0-9]*.pth)
                    if (( ! ${#saved[@]} )); then
                        printf '%s 去马赛克阶段没有保存权重，详见 %s/demosaic.log\n' "$model" "$run_dir" >&2
                        return 1
                    fi
                    weights=("$(printf '%s\n' "${saved[@]}" | sort -V | tail -n 1)")
                    printf '%s 未保存最佳权重，融合阶段使用最近的周期权重 %s\n' "$model" "${weights[0]}"
                elif (( ${#weights[@]} != 1 )); then
                    printf '%s 去马赛克阶段未生成唯一的最佳权重，详见 %s/demosaic.log\n' "$model" "$run_dir" >&2
                    return 1
                fi
                demosaic_args=(--resume_demosaic "${weights[0]}")
            fi
            run_stage "$run_dir" "$model" train_pansharpening_simulate.py 2 "$epochs" 16 5 pansharpening "${demosaic_args[@]}"
            ;;
    esac
}

failed=()
for model in "${models[@]}"; do
    if ! run_model "$model"; then
        failed+=("$model")
    fi
done

if (( ${#failed[@]} )); then
    printf '\n以下模型执行失败，请查看各自的阶段日志: %s\n' "${failed[*]}" >&2
    exit 1
fi
