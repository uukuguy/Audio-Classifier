#!/usr/bin/env bash
# 云端一键:下模型 → 提取 whisper 帧(断点续跑) → 训神经头 → 出 pred_test1.csv。
# 也支持 LoRA 微调模式 (在线前向, 不读缓存)。
# 分阶段可单独跑;每阶段 PID+artifact 双信号判终止(不靠 stdout grep)。
#
# 用法(云终端，repo 根):
#   bash cloud/run_cloud.sh smoke        # 冒烟:前 40 通 train + 全 test，冻结头
#   bash cloud/run_cloud.sh full         # 全量冻结头
#   bash cloud/run_cloud.sh extract      # 仅提取
#   bash cloud/run_cloud.sh head         # 仅训头(需缓存已就绪)
#   bash cloud/run_cloud.sh lora-smoke   # LoRA 冒烟:40 通, cap5, 10ep
#   bash cloud/run_cloud.sh lora-full    # LoRA 全量:全通, cap5, 50ep
set -euo pipefail

# AutoDL 非交互式 ssh 不加载 .bashrc, conda 不在 PATH。强制初始化。
if [ -d /root/miniconda3 ] && [[ ":$PATH:" != *":/root/miniconda3/bin:"* ]]; then
  export PATH="/root/miniconda3/bin:$PATH"
fi

MODE="${1:-smoke}"
mkdir -p tools/runs/climb

# 限线程留余量(同本机铁律；云上核多可调大)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "=== [run_cloud] mode=$MODE dev check ==="
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 0) 模型
if [ ! -s "${WHISPER_DIR:-$HOME/.cache/manual_models/whisper-large-v3}/model.safetensors" ]; then
  bash cloud/download_whisper.sh
fi

# ── LoRA 模式 (在线前向, 不需要提取缓存) ──
case "$MODE" in
  lora-smoke)
    RUN_DIR="${RUN_DIR:-tools/runs/climb/lora-smoke}"
    echo "=== [run_cloud] LoRA smoke (40 convs, cap5, 10ep) ==="
    python3 cloud/train_lora.py --convs 40 --epochs 10 --slice-cap 5 --batch-size 8 --grad-accum 2 \
      --run-dir "$RUN_DIR"
    ;;
  lora-full)
    RUN_DIR="${RUN_DIR:-tools/runs/climb/lora-full}"
    echo "=== [run_cloud] LoRA full (all convs, cap5, 50ep) ==="
    python3 cloud/train_lora.py --convs 0 --epochs 50 --slice-cap 5 --batch-size 8 --grad-accum 4 \
      --run-dir "$RUN_DIR"
    ;;
  *)
    # ── 冻结头模式 (需要缓存) ──
    RUN_DIR="${RUN_DIR:-tools/runs/climb/cloud-whisper-h001}"
    mkdir -p "$RUN_DIR"

    case "$MODE" in
      smoke)   TRAIN_CONVS=40 ;;
      full|extract|head) TRAIN_CONVS=0 ;;
      *) echo "unknown mode: $MODE"; exit 1 ;;
    esac

    # 1) 提取(断点续跑;重跑只补未完成的通)
    if [ "$MODE" != "head" ]; then
      echo "=== [run_cloud] extract test ==="
      python3 cloud/extract_whisper_cuda.py --split test --convs 0
      echo "=== [run_cloud] extract train (convs=$TRAIN_CONVS) ==="
      python3 cloud/extract_whisper_cuda.py --split train --convs "$TRAIN_CONVS"
    fi

    # 2) 训头 + 出 CSV
    if [ "$MODE" != "extract" ]; then
      echo "=== [run_cloud] train head ==="
      python3 cloud/train_head_cuda.py --epochs 15 --folds 5 --run-dir "$RUN_DIR"
    fi
    ;;
esac

# 双信号自检
RUN_DIR="${RUN_DIR:-tools/runs/climb/cloud-whisper-h001}"
if [ -s "$RUN_DIR/pred_test1.csv" ]; then
  echo "RUN_CLOUD_COMPLETE csv=$RUN_DIR/pred_test1.csv rows=$(($(wc -l < "$RUN_DIR/pred_test1.csv") - 1))"
else
  echo "RUN_CLOUD_NO_CSV (check logs)"
fi
