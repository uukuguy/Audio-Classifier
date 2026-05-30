#!/usr/bin/env bash
# 云端 setup: 解压数据 + 装依赖 + 下模型。预设 zip 在 /root/autodl-fs/finv11th_train_test_data.zip。
# 不跑训练 (避免误启长任务). 跑完手动 bash cloud/run_cloud.sh smoke.
#
# 用法 (云端, repo 根 /root/audio-classifier):
#   bash cloud/setup_cloud.sh
set -euo pipefail

# AutoDL 非交互式 ssh 不加载 .bashrc, conda 不在 PATH。强制初始化。
if [ -d /root/miniconda3 ] && [[ ":$PATH:" != *":/root/miniconda3/bin:"* ]]; then
  export PATH="/root/miniconda3/bin:$PATH"
  echo "[setup] 加 /root/miniconda3/bin 到 PATH"
fi

REPO_DIR="${REPO_DIR:-/root/audio-classifier}"
ZIP="${ZIP:-/root/autodl-fs/finv11th_train_test_data.zip}"
WHISPER_DIR="${WHISPER_DIR:-$HOME/.cache/manual_models/whisper-large-v3}"

cd "$REPO_DIR"

# 1) 解压数据到 data/ (zip 内是 train/ test/ 两层, 套一层 data/ 让脚本零改)
echo "=== [setup 1/4] 解压数据 ==="
if [ -d data/train/audio ] && [ -d data/test/audio ]; then
  echo "[setup] data/ 已存在, 跳过解压"
else
  if [ ! -s "$ZIP" ]; then
    echo "[setup] ❌ zip 不存在或为空: $ZIP"
    exit 1
  fi
  mkdir -p data
  echo "[setup] 解压 $ZIP → data/ (zip 内 train/ test/ 套一层)"
  unzip -q "$ZIP" -d data/
  echo "[setup] 解压完: train=$(ls data/train/audio/*.wav 2>/dev/null | wc -l) test=$(ls data/test/audio/*.wav 2>/dev/null | wc -l)"
fi

# 2) 装 Python 依赖
echo ""
echo "=== [setup 2/4] 装 Python 依赖 ==="
# 先看 torch/torchaudio 镜像自带版本 (AutoDL 镜像一般含 torch, torchaudio 不一定)
python3 -c "import torch; print('torch自带', torch.__version__)" 2>&1 || true

# torchaudio 必须从 pytorch 官方 wheel 源装,匹配 torch 的 CUDA 版本 (PyPI 默认是旧 CU,会撞 CUDA mismatch)
need_torchaudio=0
python3 -c "import torchaudio; print('torchaudio自带', torchaudio.__version__)" 2>&1 || need_torchaudio=1
# 即使能 import, 也要校验 CUDA 版本对齐 (torch cu128 / torchaudio cu126 会 import 时炸)
if [ "$need_torchaudio" = "0" ]; then
  python3 -c "import torch, torchaudio; import torchaudio._extension" 2>&1 || need_torchaudio=1
fi
if [ "$need_torchaudio" = "1" ]; then
  TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split(\"+\")[0])")
  CUDA_TAG=$(python3 -c "import torch; v=torch.version.cuda or \"\"; print(\"cu\"+v.replace(\".\",\"\"))")
  WHEEL_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
  echo "[setup] 装 torchaudio==${TORCH_VER} 走 ${WHEEL_INDEX} (匹配 torch 的 ${CUDA_TAG})"
  # 先卸载已有错误版本 (PyPI 装的 cu126)
  pip uninstall -y torchaudio 2>/dev/null || true
  pip install --no-cache-dir -q "torchaudio==${TORCH_VER}" --index-url "${WHEEL_INDEX}"
fi
pip install --no-cache-dir -q -r cloud/requirements.txt
echo ""
echo "[setup] 装完, 综合验证:"
python3 -c "import torch, torchaudio, transformers, sklearn, lightgbm; print('torch', torch.__version__, '/ torchaudio', torchaudio.__version__, '/ transformers', transformers.__version__, '/ cuda', torch.cuda.is_available(), '/ device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# 3) 下模型 (curl 直下, 绕 hf client)
echo ""
echo "=== [setup 3/4] 下 whisper-large-v3 ==="
# 完整性校验: >2.5GB 才算下完 (large-v3 = ~3GB). 单纯 -s 测非空不够,会让半截文件骗过检查
SZ=0
if [ -f "$WHISPER_DIR/model.safetensors" ]; then
  SZ=$(stat -c %s "$WHISPER_DIR/model.safetensors" 2>/dev/null || stat -f %z "$WHISPER_DIR/model.safetensors")
fi
if [ "$SZ" -gt 2500000000 ]; then
  echo "[setup] 模型已完整 ($((SZ/1024/1024)) MB), 跳过"
else
  echo "[setup] 模型不完整 ($((SZ/1024/1024)) MB), 调下载脚本 (含学术加速+断点续传)"
  bash cloud/download_whisper.sh
fi

# 4) 健康检查 + 下一步提示
echo ""
echo "=== [setup 4/4] 环境健康检查 ==="
df -h /
echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "[setup] ✅ 完成. 下一步跑冒烟 (前 40 通 train, ~1.2GB 缓存):"
echo "  mkdir -p tools/runs/climb/cloud-whisper-smoke"
echo "  nohup bash cloud/run_cloud.sh smoke > tools/runs/climb/cloud-whisper-smoke/run.log 2>&1 &"
echo "  echo \$! > tools/runs/climb/cloud-whisper-smoke/run.pid"
echo "  tail -f tools/runs/climb/cloud-whisper-smoke/run.log"
echo ""
echo "[setup] 全量 (确认冒烟 BC 有信号后):"
echo "  RUN_DIR=tools/runs/climb/cloud-whisper-full nohup bash cloud/run_cloud.sh full > tools/runs/climb/cloud-whisper-full/run.log 2>&1 &"
