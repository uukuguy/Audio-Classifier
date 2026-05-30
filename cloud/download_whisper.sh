#!/usr/bin/env bash
# 云端下载 whisper-large-v3。
# 主路: modelscope 客户端 (国内直连快, 多线程, 自带校验). 失败回 curl HF 兜底。
set -euo pipefail

# AutoDL 非交互 ssh 不加载 .bashrc, conda 不在 PATH
if [ -d /root/miniconda3 ] && [[ ":$PATH:" != *":/root/miniconda3/bin:"* ]]; then
  export PATH="/root/miniconda3/bin:$PATH"
fi

DEST="${WHISPER_DIR:-$HOME/.cache/manual_models/whisper-large-v3}"
mkdir -p "$DEST"

# 完整性检查: 已下完就跳
if [ -f "$DEST/model.safetensors" ]; then
  SZ=$(stat -c %s "$DEST/model.safetensors" 2>/dev/null || stat -f %z "$DEST/model.safetensors")
  if [ "$SZ" -gt 2500000000 ]; then
    echo "[dl] safetensors 已完整 ($((SZ/1024/1024)) MB), 跳过"
    exit 0
  fi
fi

echo "[dl] 主路: ModelScope (AI-ModelScope/whisper-large-v3, 国内直连)"
# 装 modelscope client (幂等,已装则瞬秒过)
pip install --no-cache-dir -q modelscope 2>&1 | tail -2 || true

python3 <<PYEOF
from modelscope import snapshot_download
import os, sys
dest = os.environ.get("WHISPER_DIR", os.path.expanduser("~/.cache/manual_models/whisper-large-v3"))
mid = "AI-ModelScope/whisper-large-v3"
print(f"[ms] downloading {mid} -> {dest}", flush=True)
# 直接下到目标目录 (会自动多线程 + 续传)
p = snapshot_download(
    mid,
    cache_dir=os.path.dirname(dest),
    local_dir=dest,
    allow_patterns=[
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer_config.json", "model.safetensors",
    ],
)
print(f"[ms] done -> {p}", flush=True)
PYEOF

# 校验
SZ=$(stat -c %s "$DEST/model.safetensors" 2>/dev/null || stat -f %z "$DEST/model.safetensors")
echo "[dl] model.safetensors = $((SZ / 1024 / 1024)) MB"
if [ "$SZ" -gt 2500000000 ]; then
  echo "[dl] ✅ OK"
  exit 0
fi

# 兜底: modelscope 失败则回 curl HF (一般不会到这一步)
echo "[dl] ModelScope 失败 ($((SZ/1024/1024)) MB), 回退 curl HF"
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi
for src in "https://huggingface.co" "https://hf-mirror.com"; do
  echo "[dl] 试 $src ..."
  if curl -fSL -C - --retry 8 --retry-all-errors --retry-delay 10 \
       --connect-timeout 20 --max-time 3600 \
       -o "$DEST/model.safetensors" \
       "$src/openai/whisper-large-v3/resolve/main/model.safetensors"; then
    SZ=$(stat -c %s "$DEST/model.safetensors")
    [ "$SZ" -gt 2500000000 ] && { echo "[dl] ✅ OK ($src)"; exit 0; }
  fi
done

echo "[dl] ❌ 所有源都失败"
exit 1
