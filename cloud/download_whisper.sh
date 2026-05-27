#!/usr/bin/env bash
# 云端下载 whisper-large-v3（curl 直下，绕 hf_hub client — 本机已验证 client HEAD 失败/curl 成功）。
# AutoDL 开了"学术资源加速"时 from_pretrained 也能直连；此脚本是兜底，保证可复现。
set -euo pipefail

ORG="openai/whisper-large-v3"
DEST="${WHISPER_DIR:-$HOME/.cache/manual_models/whisper-large-v3}"
BASE="https://hf-mirror.com/${ORG}/resolve/main"
mkdir -p "$DEST"

FILES=(
  config.json
  generation_config.json
  preprocessor_config.json
  tokenizer_config.json
  model.safetensors
)

echo "[dl] whisper-large-v3 → $DEST"
for f in "${FILES[@]}"; do
  if [ -s "$DEST/$f" ]; then
    echo "[dl] skip $f (exists)"
    continue
  fi
  echo "[dl] $f ..."
  curl -fSL --retry 3 -o "$DEST/$f" "$BASE/$f"
done

# 校验 safetensors 大小（large-v3 ≈ 3.0GB）
SZ=$(stat -c %s "$DEST/model.safetensors" 2>/dev/null || stat -f %z "$DEST/model.safetensors")
echo "[dl] model.safetensors = $((SZ / 1024 / 1024)) MB"
[ "$SZ" -gt 2500000000 ] && echo "[dl] OK" || { echo "[dl] FAIL: safetensors too small"; exit 1; }
