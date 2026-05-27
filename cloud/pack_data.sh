#!/usr/bin/env bash
# 本机运行:打包上云所需数据(label/context/audio/text)→ 单 tar，便于上传 AutoDL。
# 注意:train audio 是整通对话(大)，可先只传 label+context+test 做冒烟，确认链路后再传 train audio。
set -euo pipefail

OUT="${1:-data/cloud_upload}"
mkdir -p "$OUT"

echo "[pack] 小件(label/context/text)→ $OUT/meta.tar.gz"
tar czf "$OUT/meta.tar.gz" \
  data/train/labels \
  data/test/context \
  data/train/text \
  data/test/text 2>/dev/null || true

echo "[pack] test audio(30s 切片, 小)→ $OUT/test_audio.tar"
tar cf "$OUT/test_audio.tar" data/test/audio

echo "[pack] train audio(大!整通对话)→ $OUT/train_audio.tar"
echo "[pack]   如先做冒烟可跳过 train audio，只传前 N 通:"
echo "[pack]   ls data/train/audio/*.wav | head -40 | tar cf $OUT/train_audio_smoke.tar -T -"
tar cf "$OUT/train_audio.tar" data/train/audio

echo "[pack] 完成。上传方式见 cloud/AUTODL-CHECKLIST.md"
du -sh "$OUT"/* 2>/dev/null || true
