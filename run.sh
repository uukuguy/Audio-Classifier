#!/bin/bash
# 复赛端到端推理入口 (官方要求 /app/run.sh → /app/submit/submit.csv)
# 三源基线: ctx + whisper + hubert, orthofuse-3src 融合 (初赛 SOTA 0.71755 配方)
set -e
cd /app
export MODELS=/app/models TEST_ROOT=/xydata OUTPUT_CSV=/app/submit/submit.csv
echo "[run] audio=$(ls /xydata/audio/*.wav 2>/dev/null | wc -l) ctx=$(ls /xydata/context/*.npy 2>/dev/null | wc -l)"
python -m src.infer_e2e
echo "[run] RC=$? out=$(wc -l < /app/submit/submit.csv 2>/dev/null) lines"
