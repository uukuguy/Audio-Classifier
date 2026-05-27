#!/usr/bin/env bash
# climb eval-local.sh — 本地 OOF CV 评估，输出主指标(Macro-F1) + 5 类子分
#
# Usage: tools/climb/eval-local.sh <run_dir>
# 约定: 打印一行 JSON 到 stdout:
#   {"score": <macro_f1>, "per_sub": {"c":..,"na":..,"t":..,"i":..,"bc":..}}
# climb 读这行解析。各 paradigm 的真实评估逻辑随实现填入（按 manifest.paradigm 分派）。

set -euo pipefail
RUN_DIR="${1:?usage: eval-local.sh <run_dir>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[climb-eval] STUB — 按 paradigm 分派真实评估。" >&2
echo "[climb-eval] context-only: 复用 tests/main/eda_context_baseline.py 的 OOF Macro-F1" >&2
echo "[climb-eval] baseline-enhanced / vap-stereo: 跑 src 的 valid（30s 切片化）OOF" >&2
echo "[climb-eval] 验证集必须 = 按会话划分 + 30s 切片化（防滑窗乐观偏差）" >&2

# 真实实现示例（context-only paradigm）：
#   python3 tests/main/eda_context_baseline.py --emit-json > "$RUN_DIR/cv_metrics.json"
#   cat "$RUN_DIR/cv_metrics.json"

echo '{"score": null, "per_sub": {"c":null,"na":null,"t":null,"i":null,"bc":null}, "_stub": true}'
