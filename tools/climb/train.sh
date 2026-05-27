#!/usr/bin/env bash
# climb train.sh — 训练一个 hypothesis，产物落到 artifact_dir 的时间戳子目录
#
# Usage: tools/climb/train.sh <hypothesis_id>
# 约定:
#   - 读 .claude/climb/hypotheses.yaml 找该 id 的 paradigm + 描述
#   - 按 paradigm 分派 trainer:
#       context-only      → tests/main/eda_context_baseline.py（分钟级，本机 MPS）
#       baseline-enhanced → baselines/.../src.train（改损失/阈值/音频塔）
#       vap-stereo        → 新 src（双声道 cross-attn，云 GPU）
#   - 产物 → tools/runs/climb/<ts>-<hid>-<paradigm>/{ckpt, pred_test1.csv, manifest.json}
#   - manifest.json: {cycle, hypothesis_id, paradigm, start, end}
# 长任务用 PID+artifact 双信号判完成（见 global CLAUDE.md），不 grep stdout。

set -euo pipefail
HID="${1:?usage: train.sh <hypothesis_id>}"
echo "[climb-train] STUB — 按 hypothesis $HID 的 paradigm 分派 trainer。实现随 cycle 填。" >&2
echo "[climb-train] 产物目录: tools/runs/climb/\$(date +%Y%m%d-%H%M)-$HID-<paradigm>/" >&2
exit 0
