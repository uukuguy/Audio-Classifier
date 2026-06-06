#!/usr/bin/env bash
# 训练 mask050 ctx-LGBM ckpt (D-28 dual-model fallback short-ctx ckpt)
#
# 期望产物: models/ctx_only_mask050/ 含 5 lgbm_*.joblib + thresholds.json + feature_spec.json
#
# 用法:
#   # 本机 4 线程 (5-8h, 视机器)
#   ./tools/climb/train_mask050_ckpt.sh
#
#   # 云端 4090 (1-2h, 需先 rsync 代码上云)
#   ssh autodl "cd /root/audio-classifier && ./tools/climb/train_mask050_ckpt.sh"
#
# 训完确认:
#   ls models/ctx_only_mask050/   # 应有 7 文件
#   diff models/ctx_only/feature_spec.json models/ctx_only_mask050/feature_spec.json
#       两者除 ctx_mask_prob 字段外内容应一致
#
# 跑通后接 V1/V2 验证:
#   python3 tools/climb/build_dual_model_validation.py \
#       --ckpt_dir models/ctx_only \
#       --ckpt_dir_short models/ctx_only_mask050 \
#       --out_dir submission/dual-model-validation-$(date +%Y%m%d-%H%M)/

set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M)}"
RUN_DIR="tools/runs/climb/ctx_mask050_${RUN_TAG}"
DEST_DIR="models/ctx_only_mask050"

echo "[mask050-train] RUN_DIR=$RUN_DIR"
echo "[mask050-train] DEST_DIR=$DEST_DIR"
echo "[mask050-train] CTX_MASK_PROB=0.5"
echo "[mask050-train] OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} (本机限线程防卡机, 云端可调 8)"

# 训练 (CTX_MASK_PROB=0.5, baseline default 不污染 §2.6 HARD RULE)
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
MKL_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
CTX_MASK_PROB=0.5 \
python3 tools/climb/cycle_context.py "$RUN_DIR"

# 验证 ckpt 完整 (7 文件)
test -f "$RUN_DIR/ckpt/lgbm_c.joblib" || { echo "FAIL: ckpt 没生成"; exit 1; }
test -f "$RUN_DIR/ckpt/thresholds.json" || { echo "FAIL: thresholds 没生成"; exit 1; }
test -f "$RUN_DIR/ckpt/feature_spec.json" || { echo "FAIL: spec 没生成"; exit 1; }

# 搬到 models/ctx_only_mask050/ (跟 models/ctx_only/ 同结构)
mkdir -p "$DEST_DIR"
cp "$RUN_DIR/ckpt/"* "$DEST_DIR/"
echo "[mask050-train] copied 7 files → $DEST_DIR/"
ls -la "$DEST_DIR/"

# Sanity: spec 应注明 ctx_mask_prob=0.5
python3 -c "
import json
spec = json.load(open('$DEST_DIR/feature_spec.json'))
assert spec.get('ctx_mask_prob') == 0.5, f'mask_prob 不对: {spec.get(\"ctx_mask_prob\")}'
print('[mask050-train] ✓ feature_spec.json ctx_mask_prob =', spec['ctx_mask_prob'])
print('[mask050-train] ✓ note:', spec.get('note', '(none)'))
"

echo "[mask050-train] DONE → 接下来跑 build_dual_model_validation.py 出 V1/V2 csv"
