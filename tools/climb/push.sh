#!/usr/bin/env bash
# climb push.sh — push_mode=manual-csv
#
# 本项目公榜手动提交（每天 2 次）。push.sh 不自动提交，只:
#   1. 确认 run 目录里有 pred_test1.csv（可提交工件）
#   2. 校验 CSV 格式（segment_id,c,na,i,bc,t；1000 行 0/1）
#   3. 把 run 登记到 .claude/climb/pending-lb.json
#   4. 打印提示让用户手动提交 + 之后贴回 Macro-F1 真分
#
# Usage: tools/climb/push.sh <run_dir>
#   <run_dir> 含 pred_test1.csv + manifest.json

set -euo pipefail
RUN_DIR="${1:?usage: push.sh <run_dir>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CSV="$RUN_DIR/pred_test1.csv"
PENDING="$ROOT/.claude/climb/pending-lb.json"

[ -f "$CSV" ] || { echo "[climb-push] ERROR: $CSV 不存在，先跑 eval/infer 生成 CSV" >&2; exit 1; }

# --- 校验 CSV 格式 ---
HEADER=$(head -1 "$CSV")
EXPECT="segment_id,c,na,i,bc,t"
[ "$HEADER" = "$EXPECT" ] || { echo "[climb-push] ERROR: CSV header '$HEADER' != '$EXPECT'" >&2; exit 1; }
NROWS=$(( $(wc -l < "$CSV") - 1 ))
echo "[climb-push] CSV ok: $NROWS rows, header=$HEADER"
[ "$NROWS" -eq 1000 ] || echo "[climb-push] WARN: 期望 1000 行，实际 $NROWS（测试集 1000 段）"

# --- 登记 pending-lb（用 python 安全改 json）---
RUN_ID="$(basename "$RUN_DIR")"
PARADIGM="${CLIMB_PARADIGM:-unknown}"
PRED="${CLIMB_PRED_ONLINE:-null}"
python3 - "$PENDING" "$RUN_ID" "$PARADIGM" "$PRED" "$CSV" <<'PY'
import json, sys, datetime
pending_path, run_id, paradigm, pred, csv = sys.argv[1:6]
with open(pending_path) as f: d = json.load(f)
d.setdefault("pending", [])
d["pending"] = [p for p in d["pending"] if p.get("run_id") != run_id]
d["pending"].append({
    "run_id": run_id, "paradigm": paradigm,
    "predicted_online": None if pred == "null" else float(pred),
    "csv_path": csv, "pushed_at": datetime.datetime.now().isoformat(timespec="seconds"),
})
with open(pending_path, "w") as f: json.dump(d, f, ensure_ascii=False, indent=2)
print(f"[climb-push] 已登记 pending-lb: {run_id} (paradigm={paradigm}, pred={pred})")
PY

cat <<EOF

═══════════════════════════════════════════════════════════
  手动提交（climb 不自动提交，公榜每天 2 次）
═══════════════════════════════════════════════════════════
  提交文件:  $CSV
  run_id:    $RUN_ID
  提交后请贴回 Macro-F1 真分（任意格式，climb 自动 parse）:
    例:  $RUN_ID 0.7213
         lb $RUN_ID = 0.7213 [c=.97 na=.82 t=.61 i=.50 bc=.38]
═══════════════════════════════════════════════════════════
EOF
