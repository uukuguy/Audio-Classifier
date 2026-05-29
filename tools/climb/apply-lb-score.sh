#!/usr/bin/env bash
# climb apply-lb-score.sh — 用户贴公榜真分时调用
#
# 解析松格式真分 → 匹配 pending-lb 的 run → 更新 runs.csv online_* + calibration gap
# → 从 pending 移除。
#
# Usage: tools/climb/apply-lb-score.sh "<用户粘贴的真分文本>"
#   接受: "sub-h001 0.7213"  /  "lb sub-h001 = 0.7213 [c=.97 na=.82 t=.61 i=.5 bc=.38]"

set -euo pipefail
PASTE="${1:?usage: apply-lb-score.sh \"<lb paste>\"}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLIMB="$ROOT/docs/status/climb"

python3 - "$CLIMB" "$PASTE" <<'PY'
import json, re, sys, datetime, csv as csvmod
from pathlib import Path

climb = Path(sys.argv[1]); paste = sys.argv[2]
SUBS = ["c", "na", "t", "i", "bc"]

# --- parse 主分: 第一个 0.xxxx 形态的浮点 ---
m = re.search(r'(\d\.\d{2,6})', paste)
if not m:
    print(f"[climb-lb] 无法从 '{paste}' 解析 Macro-F1 主分", file=sys.stderr); sys.exit(1)
online = float(m.group(1))

# --- parse run_id: 含 run_tag_marker(-climb-) 或 sub- 的 token，否则取 pending 里最新 ---
run_id = None
for tok in re.split(r'[\s,=\[\]]+', paste):
    if ('-climb-' in tok) or tok.startswith('sub-') or re.match(r'^[a-z].*h\d', tok):
        run_id = tok; break

# --- parse 子分 c=.97 na=.82 ... (可选) ---
per_sub = {}
for s in SUBS:
    mm = re.search(rf'\b{s}\s*[=:]\s*(\d?\.\d+|\d+\.?\d*)', paste, re.I)
    if mm: per_sub[s] = float(mm.group(1))

pending = json.loads((climb/"pending-lb.json").read_text())
plist = pending.get("pending", [])
if run_id is None:
    if not plist:
        print(f"[climb-lb] pending 为空且未识别 run_id，主分={online}，请显式带 run_id", file=sys.stderr); sys.exit(1)
    run_id = sorted(plist, key=lambda p: p.get("pushed_at",""))[-1]["run_id"]
    print(f"[climb-lb] 未识别 run_id，用 pending 最新: {run_id}")

entry = next((p for p in plist if p["run_id"] == run_id), None)
paradigm = entry["paradigm"] if entry else "unknown"

# --- 更新 runs.csv 的 online_* + gap ---
runs = climb/"runs.csv"
rows = list(csvmod.reader(runs.read_text().splitlines()))
header = rows[0]
local_score = None; updated = False
idx = {h: i for i, h in enumerate(header)}
# 已有最佳线上分（排除本 run），用于判 verdict
prior_best = 0.0
for r in rows[1:]:
    if not r or r[0].startswith('#') or r[0] == run_id:
        continue
    try:
        prior_best = max(prior_best, float(r[idx["online_score"]]))
    except (ValueError, IndexError):
        pass
for r in rows[1:]:
    if not r or r[0].startswith('#'): continue
    if r[0] == run_id:
        r[idx["online_score"]] = f"{online}"
        for s in per_sub:
            if f"online_{s}" in idx: r[idx[f"online_{s}"]] = f"{per_sub[s]}"
        r[idx["lb_landed_at"]] = datetime.datetime.now().isoformat(timespec="seconds")
        try: local_score = float(r[idx["local_score"]])
        except: pass
        if local_score is not None:
            r[idx["gap"]] = f"{online - local_score:.4f}"
        # 自动判 verdict（同步 research-tree 用）
        if online > prior_best:
            r[idx["verdict"]] = f"confirmed SOTA +{online - prior_best:.4f}" if prior_best > 0 else "confirmed first"
        else:
            r[idx["verdict"]] = f"falsified {online - prior_best:.4f} vs best"
        updated = True
if updated:
    with open(runs, "w", newline="") as f: csvmod.writer(f).writerows(rows)

gap = (online - local_score) if local_score is not None else None

# --- 更新 calibration ---
calib = json.loads((climb/"calibration.json").read_text())
p = calib["paradigms"].setdefault(paradigm, {"n_samples":0,"mean_gap":None,"std_gap":None,"last_3_gaps":[],"last_updated":None,"_note":""})
if gap is not None:
    g3 = (p["last_3_gaps"] + [round(gap,4)])[-3:]
    n = p["n_samples"] + 1
    allg = [x for x in g3]  # 简化：只用 last_3 估计；完整版应存全历史
    p["n_samples"] = n
    p["last_3_gaps"] = g3
    p["mean_gap"] = round(sum(g3)/len(g3), 4)
    p["std_gap"] = (round((sum((x-p["mean_gap"])**2 for x in g3)/len(g3))**0.5,4) if len(g3)>1 else None)
    p["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")
(climb/"calibration.json").write_text(json.dumps(calib, ensure_ascii=False, indent=2))

# --- 从 pending 移除 ---
pending["pending"] = [x for x in plist if x["run_id"] != run_id]
(climb/"pending-lb.json").write_text(json.dumps(pending, ensure_ascii=False, indent=2))

print(f"[climb-lb] {run_id}: online={online}" + (f" local={local_score} gap={gap:+.4f}" if gap is not None else "") + f" paradigm={paradigm}")
if per_sub: print(f"[climb-lb] 子分: {per_sub}")
print(f"[climb-lb] calibration[{paradigm}] n={calib['paradigms'][paradigm]['n_samples']} mean_gap={calib['paradigms'][paradigm]['mean_gap']}")
print(f"[climb-lb] 已从 pending 移除。climb 应据此重排假设池继续。")
PY

# --- 同步 research-tree（HARD INVARIANT #4: LB land 时 regen）---
PY_BIN="${CLIMB_PY:-python3}"
if [ -f "$ROOT/tools/climb/regen-tree.py" ]; then
  "$PY_BIN" "$ROOT/tools/climb/regen-tree.py" >/dev/null 2>&1 && echo "[climb-lb] research-tree synced" || echo "[climb-lb] WARN: regen-tree failed"
fi

# --- 确定性 target gate (climb.md §4.1): LB 落可能让 target 达成 → emit PAUSE ---
if [ -f "$ROOT/tools/climb/check-target.py" ]; then
  TGT=$("$PY_BIN" "$ROOT/tools/climb/check-target.py" 2>/dev/null); RC=$?
  if [ "$RC" = "10" ]; then
    echo ""
    echo "🎯🎯🎯 [climb-lb] TARGET MET (LB 真分越阈) — climb.md §4.1 Hard Pause 🎯🎯🎯"
    echo "$TGT" | "$PY_BIN" -c "import sys,json; print('  '+json.load(sys.stdin).get('reason',''))" 2>/dev/null || echo "  $TGT"
    echo "  → 写 handoff 汇报成果, 暂停等用户讨论下一档."
  fi
fi
