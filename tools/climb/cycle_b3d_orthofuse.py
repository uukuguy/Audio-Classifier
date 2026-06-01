"""climb cycle B3d-orthofuse — B3d × SOTA orthofuse 跨源 per-class 正交融合.

设计 (chain-first verified):
- B3d (DB-Loss + SupCon 校准头, OOF +0.031 vs ctx_lgbm_v1)
- SOTA orthofuse-20260531-0319 (ctx_lgbm_v1 × whisper per-class)
- 两者在 BC 上 corr=0.69 (正交), B3d 在 T cap1 +0.030 vs SOTA, BC test pos 完全一致 (27)

策略:
- per-class 选 cap1 strat (与 SOTA orthofuse 同协议)
- 候选 strat: sota | b3d | sota_eq_b3d (0.5+0.5) | sota_w70 (0.7sota+0.3b3d) | sota_w30 (0.3sota+0.7b3d)
- 保守 gate: 必须 cap1 该类 F1 > SOTA strat 类 F1 + 0.008 才换 (避 D-3/D-11 cap1 cherry-pick)

Push 门 (D-13):
- 融合后 cap1 macro >= SOTA cap1 0.6281 + 0.005 = 0.6330 → PUSH
- 否则 SKIP-advance

Falsified bound:
- 不调 BC 阈值 / 不动 BC strat (保 SOTA 'ctx' strat)  ← D-11 红旗死规则
- 不 grid 搜权重 (D-6 in-sample cap1 grid 不泛化)

Usage:
  python tools/climb/cycle_b3d_orthofuse.py --submit
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
LAB = ['C', 'T', 'BC', 'I', 'NA']
NUM = 5
THR_VARF = [0.05, 0.5, 0.75, 0.65, 0.25]  # 与 cycle_orthofuse.py 一致 (BC=0.75, I=0.65, NA=0.25)
GATE = 0.008  # per-class 必须超过 SOTA 这么多才换 (保守, 避 cherry-pick)
SUBMIT_COLS = ['c', 'na', 'i', 'bc', 't']
COL_TO_K = {'c': 0, 'na': 4, 'i': 3, 'bc': 2, 't': 1}


def strat_apply(name, sota, b3d):
    """sota: orthofuse 该类输出. b3d: B3d 该类输出. 返回融合."""
    if name == 'sota':
        return sota
    if name == 'b3d':
        return b3d
    if name == 'eq':
        return 0.5 * sota + 0.5 * b3d
    if name == 'w70_sota':
        return 0.7 * sota + 0.3 * b3d
    if name == 'w70_b3d':
        return 0.3 * sota + 0.7 * b3d
    raise ValueError(name)


def f1k(p, y, t):
    return f1_score(y, (p >= t).astype(int), zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submit', action='store_true', help='生成 pred_test1.csv')
    args = ap.parse_args()

    # 加载
    b3d_npz = np.load(ROOT / 'tools/runs/climb/b3d-calib-20260601-1008/probs.npz')
    sota_npz = np.load(ROOT / 'tools/runs/climb/orthofuse-20260531-0319/fused_probs.npz')
    stack = np.load(ROOT / 'tools/runs/climb/_stack_cache_s40.npz')
    whisper = np.load(ROOT / 'tools/runs/climb/whisper-fusion-20260531-0143/probs.npz')

    Y = b3d_npz['Y']
    G = b3d_npz['G']
    order = whisper['order']

    # cap1 = 每通首窗 (order=0), 与 cycle_orthofuse.py 一致 (BUG FIX: 之前用 order max 是错的)
    cap1_idx = np.where(order == 0)[0]
    Y_cap1 = Y[cap1_idx]
    print(f"[b3d-orthofuse] cap1 size = {len(cap1_idx)}", file=sys.stderr)

    # SOTA orthofuse 用 ctx_lgbm_v1 + whisper, strat 已知 (cv_metrics.json)
    sota_strat = {'C': 'ctx', 'T': 'w70', 'BC': 'ctx', 'I': 'whisper', 'NA': 'ctx'}

    def apply_sota_strat(ctx, wsp, k):
        s = sota_strat[LAB[k]]
        if s == 'ctx': return ctx
        if s == 'whisper': return wsp
        if s == 'w70': return 0.7 * ctx + 0.3 * wsp

    # 各源 cap1
    ctx_cap1 = stack['oof_lgbm_v1'][cap1_idx]
    wsp_cap1 = whisper['oof'][cap1_idx]
    b3d_cap1 = b3d_npz['oof'][cap1_idx]

    # 各源 test
    ctx_te = stack['te_lgbm_v1']
    wsp_te = whisper['test']
    b3d_te = b3d_npz['test']

    # SOTA per-class output (cap1 + test)
    sota_cap1 = np.stack([apply_sota_strat(ctx_cap1[:, k], wsp_cap1[:, k], k) for k in range(NUM)], axis=1)
    sota_te = np.stack([apply_sota_strat(ctx_te[:, k], wsp_te[:, k], k) for k in range(NUM)], axis=1)

    # per-class strat search (B3d 是否更强): {sota, b3d, eq, w70_sota, w70_b3d}
    print(f"\n[b3d-orthofuse] === per-class strat decision ===", file=sys.stderr)
    print(f"{'class':<5}{'sota':>10}{'b3d':>10}{'eq':>10}{'w70_s':>10}{'w70_b':>10}{'选':>10}{'Δ vs sota':>12}", file=sys.stderr)

    chosen_strat = {}
    cap1_per = []
    test_fused = np.zeros_like(sota_te)
    for k in range(NUM):
        scores = {}
        # BC 强制守 SOTA strat (D-11 红旗死规则)
        if LAB[k] == 'BC':
            chosen_strat[LAB[k]] = 'sota'
            cap1_per.append(f1k(sota_cap1[:, k], Y_cap1[:, k], THR_VARF[k]))
            test_fused[:, k] = sota_te[:, k]
            print(f"{LAB[k]:<5}" + ' ' * 50 + f"{'sota (D-11 守)':>10}{0.0:>12.4f}", file=sys.stderr)
            continue

        for strat in ['sota', 'b3d', 'eq', 'w70_sota', 'w70_b3d']:
            p_cap1 = strat_apply(strat, sota_cap1[:, k], b3d_cap1[:, k])
            scores[strat] = f1k(p_cap1, Y_cap1[:, k], THR_VARF[k])

        best = max(scores, key=lambda s: scores[s])
        # gate: 必须超 sota_strat F1 + GATE 才换
        if scores[best] >= scores['sota'] + GATE:
            chosen_strat[LAB[k]] = best
        else:
            chosen_strat[LAB[k]] = 'sota'

        # apply chosen
        p_cap1 = strat_apply(chosen_strat[LAB[k]], sota_cap1[:, k], b3d_cap1[:, k])
        cap1_per.append(f1k(p_cap1, Y_cap1[:, k], THR_VARF[k]))
        p_te = strat_apply(chosen_strat[LAB[k]], sota_te[:, k], b3d_te[:, k])
        test_fused[:, k] = p_te

        line = f"{LAB[k]:<5}" + ''.join(f"{scores[s]:>10.4f}" for s in ['sota','b3d','eq','w70_sota','w70_b3d'])
        line += f"{chosen_strat[LAB[k]]:>10}{scores[chosen_strat[LAB[k]]] - scores['sota']:>+12.4f}"
        print(line, file=sys.stderr)

    cap1_macro = float(np.mean(cap1_per))
    # SOTA baseline
    sota_cap1_macro = float(np.mean([f1k(sota_cap1[:, k], Y_cap1[:, k], THR_VARF[k]) for k in range(NUM)]))
    delta = cap1_macro - sota_cap1_macro
    push = cap1_macro >= sota_cap1_macro + 0.005

    print(f"\n[b3d-orthofuse] === RESULT ===", file=sys.stderr)
    print(f"[b3d-orthofuse] SOTA orthofuse cap1: {sota_cap1_macro:.4f}", file=sys.stderr)
    print(f"[b3d-orthofuse] B3d-orthofuse cap1: {cap1_macro:.4f} (Δ={delta:+.4f})", file=sys.stderr)
    print(f"[b3d-orthofuse] chosen strat: {chosen_strat}", file=sys.stderr)
    print(f"[b3d-orthofuse] D-13 push 门 (+0.005): {'✓ PUSH' if push else '✗ SKIP'}", file=sys.stderr)

    # 保存
    out_dir = ROOT / f'tools/runs/climb/b3d-orthofuse-{datetime.now().strftime("%Y%m%d-%H%M")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'fused_probs.npz', test=test_fused, sota_te=sota_te, b3d_te=b3d_te)

    metrics = {
        'cycle': 'B3d-orthofuse',
        'sota_baseline_cap1': sota_cap1_macro,
        'cap1_macro': cap1_macro,
        'cap1_delta_vs_sota': delta,
        'cap1_per': dict(zip(LAB, [float(x) for x in cap1_per])),
        'chosen_strat': chosen_strat,
        'gate_pass': push,
        'push_decision': 'PUSH' if push else 'SKIP',
        'note': 'BC strat 强制守 sota (D-11 红旗死规则). 其它类 gate +0.008 才换.',
    }
    (out_dir / 'cv_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 出 csv
    if args.submit:
        # test 0000-0999 segment_id 4 位
        test_ids = whisper['test_ids']
        pred = np.stack([(test_fused[:, k] >= THR_VARF[k]).astype(int) for k in range(NUM)], axis=1)
        csv_path = out_dir / 'pred_test1.csv'
        with open(csv_path, 'w') as f:
            f.write('segment_id,c,na,i,bc,t\n')
            for i, sid in enumerate(test_ids):
                row = [pred[i, COL_TO_K[c]] for c in SUBMIT_COLS]
                f.write(f"{sid}," + ','.join(str(x) for x in row) + '\n')
        print(f"[b3d-orthofuse] csv: {csv_path}", file=sys.stderr)

        # pos counts
        pos_counts = {col: int(sum(pred[:, COL_TO_K[col]])) for col in SUBMIT_COLS}
        print(f"[b3d-orthofuse] test pos counts: {pos_counts}", file=sys.stderr)

    print(f"\n[b3d-orthofuse] artifact: {out_dir}/", file=sys.stderr)
    return push


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
