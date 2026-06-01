"""B1 EDA — context 特征 v3 候选特征 vs labels 的单特征预测力探查.

目的: 先验证再投入. 不重训整个 ctx base, 只看每个新特征单独的 mutual info / single-feature AUC.
仅用 stride5 缓存里已有的 OOF (179867 × ctx_lgbm_v1) 做 baseline 对照, 不重新提取.

实际产物:
- /tmp/b1-eda/v3-candidate-features-{datetime}.json
- 每个候选特征对 5 类的 mutual info + 单特征 AUC + per-class informative gain

候选 v3 特征 (D-1~D-12 没用过):
  1. 1-2 阶差分 (label transition over 6 windows): 30d
  2. 突发性 burstness (run-length stats per class × 3 windows): 15d
  3. 跨类时序转移概率 (T→BC, C→NA, etc): 10d
  4. 同类 run length (per class mean/std): 10d

Usage: python tools/climb/eda_b1_v3_features.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[2]
LAB = ['C', 'T', 'BC', 'I', 'NA']
NUM = 5
CTX, TGT, STRIDE = 375, 25, 5


def featurize_v3(ctx: np.ndarray) -> dict:
    """对单个 375 chunk context 返回 v3 候选特征 dict (按类别分组, 方便 EDA)."""
    feats = {}
    oh = np.eye(NUM)[ctx]
    L = len(ctx)

    # 1. 1 阶差分 (chunk i 跟 i-1 是否不同, 6 窗): 6d
    diff = (ctx[1:] != ctx[:-1]).astype(np.float32)
    for w in (10, 25, 50, 100, 200, 375):
        feats[f'diff1_w{w}'] = float(diff[-w:].mean()) if len(diff) >= w else float(diff.mean())

    # 2. 2 阶差分 (二阶差分速率, 6 窗): 6d
    if len(diff) > 1:
        diff2 = (diff[1:] != diff[:-1]).astype(np.float32)
        for w in (10, 25, 50, 100, 200, 375):
            feats[f'diff2_w{w}'] = float(diff2[-w:].mean()) if len(diff2) >= w else float(diff2.mean())

    # 3. 突发性 (per-class chunk run-length stats, 3 窗): 5 类 × 3 窗 = 15d
    for w in (50, 100, 375):
        win = ctx[-w:] if len(ctx) >= w else ctx
        for k in range(NUM):
            # run length of class k chunks
            mask = (win == k).astype(int)
            if mask.sum() == 0:
                feats[f'burst_{LAB[k]}_w{w}'] = 0.0
            else:
                # count runs
                diffs = np.diff(mask)
                runs = (diffs == 1).sum() + (1 if mask[0] == 1 else 0)  # 启动数
                density = mask.mean()
                feats[f'burst_{LAB[k]}_w{w}'] = float(runs) / max(1, density * len(win))  # 簇集度

    # 4. 跨类转移概率 (key dialogue act transitions, last 100 chunks): 10d
    if L >= 100:
        win = ctx[-100:]
        trans = {}
        for i in range(len(win) - 1):
            key = (int(win[i]), int(win[i+1]))
            trans[key] = trans.get(key, 0) + 1
        total = max(1, sum(trans.values()))
        # 关键转移: T(1)→BC(2), C(0)→T(1), C(0)→NA(4), BC(2)→T(1), I(3)→C(0), NA(4)→C(0)
        key_trans = [(1, 2), (0, 1), (0, 4), (2, 1), (3, 0), (4, 0), (1, 0), (4, 1), (2, 0), (0, 0)]
        for src, dst in key_trans:
            feats[f'trans_{LAB[src]}{LAB[dst]}'] = trans.get((src, dst), 0) / total

    # 5. 同类 run length stats (per class mean/max in last 100): 5 × 2 = 10d
    if L >= 100:
        win = ctx[-100:]
        for k in range(NUM):
            mask = (win == k).astype(int)
            runs = []
            cur = 0
            for v in mask:
                if v == 1:
                    cur += 1
                else:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            if cur > 0:
                runs.append(cur)
            if len(runs) > 0:
                feats[f'runlen_{LAB[k]}_mean'] = float(np.mean(runs))
                feats[f'runlen_{LAB[k]}_max'] = float(max(runs))
            else:
                feats[f'runlen_{LAB[k]}_mean'] = 0.0
                feats[f'runlen_{LAB[k]}_max'] = 0.0

    return feats


def main():
    import glob
    print(f"[B1 EDA] 抽 50 通 train 做特征生成 + label 对照", file=sys.stderr)
    # 抽样 50 通 (够 mutual info), stride5
    train_files = sorted(glob.glob(str(ROOT / 'data/train/labels/*.npy')))[:50]
    print(f"[B1 EDA] {len(train_files)} 通 sampled", file=sys.stderr)

    X_rows = []
    Y_rows = []
    feat_names = None
    for fi, f in enumerate(train_files):
        a = np.load(f)
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE * 4):  # stride5×4=20 加速 EDA
            fv = featurize_v3(a[e - CTX:e].astype(int))
            if feat_names is None:
                feat_names = sorted(fv.keys())
            X_rows.append([fv[k] for k in feat_names])
            fut = set(int(x) for x in a[e:e + TGT])
            Y_rows.append([1 if k in fut else 0 for k in range(NUM)])
        if (fi + 1) % 10 == 0:
            print(f"[B1 EDA] {fi+1}/{len(train_files)} conv processed, rows so far={len(X_rows)}", file=sys.stderr)

    X = np.array(X_rows, dtype=np.float32)
    Y = np.array(Y_rows, dtype=np.int32)
    print(f"[B1 EDA] EDA dataset: X={X.shape} Y={Y.shape}, {len(feat_names)} candidate features", file=sys.stderr)

    # per-feature × per-class AUC (单特征预测力)
    print(f"\n[B1 EDA] Top 20 features per class (by single-feature AUC):", file=sys.stderr)
    out = {}
    for ci in range(NUM):
        scores = []
        for fi, fn in enumerate(feat_names):
            try:
                auc = roc_auc_score(Y[:, ci], X[:, fi])
                # 标 |auc-0.5| 看 informative
                scores.append((fn, abs(auc - 0.5) * 2, auc))
            except Exception:
                pass
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  ===== {LAB[ci]} (pos rate {Y[:, ci].mean():.3f}) =====", file=sys.stderr)
        out[LAB[ci]] = []
        for fn, info, auc in scores[:20]:
            print(f"    {fn:40s} AUC={auc:.4f} info={info:.4f}", file=sys.stderr)
            out[LAB[ci]].append({'feat': fn, 'auc': float(auc), 'info': float(info)})

    # Save
    import json
    out_path = ROOT / f'docs/status/2026-06-01-b1-eda-v3-features.json'
    out_path.write_text(json.dumps({
        'sampled_convs': len(train_files),
        'rows': len(X),
        'num_features': len(feat_names),
        'feat_names': feat_names,
        'top_per_class': out,
    }, indent=2, ensure_ascii=False))
    print(f"\n[B1 EDA] artifact: {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
