"""climb cycle — paradigm=context-only v2 (H-001b, 榨分冲前10).

相对 v1 的增量:
  1. 更丰富序列特征：transition bigram 频率、各类游程统计、tail 模式
  2. K-fold OOF（按会话）→ 阈值在 pooled OOF 上调（更稳健，防 BC 过拟合）
  3. LGBM + XGBoost 双模型 rank 平均集成
  4. 全量重训 → test 预测 → pred_test1.csv

Usage: python tools/climb/cycle_context_v2.py <run_dir> [n_folds]
最后一行 stdout: {"score":..,"per_sub":{..}}  (eval-local 契约)
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL_TO_LABELID = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
NUM = 5
CTX, TGT, STRIDE = 375, 25, 5
SEED = 42


def featurize(ctx: np.ndarray) -> np.ndarray:
    oh = np.eye(NUM)[ctx]
    feats = []
    # 多窗口 ratio
    for w in (10, 25, 50, 100, 200, 375):
        feats.extend(oh[-w:].mean(axis=0))
    # 最后 8 个原始标签
    for i in range(1, 9):
        feats.append(ctx[-i] if len(ctx) >= i else -1)
    L = len(ctx)
    # 距上次各类出现归一化距离
    for k in range(NUM):
        pos = np.where(ctx == k)[0]
        feats.append((L - 1 - pos[-1]) / L if len(pos) else 1.0)
    # 各类总频率
    for k in range(NUM):
        feats.append((ctx == k).sum() / L)
    # 切换率 + 末段切换率
    feats.append((ctx[1:] != ctx[:-1]).mean())
    feats.append((ctx[-50:][1:] != ctx[-50:][:-1]).mean() if L >= 2 else 0.0)
    # transition bigram 频率（5x5=25，末 100 chunk 内）
    tail = ctx[-100:]
    bg = np.zeros((NUM, NUM), dtype=np.float32)
    for a, b in zip(tail[:-1], tail[1:]):
        bg[a, b] += 1
    bg = bg / max(1, bg.sum())
    feats.extend(bg.flatten())
    # 末段各类最长游程（归一化）
    for k in range(NUM):
        m = (ctx[-100:] == k).astype(int)
        best = cur = 0
        for v in m:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        feats.append(best / 100.0)
    return np.array(feats, dtype=np.float32)


def build(conv_ids, label_files, with_group=False):
    X, Y, G = [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(label_files[cid])
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE):
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(featurize(a[e - CTX:e].astype(int)))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            if with_group:
                G.append(gi)
    X, Y = np.array(X), np.array(Y, dtype=int)
    return (X, Y, np.array(G)) if with_group else (X, Y)


def mk_lgbm(spw):
    return LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=48,
                          scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED)


def mk_xgb(spw):
    return XGBClassifier(n_estimators=400, learning_rate=0.04, max_depth=6,
                         scale_pos_weight=spw, n_jobs=-1, random_state=SEED,
                         eval_metric="logloss", tree_method="hist")


def rankavg(*probs):
    from scipy.stats import rankdata
    rs = [rankdata(p) / len(p) for p in probs]
    return np.mean(rs, axis=0)


def tune_threshold(y, p):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/runs/climb/_adhoc_v2")
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_dir.mkdir(parents=True, exist_ok=True)

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    X, Y, G = build(conv_ids, label_files, with_group=True)
    print(f"[ctx-v2] {len(X)} windows, feat_dim={X.shape[1]}, convs={len(conv_ids)}", file=sys.stderr)

    # --- K-fold OOF（按会话分组）---
    gkf = GroupKFold(n_splits=n_folds)
    oof = {k: np.zeros(len(X)) for k in range(NUM)}
    for fold, (tr, va) in enumerate(gkf.split(X, Y[:, 0], groups=G)):
        for k in range(NUM):
            ytr = Y[tr, k]
            spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
            pl = mk_lgbm(spw).fit(X[tr], ytr).predict_proba(X[va])[:, 1]
            px = mk_xgb(spw).fit(X[tr], ytr).predict_proba(X[va])[:, 1]
            oof[k][va] = rankavg(pl, px)
        print(f"[ctx-v2] fold {fold+1}/{n_folds} done", file=sys.stderr)

    thr, f1s = {}, {}
    for k in range(NUM):
        t, f = tune_threshold(Y[:, k], oof[k])
        thr[k], f1s[k] = t, f
        print(f"[ctx-v2] {LABELS[k]:3s} OOF thr={t:.2f} F1={f:.3f}", file=sys.stderr)
    macro = float(np.mean(list(f1s.values())))
    print(f"[ctx-v2] OOF Macro-F1 = {macro:.4f}  (v1 单split=0.5908, gap+0.12→est线上~{macro+0.12:.3f})", file=sys.stderr)

    # --- 全量重训 → test ---
    test_files = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_files]
    Xte = np.array([featurize(np.load(p).astype(int)) for p in test_files])
    preds = {}
    for k in range(NUM):
        spw = (len(Y) - Y[:, k].sum()) / max(1, Y[:, k].sum())
        pl = mk_lgbm(spw).fit(X, Y[:, k]).predict_proba(Xte)[:, 1]
        px = mk_xgb(spw).fit(X, Y[:, k]).predict_proba(Xte)[:, 1]
        preds[k] = (rankavg(pl, px) >= thr[k]).astype(int)

    csv_path = run_dir / "pred_test1.csv"
    with open(csv_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT_COLS) + "\n")
        for i, sid in enumerate(seg_ids):
            f.write(",".join([sid] + [str(int(preds[COL_TO_LABELID[c]][i])) for c in SUBMIT_COLS]) + "\n")
    print(f"[ctx-v2] wrote {csv_path}", file=sys.stderr)

    per_sub = {c: round(f1s[COL_TO_LABELID[c]], 4) for c in ["c", "na", "t", "i", "bc"]}
    (run_dir / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "context-only", "hypothesis_id": "H-001b", "cv_macro_f1": round(macro, 4),
        "per_sub_f1": per_sub, "thresholds": {LABELS[k]: round(thr[k], 2) for k in range(NUM)},
        "method": f"{n_folds}-fold OOF (GroupKFold by conv) + LGBM+XGB rank-avg + richer feats",
    }, ensure_ascii=False, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps({
        "cycle": 2, "hypothesis_id": "H-001b", "paradigm": "context-only",
        "start": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"score": round(macro, 4), "per_sub": per_sub}))


if __name__ == "__main__":
    main()
