"""climb cycle H-ENS — 算法正交集成 (榜单启发: 融合增益需算法正交强模型).

诊断 (2026-05-30): 现有 context 变体不正交(都是LGBM+不同阈值/seed,5seed已榨干)。
榜单分布(前10 0.73-0.75/我们0.712, 增益1-3分部分来自融合)→ 正确做法=造算法正交
强模型(LGBM/XGB/CatBoost/MLP over 同context特征, 同~0.71但归纳偏置不同→真正交)。

做法:
  4算法各 5fold conv级 OOF (cap1可信评估) + test 概率
  概率平均融合 → cap1 macro vs 单模型
  阈值: 变体F固定(C.05/T.5/BC.75/I.65/NA.25) — 守铁律不cap1激进调
  出 test 概率存盘(供后续阈值/融合实验, 解决"无context test概率"问题)

判读: 融合 cap1 > 最佳单模型 +0.005 = 算法正交有增益(榜单路径成立)
Usage: python tools/climb/cycle_algo_ensemble.py [--folds 5] [--submit]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, "tools/climb")
from cycle_context import featurize

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM, CTX, TGT, SEED = 5, 375, 25, 42
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}


def make_clf(algo, spw):
    if algo == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
    if algo == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             scale_pos_weight=spw, n_jobs=4, verbosity=0, random_state=SEED,
                             tree_method="hist")
    if algo == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                  scale_pos_weight=spw, thread_count=4, verbose=0,
                                  random_seed=SEED)
    if algo == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # MLP 不支持 scale_pos_weight/sample_weight → 用 StandardScaler + 训练时 oversample
        # (sklearn MLPClassifier.fit 无 sample_weight 参数, 改在 oof_one 里对正类过采样)
        return Pipeline([("sc", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                               alpha=1e-3, early_stopping=True,
                                               n_iter_no_change=15, random_state=SEED))])
    raise ValueError(algo)


def _balance_idx(y, rng, ratio=3.0):
    """对正类过采样到 neg/pos ≤ ratio (MLP 无 sample_weight 的替代). 返回训练索引."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    target_pos = int(len(neg) / ratio)
    if target_pos > len(pos):
        reps = rng.choice(pos, size=target_pos - len(pos), replace=True)
        idx = np.concatenate([np.arange(len(y)), reps])
    else:
        idx = np.arange(len(y))
    rng.shuffle(idx)
    return idx


def build(stride):
    conv = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    X, Y, G, order = [], [], [], []
    for gi, cid in enumerate(conv):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        o = 0
        for e in range(CTX, len(a) - TGT + 1, stride):
            X.append(featurize(a[e - CTX:e]))
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi); order.append(o); o += 1
    return np.array(X, dtype=np.float32), np.array(Y), np.array(G), np.array(order), conv


def oof_one(algo, X, Y, G, conv, folds):
    """单算法 5fold OOF (全部类). 返回 [N,5] OOF 概率.
    fold 划分用 i%folds==fi (与 cap1_macro 一致, 复现性). conv 顺序固定不 shuffle."""
    rng = np.random.default_rng(SEED)
    oof = np.zeros((len(X), NUM))
    for fi in range(folds):
        val = {i for i in range(len(conv)) if i % folds == fi}
        tr = np.array([i for i in range(len(X)) if G[i] not in val])
        va = np.array([i for i in range(len(X)) if G[i] in val])
        for k in range(NUM):
            y = Y[tr, k]; spw = (len(y) - y.sum()) / max(1, y.sum())
            c = make_clf(algo, spw)
            if algo == "mlp":
                # MLP 无 sample_weight → 正类过采样到 neg/pos≤3 让它学稀有类
                bidx = _balance_idx(y, rng, ratio=3.0)
                c.fit(X[tr][bidx], y[bidx])
            else:
                c.fit(X[tr], y)
            oof[va, k] = c.predict_proba(X[va])[:, 1]
    return oof


def cap1_macro(oof, Y, G, conv, thr):
    seen, cap1 = set(), []
    for i in range(len(G)):
        if G[i] not in seen:
            cap1.append(i); seen.add(G[i])
    cap1 = np.array(cap1)
    per = {k: f1_score(Y[cap1, k], (oof[cap1, k] >= thr[k]).astype(int), zero_division=0)
           for k in range(NUM)}
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--algos", default="lgbm,xgb,cat,mlp")
    ap.add_argument("--stride", type=int, default=40)
    args = ap.parse_args()

    X, Y, G, order, conv = build(args.stride)
    print(f"[ens] {len(conv)} convs, {len(X)} samples", file=sys.stderr)

    algos = args.algos.split(",")
    oofs = {}
    for algo in algos:
        oof = oof_one(algo, X, Y, G, conv, args.folds)
        m, per = cap1_macro(oof, Y, G, conv, THR_VARF)
        oofs[algo] = oof
        print(f"[{algo:<5}] cap1 macro={m:.4f} | " +
              " ".join(f"{LAB[k]}={per[k]:.3f}" for k in range(NUM)), file=sys.stderr)

    # 概率平均融合
    ens = np.mean([oofs[a] for a in algos], axis=0)
    m_ens, per_ens = cap1_macro(ens, Y, G, conv, THR_VARF)
    best_single = max(cap1_macro(oofs[a], Y, G, conv, THR_VARF)[0] for a in algos)

    print(f"\n=== 算法正交集成 (cap1, 变体F固定阈值) ===")
    for a in algos:
        ma, _ = cap1_macro(oofs[a], Y, G, conv, THR_VARF)
        print(f"  {a:<6} {ma:.4f}")
    print(f"  ENSEMBLE {m_ens:.4f} ({m_ens-best_single:+.4f} vs best single)")
    print(f"  per-class: " + " ".join(f"{LAB[k]}={per_ens[k]:.3f}" for k in range(NUM)))
    print(f"\n判读: 集成 > best single +0.005 = 算法正交有增益(榜单融合路径成立)")
    print(f"  cap1 gap +0.055 → 线上估 {m_ens+0.055:.4f} (变体F线上0.712)")
    print(json.dumps({"cycle": "H-ENS", "ensemble_cap1": round(m_ens, 4),
                      "best_single": round(best_single, 4), "gain": round(m_ens - best_single, 4),
                      "per_class": {LAB[k]: round(per_ens[k], 4) for k in range(NUM)}}))


if __name__ == "__main__":
    main()
