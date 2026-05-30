"""climb cycle H-T6 — context 高次时序导数特征 (用户提示: 一二阶导/趋势).

动机: H-V7 增强 context 时序特征(计数/间隔/周期)零增益, 但漏了**导数类**——
计数抓"有多少", 导数抓"势头/动量"(某类正在加速上升 → 可能预示即将事件)。
关键优势: 纯 label 序列特征, train/test 完全同分布 (test context 也是 375 chunk label),
无 H-T6 发现的 train/test 文本格式 bug 风险。

特征 (每类 k ∈ {C,T,BC,I,NA}):
  一阶导 (变化率): 近窗频率 - 远窗频率 = 上升/下降趋势
  二阶导 (加速度): 一阶导本身的变化
  趋势斜率: 对分段窗频率序列做线性拟合斜率
  动量: 最近窗 vs 历史均值的偏离

对照:
  A. baseline v1 (46维)
  B. v1 + deriv (导数特征)
评估: 全切片 OOF 高分辨率 (相对比较) + cap1 参考. 看每类增量.

Usage: python tools/climb/cycle_deriv_feats.py [--folds 5] [--stride 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctx_v1

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM, CTX, TGT, SEED = 5, 375, 25, 42


def deriv_feats(ctx: np.ndarray) -> np.ndarray:
    """context 高次时序导数特征 (纯 label, train/test 同分布)."""
    L = len(ctx)
    oh = np.eye(NUM)[ctx]                                   # [L, 5] one-hot
    f = []

    # 分段窗频率序列 (从远到近 5 段, 每段 75 chunk = 6s)
    seg = 75
    nseg = L // seg                                         # ~5
    seg_freq = np.zeros((nseg, NUM))
    for s in range(nseg):
        seg_freq[s] = oh[s * seg:(s + 1) * seg].mean(axis=0)  # [NUM] 该段各类频率

    for k in range(NUM):
        fr = seg_freq[:, k]                                 # 该类频率随时间序列 (远→近)
        # 一阶导 (相邻段差分): 末段斜率 = 最近趋势
        d1 = np.diff(fr) if len(fr) >= 2 else np.array([0.0])
        f.append(float(d1[-1]))                             # 最近一阶导(上升/下降)
        f.append(float(d1.mean()))                          # 平均一阶导(整体趋势)
        # 二阶导 (一阶导的差分): 加速度
        d2 = np.diff(d1) if len(d1) >= 2 else np.array([0.0])
        f.append(float(d2[-1]))                             # 最近加速度
        # 趋势斜率 (线性拟合)
        if len(fr) >= 2:
            x = np.arange(len(fr))
            slope = float(np.polyfit(x, fr, 1)[0])
        else:
            slope = 0.0
        f.append(slope)
        # 动量: 最近段 vs 全程均值偏离
        f.append(float(fr[-1] - fr.mean()) if len(fr) else 0.0)

    # 多尺度近窗变化率 (短窗 vs 长窗频率比 = 该类是否"刚活跃起来")
    for k in range(NUM):
        for (short, long) in ((25, 100), (50, 200), (12, 50)):
            fs = (ctx[-short:] == k).mean() if L >= short else 0.0
            fl = (ctx[-long:] == k).mean() if L >= long else 0.0
            f.append(float(fs - fl))                        # 短窗-长窗 = 近期变化率

    # 转换动态: 转换率的导数 (对话节奏在加快还是放缓)
    trans = (ctx[1:] != ctx[:-1]).astype(float)
    for w in (50, 100):
        if L > w:
            recent_tr = trans[-w:].mean()
            older_tr = trans[-2 * w:-w].mean() if L > 2 * w else recent_tr
            f.append(float(recent_tr - older_tr))          # 转换率变化(节奏加速?)
        else:
            f.append(0.0)

    return np.array(f, dtype=np.float32)


def build(conv_ids, augment):
    X, Y, G = [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        for e in range(CTX, len(a) - TGT + 1, args.stride):
            ctx = a[e - CTX:e]
            base = ctx_v1(ctx)
            feat = np.concatenate([base, deriv_feats(ctx)]) if augment else base
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(feat)
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
    return np.array(X, dtype=np.float32), np.array(Y), np.array(G)


def oof_all(X, Y, G, conv_ids, folds):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(conv_ids))
    oof = np.zeros((len(X), NUM))
    for fi in range(folds):
        val = {perm[i] for i in range(len(conv_ids)) if i % folds == fi}
        tr = [i for i in range(len(X)) if G[i] not in val]
        va = [i for i in range(len(X)) if G[i] in val]
        for k in range(NUM):
            y = Y[tr, k]
            spw = (len(y) - y.sum()) / max(1, y.sum())
            c = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
            c.fit(X[tr], y)
            oof[va, k] = c.predict_proba(X[va])[:, 1]
    return oof


def best_f1(p, yt):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(yt, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return bf


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=40)
    args = ap.parse_args()

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    print(f"[deriv] {len(conv_ids)} convs stride={args.stride}", file=sys.stderr)

    res = {}
    for mode, label in [(False, "A_baseline"), (True, "B_deriv")]:
        X, Y, G = build(conv_ids, mode)
        oof = oof_all(X, Y, G, conv_ids, args.folds)
        per = {LAB[k]: round(best_f1(oof[:, k], Y[:, k]), 4) for k in range(NUM)}
        macro = round(float(np.mean(list(per.values()))), 4)
        res[label] = {"dim": X.shape[1], "macro": macro, "per": per}
        print(f"[{label:<12}] dim={X.shape[1]} macro={macro} | " +
              " ".join(f"{k}={per[k]:.3f}" for k in ["C", "T", "BC", "I", "NA"]), file=sys.stderr)

    print("\n=== 导数特征增量 (全切片OOF相对比较) ===")
    a = res["A_baseline"]["per"]
    for label, r in res.items():
        deltas = " ".join(f"{k}={r['per'][k]:.3f}({r['per'][k]-a[k]:+.3f})" for k in ["C", "T", "BC", "I", "NA"])
        print(f"  {label:<12} macro={r['macro']:.4f}({r['macro']-res['A_baseline']['macro']:+.4f}) | {deltas}")
    print(json.dumps({"cycle": "H-T6-deriv", "results": res}))


if __name__ == "__main__":
    main()
