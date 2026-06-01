"""cycle 19b — LGBM 超参 sweep, 在 stride5 全量 5-fold OOF 选最优 (避 D-3 cap1 cherry-pick).

当前 lgbm_v1 base: n_estimators=300, learning_rate=0.05, num_leaves=31, fold OOF cap1=0.6228.

sweep 维度 (~36 组合):
  n_estimators: [200, 300, 500, 800]
  learning_rate: [0.03, 0.05, 0.08]
  num_leaves: [31, 63]
  feature_fraction: [1.0, 0.8] (列子采样, 增加多样性)

判据 (避 D-3/D-9/D-11 cap1 cherry-pick):
  - 不在 cap1 上选 best (cap1 369 样本对 36 组合搜 = 100% 过拟合)
  - 用 OOF 全量 (179867 样本) 5-fold macro-F1 选 best
  - 选出 best 后, **必须**: cap1 macro > lgbm_v1 cap1 0.6228 + 0.005 (gate, 守稳定增益)

不重训音频 head (whisper/hubert 已固定), 只重训 LGBM. ~30min 本机.
若 best > +0.005 → 替换 _stack_cache_s40.npz oof_lgbm_v1/te_lgbm_v1 → 跑 nsrc 三源融合看 cap1 fused 升多少.
"""
from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat

NUM, CTX, TGT = 5, 375, 25
STRIDE = 5  # 同 _stack_cache_s40 用 stride40? 先看 cache 实际 stride
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SEED = 42
np.random.seed(SEED)


def cap1_idx(G):
    seen, cap1 = set(), []
    for i in range(len(G)):
        if int(G[i]) not in seen:
            cap1.append(i); seen.add(int(G[i]))
    return np.array(cap1)


def f1k(p, y, thr):
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}


def eval_oof(oof, Y, G):
    """OOF 全量 + cap1 双指标"""
    macro_full = np.mean([f1k(oof[:, k], Y[:, k], THR_VARF[k]) for k in range(NUM)])
    cap1 = cap1_idx(G)
    macro_cap1 = np.mean([f1k(oof[cap1, k], Y[cap1, k], THR_VARF[k]) for k in range(NUM)])
    per_class_cap1 = {LAB[k]: f1k(oof[cap1, k], Y[cap1, k], THR_VARF[k]) for k in range(NUM)}
    return macro_full, macro_cap1, per_class_cap1


def fit_lgbm_with_params(X, y, params):
    spw = (len(y) - y.sum()) / max(1, y.sum())
    clf = LGBMClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        feature_fraction=params["feature_fraction"],
        scale_pos_weight=spw,
        n_jobs=-1, verbose=-1, random_state=SEED,
    )
    clf.fit(X, y)
    return clf


def oof_predict_5fold(X, Y, G, params):
    """5-fold OOF, 按 G (通号) 分组防泄漏."""
    oof = np.zeros((len(X), NUM), dtype=np.float64)
    gkf = GroupKFold(n_splits=5)
    for fold_id, (tr, va) in enumerate(gkf.split(X, Y[:, 0], G)):
        for k in range(NUM):
            clf = fit_lgbm_with_params(X[tr], Y[tr, k], params)
            oof[va, k] = clf.predict_proba(X[va])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=40, help="40=快, 5=全量 (~30min/组合 太慢)")
    ap.add_argument("--quick", action="store_true", help="只跑 4 组合冒烟")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[sweep] 加载数据 (stride={args.stride})...")
    # build X, Y, G 从 train labels 直接 (跳过 stride5 全量 cache, 仍用同步采样)
    label_files = sorted(glob.glob("data/train/labels/*.npy"))
    X, Y, G = [], [], []
    for gi, f in enumerate(label_files):
        a = np.load(f)
        for e in range(CTX, a.shape[0] - TGT + 1, args.stride):
            X.append(ctxfeat(a[e - CTX:e].astype(int)))
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
    X = np.array(X); Y = np.array(Y, dtype=int); G = np.array(G)
    print(f"[sweep] 共 {len(X)} 窗 ({len(label_files)} 通, stride={args.stride})  load_t={time.time()-t0:.1f}s")

    # baseline (lgbm_v1 现 default)
    BASELINE = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "feature_fraction": 1.0}

    if args.quick:
        sweeps = [
            BASELINE,
            {**BASELINE, "n_estimators": 500},
            {**BASELINE, "learning_rate": 0.08, "num_leaves": 63},
            {**BASELINE, "feature_fraction": 0.8, "n_estimators": 500},
        ]
    else:
        # 完整 sweep ~36 组合
        sweeps = []
        for ne in [200, 300, 500, 800]:
            for lr in [0.03, 0.05, 0.08]:
                for nl in [31, 63]:
                    for ff in [1.0, 0.8]:
                        sweeps.append({"n_estimators": ne, "learning_rate": lr, "num_leaves": nl, "feature_fraction": ff})

    print(f"[sweep] {len(sweeps)} 组合, 5-fold OOF each")
    results = []
    for i, p in enumerate(sweeps):
        ts = time.time()
        oof = oof_predict_5fold(X, Y, G, p)
        mfull, mcap1, pc = eval_oof(oof, Y, G)
        dt = time.time() - ts
        is_base = (p == BASELINE)
        tag = " ← BASELINE" if is_base else ""
        print(f"[{i+1}/{len(sweeps)}] {p}  full_macro={mfull:.4f}  cap1_macro={mcap1:.4f}  ({dt:.0f}s){tag}")
        results.append({"params": p, "full_macro": mfull, "cap1_macro": mcap1, "per_class_cap1": pc, "is_base": is_base})

    # 选 OOF full_macro 最高 (避 cap1 369 cherry-pick)
    best = max(results, key=lambda r: r["full_macro"])
    base = next(r for r in results if r["is_base"])
    print()
    print("=== 结果 ===")
    print(f"BASELINE: full_macro={base['full_macro']:.4f}  cap1_macro={base['cap1_macro']:.4f}")
    print(f"BEST:     full_macro={best['full_macro']:.4f}  cap1_macro={best['cap1_macro']:.4f}")
    print(f"BEST params: {best['params']}")
    print(f"Δ full_macro: {best['full_macro']-base['full_macro']:+.4f}")
    print(f"Δ cap1_macro: {best['cap1_macro']-base['cap1_macro']:+.4f}")
    print(f"BEST cap1 per-class: {best['per_class_cap1']}")

    # gate: cap1 涨 +0.005 才采纳 (避 D-3/D-9 cap1 噪声)
    gate = 0.005
    if best['cap1_macro'] - base['cap1_macro'] > gate:
        print(f"\n★ best cap1 涨 +{best['cap1_macro']-base['cap1_macro']:.4f} > +{gate} gate, 推荐重训 lgbm_v1 替 _stack_cache")
    else:
        print(f"\n× best cap1 仅 +{best['cap1_macro']-base['cap1_macro']:.4f} < +{gate} gate, 不替换 (避陷阱)")

    print(f"\n[sweep] total time {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
