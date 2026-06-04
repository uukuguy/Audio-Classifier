"""F0/spectral 特征 → 5fold lgbm head → cap1 + OOF probs.

数据: data/cache/f0_features.npz (57d 声学特征, 2073 段)
模型: lgbm per-class, 5fold conv-level
输出: probs.npz (oof + test) → 入 orthofuse 看 SOTA 是否能涨

Usage:
  OMP_NUM_THREADS=8 python3 tools/climb/train_f0_head.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SEED = 42
N_FOLDS = 5
RUN_DIR = Path(f"tools/runs/climb/f0-head-{time.strftime('%Y%m%d-%H%M')}")


def cap1_idx(G, order, is_aug):
    """每通取首窗 (order=0) + 原始 (is_aug=0) 作为 cap1 评估集."""
    seen, cap1 = set(), []
    for i, (g, o, a) in enumerate(zip(G, order, is_aug)):
        if int(o) == 0 and int(a) == 0 and int(g) not in seen:
            cap1.append(i); seen.add(int(g))
    return np.array(cap1)


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[f0-head] loading f0 features...", file=sys.stderr, flush=True)
    d = np.load("data/cache/f0_features.npz")
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(int)
    G = d["G"]
    order = d["order"]
    is_aug = d["is_aug"]
    Xt = d["Xt"].astype(np.float32)
    test_ids = d["test_ids"]

    n_convs = int(G.max()) + 1
    print(f"[f0-head] X={X.shape} Y={Y.shape} convs={n_convs} test={Xt.shape}", file=sys.stderr, flush=True)
    print(f"[f0-head] BC 正例 (含增强) = {Y[:, 2].sum()}", file=sys.stderr, flush=True)

    # cap1 for eval
    cap1 = cap1_idx(G, order, is_aug)
    Y_c1 = Y[cap1]
    print(f"[f0-head] cap1 N={len(cap1)} (每通 order=0 原始)", file=sys.stderr, flush=True)

    # 5fold conv-level
    rng = np.random.default_rng(SEED)
    conv_perm = rng.permutation(n_convs)
    fold_of = np.zeros(n_convs, dtype=int)
    for i, c in enumerate(conv_perm):
        fold_of[c] = i % N_FOLDS

    oof = np.zeros_like(Y, dtype=np.float32)
    test_probs = np.zeros((len(Xt), 5), dtype=np.float32)

    t_total = time.time()
    for k in range(5):
        # pos_weight
        ratio = (Y[:, k] == 0).sum() / max(1, Y[:, k].sum())
        spw = min(ratio, 10.0)
        for fi in range(N_FOLDS):
            val_convs = set(np.where(fold_of == fi)[0].tolist())
            tr_mask = np.array([g not in val_convs for g in G])
            va_mask = np.array([g in val_convs and a == 0 for g, a in zip(G, is_aug)])
            # train 含增强, val 仅原始
            clf = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED,
            )
            clf.fit(X[tr_mask], Y[tr_mask, k])
            oof[va_mask, k] = clf.predict_proba(X[va_mask])[:, 1]
            test_probs[:, k] += clf.predict_proba(Xt)[:, 1] / N_FOLDS
        print(f"[f0-head] class {LAB[k]} 5fold done", file=sys.stderr, flush=True)

    print(f"[f0-head] all done in {time.time()-t_total:.0f}s", file=sys.stderr, flush=True)

    # cap1 eval
    print("\n=== F0/spectral 单源 cap1 ===")
    f1_per = {}
    for k in range(5):
        pred = (oof[cap1, k] >= THR_VARF[k]).astype(int)
        f1_per[k] = float(f1_score(Y_c1[:, k], pred, zero_division=0))
    macro = float(np.mean(list(f1_per.values())))
    print(f"varF macro={macro:.4f} | " + " ".join(f"{LAB[k]}={f1_per[k]:.3f}" for k in range(5)))

    # 概率分布看 BC 是否有信号 (D-19 教训)
    print("\n=== BC 概率分布 (cap1 + 全 OOF) ===")
    bc_cap1 = oof[cap1, 2]
    print(f"cap1 BC: min={bc_cap1.min():.3f} max={bc_cap1.max():.3f} mean={bc_cap1.mean():.3f}")
    print(f"  真正例 N={Y_c1[:,2].sum()}, prob mean={bc_cap1[Y_c1[:,2]==1].mean():.3f}")
    print(f"  真负例 N={(Y_c1[:,2]==0).sum()}, prob mean={bc_cap1[Y_c1[:,2]==0].mean():.3f}")
    bc_full = oof[:, 2]
    print(f"全 OOF BC: max={bc_full.max():.3f} q99={np.quantile(bc_full, 0.99):.3f}")
    bc_y = Y[:, 2]
    print(f"  真正例 N={bc_y.sum()}, prob mean={bc_full[bc_y==1].mean():.3f}")
    print(f"  真负例 N={(bc_y==0).sum()}, prob mean={bc_full[bc_y==0].mean():.3f}")

    # 保存 probs.npz (orthofuse 格式)
    np.savez_compressed(
        RUN_DIR / "probs.npz",
        oof=oof, test=test_probs, Y=Y.astype(np.int8),
        G=G.astype(np.int16), order=order.astype(np.int16),
    )
    print(f"\n[f0-head] saved {RUN_DIR}/probs.npz", file=sys.stderr, flush=True)

    (RUN_DIR / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "f0-spectral-head-lgbm",
        "feat_dim": int(X.shape[1]),
        "n_train": int(X.shape[0]),
        "n_train_aug": int((is_aug > 0).sum()),
        "n_test": int(Xt.shape[0]),
        "cap1_macro_f1": round(macro, 4),
        "per_sub_cap1": {LAB[k]: round(f1_per[k], 4) for k in range(5)},
        "bc_prob_stats": {
            "max": float(bc_full.max()),
            "pos_mean": float(bc_full[bc_y==1].mean()),
            "neg_mean": float(bc_full[bc_y==0].mean()),
            "pos_neg_diff": float(bc_full[bc_y==1].mean() - bc_full[bc_y==0].mean()),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
