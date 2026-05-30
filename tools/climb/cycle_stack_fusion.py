"""climb cycle H-STACK — 加权/stacking 融合 (冲 0.75, 修上轮等权融合 -0.023 的根因).

上轮 (H-ENS) 算法正交集成失败 = (1) MLP 坏(BC=0 拖垮等权) (2) 三树不正交 (3) 等权本身错.
本轮修三点:
  1. MLP 已修(过采样学稀有类, 见 cycle_algo_ensemble._balance_idx)
  2. 加真正交成员: 除换算法, 再加"不同特征子集"(v1 ctx 特征 vs v2 富序列特征)
  3. 不等权: per-class 在 OOF 上搜最优融合 (权重网格 + stacking logistic meta)

成员 (每类独立, 都出 OOF[N] + test[M] 概率):
  M1 lgbm_v1   : LGBM over v1 特征 (= 变体F 基座, 强 baseline)
  M2 xgb_v1    : XGB  over v1 特征 (算法正交)
  M3 lgbm_v2   : LGBM over v2 富特征 (特征正交 — 不同输入 → 真正交)
  M4 mlp_v1    : MLP  over v1 特征 (神经归纳偏置, 过采样修 BC=0)

融合策略 (per-class 选最优, cap1 CV 裁判):
  A 等权平均   : mean(members) — 上轮基线
  B 权重网格   : 在 OOF 上 grid-search 凸组合权重 (粗网格防过拟合)
  C stacking   : logistic meta-learner over 成员 OOF 概率 (5fold nested 防泄漏)

判读: 最优融合 cap1 > lgbm_v1 单模 +0.005 = 正交融合成立(榜单路径). 守阈值铁律(变体F固定阈值).
Usage: OMP_NUM_THREADS=4 python tools/climb/cycle_stack_fusion.py [--folds 5] [--stride 40] [--submit]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as v1feat  # noqa: E402
from cycle_context_v2 import featurize as v2feat  # noqa: E402

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM, CTX, TGT, SEED = 5, 375, 25, 42
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}

# 成员定义: (name, algo, feat_fn)
MEMBERS = [
    ("lgbm_v1", "lgbm", "v1"),
    ("xgb_v1", "xgb", "v1"),
    ("lgbm_v2", "lgbm", "v2"),
    ("mlp_v1", "mlp", "v1"),
]


def _balance_idx(y, rng, ratio=3.0):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
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
    if algo == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([("sc", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                               alpha=1e-3, early_stopping=True,
                                               n_iter_no_change=15, random_state=SEED))])
    raise ValueError(algo)


def build(stride):
    conv = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    Xv1, Xv2, Y, G = [], [], [], []
    for gi, cid in enumerate(conv):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        for e in range(CTX, len(a) - TGT + 1, stride):
            ctx = a[e - CTX:e]
            Xv1.append(v1feat(ctx)); Xv2.append(v2feat(ctx))
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
    feats = {"v1": np.array(Xv1, dtype=np.float32), "v2": np.array(Xv2, dtype=np.float32)}
    return feats, np.array(Y), np.array(G), conv


def build_test():
    files = sorted(glob.glob("data/test/context/*.npy"))
    seg = [Path(p).stem for p in files]
    Xv1, Xv2 = [], []
    for p in files:
        ctx = np.load(p).astype(int)
        Xv1.append(v1feat(ctx)); Xv2.append(v2feat(ctx))
    return {"v1": np.array(Xv1, dtype=np.float32), "v2": np.array(Xv2, dtype=np.float32)}, seg


def member_oof_and_test(name, algo, fk, feats, Y, G, conv, Xte, folds):
    """单成员: 5fold OOF [N,5] + 全量重训 test [M,5]."""
    X = feats[fk]; rng = np.random.default_rng(SEED)
    oof = np.zeros((len(X), NUM)); te = np.zeros((len(Xte[fk]), NUM))
    for fi in range(folds):
        val = {i for i in range(len(conv)) if i % folds == fi}
        tr = np.array([i for i in range(len(X)) if G[i] not in val])
        va = np.array([i for i in range(len(X)) if G[i] in val])
        for k in range(NUM):
            y = Y[tr, k]; spw = (len(y) - y.sum()) / max(1, y.sum())
            c = make_clf(algo, spw)
            if algo == "mlp":
                bidx = _balance_idx(y, rng, ratio=3.0)
                c.fit(X[tr][bidx], y[bidx])
            else:
                c.fit(X[tr], y)
            oof[va, k] = c.predict_proba(X[va])[:, 1]
    # 全量重训 → test
    for k in range(NUM):
        y = Y[:, k]; spw = (len(y) - y.sum()) / max(1, y.sum())
        c = make_clf(algo, spw)
        if algo == "mlp":
            bidx = _balance_idx(y, rng, ratio=3.0)
            c.fit(X[bidx], y[bidx])
        else:
            c.fit(X, y)
        te[:, k] = c.predict_proba(Xte[fk])[:, 1]
    return oof, te


def cap1_idx(G, conv):
    seen, cap1 = set(), []
    for i in range(len(G)):
        if G[i] not in seen:
            cap1.append(i); seen.add(G[i])
    return np.array(cap1)


def class_f1(p, y, thr):
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


def grid_weights(member_oof_k, y_cap1, cap1, thr):
    """per-class 在 cap1 OOF 上粗网格搜凸组合权重. 返回 (best_w, best_f1)."""
    n = len(member_oof_k)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_w, best_f = None, -1.0
    for combo in product(grid, repeat=n):
        s = sum(combo)
        if s == 0:
            continue
        w = np.array(combo) / s
        p = sum(w[j] * member_oof_k[j] for j in range(n))
        f = class_f1(p[cap1], y_cap1, thr)
        if f > best_f:
            best_f, best_w = f, w
    return best_w, best_f


def _grid_on_subset(member_oof_k, y_sub, idx_sub, thr):
    """在给定 cap1 子集 idx_sub 上 grid-search 权重, 返回最优 w (不评估)."""
    n = len(member_oof_k)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_w, best_f = None, -1.0
    for combo in product(grid, repeat=n):
        s = sum(combo)
        if s == 0:
            continue
        w = np.array(combo) / s
        p = sum(w[j] * member_oof_k[j][idx_sub] for j in range(n))
        f = f1_score(y_sub, (p >= thr).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_w = f, w
    return best_w


def nested_grid_cv(member_oof_k, Y_k, G, conv, cap1, thr, folds=5):
    """决定性过拟合验证: cap1 的 conv 分 folds 折, 训练折 grid 搜权重, 留出折评估.
    留出折汇总 F1 = grid 权重的泛化估计. 若 << in-sample grid → 过拟合(同 ti-robust)."""
    rng = np.random.default_rng(SEED)
    cap1_convs = [int(G[i]) for i in cap1]  # cap1 每样本的 conv id (=本身, 每通1样本)
    order = np.array(sorted(set(cap1_convs)))
    rng.shuffle(order)
    fold_of = {c: i % folds for i, c in enumerate(order)}
    held_pred, held_y = [], []
    for fi in range(folds):
        tr_pos = np.array([p for p, i in enumerate(cap1) if fold_of[int(G[i])] != fi])
        va_pos = np.array([p for p, i in enumerate(cap1) if fold_of[int(G[i])] == fi])
        if len(va_pos) == 0 or len(tr_pos) == 0:
            continue
        # 成员 OOF 在 cap1 上的值
        moof_cap1 = [m[cap1] for m in member_oof_k]
        y_tr = Y_k[cap1][tr_pos]
        w = _grid_on_subset(moof_cap1, y_tr, tr_pos, thr)
        p_va = sum(w[j] * moof_cap1[j][va_pos] for j in range(len(moof_cap1)))
        held_pred.append((p_va >= thr).astype(int))
        held_y.append(Y_k[cap1][va_pos])
    if not held_pred:
        return 0.0
    return f1_score(np.concatenate(held_y), np.concatenate(held_pred), zero_division=0)


def stack_meta(member_oof_k, Y_k, G, conv, cap1, thr, folds):
    """logistic meta over 成员 OOF 概率, nested 5fold 防泄漏 → cap1 meta-OOF F1 + 全量 meta."""
    Z = np.column_stack(member_oof_k)  # [N, n_members]
    meta_oof = np.zeros(len(Z))
    for fi in range(folds):
        val = {i for i in range(len(conv)) if i % folds == fi}
        tr = np.array([i for i in range(len(Z)) if G[i] not in val])
        va = np.array([i for i in range(len(Z)) if G[i] in val])
        spw = (Y_k[tr] == 0).sum() / max(1, (Y_k[tr] == 1).sum())
        lr = LogisticRegression(max_iter=1000, C=1.0,
                                class_weight={0: 1.0, 1: float(spw)})
        lr.fit(Z[tr], Y_k[tr])
        meta_oof[va] = lr.predict_proba(Z[va])[:, 1]
    f = class_f1(meta_oof[cap1], Y_k[cap1], thr)
    # 全量 meta (用于 test)
    spw = (Y_k == 0).sum() / max(1, (Y_k == 1).sum())
    meta_full = LogisticRegression(max_iter=1000, C=1.0,
                                   class_weight={0: 1.0, 1: float(spw)})
    meta_full.fit(Z, Y_k)
    return f, meta_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--cached", action="store_true", help="复用缓存的成员 OOF/test 概率")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    feats, Y, G, conv = build(args.stride)
    Xte, seg = build_test()
    cap1 = cap1_idx(G, conv)
    print(f"[stack] {len(conv)} convs, {len(Y)} samples, cap1={len(cap1)}, test={len(seg)}",
          file=sys.stderr)

    cache = Path(f"tools/runs/climb/_stack_cache_s{args.stride}.npz")
    # 1) 每成员 OOF + test (缓存复用, 省 13min 重训)
    moof, mte = {}, {}
    if args.cached and cache.exists():
        z = np.load(cache)
        for name, _, _ in MEMBERS:
            moof[name], mte[name] = z[f"oof_{name}"], z[f"te_{name}"]
        Y, G = z["Y"], z["G"]
        print(f"[stack] loaded cached members from {cache}", file=sys.stderr)
    else:
        for name, algo, fk in MEMBERS:
            oof, te = member_oof_and_test(name, algo, fk, feats, Y, G, conv, Xte, args.folds)
            moof[name], mte[name] = oof, te
            m = np.mean([class_f1(oof[cap1, k], Y[cap1, k], THR_VARF[k]) for k in range(NUM)])
            print(f"[member {name:<9}] cap1 macro={m:.4f} | " +
                  " ".join(f"{LAB[k]}={class_f1(oof[cap1,k],Y[cap1,k],THR_VARF[k]):.3f}"
                           for k in range(NUM)), file=sys.stderr)
        save = {f"oof_{n}": moof[n] for n, _, _ in MEMBERS}
        save.update({f"te_{n}": mte[n] for n, _, _ in MEMBERS})
        save.update({"Y": Y, "G": G})
        np.savez(cache, **save)
        print(f"[stack] cached members to {cache}", file=sys.stderr)

    names = [m[0] for m in MEMBERS]
    base = "lgbm_v1"

    # 2) per-class 选最优融合策略 (A 等权 / B 权重网格 / C stacking)
    per_strat = {}  # k -> (strat, f1, payload)
    test_prob = np.zeros((len(seg), NUM))
    for k in range(NUM):
        thr = THR_VARF[k]
        moof_k = [moof[n][:, k] for n in names]
        mte_k = [mte[n][:, k] for n in names]
        y_cap1 = Y[cap1, k]

        # A 等权
        p_eq = np.mean(moof_k, axis=0)
        f_eq = class_f1(p_eq[cap1], y_cap1, thr)
        # B 权重网格
        w_b, f_b = grid_weights(moof_k, y_cap1, cap1, thr)
        # C stacking
        f_c, meta_full = stack_meta(moof_k, Y[:, k], G, conv, cap1, thr, args.folds)
        # baseline 单模
        f_base = class_f1(moof[base][cap1, k], y_cap1, thr)

        # ★决定性: nested CV 验 grid 是否过拟合 cap1 (留出折泛化 F1)
        f_grid_nested = nested_grid_cv(moof_k, Y[:, k], G, conv, cap1, thr, args.folds)

        cand = [("equal", f_eq, None), ("grid", f_b, w_b), ("stack", f_c, meta_full),
                ("single", f_base, None)]
        strat, f_best, payload = max(cand, key=lambda x: x[1])
        per_strat[k] = (strat, f_best, payload)
        print(f"[fuse {LAB[k]:>3}] eq={f_eq:.3f} grid={f_b:.3f}(nested={f_grid_nested:.3f}) "
              f"stack={f_c:.3f} single={f_base:.3f} → pick {strat} ({f_best:.3f})",
              file=sys.stderr)

        # test 概率按所选策略
        if strat == "equal":
            test_prob[:, k] = np.mean(mte_k, axis=0)
        elif strat == "grid":
            test_prob[:, k] = sum(payload[j] * mte_k[j] for j in range(len(mte_k)))
        elif strat == "stack":
            Zte = np.column_stack(mte_k)
            test_prob[:, k] = payload.predict_proba(Zte)[:, 1]
        else:  # single
            test_prob[:, k] = mte[base][:, k]

    macro = float(np.mean([per_strat[k][1] for k in range(NUM)]))
    base_macro = float(np.mean([class_f1(moof[base][cap1, k], Y[cap1, k], THR_VARF[k])
                                for k in range(NUM)]))
    eq_macro = float(np.mean([class_f1(np.mean([moof[n][:, k] for n in names], axis=0)[cap1],
                                       Y[cap1, k], THR_VARF[k]) for k in range(NUM)]))

    print(f"\n=== STACK FUSION (cap1, 变体F固定阈值) ===")
    print(f"  baseline lgbm_v1   {base_macro:.4f}")
    print(f"  equal-weight       {eq_macro:.4f} ({eq_macro-base_macro:+.4f})")
    print(f"  per-class-best     {macro:.4f} ({macro-base_macro:+.4f} vs base)")
    print(f"  strat per class: " + " ".join(f"{LAB[k]}={per_strat[k][0]}" for k in range(NUM)))
    print(f"  cap1 gap +0.072(变体F) → 线上估 {macro+0.072:.4f}")
    print(f"\n判读: per-class-best > base +0.005 = 正交融合成立")

    out = {"cycle": "H-STACK", "base_cap1": round(base_macro, 4),
           "equal_cap1": round(eq_macro, 4), "best_cap1": round(macro, 4),
           "gain_vs_base": round(macro - base_macro, 4),
           "strat": {LAB[k]: per_strat[k][0] for k in range(NUM)},
           "per_class": {LAB[k]: round(per_strat[k][1], 4) for k in range(NUM)}}

    if args.submit:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        run = Path(f"tools/runs/climb/stack-fusion-{ts}")
        run.mkdir(parents=True, exist_ok=True)
        with open(run / "pred_test1.csv", "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(seg):
                row = [sid] + [str(int(test_prob[i, COL2K[c]] >= THR_VARF[COL2K[c]]))
                               for c in SUBMIT]
                f.write(",".join(row) + "\n")
        (run / "cv_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
        cnts = {c: int((test_prob[:, COL2K[c]] >= THR_VARF[COL2K[c]]).sum()) for c in SUBMIT}
        out["pos_counts"] = cnts
        out["csv"] = str(run / "pred_test1.csv")
        print(f"[submit] wrote {run}/pred_test1.csv  pos={cnts}", file=sys.stderr)

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
