"""climb cycle H-V7 — 攻 BC: 增强 context 时序特征 + 二阶 precision 模型.

诊断依据 (2026-05-30):
  - BC 可预测信号在 context 标签时序里 (过去2s有BC→未来BC r=+0.134), 不在音频 (|r|<0.04)
  - BC 是 PRECISION 灾难: thr0.3 时 R=0.83 (能感知BC要来) 但 P=0.05 (假警报淹没)
  - 当前 featurize 只用了粗统计 (类频率/距上次/转换率), 没用 BC 突发性/周期/节奏

本实验 (不改 baseline featurize, 新增 bc_temporal_feats):
  1. 增强 BC 时序特征: 多窗短突发性 / BC间隔分布 / 周期性 / 说话人切换耦合 / 自相关
  2. 对比 3 个配置的 BC F1 (cap1 可信 CV, 会话级 5fold OOF):
     A. baseline featurize (复现 0.222 锚点)
     B. featurize + bc_temporal_feats (单阶 LGBM)
     C. 二阶: stage1 高recall召回候选 → stage2 增强特征判真假 (攻 precision)

Usage:
  python tools/climb/cycle_bc_temporal.py [--folds 5] [--stride 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, "tools/climb")
from cycle_context import featurize  # baseline 特征 (不改)

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM, CTX, TGT, SEED = 5, 375, 25, 42
BC = 2


def bc_temporal_feats(ctx: np.ndarray) -> np.ndarray:
    """BC 专属时序特征 (诊断驱动: 突发性/间隔/周期/耦合). ctx = 375 chunk 标签序列."""
    L = len(ctx)
    f = []
    bc_pos = np.where(ctx == BC)[0]

    # 1. 多窗短突发性 (BC 聚集性 — 诊断 r=+0.134)
    for w in (12, 25, 50, 75, 125):  # ~1s/2s/4s/6s/10s @ 80ms
        f.append(float((ctx[-w:] == BC).sum()) if L >= w else 0.0)         # 窗内 BC 计数
        f.append(float((ctx[-w:] == BC).any()) if L >= w else 0.0)         # 窗内有无 BC

    # 2. BC 间隔分布 (节奏)
    if len(bc_pos) >= 2:
        gaps = np.diff(bc_pos)
        f.extend([float(gaps.mean()), float(gaps.std()), float(gaps.min()), float(gaps.max())])
        f.append(float(L - 1 - bc_pos[-1]))                                # 距最近 BC
        f.append(float(bc_pos[-1] - bc_pos[-2]))                           # 最近两次 BC 间隔
    else:
        f.extend([L, 0.0, L, L, float(L - 1 - bc_pos[-1]) if len(bc_pos) else L, L])

    # 3. BC 周期性 (自相关 — 上次 BC 后多久, 是否到了"该再 BC"的节律点)
    last_bc = (L - 1 - bc_pos[-1]) if len(bc_pos) else L
    mean_gap = float(np.diff(bc_pos).mean()) if len(bc_pos) >= 2 else L
    f.append(last_bc / max(1.0, mean_gap))                                 # 距上次/平均间隔 (>1=超期该来了)

    # 4. 说话人切换与 BC 的耦合 (BC 常发生在对方持续说话时 = C/NA 主导段)
    #    用 C(0)/NA(4) 在近窗的占比 + 最近是否 C→ 看是否"对方在说"
    for w in (12, 25, 50):
        seg = ctx[-w:] if L >= w else ctx
        f.append(float((seg == 0).mean()))                                 # C (本方继续) 占比
        f.append(float((seg == 4).mean()))                                 # NA (对方说?) 占比
    # 最近 5 标签是否含 BC 的前兆模式 (C/NA 连续段)
    f.append(float(len(set(ctx[-10:])) == 1))                              # 近10是否单一类(稳定段)

    # 5. BC 总体节律
    f.append(float(len(bc_pos)) / L)                                       # 全局 BC 密度
    f.append(float((ctx[-50:] == BC).sum()) / max(1.0, (ctx == BC).sum())) # 近期/总体 BC 比 (是否进入活跃期)

    return np.array(f, dtype=np.float32)


def build(conv_ids, augment: bool):
    X, Y, G = [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        # cap1 风格但训练用多切片 (stride), 评估时取每通首切片
        order = 0
        for e in range(CTX, len(a) - TGT + 1, args.stride):
            ctx = a[e - CTX:e]
            base = featurize(ctx)
            feat = np.concatenate([base, bc_temporal_feats(ctx)]) if augment else base
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(feat)
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
            order += 1
    return np.array(X), np.array(Y), np.array(G)


def oof_bc(X, Y, G, conv_ids, folds, stage2=False):
    """返回 BC 的 OOF 概率 (会话级 folds-fold)."""
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(conv_ids))
    oof = np.zeros(len(X))
    for fi in range(folds):
        val_g = {perm[i] for i in range(len(conv_ids)) if i % folds == fi}
        tr = [i for i in range(len(X)) if G[i] not in val_g]
        va = [i for i in range(len(X)) if G[i] in val_g]
        ytr = Y[tr, BC]
        spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
        if not stage2:
            clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
            clf.fit(X[tr], ytr)
            oof[va] = clf.predict_proba(X[va])[:, 1]
        else:
            # 二阶: stage1 高recall召回 (低阈值), stage2 在候选上判真假
            s1 = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
            s1.fit(X[tr], ytr)
            p1_tr = s1.predict_proba(X[tr])[:, 1]
            # stage1 召回候选 (recall高的低阈值, 取 train 上 R~0.9 的阈值)
            cand_thr = np.quantile(p1_tr[ytr == 1], 0.1)  # 让 90% 真BC进候选
            cand_tr = [tr[i] for i in range(len(tr)) if p1_tr[i] >= cand_thr]
            ycand = Y[cand_tr, BC]
            spw2 = (len(ycand) - ycand.sum()) / max(1, ycand.sum())
            s2 = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=63,
                                scale_pos_weight=spw2, n_jobs=4, verbose=-1, random_state=SEED)
            s2.fit(X[cand_tr], ycand)
            # 推理: stage1 召回 → stage2 概率 (非候选=0)
            p1_va = s1.predict_proba(X[va])[:, 1]
            p2_va = s2.predict_proba(X[va])[:, 1]
            oof[va] = np.where(p1_va >= cand_thr, p2_va, 0.0)
    return oof


def bc_f1_full(oof, Y):
    """全切片 OOF 评估 (高分辨率, 6000+ BC正例). 用于相对比较不同特征.
    注: 这是滑窗分布, 调出的绝对阈值不可搬 test (阈值铁律), 但 A/B/C 相对比较安全."""
    yt = Y[:, BC]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(yt, (oof >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    p, r, _, _ = precision_recall_fscore_support(yt, (oof >= best_t).astype(int),
                                                  average='binary', zero_division=0)
    return best_f, best_t, p, r, int(yt.sum()), len(yt)


def cap1_bc_f1(oof, Y, G, conv_ids):
    """cap1 评估: 每通取首切片. 9 BC 正例太稀疏=离散跳变, 仅作参考不作决策."""
    seen, cap1 = set(), []
    for i in range(len(G)):
        if G[i] not in seen:
            cap1.append(i); seen.add(G[i])
    cap1 = np.array(cap1)
    yt = Y[cap1, BC]
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(yt, (oof[cap1] >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_f, best_t, int(yt.sum()), len(cap1)


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=40)
    args = ap.parse_args()

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    print(f"[bc-temporal] {len(conv_ids)} convs, stride={args.stride}, folds={args.folds}", file=sys.stderr)

    print("[bc-temporal] building baseline features...", file=sys.stderr)
    Xb, Y, G = build(conv_ids, augment=False)
    print(f"[bc-temporal] building augmented features...", file=sys.stderr)
    Xa, _, _ = build(conv_ids, augment=True)
    print(f"[bc-temporal] samples={len(Xb)} base_dim={Xb.shape[1]} aug_dim={Xa.shape[1]} "
          f"BC_rate={Y[:,BC].mean():.3f}", file=sys.stderr)

    results = {}
    for name, X, s2 in [("A_baseline", Xb, False), ("B_aug_1stage", Xa, False), ("C_aug_2stage", Xa, True)]:
        oof = oof_bc(X, Y, G, conv_ids, args.folds, stage2=s2)
        f, t, p, r, npos, ntot = bc_f1_full(oof, Y)            # 高分辨率全切片 (相对比较)
        cf, ct, cpos, ncap = cap1_bc_f1(oof, Y, G, conv_ids)   # cap1 参考
        results[name] = {"bc_f1_full": round(f, 4), "thr": round(t, 2), "P": round(p, 3),
                         "R": round(r, 3), "bc_f1_cap1": round(cf, 4)}
        print(f"[{name:<14}] full BC F1={f:.4f} @thr{t:.2f} (P={p:.3f} R={r:.3f}) "
              f"[{npos}/{ntot}] | cap1 ref={cf:.3f}", file=sys.stderr)

    print("\n=== BC F1 相对比较 (全切片高分辨率; LGBM/VAP 锚点 0.222) ===")
    base = results["A_baseline"]["bc_f1_full"]
    for name, rr in results.items():
        delta = rr["bc_f1_full"] - base
        print(f"  {name:<14} BC_full={rr['bc_f1_full']:.4f} ({delta:+.4f} vs A) "
              f"P={rr['P']} R={rr['R']} | cap1={rr['bc_f1_cap1']}")
    print(json.dumps({"cycle": "H-V7", "bc_results": results}))


if __name__ == "__main__":
    main()
