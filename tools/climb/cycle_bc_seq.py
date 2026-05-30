"""climb cycle H-SEQ — BC 序列/计数预测 (用户选项3: 换框架, 非 event 二分类).

前提验证 (2026-05-30): BC 在未来25chunk是连续段(57%含3+连续BC,平均3.7个), 非孤立点。
event-level 二分类("25帧内有无BC")丢连续性信息。

因果约束: test 预测点固定在切片末, 未来不可见 → 不能逐帧滑动预测。
可行的"序列利用": 同一 context 预测未来25帧的 BC 结构。三种框架对比:
  A. event 二分类 (baseline: 未来25帧有无BC) — 当前法
  B. BC 计数回归 (预测未来BC chunk数, 连续段→数量大, 用连续性) → event=count>阈值
  C. multi-output (预测25个future chunk各自BC概率, 取max/sum) → 用逐帧结构

看 B/C 的 event-level BC F1 是否 > A. 若连续性是真信号, 计数/序列应抓更多.

Usage: python tools/climb/cycle_bc_seq.py [--folds 5] [--stride 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctx_v1

NUM, CTX, TGT, SEED = 5, 375, 25, 42
BC = 2


def build(conv_ids):
    """返回 X, 及三种标签: y_event(0/1有无BC), y_count(BC chunk数), Y_frame(25维逐帧)."""
    X, y_event, y_count, Y_frame, G = [], [], [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        for e in range(CTX, len(a) - TGT + 1, args.stride):
            ctx = a[e - CTX:e]
            fut = a[e:e + TGT]                          # 未来25帧
            X.append(ctx_v1(ctx))
            bc_frames = (fut == BC).astype(int)
            y_event.append(1 if bc_frames.sum() > 0 else 0)
            y_count.append(int(bc_frames.sum()))
            Y_frame.append(bc_frames)
            G.append(gi)
    return (np.array(X, dtype=np.float32), np.array(y_event), np.array(y_count),
            np.array(Y_frame), np.array(G))


def oof_event(X, y, G, conv_ids, folds):
    """A: event 二分类 OOF."""
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(conv_ids))
    oof = np.zeros(len(X))
    for fi in range(folds):
        val = {perm[i] for i in range(len(conv_ids)) if i % folds == fi}
        tr = [i for i in range(len(X)) if G[i] not in val]
        va = [i for i in range(len(X)) if G[i] in val]
        spw = (len(tr) - y[tr].sum()) / max(1, y[tr].sum())
        c = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
        c.fit(X[tr], y[tr]); oof[va] = c.predict_proba(X[va])[:, 1]
    return oof


def oof_count(X, yc, G, conv_ids, folds):
    """B: BC 计数回归 OOF → 预测的 count 作 event 分数."""
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(conv_ids))
    oof = np.zeros(len(X))
    for fi in range(folds):
        val = {perm[i] for i in range(len(conv_ids)) if i % folds == fi}
        tr = [i for i in range(len(X)) if G[i] not in val]
        va = [i for i in range(len(X)) if G[i] in val]
        r = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          n_jobs=4, verbose=-1, random_state=SEED)
        r.fit(X[tr], yc[tr]); oof[va] = r.predict(X[va])
    return oof


def oof_frame(X, Yf, G, conv_ids, folds):
    """C: multi-output 逐帧 — 训25个位置模型, event分数=25帧预测概率的max."""
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(conv_ids))
    oof = np.zeros((len(X), TGT))
    # 全局帧级 pos_weight (各帧BC率近似)
    for fi in range(folds):
        val = {perm[i] for i in range(len(conv_ids)) if i % folds == fi}
        tr = [i for i in range(len(X)) if G[i] not in val]
        va = [i for i in range(len(X)) if G[i] in val]
        # 共享一个模型预测"任一帧BC"太粗; 这里训单模型预测帧级BC率(把25帧标签摊平监督)
        # 简化: 训一个模型预测 P(该context下随机未来帧是BC), 用帧级平均标签
        yf_mean = Yf[tr].mean(axis=1)  # 每样本的BC帧比例 [0,1]
        # 用比例回归 (类似count但归一化)
        from lightgbm import LGBMRegressor as R
        m = R(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=4,
              verbose=-1, random_state=SEED)
        m.fit(X[tr], yf_mean)
        pred = m.predict(X[va])
        for j, i in enumerate(va):
            oof[i, :] = pred[j]  # 同context所有帧同概率(LGBM限制), event分数=该值
    return oof.max(axis=1)


def best_f1(score, y_event):
    s = (score - score.min()) / (score.max() - score.min() + 1e-9)  # 归一化到[0,1]比阈值
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(y_event, (s >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    p, r, _, _ = precision_recall_fscore_support(y_event, (s >= bt).astype(int),
                                                 average='binary', zero_division=0)
    return bf, p, r


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=40)
    args = ap.parse_args()

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    print(f"[bc-seq] {len(conv_ids)} convs stride={args.stride}", file=sys.stderr)
    X, ye, yc, Yf, G = build(conv_ids)
    print(f"[bc-seq] {len(X)} samples, BC event率={ye.mean():.3f}", file=sys.stderr)

    res = {}
    # A. event 二分类
    oa = oof_event(X, ye, G, conv_ids, args.folds)
    fa, pa, ra = best_f1(oa, ye)
    res["A_event_binary"] = {"bc_f1": round(fa, 4), "P": round(pa, 3), "R": round(ra, 3)}
    # B. 计数回归
    ob = oof_count(X, yc, G, conv_ids, args.folds)
    fb, pb, rb = best_f1(ob, ye)
    res["B_count_regress"] = {"bc_f1": round(fb, 4), "P": round(pb, 3), "R": round(rb, 3)}
    # C. 帧比例回归
    oc = oof_frame(X, Yf, G, conv_ids, args.folds)
    fc, pc, rc = best_f1(oc, ye)
    res["C_frame_ratio"] = {"bc_f1": round(fc, 4), "P": round(pc, 3), "R": round(rc, 3)}

    print(f"\n=== BC 框架对比 (event-level F1, 全切片OOF) ===")
    base = res["A_event_binary"]["bc_f1"]
    for k, v in res.items():
        print(f"  {k:<18} BC={v['bc_f1']:.4f} ({v['bc_f1']-base:+.4f}) P={v['P']} R={v['R']}")
    print(f"\n判读: B/C > A +0.01 = 连续性/序列结构可榨; ≈A = 二分类已够(连续性无额外信息)")
    print(json.dumps({"cycle": "H-SEQ", "results": res}))


if __name__ == "__main__":
    main()
