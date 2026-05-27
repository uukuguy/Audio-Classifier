"""30s 切片化验证集 — 修方法论漏洞（CONTEXT Decision 4）。

问题: cycle_context.py 的 valid 用 stride=5 密集滑窗(同一长对话被切成几千个
高度重叠窗)，但 test 是 1000 个独立 30s 切片(context 恒 375 chunk)。分布错配 →
滑窗 CV 不可信(实测 gap 线上 0.7108 − 滑窗 CV 0.5908 = +0.12)。

本脚本构造**切片化验证集**: valid 对话里切**不重叠** 375+25 片段(模拟 test 独立
片段分布)，让 OOF CV 逼近线上真分。同时对照打印滑窗 CV，量化两种验证协议的差异。

会话级 K-fold OOF:
  - 每折: train 对话密集滑窗训 LGBM，held-out 对话切独立片段做 valid
  - 阈值在**切片 valid 分布**上调(而非滑窗) → 鲁棒
  - OOF 拼接全部 held-out 切片 → 最终 Macro-F1 + per-class

Usage: python tools/climb/sliced_cv.py [--folds 5] [--slice-stride 400] [--per-conv-cap 0]
输出: 滑窗 CV vs 切片 CV 对照 + per-class + 推荐阈值(stderr)
      最后一行 stdout = JSON {"sliced_cv":.., "window_cv":.., "per_sub":{..}, "thresholds":{..}}
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

# 复用 cycle_context 的 featurize / 常量，保证 train/valid/test 特征一致
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycle_context import (  # noqa: E402
    CTX,
    LABELS,
    NUM,
    STRIDE,
    TGT,
    featurize,
)

SEED = 42


def build_windows(label_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """密集滑窗(stride=5)，用于训练 — 数据量大无妨。"""
    X, Y = [], []
    for e in range(CTX, label_arr.shape[0] - TGT + 1, STRIDE):
        fut = set(int(x) for x in label_arr[e:e + TGT])
        X.append(featurize(label_arr[e - CTX:e].astype(int)))
        Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(X), np.array(Y, dtype=int)


def build_slices_all(
    label_arr: np.ndarray, slice_stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """切全部不重叠 375+25 片段(模拟 test 独立 30s 切片分布)，用于 valid。

    test 是 1000 个独立片段，每个 context 恰好 375 chunk。验证集应同分布:
    从长对话切互不重叠的 (375 context + 25 target) 片段。返回的 slice_order 是
    每个片段在本通对话内的序号(0-based)，供后期按 per-conv-cap 子采样(cap=1
    取序号<1，cap=5 取序号<5，all 取全部)。
    """
    starts = list(range(CTX, label_arr.shape[0] - TGT + 1, slice_stride))
    X, Y, order = [], [], []
    for j, e in enumerate(starts):
        fut = set(int(x) for x in label_arr[e:e + TGT])
        X.append(featurize(label_arr[e - CTX:e].astype(int)))
        Y.append([1 if k in fut else 0 for k in range(NUM)])
        order.append(j)
    if not X:
        return np.empty((0, 0)), np.empty((0, NUM), dtype=int), np.empty((0,), dtype=int)
    return np.array(X), np.array(Y, dtype=int), np.array(order, dtype=int)


def fit_lgbm(X: np.ndarray, y: np.ndarray):
    spw = (len(y) - y.sum()) / max(1, y.sum())
    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED,
    )
    clf.fit(X, y)
    return clf


def tune_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def macro_from_oof(
    oof_prob: dict[int, np.ndarray],
    oof_true: dict[int, np.ndarray],
    mask: np.ndarray | None = None,
) -> tuple[float, dict, dict, int]:
    """OOF 概率 → 逐类调阈值 → Macro-F1。mask 选子集(按 per-conv-cap 采样)。"""
    thr, per_f1 = {}, {}
    n = 0
    for k in range(NUM):
        p = oof_prob[k] if mask is None else oof_prob[k][mask]
        y = oof_true[k] if mask is None else oof_true[k][mask]
        n = len(y)
        t, _ = tune_threshold(y, p)
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        thr[k], per_f1[k] = t, f
    macro = float(np.mean(list(per_f1.values())))
    return macro, thr, per_f1, n


CAPS = [1, 5, 0]  # 0 = all；一次跑出三档采样密度对照
ANCHOR = 0.7108


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-stride", type=int, default=400,
                    help="valid 切片步长(>=400 保证不重叠，越大越接近独立抽样)")
    args = ap.parse_args()

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    perm = np.random.default_rng(SEED).permutation(len(conv_ids))
    conv_ids = [conv_ids[i] for i in perm]
    folds = np.array_split(np.arange(len(conv_ids)), args.folds)
    print(f"[sliced-cv] {len(conv_ids)} convs, {args.folds}-fold conv-level OOF", file=sys.stderr)
    print(f"[sliced-cv] valid slice-stride={args.slice_stride}, caps={CAPS} (0=all)", file=sys.stderr)

    arrs = {cid: np.load(label_files[cid]).astype(int) for cid in conv_ids}

    # OOF: 切片(带 per-conv 序号) + 滑窗 两套
    oof_slice_p = {k: [] for k in range(NUM)}
    oof_slice_y = {k: [] for k in range(NUM)}
    oof_slice_order: list[np.ndarray] = []
    oof_win_p = {k: [] for k in range(NUM)}
    oof_win_y = {k: [] for k in range(NUM)}

    for fi, val_idx in enumerate(folds):
        val_ids = [conv_ids[i] for i in val_idx]
        tr_set = set(val_ids)
        tr_ids = [c for c in conv_ids if c not in tr_set]

        tr_win = [build_windows(arrs[c]) for c in tr_ids]
        Xtr = np.vstack([x for x, _ in tr_win])
        Ytr = np.vstack([y for _, y in tr_win])
        del tr_win

        Xvs_parts, Yvs_parts, order_parts, Xvw_parts, Yvw_parts = [], [], [], [], []
        for cid in val_ids:
            Xs, Ys, order = build_slices_all(arrs[cid], args.slice_stride)
            if len(Xs):
                Xvs_parts.append(Xs)
                Yvs_parts.append(Ys)
                order_parts.append(order)
            Xw, Yw = build_windows(arrs[cid])
            Xvw_parts.append(Xw)
            Yvw_parts.append(Yw)
        Xvs, Yvs = np.vstack(Xvs_parts), np.vstack(Yvs_parts)
        order_fold = np.concatenate(order_parts)
        Xvw, Yvw = np.vstack(Xvw_parts), np.vstack(Yvw_parts)
        oof_slice_order.append(order_fold)

        for k in range(NUM):
            clf = fit_lgbm(Xtr, Ytr[:, k])
            oof_slice_p[k].append(clf.predict_proba(Xvs)[:, 1])
            oof_slice_y[k].append(Yvs[:, k])
            oof_win_p[k].append(clf.predict_proba(Xvw)[:, 1])
            oof_win_y[k].append(Yvw[:, k])
        print(f"[sliced-cv] fold {fi + 1}/{args.folds}: train_win={len(Xtr)} "
              f"valid_slice={len(Xvs)} valid_win={len(Xvw)}", file=sys.stderr)

    oof_slice_p = {k: np.concatenate(v) for k, v in oof_slice_p.items()}
    oof_slice_y = {k: np.concatenate(v) for k, v in oof_slice_y.items()}
    oof_win_p = {k: np.concatenate(v) for k, v in oof_win_p.items()}
    oof_win_y = {k: np.concatenate(v) for k, v in oof_win_y.items()}
    slice_order = np.concatenate(oof_slice_order)

    win_macro, win_thr, win_f1, _ = macro_from_oof(oof_win_p, oof_win_y)

    # 三档采样密度对照: cap=1 取每通序号<1，cap=5 取序号<5，all 取全部
    results = {}
    for cap in CAPS:
        mask = slice_order < cap if cap else None
        m, thr, f1, n = macro_from_oof(oof_slice_p, oof_slice_y, mask)
        results[cap] = {"macro": m, "thr": thr, "f1": f1, "n": n}

    # ===== 对照打印 =====
    print("\n[sliced-cv] ===== 验证协议对照 (per-class F1) =====", file=sys.stderr)
    cap_names = {1: "cap1", 5: "cap5", 0: "all"}
    hdr = f"{'class':6s} {'win':>7s}" + "".join(f"{cap_names[c]:>8s}" for c in CAPS)
    print(hdr, file=sys.stderr)
    for k in range(NUM):
        row = f"{LABELS[k]:6s} {win_f1[k]:7.3f}" + "".join(
            f"{results[c]['f1'][k]:8.3f}" for c in CAPS)
        print(row, file=sys.stderr)
    macro_row = f"{'MACRO':6s} {win_macro:7.4f}" + "".join(
        f"{results[c]['macro']:8.4f}" for c in CAPS)
    print(macro_row, file=sys.stderr)
    nrow = f"{'n_val':6s} {'-':>7s}" + "".join(f"{results[c]['n']:8d}" for c in CAPS)
    print(nrow, file=sys.stderr)

    print(f"\n[sliced-cv] 线上锚 {ANCHOR}。各协议 gap (线上−CV):", file=sys.stderr)
    print(f"  滑窗(旧)      CV={win_macro:.4f}  gap={ANCHOR - win_macro:+.4f}", file=sys.stderr)
    for c in CAPS:
        print(f"  切片 {cap_names[c]:5s}    CV={results[c]['macro']:.4f}  "
              f"gap={ANCHOR - results[c]['macro']:+.4f}", file=sys.stderr)
    best_cap = min(CAPS, key=lambda c: abs(ANCHOR - results[c]["macro"]))
    print(f"\n[sliced-cv] 最逼近线上的协议: 切片 {cap_names[best_cap]} "
          f"(|gap|={abs(ANCHOR - results[best_cap]['macro']):.4f})", file=sys.stderr)

    out = {
        "window_cv": round(win_macro, 4),
        "online_anchor": ANCHOR,
        "window_gap": round(ANCHOR - win_macro, 4),
        "sliced": {cap_names[c]: {
            "cv": round(results[c]["macro"], 4),
            "gap": round(ANCHOR - results[c]["macro"], 4),
            "n_val": results[c]["n"],
            "per_sub": {sub: round(results[c]["f1"][i], 4)
                        for sub, i in {"c": 0, "na": 4, "t": 1, "i": 3, "bc": 2}.items()},
            "thresholds": {LABELS[k]: round(results[c]["thr"][k], 2) for k in range(NUM)},
        } for c in CAPS},
        "best_cap": cap_names[best_cap],
        "n_folds": args.folds,
        "slice_stride": args.slice_stride,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
