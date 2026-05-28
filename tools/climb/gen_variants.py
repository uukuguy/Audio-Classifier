"""本机零算力变体生成 — 用 cap1 切片 CV 调阈值 + 多 seed 集成，产出多个提交 CSV。

目的:在不上云的前提下，用便宜变体探明"切片阈值 / 多 seed 集成 / context-v2 特征"
线上是否真有用。每个变体打印 cap1 切片 CV 预估 + 与 cycle1 已提交 CSV 的正例数 diff
(diff 越大风险越高)，供按提交配额挑选。

变体(均 cycle1 纯上下文 LGBM 范式，改阈值策略/集成):
  B  切片阈值     : cycle1 模型 + cap1 切片 CV 调出的阈值
  C  多seed集成   : 5 seed LGBM rank 平均 + 切片阈值
  D  +v2特征      : C 基础上并入 context-v2 特征

实验旋钮走 CLI/env，不改 cycle_context.py baseline 默认值(HARD RULE)。

Usage: python tools/climb/gen_variants.py --variants B,C,D --seeds 5
输出: tools/runs/climb/variant-<X>-<ts>/pred_test1.csv + cv_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, "tools/climb")
from cycle_context import CTX, LABELS, NUM, STRIDE, TGT  # noqa: E402
from cycle_context import featurize as ctxfeat  # noqa: E402
from cycle_context_v2 import featurize as v2feat  # noqa: E402

SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
ANCHOR_CSV = "tools/runs/climb/20260527-1636-h001-context-only/pred_test1.csv"
SLICE_STRIDE = 400


def build_windows(arr, feat_fn):
    X, Y = [], []
    for e in range(CTX, arr.shape[0] - TGT + 1, STRIDE):
        fut = set(int(x) for x in arr[e:e + TGT])
        X.append(feat_fn(arr[e - CTX:e].astype(int)))
        Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(X), np.array(Y, dtype=int)


def build_cap1_slices(arr, feat_fn):
    """每通取序号 0 的不重叠片段(模拟 test 独立片段, cap1)。"""
    starts = list(range(CTX, arr.shape[0] - TGT + 1, SLICE_STRIDE))
    if not starts:
        return np.empty((0, 0)), np.empty((0, NUM), dtype=int)
    e = starts[0]
    fut = set(int(x) for x in arr[e:e + TGT])
    X = [feat_fn(arr[e - CTX:e].astype(int))]
    Y = [[1 if k in fut else 0 for k in range(NUM)]]
    return np.array(X), np.array(Y, dtype=int)


def fit(X, y, seed):
    spw = (len(y) - y.sum()) / max(1, y.sum())
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=seed)
    clf.fit(X, y)
    return clf


def rank_avg(prob_list):
    """多 seed 概率 → rank 平均。⚠对稀有类有害(C 实测 BC 崩):
    海量负例稀释 rank，假正例被推到高分段。仅 C/NA 这类高频类可用。"""
    ranks = [np.argsort(np.argsort(p)) / (len(p) - 1) for p in prob_list]
    return np.mean(ranks, axis=0)


def prob_avg(prob_list):
    """多 seed 概率 → 概率平均。保留稀有类绝对置信度，不被负例稀释。"""
    return np.mean(prob_list, axis=0)


def aggregate(prob_list, agg):
    if len(prob_list) == 1:
        return prob_list[0]
    return rank_avg(prob_list) if agg == "rank" else prob_avg(prob_list)


def tune_thr(y, p):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, float(t)
    return bt, bf


def load_anchor_counts():
    import csv
    import os
    if not os.path.exists(ANCHOR_CSV):
        return None
    rows = list(csv.reader(open(ANCHOR_CSV)))[1:]
    return {c: sum(int(r[i + 1]) for r in rows) for i, c in enumerate(SUBMIT)}


def gen_variant(variant, conv_ids, label_files, test_files, seeds, folds=5):
    """阈值用 cap1 切片 OOF 调(无泄漏)，test 预测用全量重训。

    返回 (per_sub_cv, thresholds, preds_by_col, macro)。
    """
    # 变体配置: B=单模型切片阈值 / C=5seed rank平均 / E=5seed 概率平均(修C的rank缺陷)
    #          F=5seed 概率平均 + cycle1原阈值(只吃集成降方差,不让切片CV砸NA近全正)
    #          D=5seed rank + v2特征(慢,已弃)
    feat_fn = (lambda c: np.concatenate([ctxfeat(c), v2feat(c)])) if variant == "D" else ctxfeat
    n_seed = seeds if variant in ("C", "D", "E", "F") else 1
    agg = "rank" if variant in ("C", "D") else "prob"  # E/F 用概率平均
    # F 用 cycle1 已验证原阈值(线上0.7108),不被切片CV调动(阈值铁律: 近全正类别偏离低阈值)
    fixed_thr = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25} if variant == "F" else None
    arrs = {c: np.load(label_files[c]).astype(int) for c in conv_ids}

    # 预构造每通的滑窗(train) + cap1 切片(valid)
    win = {c: build_windows(arrs[c], feat_fn) for c in conv_ids}
    sli = {c: build_cap1_slices(arrs[c], feat_fn) for c in conv_ids}

    # 会话级 fold
    rng = np.random.default_rng(42)
    order = list(conv_ids)
    rng.shuffle(order)
    fold_of = {c: i % folds for i, c in enumerate(order)}

    # cap1 OOF: 每折训非本折，预测本折 cap1 切片(无泄漏)
    oof_p = {k: [] for k in range(NUM)}
    oof_y = {k: [] for k in range(NUM)}
    for fi in range(folds):
        tr = [c for c in conv_ids if fold_of[c] != fi]
        va = [c for c in conv_ids if fold_of[c] == fi and len(sli[c][0])]
        if not va:
            continue
        Xtr = np.vstack([win[c][0] for c in tr])
        Ytr = np.vstack([win[c][1] for c in tr])
        Xva = np.vstack([sli[c][0] for c in va])
        Yva = np.vstack([sli[c][1] for c in va])
        for k in range(NUM):
            seed_probs = [fit(Xtr, Ytr[:, k], 42 + s).predict_proba(Xva)[:, 1]
                          for s in range(n_seed)]
            oof_p[k].append(aggregate(seed_probs, agg))
            oof_y[k].append(Yva[:, k])

    thr, cv_f1 = {}, {}
    for k in range(NUM):
        yk = np.concatenate(oof_y[k])
        pk = np.concatenate(oof_p[k])
        t, f = tune_thr(yk, pk)
        # F: 用 cycle1 固定阈值出预测,但 cv_f1 仍按切片 CV 该阈值下的 F1 报(诚实估计)
        if fixed_thr is not None:
            t = fixed_thr[k]
            f = f1_score(yk, (pk >= t).astype(int), zero_division=0)
        thr[k], cv_f1[k] = t, f
    macro = float(np.mean(list(cv_f1.values())))

    # test: 全量重训(多 seed rank 平均) + OOF 阈值
    Xall = np.vstack([win[c][0] for c in conv_ids])
    Yall = np.vstack([win[c][1] for c in conv_ids])
    Xte = np.array([feat_fn(np.load(p).astype(int)) for p in test_files])
    preds = {}
    for k in range(NUM):
        seed_te = [fit(Xall, Yall[:, k], 42 + s).predict_proba(Xte)[:, 1]
                   for s in range(n_seed)]
        pte = aggregate(seed_te, agg)
        preds[k] = (pte >= thr[k]).astype(int)
    return cv_f1, thr, preds, macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="B,C,D")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    test_files = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_files]
    anchor = load_anchor_counts()
    ts = datetime.now().strftime("%Y%m%d-%H%M")

    summary = []
    for v in args.variants.split(","):
        v = v.strip()
        print(f"\n[gen] ===== 变体 {v} =====", file=sys.stderr)
        cv_f1, thr, preds, macro = gen_variant(v, conv_ids, label_files, test_files, args.seeds)
        cnts = {c: int(preds[COL2K[c]].sum()) for c in SUBMIT}
        diff = {c: cnts[c] - anchor[c] for c in SUBMIT} if anchor else {}

        print(f"[gen] {v} cap1切片CV macro={macro:.4f} | " +
              " ".join(f"{LABELS[k]}={cv_f1[k]:.3f}@{thr[k]:.2f}" for k in range(NUM)), file=sys.stderr)
        print(f"[gen] {v} 正例数: " + " ".join(f"{c}={cnts[c]}({diff.get(c, 0):+d})" for c in SUBMIT),
              file=sys.stderr)

        run = Path(f"tools/runs/climb/variant-{v}-{ts}")
        run.mkdir(parents=True, exist_ok=True)
        with open(run / "pred_test1.csv", "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(seg_ids):
                f.write(",".join([sid] + [str(int(preds[COL2K[c]][i])) for c in SUBMIT]) + "\n")
        (run / "cv_metrics.json").write_text(json.dumps({
            "variant": v, "cap1_macro_f1": round(macro, 4),
            "per_sub": {LABELS[k]: round(cv_f1[k], 4) for k in range(NUM)},
            "thresholds": {LABELS[k]: round(thr[k], 2) for k in range(NUM)},
            "pos_counts": cnts, "diff_vs_cycle1": diff,
        }, ensure_ascii=False, indent=2))
        summary.append((v, macro, str(run / "pred_test1.csv"), cnts, diff))

    print("\n[gen] ===== 汇总(按 cap1 切片 CV 排序) =====", file=sys.stderr)
    for v, m, path, cnts, diff in sorted(summary, key=lambda x: -x[1]):
        d = " ".join(f"{c}{diff.get(c, 0):+d}" for c in SUBMIT) if diff else ""
        print(f"[gen] {v}: CV={m:.4f}  diff[{d}]  {path}", file=sys.stderr)
    print(json.dumps({"variants": [{"v": v, "cap1_cv": round(m, 4), "csv": p}
                                    for v, m, p, _, _ in summary]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
