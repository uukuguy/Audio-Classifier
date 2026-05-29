"""变体 G = 变体 F (SOTA 0.7124) + 只给 T/I 叠 ASR 文本特征。最低风险纯增量。

诊断 workflow 定论 (2026-05-29): 文本特征帮 T(0.54→0.58)/I(0.44→0.49) 但污染 BC。
SOTA 变体 F = 46维 ctxfeat + 5seed 概率平均 + cycle1 固定阈值。

变体 G 设计 (C/NA/BC 逐位等于 SOTA, 只 T/I 增益):
  - C(0)/NA(4)/BC(2): 46维 ctxfeat (完全等于变体 F, BC 回 0.217)
  - T(1)/I(3):        46维 ctxfeat + 21维 text_feats
  - 所有类: 5seed 概率平均 (变体 F 一致), cycle1 固定阈值 (阈值铁律)
  - cap1 切片 CV 报告 (可信协议, 非滑窗)

阈值: cycle1 固定 {C:0.05, T:0.50, BC:0.75, I:0.65, NA:0.25} — 与变体 F 完全一致。

Usage: python tools/climb/gen_variant_g.py [--seeds 5]
输出: tools/runs/climb/variant-G-<ts>/pred_test1.csv + cv_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools/climb")
from cycle_context import CTX, LABELS, NUM, STRIDE, TGT  # noqa: E402
from cycle_context import featurize as ctxfeat  # noqa: E402
from cycle_text_fusion import CHUNK_MS, text_feats  # noqa: E402
from gen_variants import COL2K, SUBMIT, fit, prob_avg, load_anchor_counts  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

# default = T,I get text. Overridable via --text-classes (experiment value, never baked).
DEFAULT_TEXT_CLASSES = {1, 3}
FIXED_THR = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}  # cycle1 = variant F
SLICE_STRIDE = 400
CTX_DIM = 46


def build_windows_with_text(arr, utts):
    """滑窗: 返回 (X_ctx[46], X_full[67=46+21], Y). X_full 给 T/I, X_ctx 给其余。"""
    Xc, Xf, Y = [], [], []
    for e in range(CTX, arr.shape[0] - TGT + 1, STRIDE):
        ctx = arr[e - CTX:e].astype(int)
        cf = ctxfeat(ctx)
        tf = text_feats(utts, e * CHUNK_MS)
        Xc.append(cf)
        Xf.append(np.concatenate([cf, tf]))
        fut = set(int(x) for x in arr[e:e + TGT])
        Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(Xc, dtype=np.float32), np.array(Xf, dtype=np.float32), np.array(Y, dtype=int)


def build_cap1_with_text(arr, utts):
    """cap1 切片(每通序号0): 返回 (X_ctx, X_full, Y)。"""
    starts = list(range(CTX, arr.shape[0] - TGT + 1, SLICE_STRIDE))
    if not starts:
        return (np.empty((0, CTX_DIM), dtype=np.float32),
                np.empty((0, CTX_DIM + 21), dtype=np.float32),
                np.empty((0, NUM), dtype=int))
    e = starts[0]
    cf = ctxfeat(arr[e - CTX:e].astype(int))
    tf = text_feats(utts, e * CHUNK_MS)
    fut = set(int(x) for x in arr[e:e + TGT])
    Y = [[1 if k in fut else 0 for k in range(NUM)]]
    return (np.array([cf], dtype=np.float32),
            np.array([np.concatenate([cf, tf])], dtype=np.float32),
            np.array(Y, dtype=int))


def feat_for_class(k, Xc, Xf, text_classes):
    return Xf if k in text_classes else Xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--text-classes", default="1,3",
                    help="class ids that get text feats (default 1,3 = T,I)")
    ap.add_argument("--tag", default="G", help="variant tag for run dir")
    args = ap.parse_args()
    n_seed = args.seeds
    text_classes = {int(x) for x in args.text_classes.split(",") if x.strip()}
    print(f"[G] TEXT_CLASSES = {sorted(text_classes)} ({[LABELS[k] for k in sorted(text_classes)]})",
          file=sys.stderr)

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    test_files = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_files]
    anchor = load_anchor_counts()
    ts = datetime.now().strftime("%Y%m%d-%H%M")

    print(f"[G] building windows + cap1 slices ({len(conv_ids)} convs)...", file=sys.stderr)
    arrs = {c: np.load(label_files[c]).astype(int) for c in conv_ids}
    utts = {c: json.load(open(f"data/train/text/{c}.json")).get("utterances", []) for c in conv_ids}
    win = {c: build_windows_with_text(arrs[c], utts[c]) for c in conv_ids}
    sli = {c: build_cap1_with_text(arrs[c], utts[c]) for c in conv_ids}
    print("[G] features built", file=sys.stderr)

    # session-level fold (same seed/shuffle as variant F)
    rng = np.random.default_rng(42)
    order = list(conv_ids)
    rng.shuffle(order)
    folds = 5
    fold_of = {c: i % folds for i, c in enumerate(order)}

    # cap1 OOF (no leak): train non-fold windows, predict fold cap1 slice
    oof_p = {k: [] for k in range(NUM)}
    oof_y = {k: [] for k in range(NUM)}
    for fi in range(folds):
        tr = [c for c in conv_ids if fold_of[c] != fi]
        va = [c for c in conv_ids if fold_of[c] == fi and len(sli[c][2])]
        if not va:
            continue
        Xc_tr = np.vstack([win[c][0] for c in tr])
        Xf_tr = np.vstack([win[c][1] for c in tr])
        Ytr = np.vstack([win[c][2] for c in tr])
        Xc_va = np.vstack([sli[c][0] for c in va])
        Xf_va = np.vstack([sli[c][1] for c in va])
        Yva = np.vstack([sli[c][2] for c in va])
        for k in range(NUM):
            Xtr_k = feat_for_class(k, Xc_tr, Xf_tr, text_classes)
            Xva_k = feat_for_class(k, Xc_va, Xf_va, text_classes)
            seed_probs = [fit(Xtr_k, Ytr[:, k], 42 + s).predict_proba(Xva_k)[:, 1]
                          for s in range(n_seed)]
            oof_p[k].append(prob_avg(seed_probs))
            oof_y[k].append(Yva[:, k])
        print(f"[G] fold {fi+1}/{folds}", file=sys.stderr)

    cv_f1 = {}
    for k in range(NUM):
        yk = np.concatenate(oof_y[k])
        pk = np.concatenate(oof_p[k])
        cv_f1[k] = f1_score(yk, (pk >= FIXED_THR[k]).astype(int), zero_division=0)
        tag = "ctx+text" if k in text_classes else "ctx-only"
        print(f"[G] {LABELS[k]:3s} F1={cv_f1[k]:.4f}@{FIXED_THR[k]:.2f} [{tag}]", file=sys.stderr)
    macro = float(np.mean(list(cv_f1.values())))
    print(f"[G] cap1 切片CV macro={macro:.4f} (变体F SOTA cap1≈0.650)", file=sys.stderr)

    # test: full retrain 5seed prob-avg + fixed thr
    Xc_all = np.vstack([win[c][0] for c in conv_ids])
    Xf_all = np.vstack([win[c][1] for c in conv_ids])
    Yall = np.vstack([win[c][2] for c in conv_ids])
    te_arrs = [np.load(p).astype(int) for p in test_files]
    te_utts = [json.load(open(f"data/test/text/{Path(p).stem}.json")) for p in test_files]
    Xc_te, Xf_te = [], []
    for arr, tj in zip(te_arrs, te_utts):
        cf = ctxfeat(arr)
        tf = text_feats(tj.get("utterances", []), int(tj.get("end_ms", 30000)))
        Xc_te.append(cf)
        Xf_te.append(np.concatenate([cf, tf]))
    Xc_te = np.array(Xc_te, dtype=np.float32)
    Xf_te = np.array(Xf_te, dtype=np.float32)

    preds = {}
    for k in range(NUM):
        Xk = feat_for_class(k, Xc_all, Xf_all, text_classes)
        Xte_k = feat_for_class(k, Xc_te, Xf_te, text_classes)
        seed_te = [fit(Xk, Yall[:, k], 42 + s).predict_proba(Xte_k)[:, 1] for s in range(n_seed)]
        preds[k] = (prob_avg(seed_te) >= FIXED_THR[k]).astype(int)

    cnts = {c: int(preds[COL2K[c]].sum()) for c in SUBMIT}
    diff = {c: cnts[c] - anchor[c] for c in SUBMIT} if anchor else {}
    print(f"[G] 正例数: " + " ".join(f"{c}={cnts[c]}({diff.get(c, 0):+d})" for c in SUBMIT), file=sys.stderr)

    run = Path(f"tools/runs/climb/variant-G-{ts}")
    run.mkdir(parents=True, exist_ok=True)
    with open(run / "pred_test1.csv", "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(seg_ids):
            f.write(",".join([sid] + [str(int(preds[COL2K[c]][i])) for c in SUBMIT]) + "\n")
    (run / "cv_metrics.json").write_text(json.dumps({
        "variant": "G", "cap1_macro_f1": round(macro, 4),
        "per_sub": {LABELS[k]: round(cv_f1[k], 4) for k in range(NUM)},
        "thresholds": {LABELS[k]: FIXED_THR[k] for k in range(NUM)},
        "pos_counts": cnts, "diff_vs_cycle1": diff,
        "method": "变体F基座(C/NA/BC纯ctx 5seed概率平均)+T/I叠ASR文本, cycle1固定阈值, cap1切片CV",
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"variant": "G", "cap1_cv": round(macro, 4),
                      "per_sub": {LABELS[k]: round(cv_f1[k], 4) for k in range(NUM)},
                      "csv": str(run / "pred_test1.csv")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
