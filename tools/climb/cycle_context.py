"""climb cycle — paradigm=context-only (H-001).

纯上下文标签 LGBM 多标签。流程:
  1. 按会话划分 train/valid（防泄漏）
  2. valid 上逐类调阈值（max per-class F1）→ 得 Macro-F1 CV + 5 类子分 + 5 阈值
  3. 全量 train 重训 5 个 LGBM
  4. test context（data/test/context/*.npy, 375-chunk）featurize → 预测 → 套阈值
  5. 写 pred_test1.csv（segment_id,c,na,i,bc,t）到 run 目录 + manifest

Usage: python tools/climb/cycle_context.py <run_dir>
输出: <run_dir>/pred_test1.csv, <run_dir>/cv_metrics.json, <run_dir>/manifest.json
最后一行 stdout 是 eval-local 契约 JSON: {"score":..,"per_sub":{..}}
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
# 提交 CSV 列顺序（config multi_targets 小写）: c,na,i,bc,t
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL_TO_LABELID = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
NUM = 5
CTX, TGT, STRIDE = 375, 25, 5
SEED = 42


def featurize(ctx: np.ndarray) -> np.ndarray:
    """375-chunk 上下文标签 → 手工特征（train/test 一致）。"""
    oh = np.eye(NUM)[ctx]
    feats = []
    for w in (10, 25, 50, 100, 200, 375):
        feats.extend(oh[-w:].mean(axis=0))
    for i in range(1, 6):
        feats.append(ctx[-i] if len(ctx) >= i else -1)
    L = len(ctx)
    for k in range(NUM):
        pos = np.where(ctx == k)[0]
        feats.append((L - 1 - pos[-1]) / L if len(pos) else 1.0)
    for k in range(NUM):
        feats.append((ctx == k).sum() / L)
    feats.append((ctx[1:] != ctx[:-1]).mean())
    return np.array(feats, dtype=np.float32)


def build_train(conv_ids, label_files, mask_prob: float = 0.0,
                keep_choices: tuple = (50, 100, 200, 300, 375)):
    """构 train 窗口. 默认 baseline 不 mask.

    mask_prob > 0 时, 每个窗口以概率 mask_prob 随机截短到 keep_choices 之一,
    模拟 D-26 复赛动态时长. 截短后前段 NA pad 回 375 chunk (跟 normalize_ctx_to_375 一致).
    实验值通过 env var 传入, 不进 baseline default (CLAUDE.md §2.6).
    """
    NA_LABEL = 4
    rng = np.random.RandomState(SEED)
    X, Y = [], []
    for cid in conv_ids:
        a = np.load(label_files[cid])
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE):
            fut = set(int(x) for x in a[e:e + TGT])
            ctx_window = a[e - CTX:e].astype(int)
            # 训练时模拟变长 (T2)
            if mask_prob > 0 and rng.random() < mask_prob:
                keep = int(rng.choice(keep_choices))
                if keep < CTX:
                    pad = np.full(CTX - keep, NA_LABEL, dtype=int)
                    ctx_window = np.concatenate([pad, ctx_window[-keep:]])
            X.append(featurize(ctx_window))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(X), np.array(Y, dtype=int)


def tune_threshold(y, p):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def fit_lgbm(X, y):
    spw = (len(y) - y.sum()) / max(1, y.sum())
    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED,
    )
    clf.fit(X, y)
    return clf


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/runs/climb/_adhoc")
    run_dir.mkdir(parents=True, exist_ok=True)

    # T2 实验: env var 读 mask_prob (baseline default = 0 = 不 mask, §2.6)
    mask_prob = float(os.environ.get("CTX_MASK_PROB", "0.0"))
    if mask_prob > 0:
        print(f"[cycle-ctx] T2 mask 实验: CTX_MASK_PROB={mask_prob} "
              f"(每窗 {mask_prob*100:.0f}% 概率随机截短到 {{50,100,200,300,375}} 之一)",
              file=sys.stderr)

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    random.Random(SEED).shuffle(conv_ids)
    n_val = max(1, int(len(conv_ids) * 0.15))
    val_ids, train_ids = conv_ids[:n_val], conv_ids[n_val:]
    print(f"[cycle-ctx] conv: train={len(train_ids)} valid={len(val_ids)}", file=sys.stderr)

    Xtr, Ytr = build_train(train_ids, label_files, mask_prob=mask_prob)
    # valid 集**不 mask** — 保持 baseline 评估口径一致 (cap1 真分关心 30s 完整 ctx)
    Xva, Yva = build_train(val_ids, label_files, mask_prob=0.0)
    print(f"[cycle-ctx] windows: train={len(Xtr)} valid={len(Xva)} (valid 不 mask)",
          file=sys.stderr)

    # --- 1) valid 上调阈值 + 记 CV 指标（per class id 0..4） ---
    thr_by_labelid, f1_by_labelid = {}, {}
    for k in range(NUM):
        clf = fit_lgbm(Xtr, Ytr[:, k])
        p = clf.predict_proba(Xva)[:, 1]
        t, f = tune_threshold(Yva[:, k], p)
        thr_by_labelid[k], f1_by_labelid[k] = t, f
        print(f"[cycle-ctx] {LABELS[k]:3s} thr={t:.2f} F1={f:.3f}", file=sys.stderr)
    macro_f1 = float(np.mean(list(f1_by_labelid.values())))
    print(f"[cycle-ctx] CV Macro-F1 (tuned) = {macro_f1:.4f}", file=sys.stderr)

    # --- 2) 全量重训（train+valid 所有 conv, 同样 mask）---
    Xall, Yall = build_train(conv_ids, label_files, mask_prob=mask_prob)
    print(f"[cycle-ctx] retrain on all: {len(Xall)} windows (mask_prob={mask_prob})",
          file=sys.stderr)
    final_clfs = {k: fit_lgbm(Xall, Yall[:, k]) for k in range(NUM)}

    # --- 2b) dump LGBM ckpt + thresholds 供 docker infer load 用 ---
    ckpt_dir = run_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    for k in range(NUM):
        joblib.dump(final_clfs[k], ckpt_dir / f"lgbm_{LABELS[k].lower()}.joblib")
    (ckpt_dir / "thresholds.json").write_text(json.dumps(
        {LABELS[k].lower(): round(thr_by_labelid[k], 4) for k in range(NUM)},
        ensure_ascii=False, indent=2,
    ))
    (ckpt_dir / "feature_spec.json").write_text(json.dumps({
        "paradigm": "context-only",
        "feature_dim": int(Xall.shape[1]),
        "ctx_chunks": CTX,
        "label_order": [LABELS[k] for k in range(NUM)],
        "submit_cols": SUBMIT_COLS,
        "col_to_labelid": COL_TO_LABELID,
        "featurize_fn": "tools.climb.cycle_context.featurize",
        "ctx_mask_prob": mask_prob,
        "ctx_mask_keep_choices": [50, 100, 200, 300, 375] if mask_prob > 0 else None,
        "note": ("baseline 训练 (mask=0)" if mask_prob == 0 else
                 f"T2 变长 mask 训练 (CTX_MASK_PROB={mask_prob}), "
                 "infer 端配 normalize_ctx_to_375 适配 D-26 复赛动态时长"),
    }, ensure_ascii=False, indent=2))
    print(f"[cycle-ctx] dumped ckpt to {ckpt_dir}/ (5 lgbm + thresholds + spec)", file=sys.stderr)

    # --- 3) test context 预测 ---
    test_ctx_files = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_ctx_files]
    Xte = np.array([featurize(np.load(p).astype(int)) for p in test_ctx_files])
    preds_by_labelid = {}
    for k in range(NUM):
        p = final_clfs[k].predict_proba(Xte)[:, 1]
        preds_by_labelid[k] = (p >= thr_by_labelid[k]).astype(int)
    print(f"[cycle-ctx] test predicted: {len(seg_ids)} segments", file=sys.stderr)

    # --- 4) 写 pred_test1.csv（列序 c,na,i,bc,t）---
    csv_path = run_dir / "pred_test1.csv"
    with open(csv_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT_COLS) + "\n")
        for i, sid in enumerate(seg_ids):
            row = [sid] + [str(int(preds_by_labelid[COL_TO_LABELID[c]][i])) for c in SUBMIT_COLS]
            f.write(",".join(row) + "\n")
    print(f"[cycle-ctx] wrote {csv_path}", file=sys.stderr)

    # --- CV metrics + manifest ---
    per_sub = {c: round(f1_by_labelid[COL_TO_LABELID[c]], 4) for c in ["c", "na", "t", "i", "bc"]}
    cv = {
        "paradigm": "context-only", "hypothesis_id": "H-001",
        "cv_macro_f1": round(macro_f1, 4), "per_sub_f1": per_sub,
        "thresholds": {LABELS[k]: round(thr_by_labelid[k], 2) for k in range(NUM)},
        "n_train_windows": int(len(Xall)), "valid_split": "conv-split 15%",
        "note": "首个 cycle 单 split 调阈值（K-fold OOF defer）；CV 为滑窗乐观估计",
    }
    (run_dir / "cv_metrics.json").write_text(json.dumps(cv, ensure_ascii=False, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps({
        "cycle": 1, "hypothesis_id": "H-001", "paradigm": "context-only",
        "start": datetime.now().isoformat(timespec="seconds"), "end": None,
    }, ensure_ascii=False, indent=2))

    # eval-local 契约: 最后一行 stdout
    print(json.dumps({"score": round(macro_f1, 4),
                      "per_sub": per_sub}))


if __name__ == "__main__":
    main()
