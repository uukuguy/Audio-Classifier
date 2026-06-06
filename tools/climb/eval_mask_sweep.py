"""mask sweep: 找 ctx LGBM 训练时 mask_prob 的甜点.

实验设计:
  6 个 mask_prob × 5 个 keep_chunks = 30 个 ctx-only macro F1 数据点

  mask_prob ∈ {0.0, 0.2, 0.3, 0.4, 0.5, 0.7}
  keep_chunks ∈ {63, 125, 188, 250, 375}  (5s, 10s, 15s, 20s, 30s)

  评估: 从 train 切 cap0-cap4 (5 窗口/通) 当伪 test, 用 train Y 算 macro F1

输出:
  tools/runs/climb/mask-sweep-YYYYMMDD-HHMM/
    results.json   # 30 个数据点
    matrix.txt     # 人读矩阵
    best.json      # 加权挑选最优 mask_prob (按假设分布)

Usage:
  OMP_NUM_THREADS=4 python3 tools/climb/eval_mask_sweep.py
"""
from __future__ import annotations

import glob
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from dynamic_ctx_utils import simulate_truncated_context, CTX_FULL  # noqa: E402
from cycle_context import featurize  # noqa: E402

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
NUM = 5
TGT = 25
SEED = 42

MASK_PROBS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7]
KEEP_CHUNKS = [63, 125, 188, 250, 375]  # 5s 10s 15s 20s 30s


def build_train_with_mask(train_ids, label_files, mask_prob: float,
                          keep_choices: tuple = (50, 100, 200, 300, 375),
                          stride: int = 5):
    """跟 cycle_context.build_train(mask_prob) 一致."""
    NA = 4
    rng = np.random.RandomState(SEED)
    X, Y = [], []
    for cid in train_ids:
        a = np.load(label_files[cid]).astype(int)
        for e in range(CTX_FULL, a.shape[0] - TGT + 1, stride):
            fut = set(int(x) for x in a[e:e + TGT])
            ctx_window = a[e - CTX_FULL:e]
            if mask_prob > 0 and rng.random() < mask_prob:
                keep = int(rng.choice(keep_choices))
                if keep < CTX_FULL:
                    pad = np.full(CTX_FULL - keep, NA, dtype=int)
                    ctx_window = np.concatenate([pad, ctx_window[-keep:]])
            X.append(featurize(ctx_window))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(X), np.array(Y, dtype=int)


def build_eval_set(eval_ids, label_files, cap_n=5, stride=50):
    """每通最多 cap_n 窗口."""
    ctx_list, Y_list = [], []
    for cid in eval_ids:
        a = np.load(label_files[cid]).astype(int)
        if a.shape[0] < CTX_FULL + TGT:
            continue
        n = 0
        for e in range(CTX_FULL, a.shape[0] - TGT + 1, stride):
            ctx_list.append(a[e - CTX_FULL:e])
            fut = set(int(x) for x in a[e:e + TGT])
            Y_list.append([1 if k in fut else 0 for k in range(NUM)])
            n += 1
            if n >= cap_n:
                break
    return ctx_list, np.array(Y_list, dtype=int)


def fit_5_lgbm(X, Y):
    clfs = []
    for k in range(NUM):
        y = Y[:, k]
        spw = (len(y) - y.sum()) / max(1, y.sum())
        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED,
        )
        clf.fit(X, y)
        clfs.append(clf)
    return clfs


def predict_ctx(clfs, X):
    P = np.zeros((len(X), NUM), dtype=np.float32)
    for k in range(NUM):
        P[:, k] = clfs[k].predict_proba(X)[:, 1]
    return P


def macro_f1_from_probs(P, Y):
    f1s = [f1_score(Y[:, k], (P[:, k] >= THR_VARF[k]).astype(int), zero_division=0)
           for k in range(NUM)]
    return float(np.mean(f1s)), f1s


def main():
    t0 = time.time()
    print("=== mask sweep: 找 ctx LGBM mask_prob 甜点 ===\n", file=sys.stderr)

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    random.Random(SEED).shuffle(conv_ids)
    n_eval = max(20, int(len(conv_ids) * 0.15))
    eval_ids, train_ids = conv_ids[:n_eval], conv_ids[n_eval:]
    print(f"split: train={len(train_ids)} eval={len(eval_ids)}\n", file=sys.stderr)

    # 评估集 (用所有 mask_prob 共享, 不变)
    ctx_eval_list, Y_eval = build_eval_set(eval_ids, label_files, cap_n=5, stride=50)
    print(f"eval samples: {len(ctx_eval_list)}", file=sys.stderr)
    pos_dist = " ".join(f"{LABELS[k]}={Y_eval[:, k].sum()}" for k in range(NUM))
    print(f"Y_eval pos: {pos_dist}\n", file=sys.stderr)

    # 6 mask_prob × 5 keep_chunks 矩阵
    matrix = {}  # {mask_prob: {keep: macro_f1}}
    per_class = {}  # {mask_prob: {keep: [C T BC I NA]}}
    for mp in MASK_PROBS:
        t1 = time.time()
        print(f"=== mask_prob={mp} ===", file=sys.stderr)
        Xtr, Ytr = build_train_with_mask(train_ids, label_files, mask_prob=mp, stride=5)
        print(f"  train windows: {Xtr.shape}", file=sys.stderr)
        clfs = fit_5_lgbm(Xtr, Ytr)
        train_sec = time.time() - t1

        matrix[mp] = {}
        per_class[mp] = {}
        for keep in KEEP_CHUNKS:
            X_sim = np.array([
                featurize(simulate_truncated_context(ctx, keep_chunks=keep, mode="pad_na_left"))
                for ctx in ctx_eval_list
            ])
            P_sim = predict_ctx(clfs, X_sim)
            macro, per = macro_f1_from_probs(P_sim, Y_eval)
            matrix[mp][keep] = macro
            per_class[mp][keep] = per

        eval_sec = time.time() - t1 - train_sec
        print(f"  done in train={train_sec:.0f}s eval={eval_sec:.0f}s\n", file=sys.stderr)

    # 输出矩阵
    out_dir = Path(f"tools/runs/climb/mask-sweep-{time.strftime('%Y%m%d-%H%M')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_txt = ["=== mask sweep: macro F1 矩阵 (内部 cross-context 评估, train cap5) ==="]
    matrix_txt.append("")
    header = f"  {'mask_prob':>10s} | " + " ".join(f"{k:>7s}" for k in [f"{c*80/1000:.0f}s" for c in KEEP_CHUNKS])
    matrix_txt.append(header)
    matrix_txt.append("  " + "-" * (12 + 8 * len(KEEP_CHUNKS)))
    for mp in MASK_PROBS:
        row = f"  {mp:>10.2f} | " + " ".join(f"{matrix[mp][k]:>7.4f}" for k in KEEP_CHUNKS)
        matrix_txt.append(row)
    matrix_txt.append("")

    # Δ vs mask=0 行
    matrix_txt.append("=== Δ vs baseline (mask_prob=0) ===")
    matrix_txt.append(header)
    matrix_txt.append("  " + "-" * (12 + 8 * len(KEEP_CHUNKS)))
    for mp in MASK_PROBS:
        if mp == 0.0:
            continue
        row = f"  {mp:>10.2f} | " + " ".join(
            f"{matrix[mp][k] - matrix[0.0][k]:+7.4f}" for k in KEEP_CHUNKS
        )
        matrix_txt.append(row)

    # 加权选最优 (假设分布)
    matrix_txt.append("")
    matrix_txt.append("=== 加权选最优 (3 个假设分布) ===")
    weights_scenarios = {
        "均匀 (0,30]s": {k: 1 / len(KEEP_CHUNKS) for k in KEEP_CHUNKS},
        "短偏 (S形)":   {63: 0.30, 125: 0.30, 188: 0.20, 250: 0.15, 375: 0.05},
        "长偏 (公榜复刻)": {63: 0.05, 125: 0.10, 188: 0.15, 250: 0.30, 375: 0.40},
    }
    for sc_name, w in weights_scenarios.items():
        matrix_txt.append(f"  {sc_name}:")
        for mp in MASK_PROBS:
            weighted = sum(matrix[mp][k] * w[k] for k in KEEP_CHUNKS)
            matrix_txt.append(f"    mask={mp:.2f}: {weighted:.4f}")

    matrix_text = "\n".join(matrix_txt)
    print("\n" + matrix_text)
    (out_dir / "matrix.txt").write_text(matrix_text)

    # 落盘 JSON
    results_json = {
        "_note": "mask sweep: ctx LGBM 训练 mask_prob 甜点扫描. cross-context macro F1 (cap5).",
        "mask_probs": MASK_PROBS,
        "keep_chunks": KEEP_CHUNKS,
        "n_eval_samples": len(ctx_eval_list),
        "matrix": {str(mp): {str(k): matrix[mp][k] for k in KEEP_CHUNKS} for mp in MASK_PROBS},
        "per_class": {str(mp): {str(k): per_class[mp][k] for k in KEEP_CHUNKS}
                      for mp in MASK_PROBS},
        "wall_time_sec": time.time() - t0,
    }
    (out_dir / "results.json").write_text(json.dumps(results_json, indent=2, ensure_ascii=False))
    print(f"\n落盘: {out_dir}/", file=sys.stderr)
    print(f"  matrix.txt + results.json", file=sys.stderr)
    print(f"  总耗时 {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
