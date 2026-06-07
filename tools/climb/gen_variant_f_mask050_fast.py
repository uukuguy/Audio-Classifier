"""variant-F mask050 (fast) — 跳过 cap1 OOF, 直接全量 5 seed retrain → test probs.

跟 gen_variant_f_mask050.py 区别:
  - 不算 cap1 OOF (不需要 cv eval, V2 ctx-only 已公榜证 mask050 在动态长度上 +0.010)
  - build_windows 全局 1 次 (mask 随机性来自单一 rng, 不每 seed 重抽)
  - 5 seed 多样性来自 LGBM random_state (跟 variant-F 同款)
  - test 同时算两套: test (全 30s) + test_v2 (V2 截短规则)

预估耗时: ~10 min (vs 原版 ~4h).

Usage:
  OMP_NUM_THREADS=4 python3 tools/climb/gen_variant_f_mask050_fast.py [--seeds 5] [--mask-prob 0.5]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/climb"))

from cycle_context import CTX, LABELS, NUM, STRIDE, TGT  # noqa: E402
from cycle_context import featurize as ctxfeat  # noqa: E402

warnings.filterwarnings("ignore")

NA_LABEL = 4
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
FIXED_THR = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}  # cycle1 / variant-F
KEEP_CHOICES = (50, 100, 200, 300, 375)


def fit(X, y, seed):
    # 跟 gen_variants.py / cycle_context.py 严格同纲: scale_pos_weight 必须有,
    # 否则稀有类 (BC/I/T) 假负爆 + 多数类 (C/NA) 假正爆.
    spw = (len(y) - y.sum()) / max(1, y.sum())
    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=seed,
    )
    clf.fit(X, y)
    return clf


def build_all_windows(label_files, conv_ids, mask_prob: float, master_seed: int = 42):
    """全量 build_windows, mask 用单 rng 共享 (跟原 cycle_context.build_train 一致)."""
    rng = np.random.RandomState(master_seed)
    X_all, Y_all = [], []
    t0 = time.time()
    for cid in conv_ids:
        arr = np.load(label_files[cid]).astype(int)
        for e in range(CTX, arr.shape[0] - TGT + 1, STRIDE):
            fut = set(int(x) for x in arr[e:e + TGT])
            ctx_window = arr[e - CTX:e].astype(int)
            if mask_prob > 0 and rng.random() < mask_prob:
                keep = int(rng.choice(KEEP_CHOICES))
                if keep < CTX:
                    pad = np.full(CTX - keep, NA_LABEL, dtype=int)
                    ctx_window = np.concatenate([pad, ctx_window[-keep:]])
            X_all.append(ctxfeat(ctx_window))
            Y_all.append([1 if k in fut else 0 for k in range(NUM)])
    X = np.array(X_all, dtype=np.float32)
    Y = np.array(Y_all, dtype=np.int8)
    print(f"  build_windows done in {time.time()-t0:.1f}s, shape X={X.shape} Y={Y.shape}",
          file=sys.stderr)
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mask-prob", type=float, default=0.5)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = list(label_files.keys())
    test_files = sorted(glob.glob("data/test/context/*.npy"))

    print(f"[mask050-fast] mask_prob={args.mask_prob} seeds={args.seeds} "
          f"n_conv={len(conv_ids)} n_test={len(test_files)}", file=sys.stderr)

    # 1. 全量 build_windows (1 次)
    print("[mask050-fast] building all windows...", file=sys.stderr)
    X_tr, Y_tr = build_all_windows(label_files, conv_ids, args.mask_prob, master_seed=42)

    # 2. 构 test 特征 (两套: full 30s + V2 截短)
    print("[mask050-fast] building test features (full + v2)...", file=sys.stderr)
    seg_ids = [Path(p).stem for p in test_files]
    Xte_full = np.array([ctxfeat(np.load(p).astype(int)) for p in test_files], dtype=np.float32)
    Xte_v2_list = []
    for p in test_files:
        sid = int(Path(p).stem)
        ctx = np.load(p).astype(int)
        if sid % 2 == 0:
            ctx_eff = ctx[-125:] if len(ctx) >= 125 else ctx
            pad = np.full(CTX - len(ctx_eff), NA_LABEL, dtype=int)
            ctx_v2 = np.concatenate([pad, ctx_eff])
        else:
            ctx_v2 = ctx if len(ctx) == CTX else ctx[-CTX:]
        Xte_v2_list.append(ctxfeat(ctx_v2))
    Xte_v2 = np.array(Xte_v2_list, dtype=np.float32)
    print(f"  Xte_full={Xte_full.shape} Xte_v2={Xte_v2.shape}", file=sys.stderr)

    # 3. 5 seed × 5 class fit + predict
    print(f"[mask050-fast] training {args.seeds} seed × {NUM} class on {len(X_tr)} windows...",
          file=sys.stderr)
    preds_full = np.zeros((len(test_files), NUM), dtype=np.float32)
    preds_v2 = np.zeros((len(test_files), NUM), dtype=np.float32)
    for k in range(NUM):
        t_class = time.time()
        seed_te_full = []
        seed_te_v2 = []
        for s in range(args.seeds):
            clf = fit(X_tr, Y_tr[:, k], seed=42 + s)
            seed_te_full.append(clf.predict_proba(Xte_full)[:, 1])
            seed_te_v2.append(clf.predict_proba(Xte_v2)[:, 1])
        preds_full[:, k] = np.mean(seed_te_full, axis=0)
        preds_v2[:, k] = np.mean(seed_te_v2, axis=0)
        print(f"  class {LABELS[k]} done in {time.time()-t_class:.1f}s", file=sys.stderr)

    # 4. 二值化 (cycle1 阈值) — 双 csv (full + v2)
    pred_bin_full = np.zeros_like(preds_full, dtype=int)
    pred_bin_v2 = np.zeros_like(preds_v2, dtype=int)
    for k in range(NUM):
        pred_bin_full[:, k] = (preds_full[:, k] >= FIXED_THR[k]).astype(int)
        pred_bin_v2[:, k] = (preds_v2[:, k] >= FIXED_THR[k]).astype(int)

    pos_full = {c: int(pred_bin_full[:, COL2K[c]].sum()) for c in SUBMIT}
    pos_v2 = {c: int(pred_bin_v2[:, COL2K[c]].sum()) for c in SUBMIT}

    # 5. 输出
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / f"tools/runs/climb/variant-F-mask050-fast-{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 单独 csv 用于 ctx-only 直投 sanity (跟 V1=0.7108 / V2=0.7209 对照)
    for tag, pred_bin, pos in [("full_30s", pred_bin_full, pos_full),
                                ("v2_truncated", pred_bin_v2, pos_v2)]:
        csv_path = out_dir / f"pred_test1_{tag}.csv"
        with open(csv_path, "w", newline="\n") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(seg_ids):
                row = ",".join(str(pred_bin[i, COL2K[c]]) for c in SUBMIT)
                f.write(f"{sid},{row}\n")
        print(f"  wrote {csv_path}: pos={pos}", file=sys.stderr)

    # probs.npz 给 R4 全栈 dual-route 用
    np.savez_compressed(out_dir / "probs.npz",
                        test=preds_full.astype(np.float32),
                        test_v2=preds_v2.astype(np.float32),
                        seg_ids=np.array(seg_ids))

    (out_dir / "manifest.json").write_text(json.dumps({
        "variant": "F-mask050-fast",
        "mask_prob": args.mask_prob,
        "seeds": args.seeds,
        "no_cv": "跳过 cap1 OOF, V2 ctx-only 已公榜证 +0.010",
        "thresholds_used": FIXED_THR,
        "pos_full_30s": pos_full,
        "pos_v2_truncated": pos_v2,
        "note": "variant-F 同算法 (5 seed prob_avg + cycle1 阈值), train mask=0.5, test 算 full + v2 两套.",
    }, ensure_ascii=False, indent=2))

    print(f"\n[mask050-fast] DONE → {out_dir}/", file=sys.stderr)
    print(f"  pos full_30s: {pos_full}", file=sys.stderr)
    print(f"  pos v2_truncated: {pos_v2}", file=sys.stderr)


if __name__ == "__main__":
    main()
