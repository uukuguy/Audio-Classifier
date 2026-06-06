"""T2 公榜验证: R4 全栈 + mask050 重训 ctx, 在截短 ctx 上推理.

跟 build_truncated_r4.py 对照, 区别仅在 ctx 4 base 模型用 mask=0.5 重训.
出 csv 上公榜, 跟 R4_keep125 真分 0.722 对比, 验证 mask 训能否压回退化.

Usage:
  python3 tools/climb/build_r4_mask_truncated.py --keep 125    # 10s 截短
  python3 tools/climb/build_r4_mask_truncated.py --keep 375    # 30s full (mask 训能否不破 SOTA)
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dynamic_ctx_utils import simulate_truncated_context, CTX_FULL  # noqa: E402
from cycle_context import featurize  # noqa: E402

NUM = 5
TGT = 25
SEED = 42
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}


def build_train_mask(conv_ids, label_files, mask_prob: float = 0.5,
                     keep_choices: tuple = (50, 100, 200, 300, 375),
                     stride: int = 5):
    """mask 训练: 跟 cycle_context.build_train(mask_prob=0.5) 等价."""
    NA = 4
    rng = np.random.RandomState(SEED)
    X, Y = [], []
    for cid in conv_ids:
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


def fit_5_lgbm(X, Y):
    from lightgbm import LGBMClassifier
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


def make_sota_3src(ctx, wsp, hub):
    p = np.zeros_like(ctx)
    p[:, 0] = ctx[:, 0]
    p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * hub[:, 1]
    p[:, 2] = ctx[:, 2]
    p[:, 3] = (ctx[:, 3] + wsp[:, 3] + hub[:, 3]) / 3
    p[:, 4] = ctx[:, 4]
    return p


def softadd(base, src, w, cols):
    out = base.copy()
    for c in cols:
        out[:, c] = (1 - w) * base[:, c] + w * src[:, c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, required=True,
                    help="ctx 截短挡: 375=full, 125=10s, 63=5s")
    ap.add_argument("--mask-prob", type=float, default=0.5,
                    help="train mask 概率 (实验默认 0.5)")
    args = ap.parse_args()

    keep = args.keep
    keep_sec = keep * 80 / 1000

    print(f"[T2] === R4 mask{int(args.mask_prob*100):03d} @ keep={keep} ({keep_sec:.1f}s) ===",
          file=sys.stderr)

    # 1. 训 mask ctx 4 base
    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    random.Random(SEED).shuffle(conv_ids)

    print(f"[T2] training ctx LGBM (mask={args.mask_prob}) on {len(conv_ids)} convs ...",
          file=sys.stderr)
    Xtr, Ytr = build_train_mask(conv_ids, label_files, mask_prob=args.mask_prob, stride=5)
    print(f"[T2]   windows: {Xtr.shape}", file=sys.stderr)
    clfs = fit_5_lgbm(Xtr, Ytr)

    # 2. test ctx 截短 → featurize → ctx prob
    test_ctx_files = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_ctx_files]

    X_te = np.array([
        featurize(simulate_truncated_context(np.load(p).astype(int), keep_chunks=keep))
        for p in test_ctx_files
    ])
    ctx_te = predict_ctx(clfs, X_te)
    print(f"[T2] ctx_te @keep={keep}: shape={ctx_te.shape}", file=sys.stderr)

    # 3. SSL_ms / ortho 用现有 probs (audio 模型, 不受 ctx 影响)
    z3 = np.load("tools/runs/climb/orthofuse-3src-20260601-1607/fused_probs.npz")
    wsp_te, hub_te = z3["whisper_te"], z3["hubert_te"]
    wsp_ms_te = np.load("tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/probs.npz")["test"]
    e2v_ms_te = np.load("tools/runs/climb/e2v-bcaug-multiseed-20260602-1632/probs.npz")["test"]
    hub_ms_te = np.load("tools/runs/climb/hubert-bcaug-multiseed-20260602-1506/probs.npz")["test"]

    # 4. orthofuse → NSOTA07 → R4
    sota = make_sota_3src(ctx_te, wsp_te, hub_te)
    nsota_07 = softadd(sota, wsp_ms_te, 0.07, (1, 2, 3))
    r4 = softadd(softadd(nsota_07, e2v_ms_te, 0.03, (1, 2, 3)), hub_ms_te, 0.03, (1, 2, 3))

    # 5. 写 csv
    out_name = f"R4_mask{int(args.mask_prob*100):03d}_keep{keep}_ctx{int(keep_sec)}s"
    out_dir = Path(f"submission/truncated-validation-20260604/{out_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.zeros((len(seg_ids), 5), dtype=int)
    for k in range(NUM):
        pred[:, k] = (r4[:, k] >= THR_VARF[k]).astype(int)
    df = pd.DataFrame({"segment_id": seg_ids})
    for c in SUBMIT_COLS:
        df[c] = pred[:, COL2K[c]]
    csv_path = out_dir / "pred_test1.csv"
    df.to_csv(csv_path, index=False)

    pos = {c: int(df[c].sum()) for c in SUBMIT_COLS}
    print(f"\n[T2] ✓ {csv_path}", file=sys.stderr)
    print(f"     pos: {pos}", file=sys.stderr)
    print(f"     对照 R4 full 30s: c=975 na=947 i=80 bc=20 t=523 (真分 0.7458)", file=sys.stderr)
    print(f"     对照 R4 keep125 baseline (无 mask): c=974 na=996 i=37 bc=15 t=341 (真分 0.7218)",
          file=sys.stderr)

    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump({
            "name": out_name,
            "keep_chunks": keep,
            "keep_seconds": keep_sec,
            "ctx_mask_prob": args.mask_prob,
            "description": f"R4 NSOTA07 + e2v0.03 + hub0.03, ctx LGBM 用 mask={args.mask_prob} 重训, 截短到 {keep} chunk ({keep_sec:.1f}s)",
            "pos": pos,
            "reference_r4_full_30s_no_mask": {"true_score": 0.745798, "pos": {"c": 975, "na": 947, "i": 80, "bc": 20, "t": 523}},
            "reference_r4_keep125_no_mask": {"true_score": 0.721787, "pos_approx": {"c": 974, "na": 996, "i": 37, "bc": 15, "t": 341}},
            "hypothesis": "mask 训能否压回 R4 在短 ctx 上的退化 (从 0.722 提到 0.735+)",
            "if_full_30s_drops": "mask 训对 30s 完整 ctx 推理是否伤 (估 -0.005 内)",
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
