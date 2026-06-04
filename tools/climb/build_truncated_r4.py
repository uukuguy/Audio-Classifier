"""T1 公榜验证: 模拟 R4 在截短上下文下推理, 出可提交 csv.

思路:
  1. 加载公榜测试集 1 的原始 375 chunk context (data/test/context/*.npy)
  2. 模拟截短: 每段 context 截到末 N chunk + NA pad 回 375
  3. 用我们训好的 ctx LGBM 4 base 模型重推 test_ctx (短上下文版)
  4. wsp / hub / wsp_ms / e2v_ms / hub_ms test probs 不变 (它们吃 audio, 不依赖 context)
  5. 重做 orthofuse-3src → softadd → R4 csv

注意:
  - ctx 4 base 没有 ckpt 落盘, 这里**重训** v1 (300 trees LGBM) — 跟 cycle_context.py 一致, SEED=42 重现性 OK
  - 公榜 push 1 个 csv 看真分跌幅 = 真实 R4 在 N chunk 上下文下的退化

Usage:
  python tools/climb/build_truncated_r4.py --keep 125  # 模拟 10s 上下文
  → 输出 submission/truncated-validation-20260604/R4_keep125/pred_test1.csv
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
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent))
from dynamic_ctx_utils import simulate_truncated_context, CTX_FULL  # noqa: E402

NUM = 5
TGT = 25
SEED = 42
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}


def featurize(ctx: np.ndarray) -> np.ndarray:
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


def build_train(conv_ids, label_files, stride=5):
    X, Y = [], []
    for cid in conv_ids:
        a = np.load(label_files[cid]).astype(int)
        for e in range(CTX_FULL, a.shape[0] - TGT + 1, stride):
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(featurize(a[e - CTX_FULL:e]))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(X), np.array(Y, dtype=int)


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


def make_sota_3src(ctx, wsp, hub):
    """orthofuse-3src 路由 (跟 cycle_orthofuse 一致)."""
    p = np.zeros_like(ctx)
    p[:, 0] = ctx[:, 0]                                  # C
    p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * hub[:, 1]          # T
    p[:, 2] = ctx[:, 2]                                  # BC
    p[:, 3] = (ctx[:, 3] + wsp[:, 3] + hub[:, 3]) / 3    # I
    p[:, 4] = ctx[:, 4]                                  # NA
    return p


def softadd(base, src, w, cols):
    out = base.copy()
    for c in cols:
        out[:, c] = (1 - w) * base[:, c] + w * src[:, c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, required=True,
                    help="保留末 keep chunks 模拟短上下文 (375=full, 125=10s, 63=5s, 12=1s)")
    ap.add_argument("--out-name", default=None,
                    help="输出子目录名 (默认 R4_keep<N>)")
    args = ap.parse_args()

    keep = args.keep
    if keep < 1 or keep > 375:
        print(f"[err] keep must be in [1, 375], got {keep}", file=sys.stderr)
        sys.exit(1)
    keep_sec = keep * 80 / 1000

    # 1. 训 ctx 4 base (这里只用 v1 = LGBM 一个; SOTA 0.71755 用的是 v1 单独 in orthofuse-3src)
    print(f"[T1] === build R4 @ keep={keep} chunks ({keep_sec:.1f}s) ===", file=sys.stderr)
    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    train_ids = sorted(label_files)
    random.Random(SEED).shuffle(train_ids)
    # 用全量 train (不留 holdout, 跟 cycle_orthofuse 一致, 但 SEED 重现可控)

    print(f"[T1] training ctx LGBM v1 on {len(train_ids)} convs ...", file=sys.stderr)
    Xtr, Ytr = build_train(train_ids, label_files, stride=5)
    print(f"[T1]   windows: {Xtr.shape}", file=sys.stderr)
    clfs = fit_5_lgbm(Xtr, Ytr)
    print(f"[T1]   ctx model trained", file=sys.stderr)

    # 2. 加载公榜 test context, 模拟截短, 推 ctx prob
    test_ctx_files = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_ctx_files]
    n_test = len(seg_ids)
    print(f"[T1] test segments: {n_test}", file=sys.stderr)

    # 截短 + featurize
    X_te = np.array([
        featurize(simulate_truncated_context(np.load(p).astype(int), keep_chunks=keep))
        for p in test_ctx_files
    ])
    ctx_te = predict_ctx(clfs, X_te)
    print(f"[T1] ctx_te truncated@{keep}: shape={ctx_te.shape}", file=sys.stderr)

    # 3. 加载 wsp / hub / wsp_ms / e2v_ms / hub_ms test probs (audio 模型不受 context 影响)
    z3 = np.load("tools/runs/climb/orthofuse-3src-20260601-1607/fused_probs.npz")
    wsp_te, hub_te = z3["whisper_te"], z3["hubert_te"]

    wsp_ms_te = np.load("tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/probs.npz")["test"]
    e2v_ms_te = np.load("tools/runs/climb/e2v-bcaug-multiseed-20260602-1632/probs.npz")["test"]
    hub_ms_te = np.load("tools/runs/climb/hubert-bcaug-multiseed-20260602-1506/probs.npz")["test"]

    # 4. orthofuse-3src → NSOTA_07 → R4
    sota = make_sota_3src(ctx_te, wsp_te, hub_te)
    nsota_07 = softadd(sota, wsp_ms_te, 0.07, (1, 2, 3))
    r4 = softadd(softadd(nsota_07, e2v_ms_te, 0.03, (1, 2, 3)), hub_ms_te, 0.03, (1, 2, 3))

    # 5. 写 csv
    out_name = args.out_name or f"R4_keep{keep}_ctx{int(keep_sec)}s"
    out_dir = Path(f"submission/truncated-validation-20260604/{out_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.zeros((n_test, 5), dtype=int)
    for k in range(NUM):
        pred[:, k] = (r4[:, k] >= THR_VARF[k]).astype(int)
    df = pd.DataFrame({"segment_id": seg_ids})
    for c in SUBMIT_COLS:
        df[c] = pred[:, COL2K[c]]
    csv_path = out_dir / "pred_test1.csv"
    df.to_csv(csv_path, index=False)

    pos = {c: int(df[c].sum()) for c in SUBMIT_COLS}
    print(f"\n[T1] ✓ {csv_path}", file=sys.stderr)
    print(f"     pos: {pos}", file=sys.stderr)
    print(f"     R4 full (375 chunk) pos: C=975 NA=947 I=80 BC=20 T=523 (基准, 真分 0.7458)", file=sys.stderr)
    print(f"     截短到 {keep_sec:.1f}s 后 pos 差异 ↑↓ 即 R4 在该上下文长度下的预测变化", file=sys.stderr)

    # MANIFEST
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump({
            "name": out_name,
            "keep_chunks": keep,
            "keep_seconds": keep_sec,
            "description": f"R4 NSOTA_07 + e2v_ms 0.03 + hub_ms 0.03, 上下文截短到末 {keep} chunk ({keep_sec:.1f}s)",
            "pos": pos,
            "reference_full_r4": {"true_score": 0.745798, "pos": {"c":975,"na":947,"i":80,"bc":20,"t":523}},
            "note": "T1 公榜验证 D-26 复赛动态时长. wsp/hub/wsp_ms/e2v_ms/hub_ms test probs 不变 (audio 模型), 仅 ctx_te 截短重推.",
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
