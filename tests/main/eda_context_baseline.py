"""EDA: 纯上下文标签(context labels)基线能到多少 Macro-F1?

量化方案 C 的天花板 —— 仅用过去 375 chunk 的标签序列(不碰音频/文本),
手工特征 + LightGBM 多标签, 按会话划分验证, 逐类阈值调优。
这给后续 bake-off 一个"音频/文本必须超过"的地板分。

Run: python tests/main/eda_context_baseline.py
"""
from __future__ import annotations

import glob
import random
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM = 5
CTX, TGT, STRIDE = 375, 25, 5
SEED = 42


def featurize(ctx: np.ndarray) -> np.ndarray:
    """从 375-chunk 上下文标签序列抽手工特征(模仿 test context 可用信息)。"""
    oh = np.eye(NUM)[ctx]  # [L,5]
    feats = []
    # 多窗口 ratio
    for w in (10, 25, 50, 100, 200, 375):
        feats.extend(oh[-w:].mean(axis=0))
    # 最后 5 个原始标签
    for i in range(1, 6):
        feats.append(ctx[-i] if len(ctx) >= i else -1)
    # 距上次各类出现的归一化距离
    L = len(ctx)
    for k in range(NUM):
        pos = np.where(ctx == k)[0]
        feats.append((L - 1 - pos[-1]) / L if len(pos) else 1.0)
    # 各类出现次数(转换频率代理)
    for k in range(NUM):
        feats.append((ctx == k).sum() / L)
    # 标签切换次数(对话活跃度)
    feats.append((ctx[1:] != ctx[:-1]).mean())
    return np.array(feats, dtype=np.float32)


def build(conv_ids, label_files):
    X, Y = [], []
    for cid in conv_ids:
        a = np.load(label_files[cid])
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE):
            ctx = a[e - CTX:e].astype(int)
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(featurize(ctx))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(X), np.array(Y)


def tune_threshold(y, p):
    """逐类找最大化 F1 的阈值。"""
    best_t, best_f = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 19):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def main():
    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    random.Random(SEED).shuffle(conv_ids)
    n_val = max(1, int(len(conv_ids) * 0.15))
    val_ids, train_ids = conv_ids[:n_val], conv_ids[n_val:]
    print(f"conversations: train={len(train_ids)} valid={len(val_ids)}")

    Xtr, Ytr = build(train_ids, label_files)
    Xva, Yva = build(val_ids, label_files)
    print(f"windows: train={len(Xtr)} valid={len(Xva)}, feat_dim={Xtr.shape[1]}")

    f1_at_05, f1_tuned = [], []
    print(f"\n{'class':6s} {'pos%':>6s} {'F1@0.5':>8s} {'F1_tuned':>9s} {'thr':>5s}")
    for k in range(NUM):
        ytr, yva = Ytr[:, k], Yva[:, k]
        if ytr.sum() == 0:
            continue
        spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
        clf = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED,
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        f05 = f1_score(yva, (p >= 0.5).astype(int), zero_division=0)
        thr, ft = tune_threshold(yva, p)
        f1_at_05.append(f05)
        f1_tuned.append(ft)
        print(f"{LABELS[k]:6s} {100*yva.mean():6.1f} {f05:8.3f} {ft:9.3f} {thr:5.2f}")

    print(f"\n{'='*50}")
    print(f"Macro-F1 @0.5   = {np.mean(f1_at_05):.4f}")
    print(f"Macro-F1 tuned  = {np.mean(f1_tuned):.4f}  <- 纯上下文标签地板分")
    print(f"{'='*50}")
    print("对照排行榜: #10=0.7192  #3=0.7357  #1=0.7475")
    print("注: 这是 conv-split valid 上的乐观估计(滑窗而非独立30s切片), 线上会更低")


if __name__ == "__main__":
    main()
