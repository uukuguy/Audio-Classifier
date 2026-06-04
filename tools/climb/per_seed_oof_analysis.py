"""分析 multi-seed 各 seed 的 OOF cap1, 找最强 seed (vs mean).

H-D22-3 衍生: multi-seed mean 涨 +0.007~+0.020, 但单 seed 可能更强.
找最强 seed 替 mean 可能再 +0.003~0.005.

Usage:
  python3 tools/climb/per_seed_oof_analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}


def cap1_idx_simple(G):
    seen, idx = set(), []
    for i, g in enumerate(G):
        if int(g) not in seen:
            idx.append(i); seen.add(int(g))
    return np.array(idx)


def macro_f1(probs, Y):
    pred = np.zeros_like(probs, dtype=int)
    for k in range(5):
        pred[:, k] = (probs[:, k] >= THR_VARF[k]).astype(int)
    per = [f1_score(Y[:, k], pred[:, k], zero_division=0) for k in range(5)]
    return float(np.mean(per)), per


def main():
    # 加载 ctx Y/G align (cap1 同 sweep)
    zc = np.load("tools/runs/climb/_stack_cache_s40.npz")
    Yc = zc["Y"].astype(int); Gc = zc["G"]
    cap1_c = cap1_idx_simple(Gc)
    Y_cap1 = Yc[cap1_c]
    print(f"ctx cap1: N={len(cap1_c)}\n")

    for name in ["w2v2", "e2v", "whisper"]:
        path_glob = list(Path("tools/runs/climb").glob(f"{name}-bcaug-multiseed-*/per_seed_oof.npz"))
        if not path_glob:
            continue
        path = path_glob[-1]
        ms_path = path.parent / "probs.npz"

        z_seed = np.load(path)
        z_full = np.load(ms_path)
        per_seed = z_seed["per_seed"]  # (3, 55877, 5)
        seeds = z_seed["seeds"]
        G_head = z_full["G"]
        order_head = z_full["order"]
        Y_head = z_full["Y"].astype(int)

        # cap1 mask (跟 sweep_softadd_oof 同)
        cap1_mask = order_head == 0
        # 验 align
        Y_head_cap1 = Y_head[cap1_mask]
        if len(Y_head_cap1) != len(Y_cap1):
            print(f"{name}: SKIP (N 不齐 {len(Y_head_cap1)} vs {len(Y_cap1)})")
            continue
        if not (Y_head_cap1 == Y_cap1).all():
            print(f"{name}: SKIP (Y 不齐)")
            continue

        print(f"=== {name} (multi-seed: {seeds}) ===")
        # 注意: per_seed shape (3, 55877, 5), 不是 (3, full_N, 5)
        # 55877 = bcaug 子集. cap1 子集 vs ms 自己的 N 可能不同.
        # 看看是哪个子集
        if per_seed.shape[1] != len(order_head):
            print(f"  per_seed N={per_seed.shape[1]} vs head N={len(order_head)} → bcaug 子集")
            # 关键: per_seed 是 bcaug 子集. 看子集 G/order
            # bcaug 子集应该是 train 部分, 不含 ctx cap1 标准. 这个分析对不上.
            # 改成: 看 head probs.npz 的 oof 跟 Y align (这是 mean over seeds 后的)
            print(f"  无法直接算 per-seed cap1 (per_seed=bcaug 训练子集, 非 cap1)")
            print(f"  → 退路: 看 head OOF mean (全 seed 平均) cap1")
            oof_mean = z_full["oof"][cap1_mask]
            macro, per = macro_f1(oof_mean, Y_cap1)
            print(f"  全 seed mean OOF cap1: macro={macro:.4f} | " + " ".join(f"{LAB[k]}={per[k]:.3f}" for k in range(5)))
            print()
            continue

        # 各 seed cap1
        per_seed_cap1 = per_seed[:, cap1_mask, :]
        print(f"  per_seed shape after cap1 mask: {per_seed_cap1.shape}")
        for i, sd in enumerate(seeds):
            macro, per = macro_f1(per_seed_cap1[i], Y_cap1)
            print(f"  seed={sd}: cap1={macro:.4f} | " + " ".join(f"{LAB[k]}={per[k]:.3f}" for k in range(5)))
        # mean
        mean_oof = per_seed_cap1.mean(axis=0)
        macro, per = macro_f1(mean_oof, Y_cap1)
        print(f"  mean: cap1={macro:.4f} | " + " ".join(f"{LAB[k]}={per[k]:.3f}" for k in range(5)))
        # max per sample
        max_oof = per_seed_cap1.max(axis=0)
        macro, per = macro_f1(max_oof, Y_cap1)
        print(f"  max (per-sample): cap1={macro:.4f} | " + " ".join(f"{LAB[k]}={per[k]:.3f}" for k in range(5)))
        print()


if __name__ == "__main__":
    main()
