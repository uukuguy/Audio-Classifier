"""T3 cross-context 内部对照实验 (0 push, 本机能跑).

模拟复赛动态时长场景: 上下文 ∈ (0, 30]s 任意 → 测 R4/R5/SOTA-3src 等模型退化幅度.

核心思路:
  1. 公榜测试集 1 有 1000 段, 每段 context 是 375 chunk (30s)
  2. 用 simulate_truncated_context() 把 context 截短到 {12, 25, 63, 125, 250, 375}
     (= 1s, 2s, 5s, 10s, 20s, 30s) chunk
  3. 用同一个 ctx 模型 (重训或加载) 跑各截短挡的 test probs
  4. SSL_ms / Omni 等头是吃 audio 的, 不直接受 context 截短影响, 但 R4 是 ctx-orthofuse + SSL_ms 软加
  5. 评估: 各截短挡下 R4 / R5 / SOTA-3src / R1 在公榜测试集上的 **估算 macro F1**

注意:
  - 公榜测试集没 ground truth, 但**可以用 train 集做模拟**:
    - 用 train conv 切 cap1 (每通首窗) 当"测试集 1"
    - 截短 context → featurize → ctx 模型 (现训) → SSL_ms 头用现有 probs
    - 算 macro F1 vs 真标签 (train 有 Y)
  - 这是"内部 cross-context 验证", 是答辩金料 (D-26 T3)

实施分两步:
  Step A: 重训 ctx 1 base (LGBM v1) 在 train 集上, 拿到 model.pkl + train_cap1 test probs
  Step B: 截短 6 挡 × R4 / R5 / SOTA-3src 三 base = 18 个 macro F1 数据点
"""
from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from dynamic_ctx_utils import simulate_truncated_context, CTX_FULL  # noqa: E402

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
NUM = 5
TGT = 25
SEED = 42

# 截短挡 (单位 chunk = 80ms)
KEEP_CHUNKS_LIST = [12, 25, 63, 125, 250, 375]  # 1s, 2s, 5s, 10s, 20s, 30s
KEEP_LABEL = {12: "1s", 25: "2s", 63: "5s", 125: "10s", 250: "20s", 375: "30s"}


def featurize(ctx: np.ndarray) -> np.ndarray:
    """跟 cycle_context.featurize 完全一致, 复制过来避免 import 路径问题."""
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


def build_eval_set(
    eval_conv_ids: list[str],
    label_files: dict,
    cap_n: int = 5,
    stride: int = 50,
) -> tuple:
    """从 conv_ids 构建评估集 (每通 cap_n 个窗口, stride 控制采样密度).

    cap1 (cap_n=1) BC 样本太稀疏 (369 通 cap1 上只 9 个 BC 正例). 用 cap5/stride=50
    可以取更多样本同时保持窗口多样性 (cap0/1/2/3/4 各不同位置).

    Returns:
        ctx_375_list: [N x (375,)] 原始 375 chunk context (每个窗口的过去 30s)
        Y_eval: (N, 5) 多标签 0/1 (每个窗口的未来 2s)
        conv_ids: N 个 (cid, e) 元组 (对应顺序, 调试用)
    """
    ctx_list, Y_list, ids = [], [], []
    for cid in eval_conv_ids:
        a = np.load(label_files[cid]).astype(int)
        if a.shape[0] < CTX_FULL + TGT:
            continue
        # cap_n 个窗口起点: 从 CTX_FULL 起每 stride 步采一个, 最多 cap_n 个
        windows = []
        for e in range(CTX_FULL, a.shape[0] - TGT + 1, stride):
            windows.append(e)
            if len(windows) >= cap_n:
                break
        for e in windows:
            ctx = a[e - CTX_FULL:e]
            fut = set(int(x) for x in a[e:e + TGT])
            y = [1 if k in fut else 0 for k in range(NUM)]
            ctx_list.append(ctx)
            Y_list.append(y)
            ids.append((cid, e))
    return ctx_list, np.array(Y_list, dtype=int), ids


def build_train_windows(train_ids: list[str], label_files: dict, stride: int = 5):
    """train 滑窗 — 跟 cycle_context.build_train 一致, 但不限 STRIDE."""
    X, Y = [], []
    for cid in train_ids:
        a = np.load(label_files[cid]).astype(int)
        for e in range(CTX_FULL, a.shape[0] - TGT + 1, stride):
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(featurize(a[e - CTX_FULL:e]))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
    return np.array(X), np.array(Y, dtype=int)


def fit_5_lgbm(X, Y):
    """5 个 binary LGBM (一类一个), 跟 cycle_context.fit_lgbm 同."""
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


def predict_ctx(clfs, X) -> np.ndarray:
    """5 binary LGBM → (N, 5) prob."""
    P = np.zeros((len(X), NUM), dtype=np.float32)
    for k in range(NUM):
        P[:, k] = clfs[k].predict_proba(X)[:, 1]
    return P


def macro_f1_from_probs(P, Y, thr=THR_VARF) -> tuple[float, list]:
    f1s = [f1_score(Y[:, k], (P[:, k] >= thr[k]).astype(int), zero_division=0) for k in range(NUM)]
    return float(np.mean(f1s)), f1s


def main():
    print("[T3] === cross-context 内部对照 (D-26 复赛动态时长应对) ===", file=sys.stderr)
    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    random.Random(SEED).shuffle(conv_ids)
    n_eval = max(20, int(len(conv_ids) * 0.15))
    eval_ids, train_ids = conv_ids[:n_eval], conv_ids[n_eval:]
    print(f"[T3] split: train={len(train_ids)} eval={len(eval_ids)}", file=sys.stderr)

    # Step A: 重训 ctx (LGBM v1) on train_ids
    print("[T3] step A: train ctx LGBM v1 on train_ids ...", file=sys.stderr)
    Xtr, Ytr = build_train_windows(train_ids, label_files, stride=5)
    print(f"[T3]   train windows: {Xtr.shape}", file=sys.stderr)
    clfs = fit_5_lgbm(Xtr, Ytr)
    print("[T3]   ctx model trained", file=sys.stderr)

    # Step B: 在 eval_ids 上构评估集 (cap5/stride50, 每通最多 5 窗口)
    ctx_375_list, Y_eval, eval_conv_ids = build_eval_set(
        eval_ids, label_files, cap_n=5, stride=50
    )
    n_eval_samples = len(ctx_375_list)
    print(f"[T3] step B: eval cap5 windows = {n_eval_samples}", file=sys.stderr)
    _ = eval_conv_ids  # noqa

    # Step C: 各截短挡跑 ctx 模型推 prob → 算 macro F1
    print("\n[T3] === ctx-only macro F1 vs context length ===")
    print(f"  Y_eval pos dist: " + " ".join(f"{LABELS[k]}={Y_eval[:,k].sum()}" for k in range(NUM)))
    print(f"  原始 375 chunk Y_pos rate: " + " ".join(f"{LABELS[k]}={Y_eval[:,k].mean():.3f}" for k in range(NUM)))

    # 先算 base (375 chunk = full) 作 anchor
    X_base = np.array([featurize(ctx) for ctx in ctx_375_list])
    P_base = predict_ctx(clfs, X_base)
    base_macro, base_per = macro_f1_from_probs(P_base, Y_eval, THR_VARF)
    print(f"\n  base (full 30s): macro={base_macro:.4f}  per_class={base_per}")

    results = {}
    print(f"\n  {'ctx_chunks':>11s} {'ctx_seconds':>12s} {'macro_F1':>9s}  {'C':>6s} {'T':>6s} {'BC':>6s} {'I':>6s} {'NA':>6s}  ΔvsFull")
    print("  " + "-" * 92)
    for keep in KEEP_CHUNKS_LIST:
        # 模拟截短: 每个 ctx_375 截到末 keep, pad 回 375 (left NA pad)
        X_sim = np.array([
            featurize(simulate_truncated_context(ctx, keep_chunks=keep, mode="pad_na_left"))
            for ctx in ctx_375_list
        ])
        P_sim = predict_ctx(clfs, X_sim)
        macro, per = macro_f1_from_probs(P_sim, Y_eval, THR_VARF)
        delta = macro - base_macro
        ctx_sec = keep * 80 / 1000
        per_str = " ".join(f"{p:.3f}" for p in per)
        marker = " ← base" if keep == 375 else ""
        print(f"  {keep:>11d} {ctx_sec:>11.2f}s {macro:>9.4f}  {per_str}  {delta:+.4f}{marker}")
        results[keep] = {"chunks": keep, "seconds": ctx_sec, "macro_f1": macro, "per_class": per, "delta_vs_full": delta}

    # Step D: 退化分析 + 关键结论
    print("\n[T3] === 退化分析 ===")
    full_macro = results[375]["macro_f1"]
    print(f"  Full 30s context: macro F1 = {full_macro:.4f}")
    for keep in KEEP_CHUNKS_LIST:
        if keep == 375: continue
        sec = results[keep]["seconds"]
        m = results[keep]["macro_f1"]
        d = m - full_macro
        pct = d / full_macro * 100
        print(f"  截到 {sec:.1f}s ({keep:3d} chunk): macro = {m:.4f}  Δ = {d:+.4f} ({pct:+.1f}%)")

    # 落盘
    out_path = Path("tools/runs/climb/dynamic-ctx-eval-20260604/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "_note": "T3 cross-context 内部对照 (D-26 复赛动态时长). ctx-only LGBM, cap1 评估集.",
            "n_eval_samples": n_eval_samples,
            "base_full_macro": full_macro,
            "results_by_keep": results,
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\n[T3] 落盘: {out_path}")


if __name__ == "__main__":
    main()
