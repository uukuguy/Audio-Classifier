"""动态上下文长度适配工具 (T1, D-26 应对).

复赛测试集 2 上下文 ∈ (0, 30]s 任意时长. 提供 3 个工具:

1. `normalize_ctx_to_375(ctx, mode)` — 任意长度 → 强制 375 chunk (复赛 infer 入口)
2. `simulate_truncated_context(ctx_375, keep_chunks)` — 截短模拟 (公榜验证用)
3. `featurize_variable_length(ctx)` — 长度无关的鲁棒特征 (策略 3 fallback)

参考: D-26 决策, finals/FINAL-PUSH-TASKS.md T1-T2

使用约定:
  CTX = 375 chunk × 80ms = 30s
  NA = label 4 (pad 用 NA 表示"没观测过")
"""
from __future__ import annotations

import numpy as np

CTX_FULL = 375  # 30s × 1000/80 = 375
NA_LABEL = 4    # silence


def normalize_ctx_to_375(
    ctx: np.ndarray, mode: str = "pad_na_left"
) -> np.ndarray:
    """任意长度上下文 → 强制 CTX_FULL=375 chunk.

    Args:
        ctx: 形状 (N,) 的 label 序列, N ∈ [1, 375+] (复赛测试集 2 可能任意).
        mode:
            "pad_na_left": N<375 时左侧 pad NA (假设"早期未观测"). [默认]
            "pad_loop":   N<375 时左侧循环 padding (假设"模式重复").
            "truncate_only": N<375 不 pad (返回原样), N>375 截末 375.

    Returns:
        形状 (375,) 或 (N,) (truncate_only 短情况) 的 array.

    Note:
        复赛测试集 2 上下文 (0, 30]s, 即 N ∈ [1, 375] (≤ 30s) — 主要短不长.
        long case (N>375) 仅作为 safety 处理, 实际不会出现.
    """
    n = len(ctx)
    if n == CTX_FULL:
        return ctx.astype(np.int32)
    if n > CTX_FULL:
        return ctx[-CTX_FULL:].astype(np.int32)  # 截末 375 (最近因果)

    # n < CTX_FULL: 需要 pad
    if mode == "pad_na_left":
        pad = np.full(CTX_FULL - n, NA_LABEL, dtype=np.int32)
        return np.concatenate([pad, ctx.astype(np.int32)])
    elif mode == "pad_loop":
        # 循环填充 (前面用 ctx 自身循环填)
        reps = CTX_FULL // n + 1
        padded = np.tile(ctx, reps)
        return padded[-CTX_FULL:].astype(np.int32)
    elif mode == "truncate_only":
        return ctx.astype(np.int32)
    else:
        raise ValueError(f"unknown mode: {mode}")


def simulate_truncated_context(
    ctx_375: np.ndarray, keep_chunks: int, mode: str = "pad_na_left"
) -> np.ndarray:
    """模拟复赛截短: 拿 375 chunk 上下文, 模拟"只看到末 keep_chunks chunk"的复赛场景.

    用于公榜验证: 把测试集 1 的 context (375 chunk) 截短到 keep_chunks
    重 pad 回 375, 看真分如何退化.

    Args:
        ctx_375: 形状 (375,) 的 label 序列 (公榜测试集 1).
        keep_chunks: 保留末 keep_chunks 个 chunk (模拟复赛短上下文).
            keep_chunks=375: 等于原始 (无截短)
            keep_chunks=250: 模拟 20s 上下文
            keep_chunks=125: 模拟 10s 上下文
            keep_chunks=63:  模拟 5s 上下文
            keep_chunks=12:  模拟 1s 上下文
        mode: pad 模式, 转给 normalize_ctx_to_375

    Returns:
        形状 (375,) 的 array (短上下文截末段 + pad 回 375).

    Example:
        >>> ctx = np.load("data/test/context/seg_0001.npy")  # (375,)
        >>> # 模拟"只有末 5s 上下文" → pad NA 回 375 → 让模型预测
        >>> ctx_simulated = simulate_truncated_context(ctx, keep_chunks=63)
        >>> # 用 ctx_simulated 推理, 看 macro F1 vs 原 ctx 的差距
    """
    if keep_chunks >= len(ctx_375):
        return ctx_375.astype(np.int32)
    # 取末 keep_chunks (最近因果 keep_chunks chunks)
    short_ctx = ctx_375[-keep_chunks:]
    # pad 回 375
    return normalize_ctx_to_375(short_ctx, mode=mode)


def featurize_variable_length(ctx: np.ndarray) -> np.ndarray:
    """策略 3 fallback: 长度无关的鲁棒特征 (放弃 _w375 全局, 只用短窗 ≤100 chunk).

    跟 cycle_context.featurize() 区别:
      - 短窗 (_w10 ... _w100) 保留, 这些在任意长度上都稳
      - 长窗 (_w200, _w375) 改成"末段 min(len, 200)" 跟 "末段 min(len, 375)"
      - 全局统计 (类别分布 / 切换率) 改为 max(len, 50) 以下时返回 0 (无意义)

    用于策略 3: 重训 ctx 4 base, 完全去掉 30s 假设. 单点真分预估损失 0.02-0.04.

    Args:
        ctx: 形状 (N,) 的 label 序列, N ∈ [1, 任意]

    Returns:
        长度无关的 feature vector (固定维度 = 与原 featurize 相同).
    """
    NUM = 5
    oh = np.eye(NUM)[ctx]
    L = len(ctx)
    feats = []
    # 滚动窗 (跟原 featurize 相同, 但 _w200/_w375 自动截到可用长度)
    for w in (10, 25, 50, 100, 200, 375):
        effective_w = min(w, L)
        if effective_w == 0:
            feats.extend([0.0] * NUM)
        else:
            feats.extend(oh[-effective_w:].mean(axis=0))
    # 末段 5 chunk (跟原 featurize 同)
    for i in range(1, 6):
        feats.append(int(ctx[-i]) if L >= i else -1)
    # 末次类别距离 (用 L 归一化, 短上下文里仍有意义)
    for k in range(NUM):
        pos = np.where(ctx == k)[0]
        feats.append(float((L - 1 - pos[-1]) / max(L, 1)) if len(pos) else 1.0)
    # 类别分布 (短上下文里波动大但仍有信息)
    for k in range(NUM):
        feats.append(float((ctx == k).sum() / max(L, 1)))
    # 切换率 (任意长度都稳)
    feats.append(float((ctx[1:] != ctx[:-1]).mean()) if L > 1 else 0.0)
    return np.array(feats, dtype=np.float32)


if __name__ == "__main__":
    # 单元测试
    rng = np.random.RandomState(42)
    ctx_375 = rng.randint(0, 5, size=375).astype(np.int32)

    # 1. normalize
    assert normalize_ctx_to_375(ctx_375).shape == (375,)
    assert normalize_ctx_to_375(ctx_375[:100], mode="pad_na_left").shape == (375,)
    assert (normalize_ctx_to_375(ctx_375[:100], mode="pad_na_left")[:275] == NA_LABEL).all()
    assert normalize_ctx_to_375(ctx_375[:50], mode="pad_loop").shape == (375,)
    print("✓ normalize_ctx_to_375")

    # 2. simulate truncated
    sim = simulate_truncated_context(ctx_375, keep_chunks=100)
    assert sim.shape == (375,)
    assert (sim[:275] == NA_LABEL).all()  # 前 275 是 pad
    assert (sim[275:] == ctx_375[-100:]).all()  # 后 100 是原末段
    print("✓ simulate_truncated_context")

    # 3. featurize variable length
    feat_full = featurize_variable_length(ctx_375)
    feat_short = featurize_variable_length(ctx_375[-50:])
    assert feat_full.shape == feat_short.shape
    print(f"✓ featurize_variable_length: dim={feat_full.shape[0]}")

    print("\nAll unit tests passed.")
