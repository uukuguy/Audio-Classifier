"""源 1: Context LGBM (变长自适应 featurize)。

逐类 5 个 LGBM (c/t/bc/i/na)。featurize_variable_length 处理任意长度 ctx
(复赛 5-30s 不定长, 短上下文不退化 — v1→v2 +0.028 的来源, 初赛 D-26~D-31 验证)。
"""
from __future__ import annotations

import glob
from pathlib import Path

import joblib  # 加载我方自训 LGBM ckpt (本机 git source of truth, 非外部数据 — pickle 安全)
import numpy as np

from src.common import LAB, MODELS, NA_LABEL, NUM, TEST_ROOT

CTX_FULL = 375  # 30s × 1000/80 = 375 chunk (训练时 ctx 满长)


def normalize_ctx_to_375(ctx: np.ndarray) -> np.ndarray:
    """任意长度 ctx → 375 chunk (mode=pad_na_left)。

    ★ 复赛 5-30s 不定长必须 normalize: 训练时变长 ctx = 左 pad NA 到 375
    (gen_variants.py: keep + pad NA), featurize 的位置/占比特征 (分母 L) 假设 L=375。
    推理给原始短 ctx (L=真实长度) → 这些特征系统性错位 → ctx LGBM 崩 (实测 BC +0.165/I +0.159)。
    内联自 tools/climb/dynamic_ctx_utils.py (镜像不带 tools/)。
    """
    n = len(ctx)
    if n == CTX_FULL:
        return ctx.astype(np.int32)
    if n > CTX_FULL:
        return ctx[-CTX_FULL:].astype(np.int32)  # 截末 375 (最近因果)
    pad = np.full(CTX_FULL - n, NA_LABEL, dtype=np.int32)  # 左 pad NA (早期未观测)
    return np.concatenate([pad, ctx.astype(np.int32)])


def featurize(ctx: np.ndarray) -> np.ndarray:
    """featurize_variable_length — 任意长度 ctx 自适应, 短上下文不退化。

    L_eff = 非 NA 有效长度; 6 个滚动窗 one-hot 均值 + 末 5 标签 + 各类最近距离 +
    各类占比 + transition 率。共 46 维 (feature_spec.json)。
    """
    L_eff = int((ctx != NA_LABEL).sum())
    if L_eff == 0:
        L_eff = len(ctx)
    oh = np.eye(NUM)[ctx]
    feats: list = []
    for w in (10, 25, 50, 100, 200, 375):
        eff_w = min(w, L_eff)
        feats.extend(oh[-eff_w:].mean(axis=0))
    for i in range(1, 6):
        feats.append(ctx[-i] if len(ctx) >= i else -1)
    L = len(ctx)
    for k in range(NUM):
        pos = np.where(ctx == k)[0]
        feats.append((L - 1 - pos[-1]) / max(L, 1) if len(pos) else 1.0)
    for k in range(NUM):
        feats.append((ctx == k).sum() / max(L, 1))
    feats.append((ctx[1:] != ctx[:-1]).mean() if L > 1 else 0.0)
    return np.array(feats, dtype=np.float32)


def infer_context() -> tuple[np.ndarray, list[str]]:
    """加载 5 个 LGBM, 对 test context 逐类预测概率。返回 (probs[N,5], seg_ids)。"""
    ctx_dir = MODELS / "ctx_only"
    clfs = {k: joblib.load(ctx_dir / f"lgbm_{LAB[k].lower()}.joblib") for k in range(NUM)}

    test_files = sorted(glob.glob(str(TEST_ROOT / "context/*.npy")))
    if not test_files:
        print(f"WARNING: no test context at {TEST_ROOT}/context/", flush=True)
        return np.zeros((0, NUM), dtype=np.float32), []

    seg_ids = [Path(p).stem for p in test_files]
    Xte = np.array([featurize(normalize_ctx_to_375(np.load(p).astype(int))) for p in test_files])
    if Xte.ndim == 1:
        Xte = Xte.reshape(1, -1)

    probs = np.zeros((len(Xte), NUM), dtype=np.float32)
    for k in range(NUM):
        try:
            probs[:, k] = clfs[k].predict_proba(Xte)[:, 1]
        except Exception as e:  # noqa: BLE001
            print(f"LGBM predict failed for {LAB[k]}: {e}, raw fallback", flush=True)
            probs[:, k] = clfs[k].predict(Xte, raw_score=True)
    return probs, seg_ids
