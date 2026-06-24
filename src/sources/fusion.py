"""三源融合 — 初赛 orthofuse-3src per-class 策略 (真分 0.71755)。

来源: tools/runs/climb/orthofuse-3src-20260601-1607/cv_metrics.json + build_5submissions.py:96。
per-class 固定权重 (无在线搜索, 私有测试集直接套用):
  C  = ctx                              (whisper/hubert 对 C 无增量)
  T  = 0.7*wsp + 0.3*hub                (whisper_hubert_70 — T 靠音频)
  BC = ctx                              (BC 冻结上限 0.22, ctx 最好)
  I  = (ctx + wsp + hub) / 3            (三源等权)
  NA = ctx                              (NA 靠 ctx)
不带 env 软加权 (那是 e2v/Omni 多源叠加, 三源基线不需要)。
"""
from __future__ import annotations

import numpy as np


def fuse_orthofuse_3src(ctx_p: np.ndarray, wsp_p: np.ndarray, hub_p: np.ndarray) -> np.ndarray:
    """ctx × whisper × hubert per-class 正交融合 (初赛 SOTA 配方)。"""
    p = np.zeros_like(ctx_p)
    p[:, 0] = ctx_p[:, 0]                                   # C  = ctx
    p[:, 1] = 0.7 * wsp_p[:, 1] + 0.3 * hub_p[:, 1]         # T  = whisper_hubert_70
    p[:, 2] = ctx_p[:, 2]                                   # BC = ctx
    p[:, 3] = (ctx_p[:, 3] + wsp_p[:, 3] + hub_p[:, 3]) / 3  # I  = 三源等权
    p[:, 4] = ctx_p[:, 4]                                   # NA = ctx
    return p
