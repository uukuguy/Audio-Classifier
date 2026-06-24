"""复赛端到端推理 — 共享常量 / 设备选择 / IO。

三源基线 (ctx + whisper + hubert)。融合策略来自初赛 orthofuse-3src (真分 0.71755)。
评测环境有 GPU (官方 baseline 支持 CUDA) → 走 cuda fp16; 无 GPU fallback cpu fp32。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# === 路径 (评测环境无网络, 全部 Docker 内 /app 子路径; 本机测试传 env 覆盖) ===
TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/xydata"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", "/app/submit/submit.csv"))
MODELS = Path(os.environ.get("MODELS", "/app/models"))

# === 设备 (GPU 优先 + CPU fallback, v2 核心成果保留) ===
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32

# === 标签 / 提交列 (官方: segment_id,c,na,i,bc,t) ===
LAB = ["C", "T", "BC", "I", "NA"]          # 概率数组列序 (训练时的顺序)
SUBMIT = ["c", "na", "i", "bc", "t"]       # 官方 CSV 列序
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}  # 提交列 → 概率数组列下标

# === 阈值 (初赛 orthofuse-3src SOTA THR_VARF, 真分 0.71755 验证过) ===
# 注意: 这套是【融合后】阈值, 不同于 ctx-only 阶段的 thresholds.json。
# C/NA 低阈 (大类恒正安全), BC/I/T 中高阈。违此值=违阈值铁律 (见 CLAUDE.md)。
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}  # {C,T,BC,I,NA}

# === 音频 / 上下文常量 ===
SR16, CTX_SEC, DS_FRAMES = 16000, 8, 80    # 16k 重采样 / 末 8s / 池化到 80 帧
NUM = 5
NA_LABEL = 4


def write_submit(seg_ids: list[str], probs: np.ndarray) -> dict[str, int]:
    """按官方格式写 submit.csv (硬 0/1, THR_VARF 阈值), 返回各类正例数。"""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(seg_ids):
            vals = [str(int(probs[i, COL2K[c]] >= THR_VARF[COL2K[c]])) for c in SUBMIT]
            f.write(f"{sid}," + ",".join(vals) + "\n")
    return {c: int((probs[:, COL2K[c]] >= THR_VARF[COL2K[c]]).sum()) for c in SUBMIT}


def write_empty_submit() -> None:
    """无测试数据时写空 header (评测兜底, 不崩)。"""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
