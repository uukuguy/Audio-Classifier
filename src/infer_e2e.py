"""复赛端到端推理主入口 — 三源基线 (ctx + whisper + hubert)。

run.sh → python -m src.infer_e2e。挂载私有测试集2 (/xydata, 5-30s 不定长) →
现场推理三源 → orthofuse-3src 融合 → /app/submit/submit.csv。
GPU 优先 (评测有 GPU) + CPU fallback。融合/阈值 = 初赛 orthofuse-3src SOTA 0.71755 配方。

(初赛 src/infer.py = ctx-only dual-model 骨架, D-30 已证伪, 保留不动; 本文件是复赛端到端新入口。)
"""
from __future__ import annotations

import sys
import time
import traceback

import torch

from src.common import DEV, MODELS, OUTPUT_CSV, TEST_ROOT, write_empty_submit, write_submit
from src.sources.context import infer_context
from src.sources.fusion import fuse_orthofuse_3src
from src.sources.ssl_encoder import infer_ssl


def main() -> None:
    t0 = time.time()
    print(f"[infer] DEV={DEV} CUDA={torch.cuda.is_available()} TEST_ROOT={TEST_ROOT}", flush=True)
    try:
        print("[infer] 1/4 context LGBM...", flush=True)
        ctx_probs, seg_ids = infer_context()
        print(f"  ctx {ctx_probs.shape}", flush=True)
        if len(seg_ids) == 0:
            print("[infer] 无测试数据, 写空 submit.csv", flush=True)
            write_empty_submit()
            return

        print("[infer] 2/4 whisper...", flush=True)
        wsp_probs = infer_ssl("whisper", MODELS / "wsp_head")
        print(f"  wsp {wsp_probs.shape}", flush=True)

        print("[infer] 3/4 hubert...", flush=True)
        hub_probs = infer_ssl("hubert", MODELS / "hub_head")
        print(f"  hub {hub_probs.shape}", flush=True)

        n = len(seg_ids)
        print("[infer] 4/4 orthofuse-3src...", flush=True)
        fused = fuse_orthofuse_3src(ctx_probs, wsp_probs[:n], hub_probs[:n])

        pos = write_submit(seg_ids, fused)
        print(f"[infer] DONE {OUTPUT_CSV} n={n} pos={pos} elapsed={time.time()-t0:.1f}s", flush=True)
    except Exception:  # noqa: BLE001 — 任何异常都写空 csv 兜底 (评测不崩)
        traceback.print_exc()
        write_empty_submit()
        sys.exit(1)


if __name__ == "__main__":
    main()
