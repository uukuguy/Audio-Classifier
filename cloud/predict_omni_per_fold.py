"""5 fold Omni ckpt 各自推理 test → 5 个独立 test probs.

用户洞察 (2026-06-02): CV 5fold 默认 mean 把多样性抹平, 单 fold 推理保留各自特点.
NEW SOTA cand2 = SOTA + Omni_5fold_avg 软加 0.2 = 0.728524
本脚本: 5 个 fold ckpt 各自跑 test → 5 个独立 test probs → 当 5 个独立源融合.

输出: probs_perfold.npz 包含 5 × (1000, 5) test probs

Usage (云端):
  python3 cloud/predict_omni_per_fold.py
"""
from __future__ import annotations
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "cloud")
sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# 复用 train_omni_head 的 dataset / model / collate / processor
from train_omni_head import (  # noqa: E402
    OmniMultimodalDataset,
    OmniHeadLoRA,
    build_thinker_with_lora,
    pad_stack_collate,
    DEV,
)

RUN_DIR = Path("tools/runs/climb/omni-lora-20260602-1002")
CKPT_FILES = [RUN_DIR / f"fold{fi}.pt" for fi in range(5)]


def main():
    print(f"[per-fold] dev={DEV}, RUN_DIR={RUN_DIR}", file=sys.stderr)
    for ck in CKPT_FILES:
        assert ck.exists(), f"missing {ck}"
    print(f"[per-fold] 5 fold ckpt all present", file=sys.stderr)

    # 1. processor 一次
    print("[per-fold] loading processor...", file=sys.stderr)
    from transformers import Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained(
        "/root/.cache/manual_models/Qwen2.5-Omni-7B"
    )

    # 2. test ds
    print("[per-fold] building test dataset...", file=sys.stderr)
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    test_ds = OmniMultimodalDataset(test_ids, "test", slice_cap=1, bc_aug_n=0,
                                    processor=processor)
    print(f"[per-fold] test_ds N={len(test_ds)}", file=sys.stderr)

    # 3. ctx_dim 必须跟训练一致 (训练时 ctxfeat 出来 46d)
    ctx_dim = test_ds[0]["ctx"].shape[0]
    print(f"[per-fold] ctx_dim={ctx_dim}", file=sys.stderr)

    # 4. 5 fold 各自推理
    from torch.utils.data import DataLoader
    per_fold_probs = np.zeros((5, len(test_ds), 5), dtype=np.float32)
    t_total = time.time()
    for fi, ckpt_path in enumerate(CKPT_FILES):
        print(f"\n[per-fold] === fold {fi+1}/5 loading thinker + ckpt {ckpt_path.name} ===",
              file=sys.stderr)
        t_fold = time.time()
        thinker = build_thinker_with_lora()
        model = OmniHeadLoRA(ctx_dim=ctx_dim, thinker=thinker)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[per-fold] loaded {len(sd)} keys, missing={len(missing)} unexpected={len(unexpected)}",
              file=sys.stderr)
        model.to(DEV)
        model.eval()

        loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                            collate_fn=pad_stack_collate, num_workers=0, pin_memory=True)
        fold_probs = []
        with torch.no_grad():
            for bi, (proc_out, ctx, _) in enumerate(loader):
                proc_out = {k: v.to(DEV, non_blocking=True) for k, v in proc_out.items()}
                ctx = ctx.to(DEV, non_blocking=True)
                p = torch.sigmoid(model(proc_out, ctx))
                fold_probs.append(p.cpu().numpy())
                if bi == 0 or (bi + 1) % 50 == 0:
                    print(f"[per-fold]   fold {fi+1} batch {bi+1}", file=sys.stderr)
        per_fold_probs[fi] = np.concatenate(fold_probs, axis=0)
        del model, thinker
        torch.cuda.empty_cache()
        print(f"[per-fold] fold {fi+1} done in {(time.time()-t_fold)/60:.1f}min", file=sys.stderr)

    # 5. save
    print(f"\n[per-fold] total {(time.time()-t_total)/60:.1f}min", file=sys.stderr)
    out_path = RUN_DIR / "probs_perfold.npz"
    np.savez_compressed(out_path, per_fold=per_fold_probs)
    print(f"[per-fold] saved {out_path}: shape {per_fold_probs.shape}", file=sys.stderr)

    # 6. fold 间差异统计 (看是否真有多样性)
    print("\n=== Fold 间 test probs 差异 ===", file=sys.stderr)
    mean_probs = per_fold_probs.mean(0)
    LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
    for k in range(5):
        per_fold_k = per_fold_probs[:, :, k]  # (5, 1000)
        std_across_folds = per_fold_k.std(axis=0).mean()  # 每段在 5 fold 上的 std, 平均
        max_diff = (per_fold_k.max(0) - per_fold_k.min(0)).mean()
        print(f"{LAB[k]}: mean std={std_across_folds:.4f}, mean max-min={max_diff:.4f}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
