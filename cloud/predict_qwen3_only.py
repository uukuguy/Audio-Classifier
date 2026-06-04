"""独立 predict_test 脚本: 用已存的 5 fold ckpt 跑 inference + 落 probs.npz.

用于补救 multi-seed 训练 train 完但 predict_test OOM 死的情况.
不重训, 只跑预测.

Usage:
  QWEN_DIR=/root/.cache/manual_models/Qwen3-1.7B python cloud/predict_qwen3_only.py \\
    --run-dir tools/runs/climb/qwen3-17b-head-ms-seed42-20260603-1059
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# 复用 train_qwen3_head 的所有类和函数
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
import train_qwen3_head as t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-cap", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    # 初始化 HIDDEN_SIZE
    t.QWEN_DIM = t._read_hidden_size(t.QWEN_DIR)

    run = Path(args.run_dir)
    if not run.exists():
        print(f"ERROR: run-dir not found: {run}", file=sys.stderr); return 1

    # 检查 ckpt 数
    ckpts = sorted(run.glob("fold*.pt"))
    if len(ckpts) < args.folds:
        print(f"ERROR: only {len(ckpts)} ckpt found, need {args.folds}", file=sys.stderr); return 1
    print(f"[predict] found {len(ckpts)} ckpt in {run}", file=sys.stderr)
    print(f"[predict] model={t.QWEN_DIR} hidden={t.QWEN_DIM}", file=sys.stderr)

    # 加载 tokenizer + 数据
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(t.QWEN_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import glob as _glob
    conv_ids = sorted(Path(p).stem for p in _glob.glob("data/train/labels/*.npy"))
    test_ids = sorted(Path(p).stem for p in _glob.glob("data/test/audio/*.wav"))
    print(f"[predict] {len(conv_ids)} train convs, {len(test_ids)} test", file=sys.stderr)

    test_ds = t.TextTurnTakingDataset(test_ids, "test", slice_cap=args.slice_cap, tokenizer=tok)
    print(f"[predict] test samples={len(test_ds)}", file=sys.stderr)

    # 加载每 fold model, 预测
    enc = t.build_qwen_with_lora()  # 共用 frozen encoder + 待加载的 LoRA
    models = []
    for ckpt_path in ckpts[:args.folds]:
        print(f"[predict] loading {ckpt_path.name}...", file=sys.stderr)
        m = t.Qwen3HeadLoRA(ctx_dim=46, encoder=enc)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # ckpt 含 LoRA + head, frozen Qwen3 base 来自 from_pretrained, strict=False
        m.load_state_dict(state, strict=False)
        m.eval()
        models.append(m)

    # predict test
    print(f"[predict] running predict_test on {len(models)} models...", file=sys.stderr)
    test_probs = t.predict_test(models, test_ds, batch_size=args.batch_size)
    print(f"[predict] test: {test_probs.shape}", file=sys.stderr)

    # OOF (从 fold splits 重算)
    from sklearn.model_selection import GroupKFold
    train_ds = t.TextTurnTakingDataset(conv_ids, "train", slice_cap=args.slice_cap, tokenizer=tok, bc_aug_n=0)
    print(f"[predict] OOF: {len(train_ds)} samples", file=sys.stderr)

    groups = np.array([train_ds.samples[i][0] for i in range(len(train_ds))])
    gkf = GroupKFold(n_splits=args.folds)
    splits = list(gkf.split(np.zeros(len(train_ds)), groups=groups))

    oof = np.zeros((len(train_ds), 5), dtype=np.float32)
    for f_idx, (_, val_idx) in enumerate(splits):
        if f_idx >= len(models): break
        oof[val_idx] = t.predict_oof(models[f_idx], train_ds, val_idx, batch_size=args.batch_size)

    Y = np.zeros((len(train_ds), 5), dtype=int)
    G = np.zeros(len(train_ds), dtype=int)
    order = np.zeros(len(train_ds), dtype=int)
    for i in range(len(train_ds)):
        cid, end_ms, split, slice_idx = train_ds.samples[i]
        lab = train_ds._labels[(cid, "train")]
        end_chunk = end_ms // 80
        future = lab[end_chunk:end_chunk + 25]
        for k in range(5):
            Y[i, k] = int((future == k).any())
        G[i] = conv_ids.index(cid)
        order[i] = slice_idx

    np.savez_compressed(
        run / "probs.npz",
        oof=oof, test=test_probs, Y=Y, G=G, order=order,
        test_ids=np.array(test_ids),
    )
    print(f"[predict] saved {run}/probs.npz", file=sys.stderr)

    # cv metrics
    from sklearn.metrics import f1_score
    THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
    pred = np.zeros_like(oof, dtype=int)
    cap1_mask = order == 0
    for k in range(5):
        pred[:, k] = (oof[:, k] >= THR_VARF[k]).astype(int)
    per = [f1_score(Y[cap1_mask, k], pred[cap1_mask, k], zero_division=0) for k in range(5)]
    macro = float(np.mean(per))
    print(f"[predict] cap1 macro={macro:.4f}", file=sys.stderr)
    (run / "cv_metrics.json").write_text(json.dumps({
        "cap1_score": macro, "per_sub": {"C": per[0], "T": per[1], "BC": per[2], "I": per[3], "NA": per[4]},
        "note": "predict-only after train OOM"
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
