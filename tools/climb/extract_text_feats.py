"""H-T3 step 1: 去重提取 Qwen3-0.6B 文本特征缓存.

关键(实测)：1.44M 滑窗朴素提取=25h；按 (conv_id, 可见utterance数) 去重后
唯一文本上下文仅 8.3%(~12万)→ MPS ~126min 可行。冻结编码器，masked mean 末层。
注：mdl.eval() 是 PyTorch eval 模式(非 Python eval())。

产物: data/cache/qwen_text/<conv_id>.npz  (key=可见utt数, val=1024d)
      data/cache/qwen_text_test.npz        (key=segment_id, val=1024d)

Usage: python tools/climb/extract_text_feats.py [--smoke N]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = str(Path.home() / ".cache/manual_models/Qwen3-0.6B")
CTX, TGT, STRIDE, CHUNK_MS = 375, 25, 5, 80
MAXLEN = 256
SPK = {1: "[SPK1]", 2: "[SPK2]"}
CACHE_DIR = Path("data/cache/qwen_text")


def build_text(utts, end_ms):
    parts = []
    for u in utts:
        if int(u.get("end_ms", 0)) <= end_ms:
            t = str(u.get("text", "")).strip()
            if t:
                parts.append(f"{SPK.get(int(u.get('channel_id', 1)), '[SPK1]')} {t}")
    return " ".join(parts[-60:]) if parts else "[SPK1] <silence>"


def ms_for_nvis(ends, nvis):
    if nvis <= 0 or not ends:
        return 0
    return ends[min(nvis, len(ends)) - 1] + 1


@torch.no_grad()
def encode_batch(texts, tok, mdl, device):
    ids = tok(texts, return_tensors="pt", truncation=True, max_length=MAXLEN, padding=True).to(device)
    out = mdl(**ids).last_hidden_state
    mask = ids["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[ext] loading Qwen3-0.6B on {device}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    mdl = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.float32).to(device)
    mdl.eval()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(glob.glob("data/train/labels/*.npy"))
    if args.smoke:
        label_files = label_files[:args.smoke]
    t0 = time.time()
    total_uniq = 0
    for fi, p in enumerate(label_files):
        cid = Path(p).stem
        out_path = CACHE_DIR / f"{cid}.npz"
        if out_path.exists() and not args.smoke:
            continue
        a = np.load(p)
        utts = json.load(open(f"data/train/text/{cid}.json")).get("utterances", [])
        ends = sorted(int(u.get("end_ms", 0)) for u in utts)
        keys = sorted({sum(1 for x in ends if x <= e * CHUNK_MS)
                       for e in range(CTX, a.shape[0] - TGT + 1, STRIDE)})
        texts = [build_text(utts, ms_for_nvis(ends, k)) for k in keys]
        feats = {}
        for i in range(0, len(texts), args.batch):
            vecs = encode_batch(texts[i:i + args.batch], tok, mdl, device)
            for j, v in enumerate(vecs):
                feats[str(keys[i + j])] = v.astype(np.float32)
        np.savez(out_path, **feats)
        total_uniq += len(keys)
        if fi % 5 == 0 or fi == len(label_files) - 1:
            el = time.time() - t0
            done, tot = fi + 1, len(label_files)
            eta = el / done * (tot - done) / 60
            msg = f"[ext] {done}/{tot} convs ({100*done/tot:.0f}%), {total_uniq} uniq, {el:.0f}s, eta {eta:.0f}min"
            print(msg, file=sys.stderr, flush=True)
            # 进度文件供 heartbeat / resume 读
            Path("data/cache/.extract_progress").write_text(
                f"{done}/{tot} {100*done/tot:.0f}% uniq={total_uniq} eta={eta:.0f}min t={el:.0f}s")

    test_files = sorted(glob.glob("data/test/text/*.json"))
    if args.smoke:
        test_files = test_files[:min(50, args.smoke * 10)]
    test_feats = {}
    seg_ids = [Path(p).stem for p in test_files]
    texts = []
    for p in test_files:
        tj = json.load(open(p))
        texts.append(build_text(tj.get("utterances", []), int(tj.get("end_ms", 30000))))
    for i in range(0, len(texts), args.batch):
        vecs = encode_batch(texts[i:i + args.batch], tok, mdl, device)
        for j, v in enumerate(vecs):
            test_feats[seg_ids[i + j]] = v.astype(np.float32)
    np.savez("data/cache/qwen_text_test.npz", **test_feats)
    print(f"[ext] DONE {total_uniq} train uniq + {len(test_feats)} test, {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
