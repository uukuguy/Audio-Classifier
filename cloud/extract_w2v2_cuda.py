"""上云 chinese-wav2vec2-large 帧特征提取（CUDA, fp16, 断点续跑）— 第四独立正交源候选。

承 extract_hubert_cuda.py 结构, 改 HubertModel→Wav2Vec2Model:
  - Wav2Vec2Model (TencentGameMate/chinese-wav2vec2-large, WenetSpeech 中文预训练, 1024维, 50Hz)
  - Wav2Vec2FeatureExtractor (与 hubert 同)
  - 双声道帧序列 → 喂神经头 (train_head_cuda.py 兼容: wd=1024)

为什么 w2v2 (用户要求 C+D 并行试更多模型, cycle 17):
  - 与 hubert 同门 (TencentGameMate/WenetSpeech 同源 SSL) 风险=高相关不正交
  - 但训练目标不同 (w2v2=contrastive predictive coding, hubert=mask predict)
  - 实测看 OOF cap1 T/I/BC 是否提供 hubert 没有的信号才能判
  - 已下载 1.2GB 在 /root/.cache/manual_models/chinese-wav2vec2-large/

断点续跑 + PID+done 双信号 (同 hubert/whisper 版)。

Usage（云终端）:
  python cloud/extract_w2v2_cuda.py --split train --convs 0
  python cloud/extract_w2v2_cuda.py --split test  --convs 0
输出: <cache>/<split>/<cid>.npz  (key: frames [W,2,DS_FRAMES,1024], ends [W])
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

CTX, TGT, STRIDE, CHUNK_MS = 375, 25, 5, 80
SR16 = 16000
CTX_SEC = 8
DS_FRAMES = 80
FEAT_DIM = 1024

MODEL_DIR = os.environ.get("W2V2_DIR", str(Path.home() / ".cache/manual_models/chinese-wav2vec2-large"))
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEV == "cuda" else torch.float32

_fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
_enc = None


def get_enc():
    global _enc
    if _enc is None:
        _enc = Wav2Vec2Model.from_pretrained(MODEL_DIR, dtype=DTYPE).to(DEV).eval()
    return _enc


@torch.no_grad()
def w2v2_frames(wav16_list: list[np.ndarray], batch: int) -> np.ndarray:
    enc = get_enc()
    out = []
    for i in range(0, len(wav16_list), batch):
        chunk = wav16_list[i:i + batch]
        feats = _fe(chunk, sampling_rate=SR16, return_tensors="pt", padding=True)
        h = enc(feats.input_values.to(DEV, DTYPE)).last_hidden_state
        ds = torch.nn.functional.adaptive_avg_pool1d(
            h.transpose(1, 2).float(), DS_FRAMES).transpose(1, 2)
        out.append(ds.cpu().numpy().astype(np.float16))
    return np.concatenate(out, axis=0)


def conv_windows(label_len: int, stride_mult: int) -> list[int]:
    return list(range(CTX, label_len - TGT + 1, STRIDE * stride_mult))


def extract_conv(cid: str, split: str, batch: int, stride_mult: int) -> tuple[np.ndarray, np.ndarray]:
    if split == "train":
        a = np.load(f"data/train/labels/{cid}.npy")
        ends = conv_windows(a.shape[0], stride_mult)
        wav_path = f"data/train/audio/{cid}.wav"
    else:
        ends = [CTX]
        wav_path = f"data/test/audio/{cid}.wav"

    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        full = wf.readframes(wf.getnframes())
    d = np.frombuffer(full, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0

    ch_segs = {0: [], 1: []}
    for e in ends:
        end = int(e * CHUNK_MS / 1000 * sr) if split == "train" else d.shape[1]
        start = max(0, end - CTX_SEC * sr)
        for ch in range(2):
            seg = d[ch, start:end]
            if len(seg) < CTX_SEC * sr:
                seg = np.pad(seg, (CTX_SEC * sr - len(seg), 0))
            w16 = torchaudio.functional.resample(torch.tensor(seg), sr, SR16).numpy()
            ch_segs[ch].append(w16)
    fr0 = w2v2_frames(ch_segs[0], batch)
    fr1 = w2v2_frames(ch_segs[1], batch)
    frames = np.stack([fr0, fr1], axis=1)
    return frames.astype(np.float16), np.array(ends, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride-mult", type=int, default=40,
                    help="stride40 轻量(同 hubert cycle 16 决策) 验正交性再扩")
    ap.add_argument("--cache", default=os.environ.get("WCACHE2", "data/w2v2_cache"))
    args = ap.parse_args()

    cache = Path(args.cache) / args.split
    done_dir = cache / "_done"
    done_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "train":
        ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    else:
        ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    if args.convs:
        ids = ids[:args.convs]

    todo = [c for c in ids if not (done_dir / c).exists()]
    print(f"[extract] split={args.split} dev={DEV} dtype={DTYPE} model={MODEL_DIR}", file=sys.stderr)
    print(f"[extract] {len(ids)} 通, 已完成 {len(ids) - len(todo)}, 待提取 {len(todo)}", file=sys.stderr)

    t_all = time.time()
    for i, cid in enumerate(todo):
        t0 = time.time()
        frames, ends = extract_conv(cid, args.split, args.batch, args.stride_mult)
        np.savez_compressed(cache / f"{cid}.npz", frames=frames, ends=ends)
        (done_dir / cid).touch()
        dt = time.time() - t0
        eta = (len(todo) - i - 1) * (time.time() - t_all) / (i + 1) / 60
        print(f"[extract] {i + 1}/{len(todo)} {cid}: {frames.shape[0]}窗 {dt:.1f}s "
              f"(ETA {eta:.0f}min)", file=sys.stderr)

    n_done = len(list(done_dir.glob("*")))
    print(f"[extract] DONE split={args.split} done={n_done}/{len(ids)} "
          f"total={(time.time() - t_all) / 60:.0f}min", file=sys.stderr)
    print(f"EXTRACT_COMPLETE split={args.split} done={n_done} target={len(ids)}")


if __name__ == "__main__":
    main()
