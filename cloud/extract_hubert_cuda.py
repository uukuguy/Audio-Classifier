"""上云 chinese-hubert-large 帧特征提取（CUDA, fp16, 断点续跑）— 第三独立正交源。

承 extract_whisper_cuda.py 结构, 改 whisper→hubert:
  - HubertModel (TencentGameMate/chinese-hubert-large, WenetSpeech 中文预训练, 1024维, 50Hz)
  - Wav2Vec2FeatureExtractor (raw 16k waveform 标准化, 非 mel)
  - 双声道帧序列 → 喂神经头 (train_head_cuda.py 兼容: 改 wd=1024)

为什么 hubert (multi-AI 3/3): 融合饱和后需全新正交源; HuBERT 对韵律/副语言敏感,
比 ASR-倾向的 whisper 更易在 T/I 提供正交信号 (中文电话域最对口)。

断点续跑 + PID+done 双信号 (同 whisper 版)。

Usage（云终端）:
  python cloud/extract_hubert_cuda.py --split train --convs 5    # 冒烟先 5 通
  python cloud/extract_hubert_cuda.py --split train --convs 0    # 全量
  python cloud/extract_hubert_cuda.py --split test  --convs 0
输出: <cache>/<split>/<cid>.npz  (key: frames [W,2,DS_FRAMES,1024], ends [W])
      <cache>/<split>/_done/<cid>
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
from transformers import HubertModel, Wav2Vec2FeatureExtractor

CTX, TGT, STRIDE, CHUNK_MS = 375, 25, 5, 80
SR16 = 16000
CTX_SEC = 8           # 末 8s 上下文 (同 whisper 版, 因果)
DS_FRAMES = 80        # 降采样到 80 帧 (与 whisper 版对齐, 神经头 cross-attn 输入一致)
FEAT_DIM = 1024       # chinese-hubert-large hidden

# --- 环境可配 (实验旋钮走 env/flag, 不碰 baseline default, HARD RULE) ---
MODEL_DIR = os.environ.get("HUBERT_DIR", str(Path.home() / ".cache/manual_models/chinese-hubert-large"))
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEV == "cuda" else torch.float32

_fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
_enc = None


def get_enc():
    global _enc
    if _enc is None:
        _enc = HubertModel.from_pretrained(MODEL_DIR, dtype=DTYPE).to(DEV).eval()
    return _enc


@torch.no_grad()
def hubert_frames(wav16_list: list[np.ndarray], batch: int) -> np.ndarray:
    """list of 1D 16k arrays (各 CTX_SEC*16k 长) → [N, DS_FRAMES, FEAT_DIM]。分 batch 防 OOM。"""
    enc = get_enc()
    out = []
    for i in range(0, len(wav16_list), batch):
        chunk = wav16_list[i:i + batch]
        # Wav2Vec2FeatureExtractor: raw waveform 标准化 + padding 到等长
        feats = _fe(chunk, sampling_rate=SR16, return_tensors="pt", padding=True)
        h = enc(feats.input_values.to(DEV, DTYPE)).last_hidden_state  # [n, T~400, 1024]
        # 降采样到 DS_FRAMES (8s@50Hz≈400帧 → 80帧, 与 whisper 版同步)
        ds = torch.nn.functional.adaptive_avg_pool1d(
            h.transpose(1, 2).float(), DS_FRAMES).transpose(1, 2)
        out.append(ds.cpu().numpy().astype(np.float16))
    return np.concatenate(out, axis=0)


def conv_windows(label_len: int, stride_mult: int) -> list[int]:
    return list(range(CTX, label_len - TGT + 1, STRIDE * stride_mult))


def extract_conv(cid: str, split: str, batch: int, stride_mult: int) -> tuple[np.ndarray, np.ndarray]:
    """一通对话 → frames [W,2,DS_FRAMES,FEAT_DIM], ends [W]。"""
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
    fr0 = hubert_frames(ch_segs[0], batch)
    fr1 = hubert_frames(ch_segs[1], batch)
    frames = np.stack([fr0, fr1], axis=1)  # [W,2,DS_FRAMES,FEAT_DIM]
    return frames.astype(np.float16), np.array(ends, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--convs", type=int, default=0, help="0=全量；>0 取前 N 通(冒烟)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride-mult", type=int, default=8, help="train 窗步长倍数(去重省算力)")
    ap.add_argument("--cache", default=os.environ.get("HCACHE", "data/hubert_cache"))
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
