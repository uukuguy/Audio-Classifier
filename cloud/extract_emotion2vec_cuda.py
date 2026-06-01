"""上云 emotion2vec_base 帧特征提取 (CUDA, 断点续跑) — 第五独立正交源 (副语言情感, 维度独特).

承 extract_hubert_cuda.py 结构, 改 HF transformers→funasr AutoModel:
  - emotion2vec_base (iic, ModelScope, 12层 768维, 50Hz, 副语言情感预训练)
  - funasr API: m.generate(wav, granularity="frame", extract_embedding=True)
    返回 list[{"key":..., "labels":..., "scores":..., "feats": [T, 768] float32}]
  - 双声道帧序列 → 喂神经头 (train_head_cuda.py 兼容 FDIM 动态探测, wd=768)

为什么 emotion2vec (cycle 17 用户要 C+D, 唯一未试维度独特模型):
  - 与 whisper/hubert/w2v2 (语义/SSL 范式) 不同, emotion2vec 训练目标=情感识别
    → 副语言信号 (语调升降/语速/情感强度) 是 turn-taking 没人提过的角度
  - 风险=单 768d 比 hubert 1024d 信息容量小, 可能弱; 但若提供正交副语言信号则三/四/五源融合有真增量

接口差异:
  - funasr API 非 HF, 不接受 batch list (per-call 单 wav), 需手动 batch loop
  - feats shape [T, 768] 已是 50Hz 帧序列, T≈sec*50, 直接 adaptive_avg_pool1d→80帧

断点续跑 + PID+done 双信号 (同 hubert/w2v2).

Usage:
  python cloud/extract_emotion2vec_cuda.py --split train --convs 5    # 冒烟
  python cloud/extract_emotion2vec_cuda.py --split train --convs 0
  python cloud/extract_emotion2vec_cuda.py --split test  --convs 0
输出: <cache>/<split>/<cid>.npz  (key: frames [W,2,DS_FRAMES,768], ends [W])
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
from funasr import AutoModel

CTX, TGT, STRIDE, CHUNK_MS = 375, 25, 5, 80
SR16 = 16000
CTX_SEC = 8
DS_FRAMES = 80
FEAT_DIM = 768

MODEL_DIR = os.environ.get("E2V_DIR", str(Path.home() / ".cache/manual_models/emotion2vec_base"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_m = None


def get_model():
    global _m
    if _m is None:
        # funasr AutoModel: device 通过 device kwarg 控制 (CUDA 自动用)
        _m = AutoModel(model=MODEL_DIR, model_revision=None, disable_update=True, device=DEV)
    return _m


@torch.no_grad()
def emotion2vec_frames(wav16_list: list[np.ndarray]) -> np.ndarray:
    """list of 1D 16k float arrays → [N, DS_FRAMES, 768]。
    funasr API per-call 单 wav, 无 batch — 手动循环。"""
    m = get_model()
    out = []
    for wav in wav16_list:
        res = m.generate(wav.astype(np.float32), granularity="frame", extract_embedding=True)
        feats = res[0]["feats"]  # [T, 768] float32
        # 降采样到 DS_FRAMES (8s@50Hz≈400帧 → 80帧, 与 whisper/hubert/w2v2 对齐)
        t = torch.from_numpy(feats).T.unsqueeze(0)  # [1, 768, T]
        ds = torch.nn.functional.adaptive_avg_pool1d(t, DS_FRAMES).squeeze(0).T  # [80, 768]
        out.append(ds.numpy().astype(np.float16))
    return np.stack(out, axis=0)  # [N, 80, 768]


def conv_windows(label_len: int, stride_mult: int) -> list[int]:
    return list(range(CTX, label_len - TGT + 1, STRIDE * stride_mult))


def extract_conv(cid: str, split: str, stride_mult: int) -> tuple[np.ndarray, np.ndarray]:
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
    fr0 = emotion2vec_frames(ch_segs[0])
    fr1 = emotion2vec_frames(ch_segs[1])
    frames = np.stack([fr0, fr1], axis=1)  # [W,2,DS_FRAMES,768]
    return frames.astype(np.float16), np.array(ends, dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--stride-mult", type=int, default=40,
                    help="stride40 轻量(同 hubert/w2v2 cycle 17 策略) 验正交性再扩")
    ap.add_argument("--cache", default=os.environ.get("E2VCACHE", "data/emotion2vec_cache"))
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
    print(f"[extract] split={args.split} dev={DEV} model={MODEL_DIR}", file=sys.stderr)
    print(f"[extract] {len(ids)} 通, 已完成 {len(ids) - len(todo)}, 待提取 {len(todo)}", file=sys.stderr)

    t_all = time.time()
    for i, cid in enumerate(todo):
        t0 = time.time()
        frames, ends = extract_conv(cid, args.split, args.stride_mult)
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
