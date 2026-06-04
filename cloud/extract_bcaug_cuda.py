"""P1.5 v2 — BC 正例音频增强帧提取 (whisper + hubert 双 encoder).

设计:
  1. 扫训练 stride40 全量 (179867 窗), 找 BC 正例窗 (~6591)
  2. 对每个 BC 正例 wav 段做 3x audio augment (变速保留 / 加噪 / SpecAug 时间掩码)
  3. 同一份增强 wav 同时喂 whisper-large-v3 + chinese-hubert-large encoder
  4. 输出 cache 文件 (新 wsp_bcaug + hub_bcaug 两个 cache 目录)

为什么不在线 LoRA 训练: 5/29 LoRA 全量 30h + 6/1 LoRA 在线 24min CPU 卡死 = 在线 mel/wav 预处理 IO 瓶颈.
为什么用 stride40 同 cycle 16: 已实证训 head 4min/fold, 速度 OK.
为什么 BC 增强而非全量增强: 只增强 BC 正例补救稀少样本 (用户拍板"BC 正例稀少要做数据增强").

Usage (云端):
  python cloud/extract_bcaug_cuda.py --encoder whisper --aug-n 3
  python cloud/extract_bcaug_cuda.py --encoder hubert --aug-n 3
输出: <cache>/train_bcaug/<cid>.npz  (frames_bcaug + ends_bcaug + is_aug + orig_end)
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

CTX, TGT, STRIDE, CHUNK_MS = 375, 25, 40, 80  # stride40 同 cycle 16
SR16 = 16000
CTX_SEC = 8
TAIL_FRAMES = 400
DS_FRAMES = 80
BC_CLASS = 2
SEED = 42

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32


def augment_wav_bc(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """音频增强 (BC 正例多样性). wav: [samples] 1D 或 [2, samples] 2D.

    组合: 加噪 (SNR 20-30dB) + gain 扰 (±3dB) + 时间掩码 (SpecAug 时域).
    不变速 (保因果时序).
    """
    x = wav.copy()
    noise_std = x.std() * rng.uniform(0.03, 0.10)
    x = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)
    x = x * float(rng.uniform(0.7, 1.4))
    last_dim = x.shape[-1]
    if last_dim > 2000:
        ml = int(rng.uniform(0.02, 0.08) * last_dim)
        st = int(rng.uniform(0, last_dim - ml))
        if x.ndim == 1:
            x[st:st + ml] = 0.0
        else:
            x[:, st:st + ml] = 0.0
    return x.astype(np.float32)


# ── encoder loaders ────────────────────────────────────────────────────────
def load_whisper_encoder():
    from transformers import WhisperFeatureExtractor, WhisperModel
    model_dir = os.environ.get("WHISPER_DIR", "/autodl-fs/data/backups/manual_models/whisper-large-v3")
    fe = WhisperFeatureExtractor.from_pretrained(model_dir)
    enc = WhisperModel.from_pretrained(model_dir, torch_dtype=DTYPE).encoder.to(DEV).eval()
    return fe, enc, "whisper", 1280


def load_hubert_encoder():
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    model_dir = os.environ.get("HUBERT_DIR", "/root/.cache/manual_models/chinese-hubert-large")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    enc = HubertModel.from_pretrained(model_dir, torch_dtype=DTYPE).to(DEV).eval()
    enc.config.mask_time_prob = 0.0
    enc.config.mask_feature_prob = 0.0
    return fe, enc, "hubert", 1024


def load_w2v2_encoder():
    """chinese-wav2vec2-large (TencentGameMate) 同 hubert 同输入 (raw 16k waveform)."""
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    model_dir = os.environ.get("W2V2_DIR", "/root/.cache/manual_models/chinese-wav2vec2-large")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    enc = Wav2Vec2Model.from_pretrained(model_dir, torch_dtype=DTYPE).to(DEV).eval()
    enc.config.mask_time_prob = 0.0
    enc.config.mask_feature_prob = 0.0
    return fe, enc, "w2v2", 1024


def load_e2v_encoder():
    """emotion2vec_base (funasr 包装, 副语言情感专用)."""
    from funasr import AutoModel
    model_dir = os.environ.get("E2V_DIR", "/root/.cache/manual_models/emotion2vec_base")
    m = AutoModel(model=model_dir)
    # funasr model 内部包装, 不直接拿 encoder. 用 m.generate(extract_embedding=True)
    return None, m, "e2v", 768


@torch.no_grad()
def whisper_frames(wav16_list: list[np.ndarray], fe, enc, batch: int) -> np.ndarray:
    """list of 1D 16k arrays → [N, 80, 1280]."""
    out = []
    for i in range(0, len(wav16_list), batch):
        chunk = wav16_list[i:i + batch]
        feats = fe(chunk, sampling_rate=SR16, return_tensors="pt")
        h = enc(feats.input_features.to(DEV, DTYPE)).last_hidden_state  # [n,1500,1280]
        tail = h[:, -TAIL_FRAMES:, :]
        ds = torch.nn.functional.adaptive_avg_pool1d(
            tail.transpose(1, 2).float(), DS_FRAMES
        ).transpose(1, 2)  # [n, 80, 1280]
        out.append(ds.cpu().numpy().astype(np.float16))
    return np.concatenate(out, axis=0)


@torch.no_grad()
def hubert_frames(wav16_list: list[np.ndarray], fe, enc, batch: int) -> np.ndarray:
    """list of 1D 16k arrays → [N, 80, 1024]. hubert: raw 16k waveform input."""
    out = []
    for i in range(0, len(wav16_list), batch):
        chunk = wav16_list[i:i + batch]
        # hubert needs same-length input
        max_len = max(len(w) for w in chunk)
        padded = [np.pad(w, (max_len - len(w), 0)) if len(w) < max_len else w for w in chunk]
        inputs = fe(padded, sampling_rate=SR16, return_tensors="pt")
        h = enc(inputs.input_values.to(DEV, DTYPE)).last_hidden_state  # [n, T_enc, 1024]
        ds = torch.nn.functional.adaptive_avg_pool1d(
            h.transpose(1, 2).float(), DS_FRAMES
        ).transpose(1, 2)  # [n, 80, 1024]
        out.append(ds.cpu().numpy().astype(np.float16))
    return np.concatenate(out, axis=0)


# w2v2 frames 用法跟 hubert 完全一样
w2v2_frames = hubert_frames


@torch.no_grad()
def e2v_frames(wav16_list: list[np.ndarray], fe, model, batch: int) -> np.ndarray:
    """list of 1D 16k arrays → [N, 80, 768]. e2v: funasr AutoModel wrapper."""
    out = []
    for w in wav16_list:
        # funasr extract_embedding mode: 返回 list[{feats: [T,768]}]
        res = model.generate(w, granularity="frame", extract_embedding=True)
        feats = res[0]["feats"]  # [T, 768] float32
        # downsample to DS_FRAMES
        feats_t = torch.from_numpy(feats).unsqueeze(0).transpose(1, 2).float()  # [1, 768, T]
        ds = torch.nn.functional.adaptive_avg_pool1d(feats_t, DS_FRAMES).transpose(1, 2)
        out.append(ds.squeeze(0).cpu().numpy().astype(np.float16))
    return np.stack(out, axis=0)


def find_bc_pos_ends(cid: str) -> list[int]:
    """Find all stride40 windows where BC is in future TGT chunks."""
    a = np.load(f"data/train/labels/{cid}.npy").astype(int)
    ends = []
    for e in range(CTX, len(a) - TGT + 1, STRIDE):
        if BC_CLASS in a[e:e + TGT]:
            ends.append(e)
    return ends


def extract_bcaug(cid: str, aug_n: int, fe, enc, frames_fn, batch: int, fdim: int) -> dict:
    """Extract BC 增强 frames for one conv.

    Returns dict:
      frames_bcaug: [N_aug, 2, 80, FEAT_DIM] fp16 (N_aug = bc_count * aug_n)
      orig_end: [N_aug] int32 (原始窗 end_chunk, 同一原始窗多个增强 = 同 end)
      aug_id: [N_aug] int8 (1..aug_n, 表示第几个增强变体)
    """
    bc_ends = find_bc_pos_ends(cid)
    if not bc_ends:
        return {"frames_bcaug": np.zeros((0, 2, DS_FRAMES, fdim), dtype=np.float16),
                "orig_end": np.zeros((0,), dtype=np.int32),
                "aug_id": np.zeros((0,), dtype=np.int8)}

    # 读 wav
    wav_path = f"data/train/audio/{cid}.wav"
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        full = wf.readframes(wf.getnframes())
    d = np.frombuffer(full, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0

    rng = np.random.default_rng(SEED + abs(hash(cid)) % (2**32))

    all_orig_ends, all_aug_ids = [], []
    ch_segs = {0: [], 1: []}
    for e in bc_ends:
        end = int(e * CHUNK_MS / 1000 * sr)
        start = max(0, end - CTX_SEC * sr)
        # 提原始 8s 双声道 wav 段
        seg_2ch = d[:, start:end]
        if seg_2ch.shape[1] < CTX_SEC * sr:
            seg_2ch = np.pad(seg_2ch, ((0, 0), (CTX_SEC * sr - seg_2ch.shape[1], 0)))

        # 生成 aug_n 个增强变体
        for aug_i in range(aug_n):
            aug_seg = augment_wav_bc(seg_2ch, rng)
            for ch in range(2):
                w16 = torchaudio.functional.resample(
                    torch.tensor(aug_seg[ch]), sr, SR16
                ).numpy()
                ch_segs[ch].append(w16)
            all_orig_ends.append(e)
            all_aug_ids.append(aug_i + 1)

    # 喂 encoder
    fr0 = frames_fn(ch_segs[0], fe, enc, batch)
    fr1 = frames_fn(ch_segs[1], fe, enc, batch)
    frames_bcaug = np.stack([fr0, fr1], axis=1)  # [N_aug, 2, 80, FEAT_DIM]
    return {
        "frames_bcaug": frames_bcaug.astype(np.float16),
        "orig_end": np.array(all_orig_ends, dtype=np.int32),
        "aug_id": np.array(all_aug_ids, dtype=np.int8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["whisper", "hubert", "w2v2", "e2v"], required=True)
    ap.add_argument("--aug-n", type=int, default=3, help="每 BC 正例生成 N 个增强变体")
    ap.add_argument("--convs", type=int, default=0, help="0=全量, >0 取前 N 通")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if args.encoder == "whisper":
        fe, enc, name, fdim = load_whisper_encoder()
        frames_fn = whisper_frames
        cache_dir = Path("/autodl-fs/data/whisper_bcaug")
    elif args.encoder == "hubert":
        fe, enc, name, fdim = load_hubert_encoder()
        frames_fn = hubert_frames
        cache_dir = Path("/autodl-fs/data/hubert_bcaug")
    elif args.encoder == "w2v2":
        fe, enc, name, fdim = load_w2v2_encoder()
        frames_fn = w2v2_frames
        cache_dir = Path("/autodl-fs/data/w2v2_bcaug")
    elif args.encoder == "e2v":
        fe, enc, name, fdim = load_e2v_encoder()
        frames_fn = e2v_frames
        cache_dir = Path("/autodl-fs/data/e2v_bcaug")
    else:
        sys.exit(f"unknown encoder: {args.encoder}")

    cache = cache_dir / "train"
    done_dir = cache / "_done"
    done_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs:
        ids = ids[:args.convs]

    todo = [c for c in ids if not (done_dir / c).exists()]
    print(f"[bcaug] encoder={name} fdim={fdim} aug_n={args.aug_n}", file=sys.stderr)
    print(f"[bcaug] {len(ids)} 通, 已完成 {len(ids) - len(todo)}, 待提取 {len(todo)}", file=sys.stderr)

    t_all = time.time()
    total_bc_wins = 0
    for i, cid in enumerate(todo):
        t0 = time.time()
        data = extract_bcaug(cid, args.aug_n, fe, enc, frames_fn, args.batch, fdim)
        n_aug = len(data["orig_end"])
        np.savez_compressed(cache / f"{cid}.npz", **data)
        (done_dir / cid).touch()
        total_bc_wins += n_aug // args.aug_n
        dt = time.time() - t0
        eta = (len(todo) - i - 1) * (time.time() - t_all) / (i + 1) / 60
        print(f"[bcaug] {i + 1}/{len(todo)} {cid}: {n_aug}增强帧 ({n_aug//args.aug_n}BC原) {dt:.1f}s (ETA {eta:.0f}min)",
              file=sys.stderr)

    n_done = len(list(done_dir.glob("*")))
    print(f"[bcaug] DONE encoder={name} done={n_done}/{len(ids)} "
          f"total_bc_wins={total_bc_wins} total={(time.time() - t_all) / 60:.0f}min", file=sys.stderr)
    print(f"BCAUG_COMPLETE encoder={name} done={n_done} target={len(ids)}")


if __name__ == "__main__":
    main()
