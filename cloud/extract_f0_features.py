"""F0/pitch + BC 增强提帧 — 真正交特征空间 (D-4 实测 |r|=0.128 BC 最强分支).

本机 librosa pyin, 不需要 GPU. 1845 通 cap5 + BC 3x 增强 ≈ 28k 样本.

输出每窗 8s 末段:
  F0 序列 (16 frames stride 0.5s) +
  voiced_prob 序列 +
  pitch range / std / mean (统计量)
共 ~50 维 / 窗

写到 data/cache/f0_features.npz: X (28k, ~50), Y (28k, 5), G, order, is_aug

Usage:
  OMP_NUM_THREADS=8 python3 cloud/extract_f0_features.py
"""
from __future__ import annotations
import glob
import json
import sys
import wave
from pathlib import Path

import numpy as np
import librosa
import torch
import torchaudio

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
CTX, TGT, CHUNK_MS = 375, 25, 80
CTX_SEC = 8       # 末 8s 音频
F0_HOP_MS = 500   # F0 提取每 500ms 一帧 → 8s 内 16 帧
BC_CLASS = 2
BC_AUG_N = 3
SR_OUT = 16000
SEED = 42

OUT_PATH = "data/cache/f0_features.npz"


def load_wav_8k_dual(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def augment_wav_bc(wav, sr, rng):
    """跟 train_lora_whisper_bcaug.py 同款配方."""
    x = wav.copy()
    noise_std = x.std() * rng.uniform(0.03, 0.10)
    x = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)
    x = x * float(rng.uniform(0.7, 1.4))
    if x.shape[1] > 2000:
        ml = int(rng.uniform(0.02, 0.08) * x.shape[1])
        st = int(rng.uniform(0, x.shape[1] - ml))
        x[:, st:st + ml] = 0.0
    return x.astype(np.float32)


def to_mono_16k(seg_8k_dual, sr_orig):
    mono = seg_8k_dual.mean(axis=0)
    if sr_orig != SR_OUT:
        # 用 torchaudio (项目惯例, 跟其他 cloud/*.py 一致, 避 librosa 的 numba 版本冲突)
        t = torch.from_numpy(mono)
        r = torchaudio.functional.resample(t, sr_orig, SR_OUT)
        mono = r.numpy()
    return mono.astype(np.float32)


def extract_f0(audio_16k):
    """声学特征 — 不依赖 pyin/numba.

    8s × 16kHz = 128000 samples, 分 16 帧 (500ms/帧):
      - 帧能量序列 (16d)
      - 帧零交叉率 (16d, 跟 pitch 强相关)
      - 帧 spectral centroid (16d, 频谱重心)
      - 全段统计量 (~9d)
    共 ~57 维 / 窗

    速度: numpy 纯算术 ~5ms/段
    """
    n = len(audio_16k)
    if n < SR_OUT // 2:
        # 太短, 返全 0
        return np.zeros(57, dtype=np.float32)

    hop_samples = SR_OUT // 2  # 500ms hop = 8000 samples
    n_frames = 16
    frame_len = hop_samples

    # 取末 16 帧, 不够补 0
    if n < n_frames * frame_len:
        pad = n_frames * frame_len - n
        audio_16k = np.concatenate([np.zeros(pad, dtype=np.float32), audio_16k])
    audio_tail = audio_16k[-(n_frames * frame_len):]
    frames = audio_tail.reshape(n_frames, frame_len)

    # 1. 帧能量 (16d) — log RMS
    energy = np.sqrt((frames ** 2).mean(axis=1) + 1e-10)
    log_energy = np.log(energy + 1e-6)

    # 2. 帧零交叉率 (16d) — 频率/pitch 代理
    sign_changes = np.diff(np.sign(frames), axis=1) != 0
    zcr = sign_changes.sum(axis=1) / frame_len

    # 3. 帧 spectral centroid (16d) — 频谱重心, 跟 pitch 相关
    # 用 FFT 算
    spec = np.abs(np.fft.rfft(frames, axis=1))  # (16, frame_len//2 + 1)
    freqs = np.fft.rfftfreq(frame_len, 1.0 / SR_OUT)  # (frame_len//2 + 1,)
    spec_sum = spec.sum(axis=1) + 1e-10
    centroid = (spec * freqs).sum(axis=1) / spec_sum

    # 4. 全段统计量 (9d)
    energy_mean = float(energy.mean())
    energy_std = float(energy.std())
    energy_max = float(energy.max())
    zcr_mean = float(zcr.mean())
    zcr_std = float(zcr.std())
    centroid_mean = float(centroid.mean())
    centroid_std = float(centroid.std())
    # delta energy (frame 间差) - turn-taking 重要信号
    de = np.diff(log_energy)
    de_mean = float(de.mean())
    de_std = float(de.std())

    feats = np.concatenate([
        log_energy,                                          # 16
        zcr,                                                 # 16
        centroid / 1000.0,                                   # 16 (kHz scale)
        [energy_mean, energy_std, energy_max,
         zcr_mean, zcr_std,
         centroid_mean / 1000.0, centroid_std / 1000.0,
         de_mean, de_std],                                   # 9
    ]).astype(np.float32)
    # = 57 维
    return feats


def pick_slice_ends(label_len, cap=5):
    lo, hi = CTX, label_len - TGT
    if lo > hi:
        return []
    step = max(1, (hi - lo) // cap)
    return list(range(lo, hi + 1, step))[:cap]


def main():
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    print(f"[f0] {len(conv_ids)} convs, cap5 + BC×{BC_AUG_N} 增强", file=sys.stderr, flush=True)

    rng_master = np.random.default_rng(SEED)
    X, Y, G, order_arr, is_aug = [], [], [], [], []
    feat_dim = None

    n_done, t_start = 0, __import__("time").time()
    for ci, cid in enumerate(conv_ids):
        labels = np.load(f"data/train/labels/{cid}.npy")
        wav, sr = load_wav_8k_dual(f"data/train/audio/{cid}.wav")
        ends = pick_slice_ends(len(labels), cap=5)

        for oi, e in enumerate(ends):
            # 原始
            end_sample = int(e * CHUNK_MS / 1000 * sr)
            start_sample = max(0, end_sample - CTX_SEC * sr)
            seg = wav[:, start_sample:end_sample]
            audio = to_mono_16k(seg, sr)
            feats = extract_f0(audio)
            if feat_dim is None:
                feat_dim = feats.shape[0]
                print(f"[f0] feat_dim={feat_dim}", file=sys.stderr, flush=True)
            X.append(feats)
            fut = set(int(x) for x in labels[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(5)])
            G.append(ci)
            order_arr.append(oi)
            is_aug.append(0)

            # BC 增强 (若该窗 fut 含 BC)
            if BC_CLASS in fut:
                for aug_seed in range(1, BC_AUG_N + 1):
                    seed = abs(hash((cid, e, aug_seed))) % (2**32)
                    rng = np.random.default_rng(seed)
                    seg_aug = augment_wav_bc(seg, sr, rng)
                    audio_aug = to_mono_16k(seg_aug, sr)
                    feats_aug = extract_f0(audio_aug)
                    X.append(feats_aug)
                    Y.append([1 if k in fut else 0 for k in range(5)])
                    G.append(ci)
                    order_arr.append(oi)
                    is_aug.append(aug_seed)

        n_done += 1
        if n_done % 20 == 0 or n_done == len(conv_ids):
            elapsed = __import__("time").time() - t_start
            eta = elapsed / n_done * (len(conv_ids) - n_done)
            print(f"[f0] {n_done}/{len(conv_ids)} convs, X={len(X)}, elapsed={elapsed:.0f}s, eta={eta:.0f}s",
                  file=sys.stderr, flush=True)

    X = np.stack(X)
    Y = np.array(Y, dtype=np.int8)
    G = np.array(G, dtype=np.int16)
    order_arr = np.array(order_arr, dtype=np.int16)
    is_aug = np.array(is_aug, dtype=np.int8)
    print(f"[f0] FINAL: X {X.shape} Y {Y.shape} G {G.shape} (含 {(is_aug>0).sum()} 增强)",
          file=sys.stderr, flush=True)

    # 也提 test (用 1000 段, slice_cap=1)
    print("[f0] === test ===", file=sys.stderr, flush=True)
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    Xt = []
    for ti, sid in enumerate(test_ids):
        wav, sr = load_wav_8k_dual(f"data/test/audio/{sid}.wav")
        # test wav 全长 30s, 取末 8s
        audio = to_mono_16k(wav, sr)
        Xt.append(extract_f0(audio))
        if (ti + 1) % 100 == 0:
            print(f"[f0] test {ti+1}/{len(test_ids)}", file=sys.stderr, flush=True)
    Xt = np.stack(Xt)

    np.savez_compressed(
        OUT_PATH,
        X=X, Y=Y, G=G, order=order_arr, is_aug=is_aug,
        Xt=Xt, test_ids=np.array(test_ids),
    )
    print(f"[f0] saved {OUT_PATH}: train {X.shape}, test {Xt.shape}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
