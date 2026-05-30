"""climb cycle H-007 — F0/韵律融合攻 BC (用户问"音频哪分支强"→韵律探针证 F0 最强).

韵律探针 (2026-05-30): F0 是音频对 BC 最强分支 (f0_var|r|0.128/f0_mean0.119,
BC前音高更低=turn-yielding), 比 VAP/mel/whisper(DL频谱漏F0)强3x。但与context同
量级(r0.13)=弱信号。本实验验关键问题: F0 与 context 是否正交互补(各r0.13独立叠加
才有融合价值)。

对照 (全量 OOF, BC F1):
  A. context-only (baseline v1)
  B. context + F0/韵律特征 (F0均值/方差/末段斜率 + pause + 能量)
看 B 是否比 A 的 BC 显著提升 = F0 带来 context 没有的正交信号.

F0 提取 9ms/窗, 全量~1min, 本地可行. 限线程.
Usage: python tools/climb/cycle_f0_fusion.py [--convs 0] [--stride 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctx_v1

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM, CTX, TGT, CHUNK_MS, SR, SEED = 5, 375, 25, 80, 8000, 42
BC = 2
WIN_SEC = 5


def load_wav(cid):
    with wave.open(f"data/train/audio/{cid}.wav", "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def f0_prosody_feats(seg2):
    """显式韵律特征 (探针证强分支: F0 + pause). seg2: [2, samples] 8kHz."""
    fl, hop = int(0.025 * SR), int(0.010 * SR)
    mono = seg2.mean(0)
    n = max(1, (len(mono) - fl) // hop + 1)
    fr = np.array([mono[i * hop:i * hop + fl] for i in range(n)])
    rms = np.sqrt((fr ** 2).mean(1) + 1e-8)
    f = []
    # 能量/停顿 (探针: pause_ratio r0.118)
    f.append(float(rms.mean()))
    f.append(float((rms < rms.mean() * 0.3).mean()))           # pause_ratio
    f.append(float(rms.var()))
    # F0 (探针: 最强分支 f0_var r0.128 / f0_mean r0.119)
    try:
        pf = torchaudio.functional.detect_pitch_frequency(
            torch.from_numpy(mono).unsqueeze(0), SR).squeeze(0).numpy()
        pf = pf[pf > 0]
        if len(pf) >= 20:
            q = len(pf) // 4
            f += [float(pf.mean()), float(pf.var()),
                  float(pf[-q:].mean() - pf[:q].mean()),       # 末-初 (下降?)
                  float(pf[-q:].mean())]                       # 末段F0 (turn-yield)
        else:
            f += [0.0, 0.0, 0.0, 0.0]
    except Exception:
        f += [0.0, 0.0, 0.0, 0.0]
    # 双声道能量
    r0 = np.sqrt((np.array([seg2[0][i*hop:i*hop+fl] for i in range(n)]) ** 2).mean(1) + 1e-8)
    r1 = np.sqrt((np.array([seg2[1][i*hop:i*hop+fl] for i in range(n)]) ** 2).mean(1) + 1e-8)
    f.append(float(abs(r0.mean() - r1.mean()) / (r0.mean() + r1.mean() + 1e-6)))
    return np.array(f, dtype=np.float32)


def build(conv_ids, augment):
    torch.set_num_threads(4)
    X, Y, G = [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        if augment:
            wav, sr = load_wav(cid)
        for e in range(CTX, len(a) - TGT + 1, args.stride):
            ctx = a[e - CTX:e]
            base = ctx_v1(ctx)
            if augment:
                end8 = int(e * CHUNK_MS / 1000 * sr)
                seg = wav[:, max(0, end8 - WIN_SEC * sr):end8]
                if seg.shape[1] < SR:
                    pros = np.zeros(8, dtype=np.float32)
                else:
                    pros = f0_prosody_feats(seg)
                feat = np.concatenate([base, pros])
            else:
                feat = base
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(feat)
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
        if augment and (gi + 1) % 30 == 0:
            print(f"[f0] {gi+1}/{len(conv_ids)} convs feat'd", file=sys.stderr)
    return np.array(X, dtype=np.float32), np.array(Y), np.array(G)


def oof_bc(X, Y, G, conv_ids, folds):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(conv_ids))
    oof = np.zeros(len(X))
    for fi in range(folds):
        val = {perm[i] for i in range(len(conv_ids)) if i % folds == fi}
        tr = [i for i in range(len(X)) if G[i] not in val]
        va = [i for i in range(len(X)) if G[i] in val]
        y = Y[tr, BC]
        spw = (len(y) - y.sum()) / max(1, y.sum())
        c = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
        c.fit(X[tr], y)
        oof[va] = c.predict_proba(X[va])[:, 1]
    return oof


def bc_f1(oof, yt):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(yt, (oof >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    p, r, _, _ = precision_recall_fscore_support(yt, (oof >= bt).astype(int),
                                                 average='binary', zero_division=0)
    return bf, bt, p, r


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=40)
    args = ap.parse_args()

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs > 0:
        conv_ids = conv_ids[:args.convs]
    print(f"[f0] {len(conv_ids)} convs stride={args.stride}", file=sys.stderr)

    res = {}
    for aug, label in [(False, "A_context"), (True, "B_context+F0")]:
        X, Y, G = build(conv_ids, aug)
        oof = oof_bc(X, Y, G, conv_ids, args.folds)
        yt = Y[:, BC]
        f, t, p, r = bc_f1(oof, yt)
        res[label] = {"bc_f1": round(f, 4), "P": round(p, 3), "R": round(r, 3), "dim": X.shape[1]}
        print(f"[{label:<14}] BC F1={f:.4f} @{t:.2f} (P={p:.3f} R={r:.3f}) dim={X.shape[1]}", file=sys.stderr)

    a = res["A_context"]["bc_f1"]
    print(f"\n=== F0 互补性 (全量 OOF BC F1) ===")
    for label, rr in res.items():
        print(f"  {label:<14} BC={rr['bc_f1']:.4f} ({rr['bc_f1']-a:+.4f}) P={rr['P']} R={rr['R']}")
    print(f"\n判读: B-A > +0.01 = F0 与 context 正交互补(值得做); ≈0 = 重叠无增量")
    print(json.dumps({"cycle": "H-007-f0", "results": res}))


if __name__ == "__main__":
    main()
