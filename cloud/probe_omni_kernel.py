"""Omni audio encoder 表征 BC 可分性探针 (用户: 下载了不试 / 2->1先提特征).

chain-first发现: Omni audio_tower(Qwen2_5OmniAudioEncoder) config=whisper-style
(d1280/32层/128mel), whisper已证伪. 本探针确认 Omni encoder 表征对 BC 是否真同
whisper~0.6 (config像≠表征同, Omni是对话/多模态数据重训的).

做法: Omni processor 处理音频 → audio_tower 提表征 → mean-pool → kernel探针
(线性 vs RBF/RFF/MLP) 测 BC AUC. 对比 VAP 的 0.64.

Usage: python cloud/probe_omni_kernel.py --convs 80 --stride 80
"""
import argparse
import glob
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(8)

sys.path.insert(0, "tools/climb")
CTX, TGT, CHUNK_MS, SEED = 375, 25, 80, 42
BC = 2
WIN_SEC = int(os.environ.get("WIN_SEC", "10"))
OMNI = os.environ.get("OMNI", "models/Qwen2.5-Omni-3B")
TARGET_SR = 16000


def load_wav_mono16k(cid, win_sec):
    """取末 win_sec 双声道→mono→16k (Omni 音频输入)."""
    with wave.open(f"data/train/audio/{cid}.wav", "rb") as wf:
        sr = wf.getframerate(); raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=80)
    ap.add_argument("--stride", type=int, default=80)
    args = ap.parse_args()

    import torchaudio
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration as TH
    from transformers import Qwen2_5OmniProcessor
    from cycle_context import featurize as ctxfeat

    dev = "cuda"
    proc = Qwen2_5OmniProcessor.from_pretrained(OMNI)
    m = TH.from_pretrained(OMNI, torch_dtype=torch.float16, device_map=dev)
    enc = m.audio_tower.eval()
    fe = proc.feature_extractor  # whisper-style mel 特征提取器
    print(f"[omni] loaded audio_tower={type(enc).__name__}, fe sr={fe.sampling_rate}", file=sys.stderr)

    conv = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))[:args.convs]
    feats, y, G = [], [], []
    for gi, cid in enumerate(conv):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        wav, sr = load_wav_mono16k(cid, WIN_SEC)
        mono = wav.mean(0)  # 双声道→mono (Omni 单通道音频)
        for e in range(CTX, len(a) - TGT + 1, args.stride):
            end = int(e * CHUNK_MS / 1000 * sr)
            seg = mono[max(0, end - WIN_SEC * sr):end]
            if len(seg) < sr:
                continue
            seg16 = torchaudio.functional.resample(torch.from_numpy(seg), sr, TARGET_SR).numpy()
            # processor 提 mel
            inp = fe(seg16, sampling_rate=TARGET_SR, return_tensors="pt")
            feat_mel = inp["input_features"].to(dev).to(torch.float16)
            with torch.inference_mode():
                # audio_tower forward: 输入 mel → encoder 表征
                am = inp["attention_mask"] if "attention_mask" in inp else None
                kw = {"attention_mask": am.to(dev)} if am is not None else {}
                try:
                    out = enc(feat_mel, **kw)
                except TypeError:
                    out = enc(feat_mel)  # 某些版本 audio_tower 不收 attention_mask
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                rep = h.mean(dim=1).squeeze(0).float().cpu().numpy()  # mean-pool 时序
            feats.append(rep)
            fut = set(int(x) for x in a[e:e + TGT])
            y.append(1 if BC in fut else 0)
            G.append(gi)
        if (gi + 1) % 20 == 0:
            print(f"[omni] {gi+1}/{len(conv)} convs, {len(feats)} samples", file=sys.stderr)

    Xv = np.array(feats, dtype=np.float32)
    y = np.array(y); G = np.array(G)
    print(f"[omni] X={Xv.shape} BC率={y.mean():.3f}", file=sys.stderr)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, roc_auc_score
    from lightgbm import LGBMClassifier

    rng = np.random.default_rng(SEED); perm = rng.permutation(len(conv))
    def oof(mk):
        o = np.zeros(len(Xv))
        for fi in range(5):
            val = {perm[i] for i in range(len(conv)) if i % 5 == fi}
            tr = [i for i in range(len(Xv)) if G[i] not in val]
            va = [i for i in range(len(Xv)) if G[i] in val]
            sc = StandardScaler().fit(Xv[tr])
            c = mk(); c.fit(sc.transform(Xv[tr]), y[tr])
            o[va] = c.predict_proba(sc.transform(Xv[va]))[:, 1]
        return o

    g = 1.0 / Xv.shape[1]
    spw = (len(y) - y.sum()) / max(1, y.sum())
    models = {
        "L_linear": lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
        "R_rff2000": lambda: make_pipeline(RBFSampler(gamma=g, n_components=2000, random_state=SEED),
                                           LogisticRegression(class_weight="balanced", max_iter=2000)),
        "M_mlp": lambda: MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300,
                                       early_stopping=True, random_state=SEED),
        "T_lgbm": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                         scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED),
    }
    print(f"\n=== Omni audio encoder 表征 BC 可分性 (对比 VAP 0.64) ===")
    res = {}
    for name, mk in models.items():
        o = oof(mk)
        auc = roc_auc_score(y, o)
        bf = max(f1_score(y, (o >= t).astype(int), zero_division=0) for t in np.linspace(0.05, 0.95, 19))
        res[name] = {"bc_f1": round(float(bf), 4), "auc": round(float(auc), 4)}
        print(f"  {name:<12} BC_F1={bf:.4f}  AUC={auc:.4f}")
    print("\n判读: AUC >> VAP 0.64 = Omni encoder 比 whisper/VAP 强(值得微调); ≈0.6 = 同命")
    print(json.dumps({"cycle": "Omni-kernel-probe", "results": res}))


if __name__ == "__main__":
    main()
