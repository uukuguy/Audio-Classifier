"""climb cycle — paradigm=audio-cheap-fusion (H-A1, 测便宜音频能否救 BC).

YAGNI 第一步：不上 SSL 神经编码器，先测"廉价声学特征 + 上下文"的 late fusion
能否提升 BC（research 说 BC = 听者声道短促 burst while speaker continues，
靠 onset/energy/双声道对比，未必需要大编码器）。

特征 = context-v2 特征 + 每声道末窗声学统计（energy/ZCR/voicing + 双声道对比）。
若 BC F1 明显涨 → 值得上神经编码器；若不涨 → SSL 编码器期望也低，省一次大投入。

Usage: python tools/climb/cycle_audio_fusion.py <run_dir> [n_folds]
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torchaudio
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

# 复用 v2 的上下文特征
sys.path.insert(0, str(Path(__file__).parent))
from cycle_context_v2 import featurize as ctx_featurize  # noqa: E402

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL_TO_LABELID = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
NUM, CTX, TGT, STRIDE, CHUNK_MS, SR = 5, 375, 25, 5, 80, 8000
SEED = 42
# 末窗声学：取预测点前若干秒（BC 靠近未来，近窗更关键）
ACOUSTIC_WINDOWS_S = (1, 2, 4)


def acoustic_feats(wav: np.ndarray) -> list[float]:
    """wav [2, T] (8k stereo) → 每声道多窗声学统计 + 双声道对比。"""
    feats = []
    per_ch_energy = []
    for ws in ACOUSTIC_WINDOWS_S:
        n = ws * SR
        for c in range(2):
            seg = wav[c, -n:] if wav.shape[1] >= n else wav[c]
            if seg.size < 2:
                feats += [0.0, 0.0, 0.0]; per_ch_energy.append(0.0); continue
            e = float((seg ** 2).mean())
            zcr = float(((seg[1:] * seg[:-1]) < 0).mean())
            voiced = float((np.abs(seg) > 0.02).mean())  # 粗 voicing 代理
            feats += [e, zcr, voiced]
            per_ch_energy.append(e)
    # 双声道对比（谁在说 = turn/BC 关键）：末2s 能量比 + 差
    e0, e1 = per_ch_energy[2], per_ch_energy[3]  # ws=2 的两声道
    feats.append(e0 / (e1 + 1e-9))
    feats.append(e0 - e1)
    feats.append(float(abs(e0 - e1) / (e0 + e1 + 1e-9)))  # 主导度
    return feats


def load_wav(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav[:2].numpy()


def build_train(conv_ids, label_files, audio_dir):
    X, Y, G = [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(label_files[cid])
        wav = load_wav(f"{audio_dir}/{cid}.wav")
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE):
            ctx = a[e - CTX:e].astype(int)
            end_sample = int(e * CHUNK_MS / 1000 * SR)
            wslice = wav[:, max(0, end_sample - 4 * SR):end_sample]
            if wslice.shape[1] < SR:  # 不足1s pad
                wslice = np.pad(wslice, ((0, 0), (SR - wslice.shape[1], 0)))
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(np.concatenate([ctx_featurize(ctx), acoustic_feats(wslice)]))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=int), np.array(G)


def mk_lgbm(spw):
    return LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=48,
                          scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED)


def tune_threshold(y, p):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, float(t)
    return bt, bf


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/runs/climb/_adhoc_audio")
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_dir.mkdir(parents=True, exist_ok=True)

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    print(f"[audio-fuse] building features ({len(conv_ids)} convs, audio IO)...", file=sys.stderr)
    X, Y, G = build_train(conv_ids, label_files, "data/train/audio")
    print(f"[audio-fuse] {len(X)} windows feat_dim={X.shape[1]} (ctx + {X.shape[1]-71 if X.shape[1]>71 else '?'} acoustic)", file=sys.stderr)

    gkf = GroupKFold(n_splits=n_folds)
    oof = {k: np.zeros(len(X)) for k in range(NUM)}
    for fold, (tr, va) in enumerate(gkf.split(X, Y[:, 0], groups=G)):
        for k in range(NUM):
            ytr = Y[tr, k]
            spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
            oof[k][va] = mk_lgbm(spw).fit(X[tr], ytr).predict_proba(X[va])[:, 1]
        print(f"[audio-fuse] fold {fold+1}/{n_folds}", file=sys.stderr)

    thr, f1s = {}, {}
    for k in range(NUM):
        t, f = tune_threshold(Y[:, k], oof[k])
        thr[k], f1s[k] = t, f
        print(f"[audio-fuse] {LABELS[k]:3s} thr={t:.2f} F1={f:.3f}", file=sys.stderr)
    macro = float(np.mean(list(f1s.values())))
    print(f"[audio-fuse] OOF Macro-F1 = {macro:.4f}  (对比 context-only：看 BC 是否涨)", file=sys.stderr)

    # test
    test_ctx = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_ctx]
    Xte = []
    for p in test_ctx:
        ctx = np.load(p).astype(int)
        wav = load_wav(f"data/test/audio/{Path(p).stem}.wav")
        Xte.append(np.concatenate([ctx_featurize(ctx), acoustic_feats(wav[:, -4 * SR:])]))
    Xte = np.array(Xte, dtype=np.float32)
    preds = {}
    for k in range(NUM):
        spw = (len(Y) - Y[:, k].sum()) / max(1, Y[:, k].sum())
        preds[k] = (mk_lgbm(spw).fit(X, Y[:, k]).predict_proba(Xte)[:, 1] >= thr[k]).astype(int)

    csv_path = run_dir / "pred_test1.csv"
    with open(csv_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT_COLS) + "\n")
        for i, sid in enumerate(seg_ids):
            f.write(",".join([sid] + [str(int(preds[COL_TO_LABELID[c]][i])) for c in SUBMIT_COLS]) + "\n")

    per_sub = {c: round(f1s[COL_TO_LABELID[c]], 4) for c in ["c", "na", "t", "i", "bc"]}
    (run_dir / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "audio-cheap-fusion", "hypothesis_id": "H-A1", "cv_macro_f1": round(macro, 4),
        "per_sub_f1": per_sub, "thresholds": {LABELS[k]: round(thr[k], 2) for k in range(NUM)},
        "method": f"{n_folds}-fold OOF, ctx feats + per-channel acoustic (energy/zcr/voicing/contrast)",
    }, ensure_ascii=False, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps({
        "cycle": 3, "hypothesis_id": "H-A1", "paradigm": "audio-cheap-fusion",
        "start": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"score": round(macro, 4), "per_sub": per_sub}))


if __name__ == "__main__":
    main()
