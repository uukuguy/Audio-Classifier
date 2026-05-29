"""VAP (Voice Activity Projection) 适配赛题 turn-taking 预测。

范式转向 (2026-05-30): whisper-large-v3 错 (ASR族+193ms前向, 全量30-63h不可行)。
VAP (Ekstedt&Skantze Interspeech2022) 是 turn-taking 学术 SOTA:
  - CPC encoder: causal(满足因果约束) + 256维轻量CNN(全量可行) + 16kHz
  - 双声道 cross-attn 融合 → out["x"] [B, T, 256] 50Hz帧序列
  - VAP预训练在对话语料,微调BC文献+0.3
  - 官方权重 baselines/VAP/example/checkpoints/VAP_state_dict.pt (23MB, 含CPC)

赛题适配架构:
  过去N秒双声道8k → 重采样16k [B,2,16000*N]
    → VAP.forward → out["x"] [B, T_frames, 256] (50Hz, causal融合)
    → 取末窗帧(预测点t在末尾) mean-pool 末M帧
    → + 历史标签 ctx 特征(46维, 复用cycle1, C/NA强信号)
    → 5类头(LayerNorm+MLP) → sigmoid 5类 (C/T/BC/I/NA)

CPC 默认冻结(快). --unfreeze 解冻微调(冻结输了, 微调是方向, 文献BC+0.3靠微调)。
CV: cap切片 5-fold 会话级。阈值: cycle1 固定(阈值铁律)。

Usage (云):
  # 速度验证(上云第一步, 实测GPU ms/段)
  python cloud/train_vap.py --convs 20 --slice-cap 5 --epochs 3 --folds 1 --win-sec 10 --run-dir tools/runs/climb/vap-speed
  # 小验证(可行性)
  python cloud/train_vap.py --convs 40 --slice-cap 5 --epochs 10 --folds 5 --win-sec 10 --run-dir tools/runs/climb/vap-smoke
  # 全量
  python cloud/train_vap.py --convs 0 --slice-cap 20 --epochs 10 --folds 5 --win-sec 10 --unfreeze --run-dir tools/runs/climb/vap-full

本机 dry-run (CPU, 验证流程不求速度):
  python cloud/train_vap.py --convs 3 --slice-cap 2 --epochs 1 --folds 2 --win-sec 8 --run-dir tools/runs/climb/_vap_dry
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

# ── VAP repo on path ──
VAP_ROOT = os.environ.get("VAP_ROOT", str(Path(__file__).parent.parent / "baselines/VAP"))
sys.path.insert(0, VAP_ROOT)
sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat  # noqa: E402

# ── constants ──
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
CTX, TGT, CHUNK_MS = 375, 25, 80
SR16 = 16000
VAP_HZ = 50           # VAP frame rate (20ms/frame)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# NOTE: MPS produces nan from step 2 (Apple MPS numerical bug in CPC conv/attn,
# confirmed 2026-05-30: CPU 18 steps clean, MPS step2 logits=nan). Use CUDA on cloud,
# or CPU locally. Do NOT trust MPS numerics for this model.
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

VAP_CKPT = os.environ.get("VAP_CKPT", str(Path(VAP_ROOT) / "example/checkpoints/VAP_state_dict.pt"))
THR_CYCLE1 = {0: 0.05, 4: 0.05, 1: 0.50, 3: 0.50, 2: 0.50}  # variant-F SOTA calcified


# ── build VAP encoder (CPC + stereo transformer, pretrained) ─────────────────
def build_vap(unfreeze: bool = False) -> nn.Module:
    """Load VAP (CPC encoder + stereo cross-attn transformer) with pretrained weights."""
    from vap.modules.encoder import EncoderCPC
    from vap.modules.modules import TransformerStereo
    from vap.modules.VAP import VAP

    enc = EncoderCPC(load_pretrained=False)  # weights come from VAP_state_dict below
    tr = TransformerStereo(dim=enc.dim)
    model = VAP(enc, tr)
    sd = torch.load(VAP_CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[vap] loaded {VAP_CKPT}: missing={len(missing)} unexpected={len(unexpected)}",
          file=sys.stderr)
    # Always start fully frozen.
    for p in model.parameters():
        p.requires_grad_(False)
    if unfreeze:
        # Unfreeze ONLY the CPC encoder (audio representation adaptation).
        # Do NOT unfreeze the stereo transformer: its alibi slopes `self.m` are
        # designed non-trainable (modules.py:128) and a blanket unfreeze makes the
        # computed alibi mask a non-leaf tensor → get_alibi_mask's requires_grad_(False)
        # crashes ("can only change requires_grad flags of leaf variables").
        # Caught locally before cloud (2026-05-30). Encoder-only fine-tune is also the
        # right strategy: adapt representation, keep turn-taking transformer intact.
        for p in model.encoder.parameters():
            p.requires_grad_(True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[vap] CPC encoder unfrozen ({n_tr:,} trainable); transformer frozen", file=sys.stderr)
    else:
        print("[vap] VAP fully frozen (only head trains)", file=sys.stderr)
    return model


# ── audio I/O ────────────────────────────────────────────────────────────────
def load_wav_8k_dual(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


# ── dataset ────────────────────────────────────────────────────────────────
def pick_slice_ends(label_len: int, cap: int) -> list[int]:
    lo, hi = CTX, label_len - TGT
    if lo > hi:
        return []
    if cap <= 0:
        return list(range(lo, hi + 1, 5 * 8))
    step = max(1, (hi - lo) // cap)
    return list(range(lo, hi + 1, step))[:cap]


class TurnTakingVAPDataset(Dataset):
    """Each sample = (waveform[2, win_samples], ctx_feat[46], target[5])."""

    def __init__(self, conv_ids: list[str], split: str, slice_cap: int, win_sec: int):
        self.split = split
        self.win_sec = win_sec
        self.win_samples = win_sec * SR16
        self.samples: list[tuple[str, int]] = []
        for cid in conv_ids:
            if split == "train":
                a = np.load(f"data/train/labels/{cid}.npy")
                ends = pick_slice_ends(len(a), slice_cap)
            else:
                ends = [CTX]
            for e in ends:
                self.samples.append((cid, e))
        self._labels: dict[str, np.ndarray] = {}
        self._wav: dict[str, tuple[np.ndarray, int]] = {}

    def _get_labels(self, cid: str) -> np.ndarray:
        if cid not in self._labels:
            if self.split == "train":
                self._labels[cid] = np.load(f"data/train/labels/{cid}.npy")
            else:
                self._labels[cid] = np.load(f"data/test/context/{cid}.npy")
        return self._labels[cid]

    def _get_wav(self, cid: str) -> tuple[np.ndarray, int]:
        if cid not in self._wav:
            path = f"data/train/audio/{cid}.wav" if self.split == "train" else f"data/test/audio/{cid}.wav"
            self._wav[cid] = load_wav_8k_dual(path)
        return self._wav[cid]

    def __len__(self) -> int:
        return len(self.samples)

    def get_target(self, idx: int) -> np.ndarray:
        """Labels-only (no audio) — for pos_weight (avoid the train_lora 1.5h bug)."""
        cid, end = self.samples[idx]
        if self.split != "train":
            return np.zeros(5, dtype=np.float32)
        labels = self._get_labels(cid)
        fut = set(int(x) for x in labels[end:end + TGT])
        return np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)

    def __getitem__(self, idx: int):
        cid, end = self.samples[idx]
        labels = self._get_labels(cid)
        wav, sr = self._get_wav(cid)

        if self.split == "train":
            ctx = ctxfeat(labels[end - CTX:end].astype(int))
            fut = set(int(x) for x in labels[end:end + TGT])
            target = np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)
            end_sample_8k = int(end * CHUNK_MS / 1000 * sr)
        else:
            ctx = ctxfeat(labels.astype(int))
            target = np.zeros(5, dtype=np.float32)
            end_sample_8k = wav.shape[1]

        # window: last win_sec before prediction point (causal)
        win_8k = self.win_sec * sr
        start_8k = max(0, end_sample_8k - win_8k)
        seg = wav[:, start_8k:end_sample_8k]  # [2, samples_8k]
        # resample 8k → 16k
        seg_t = torch.from_numpy(seg)
        seg16 = torchaudio.functional.resample(seg_t, sr, SR16)  # [2, samples_16k]
        # pad/trim to fixed win_samples (left-pad so prediction point stays at end)
        cur = seg16.shape[1]
        if cur < self.win_samples:
            seg16 = torch.nn.functional.pad(seg16, (self.win_samples - cur, 0))
        elif cur > self.win_samples:
            seg16 = seg16[:, -self.win_samples:]

        return seg16, torch.from_numpy(ctx), torch.from_numpy(target)


def collate(batch):
    wavs, ctxs, tgts = zip(*batch)
    return torch.stack(wavs), torch.stack(ctxs), torch.stack(tgts)


# ── model: VAP backbone + ctx fusion + 5-class head ──────────────────────────
class VAPTurnTaking(nn.Module):
    def __init__(self, vap: nn.Module, ctx_dim: int, vap_dim: int = 256,
                 pool_frames: int = 10):
        super().__init__()
        self.vap = vap
        self.pool_frames = pool_frames  # mean-pool last M frames at prediction point
        # LayerNorm (not BatchNorm1d): ctx is a feature vector of statistics with
        # constant/degenerate columns (var=0). BatchNorm on those + small batches
        # produces nan in training (caught locally 2026-05-30). LayerNorm is per-sample,
        # no running stats, robust to degenerate dims.
        self.cn = nn.LayerNorm(ctx_dim)
        fin = ctx_dim + vap_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fin),
            nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

    def forward(self, waveform: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        out = self.vap(waveform)          # out["x"]: [B, T, 256] fused stereo frames
        x = out["x"]
        # prediction point = end of window → pool last M frames
        audio_feat = x[:, -self.pool_frames:, :].mean(dim=1)  # [B, 256]
        c = self.cn(ctx)
        return self.head(torch.cat([c, audio_feat], dim=-1))


# ── train one fold ───────────────────────────────────────────────────────────
def train_fold(model, ds, tr_idx, epochs, pw, batch_size, lr_head, lr_vap, unfreeze):
    model.to(DEV)
    head_params = [p for n, p in model.named_parameters() if "vap" not in n and p.requires_grad]
    vap_params = [p for n, p in model.named_parameters() if "vap" in n and p.requires_grad]
    groups = [{"params": head_params, "lr": lr_head, "weight_decay": 1e-4}]
    if unfreeze and vap_params:
        groups.append({"params": vap_params, "lr": lr_vap, "weight_decay": 0.01})
    opt = torch.optim.AdamW(groups)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    # drop_last avoids a trailing size-1 batch that makes BatchNorm1d produce nan.
    loader = DataLoader(ds, batch_size=batch_size,
                        sampler=torch.utils.data.SubsetRandomSampler(tr_idx),
                        collate_fn=collate, num_workers=2, pin_memory=(DEV == "cuda"),
                        drop_last=(len(tr_idx) > batch_size))
    n_skip = 0
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for wav, ctx, tgt in loader:
            wav, ctx, tgt = wav.to(DEV), ctx.to(DEV), tgt.to(DEV)
            logits = model(wav, ctx)
            loss = crit(logits, tgt)
            if not torch.isfinite(loss):  # skip degenerate batch, don't poison weights
                n_skip += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            tot += float(loss); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            vram = torch.cuda.memory_allocated() / 1024**3 if DEV == "cuda" else 0
            print(f"[vap]   epoch {ep+1}/{epochs} loss={tot/max(1,nb):.4f} VRAM={vram:.1f}GB"
                  + (f" (skipped {n_skip} nan batches)" if n_skip else ""),
                  file=sys.stderr)
    model.eval()
    return model


@torch.no_grad()
def predict(model, ds, idx, batch_size=32):
    sub = torch.utils.data.Subset(ds, idx)
    loader = DataLoader(sub, batch_size=batch_size, collate_fn=collate,
                        num_workers=2, pin_memory=(DEV == "cuda"))
    out = []
    for wav, ctx, _ in loader:
        wav, ctx = wav.to(DEV), ctx.to(DEV)
        out.append(torch.sigmoid(model(wav, ctx)).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_test(models, test_ds, batch_size=32):
    n = len(test_ds)
    probs = np.zeros((n, 5), dtype=np.float32)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=2, pin_memory=(DEV == "cuda"))
    for m in models:
        m.to(DEV); m.eval()
        fp = []
        for wav, ctx, _ in loader:
            wav, ctx = wav.to(DEV), ctx.to(DEV)
            fp.append(torch.sigmoid(m(wav, ctx)).cpu().numpy())
        probs += np.concatenate(fp, axis=0)
        # release this model's VRAM before loading the next (avoid 5-model pileup)
        m.cpu()
        if DEV == "cuda":
            torch.cuda.empty_cache()
    return probs / len(models)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-cap", type=int, default=20)
    ap.add_argument("--win-sec", type=int, default=10, help="audio window before pred point")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-vap", type=float, default=1e-5)
    ap.add_argument("--pool-frames", type=int, default=10)
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--run-dir", default="tools/runs/climb/vap-run")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    print(f"[vap] dev={DEV} ckpt={VAP_CKPT}", file=sys.stderr)
    print(f"[vap] win_sec={args.win_sec} slice_cap={args.slice_cap} epochs={args.epochs} "
          f"folds={args.folds} batch={args.batch_size} unfreeze={args.unfreeze}", file=sys.stderr)

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs > 0:
        conv_ids = conv_ids[:args.convs]
    print(f"[vap] {len(conv_ids)} convs", file=sys.stderr)

    train_ds = TurnTakingVAPDataset(conv_ids, "train", args.slice_cap, args.win_sec)
    print(f"[vap] dataset size: {len(train_ds)}", file=sys.stderr)

    ctx_dim = train_ds[0][1].shape[0]
    print(f"[vap] ctx_dim={ctx_dim}", file=sys.stderr)

    # pos_weight (labels only)
    targets = np.array([train_ds.get_target(i) for i in range(len(train_ds))])
    pw = torch.tensor([(len(targets) - targets[:, k].sum()) / max(1, targets[:, k].sum())
                       for k in range(5)]).float().clamp(max=10).to(DEV)
    print(f"[vap] pos_weight: {[round(float(pw[k]), 2) for k in range(5)]}", file=sys.stderr)

    conv_to_idx = {cid: i for i, cid in enumerate(conv_ids)}
    sample_groups = np.array([conv_to_idx[s[0]] for s in train_ds.samples])
    n_convs = len(conv_ids)
    conv_perm = np.random.default_rng(SEED).permutation(n_convs)

    oof = np.zeros((len(train_ds), 5), dtype=np.float32)
    models = []
    t_total = time.time()
    for fi in range(args.folds):
        t0 = time.time()
        if args.folds == 1:
            # folds=1 (speed/quick run): i%1==0 would put ALL convs in val → empty
            # train. Use a 80/20 holdout instead (last 20% of shuffled convs = val).
            n_val = max(1, n_convs // 5)
            val_convs = {conv_perm[i] for i in range(n_convs - n_val, n_convs)}
        else:
            val_convs = {conv_perm[i] for i in range(n_convs) if i % args.folds == fi}
        tr_idx = [i for i in range(len(train_ds)) if sample_groups[i] not in val_convs]
        va_idx = [i for i in range(len(train_ds)) if sample_groups[i] in val_convs]
        print(f"[vap] fold {fi+1}/{args.folds}: train={len(tr_idx)} val={len(va_idx)}", file=sys.stderr)

        vap = build_vap(unfreeze=args.unfreeze)
        model = VAPTurnTaking(vap, ctx_dim=ctx_dim, vap_dim=256, pool_frames=args.pool_frames)
        model = train_fold(model, train_ds, tr_idx, args.epochs, pw,
                           args.batch_size, args.lr_head, args.lr_vap, args.unfreeze)
        if va_idx:
            oof[va_idx] = predict(model, train_ds, va_idx, args.batch_size)
        models.append(model)
        ckpt = run / f"fold{fi}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"[vap] fold {fi+1} done in {(time.time()-t0)/60:.1f}min, saved {ckpt}", file=sys.stderr)
        if fi < args.folds - 1 and DEV == "cuda":
            models[-1] = models[-1].cpu()
            torch.cuda.empty_cache()

    print(f"[vap] all folds done in {(time.time()-t_total)/60:.1f}min", file=sys.stderr)

    # cap1 slice CV (first slice per conv)
    cap1_idx, seen = [], set()
    for i, (cid, _e) in enumerate(train_ds.samples):
        if cid not in seen:
            cap1_idx.append(i); seen.add(cid)

    f1_cap1, thr_cap1 = {}, {}
    for k in range(5):
        bf, bt = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(targets[cap1_idx, k], (oof[cap1_idx, k] >= t).astype(int), zero_division=0)
            if f > bf:
                bf, bt = f, float(t)
        f1_cap1[k], thr_cap1[k] = bf, bt
    macro_cap1 = float(np.mean(list(f1_cap1.values())))
    print(f"[vap] cap1 CV macro={macro_cap1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cap1[k]:.3f}@{thr_cap1[k]:.2f}" for k in range(5)), file=sys.stderr)
    print(f"[vap] ★BC={f1_cap1[2]:.3f} (LGBM=0.222, frozen-whisper=0.20)", file=sys.stderr)

    f1_cyc = {k: float(f1_score(targets[cap1_idx, k], (oof[cap1_idx, k] >= THR_CYCLE1[k]).astype(int),
                                zero_division=0)) for k in range(5)}
    macro_cyc = float(np.mean(list(f1_cyc.values())))
    print(f"[vap] cap1 cycle1-thr macro={macro_cyc:.4f}", file=sys.stderr)

    # test prediction + CSV
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    if test_ids:
        test_ds = TurnTakingVAPDataset(test_ids, "test", 1, args.win_sec)
        probs = predict_test(models, test_ds, args.batch_size)
        SUBMIT = ["c", "na", "i", "bc", "t"]
        COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
        with open(run / "pred_test1.csv", "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(test_ids):
                f.write(",".join([sid] + [str(int(probs[i, COL2K[c]] >= THR_CYCLE1[COL2K[c]]))
                                          for c in SUBMIT]) + "\n")
        np.savez_compressed(run / "test_probs.npz", probs=probs, ids=np.array(test_ids))
        print(f"[vap] wrote pred_test1.csv ({len(test_ids)} segs)", file=sys.stderr)

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "vap-cpc-stereo", "win_sec": args.win_sec, "unfreeze": args.unfreeze,
        "cap1_macro_f1": round(macro_cap1, 4),
        "per_sub_cap1": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)},
        "cap1_cycle1_thr": round(macro_cyc, 4),
        "slice_cap": args.slice_cap, "epochs": args.epochs, "folds": args.folds,
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"cap1_score": round(macro_cap1, 4), "bc_f1": round(f1_cap1[2], 4),
                      "cap1_cycle1": round(macro_cyc, 4)}))


if __name__ == "__main__":
    main()
