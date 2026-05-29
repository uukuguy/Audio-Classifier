"""LoRA 微调 whisper-large-v3 encoder 攻 BC — 在线前向 (不读缓存)。

冻结路线 falsified (最高线上 0.671 << SOTA 0.7124):
  冻结 encoder → 小头学不到 turn-taking 特异信号 (韵律 timing / onset / 双声道交互)。
LoRA 让 encoder 微调适应任务: 可训练 ~5-15M 参数 (<2%), 显存友好。

架构:
  原始音频 (.wav, 8kHz 双声道)
    → resample → 16kHz → WhisperFeatureExtractor (log-mel)
    → whisper-large-v3 encoder + LoRA (r=32, α=16, q_proj/v_proj)
    → 帧序列 [2, 80, 1280] (双声道分别过 encoder)
    → proj → cross-attn(query 聚合) → 音频 2 向量
    + context 手工特征 (80d) → MLP head
    → 5 类 sigmoid + BCE loss

训练数据: cap5 切片 (每通 5 个独立片段 ≈ 1845 样本), 模拟 test 独立 30s 切片。
CV: 5-fold 会话级 split。阈值: cycle1 固定 (阈值铁律 3 验, 不在 OOF 调)。

Usage (云终端):
  # 冒烟
  python cloud/train_lora.py --convs 40 --epochs 10 --slice-cap 5 --run-dir tools/runs/climb/lora-smoke
  # 全量
  python cloud/train_lora.py --convs 0 --epochs 50 --slice-cap 5 --run-dir tools/runs/climb/lora-full
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
from transformers import WhisperFeatureExtractor, WhisperModel

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
CTX, TGT, CHUNK_MS = 375, 25, 80
SR16 = 16000
CTX_SEC = 8          # 末 8s 音频上下文 (与 extract 版一致)
TAIL_FRAMES = 400    # whisper 30s→1500 帧, 取末 8s≈400 帧
DS_FRAMES = 80       # 降采样到 80 帧
FEAT_DIM = 1280      # whisper-large-v3 hidden
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── env-configurable knobs (experiment values NEVER touch baseline defaults) ──
MODEL_DIR = os.environ.get("WHISPER_DIR", str(Path.home() / ".cache/manual_models/whisper-large-v3"))

# LoRA hyperparams (env-overridable for sweeps)
LORA_R = int(os.environ.get("LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
LR_LORA = float(os.environ.get("LR_LORA", "2e-4"))
LR_HEAD = float(os.environ.get("LR_HEAD", "1e-3"))
WEIGHT_DECAY_LORA = float(os.environ.get("WD_LORA", "0.01"))
WEIGHT_DECAY_HEAD = float(os.environ.get("WD_HEAD", "1e-4"))

# ── threshold presets (阈值铁律: 不在 OOF 调激进阈值) ──
THR_CYCLE1 = {0: 0.05, 4: 0.05, 1: 0.50, 3: 0.50, 2: 0.50}  # variant-F SOTA 钙化


# ── whisper encoder + LoRA wrapper ─────────────────────────────────────────
def build_encoder_with_lora() -> nn.Module:
    """Load whisper-large-v3 encoder + apply LoRA via PEFT."""
    from peft import LoraConfig, get_peft_model

    base = WhisperModel.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).encoder
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    enc = get_peft_model(base, lora_cfg)
    enc.print_trainable_parameters()
    return enc


# ── audio I/O ──────────────────────────────────────────────────────────────
_fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)


def load_wav_8k_dual(wav_path: str) -> tuple[np.ndarray, int]:
    """Load 8kHz dual-channel wav → (2, samples) float32, sr."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def audio_to_logmel(seg_8k: np.ndarray, sr_orig: int) -> torch.Tensor:
    """Resample 8k→16k, extract log-mel → [1, 128, T] tensor."""
    t = torch.tensor(seg_8k)
    r16 = torchaudio.functional.resample(t, sr_orig, SR16).numpy()
    feats = _fe([r16], sampling_rate=SR16, return_tensors="pt")
    return feats.input_features  # [1, 128, T]


# ── dataset ────────────────────────────────────────────────────────────────
def pick_slice_ends(label_len: int, cap: int) -> list[int]:
    """Pick up to `cap` independent slice endpoints from a conversation label array.
    Evenly spaced across the conversation, each >= CTX and < label_len - TGT.
    """
    lo, hi = CTX, label_len - TGT
    if lo > hi:
        return []
    if cap <= 0:
        return list(range(lo, hi + 1, 5 * 8))  # stride_mult=8 fallback
    step = max(1, (hi - lo) // cap)
    ends = list(range(lo, hi + 1, step))
    return ends[:cap]


class TurnTakingSliceDataset(Dataset):
    """Slice-based training data: raw wav → online whisper encoder (with LoRA).
    Each sample = (mel_ch0, mel_ch1, ctx_feat, target).
    """

    def __init__(self, conv_ids: list[str], split: str, slice_cap: int = 5):
        self.samples: list[tuple[str, int, str]] = []  # (cid, end_chunk, split)
        for cid in conv_ids:
            if split == "train":
                a = np.load(f"data/train/labels/{cid}.npy")
                ends = pick_slice_ends(len(a), slice_cap)
            else:
                ends = [CTX]  # test: single window per segment
            for e in ends:
                self.samples.append((cid, e, split))

        # preload labels + wav metadata (lazy load audio on __getitem__)
        self._labels: dict[tuple[str, str], np.ndarray] = {}
        self._wav_cache: dict[str, tuple[np.ndarray, int]] = {}

    def _get_wav(self, cid: str, split: str) -> tuple[np.ndarray, int]:
        if cid not in self._wav_cache:
            if split == "train":
                path = f"data/train/audio/{cid}.wav"
            else:
                path = f"data/test/audio/{cid}.wav"
            self._wav_cache[cid] = load_wav_8k_dual(path)
        return self._wav_cache[cid]

    def _get_labels(self, cid: str, split: str) -> np.ndarray:
        key = (cid, split)
        if key not in self._labels:
            if split == "train":
                self._labels[key] = np.load(f"data/train/labels/{cid}.npy")
            else:
                # test labels = context
                self._labels[key] = np.load(f"data/test/context/{cid}.npy")
        return self._labels[key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        cid, end, split = self.samples[idx]
        labels = self._get_labels(cid, split)
        wav, sr = self._get_wav(cid, split)

        # context features
        if split == "train":
            ctx = ctxfeat(labels[end - CTX:end].astype(int))
            fut = set(int(x) for x in labels[end:end + TGT])
            target = np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)
        else:
            ctx = ctxfeat(labels.astype(int))
            target = np.zeros(5, dtype=np.float32)  # dummy, not used for test

        # audio segment: last 8s before endpoint
        end_sample = int(end * CHUNK_MS / 1000 * sr) if split == "train" else wav.shape[1]
        start_sample = max(0, end_sample - CTX_SEC * sr)

        mel_ch0 = audio_to_logmel(wav[0, start_sample:end_sample], sr)  # [1, 128, T]
        mel_ch1 = audio_to_logmel(wav[1, start_sample:end_sample], sr)

        return mel_ch0.squeeze(0), mel_ch1.squeeze(0), \
               torch.from_numpy(ctx), torch.from_numpy(target)


def collate_variable_mel(batch):
    """Pad variable-length mel spectrograms to same time dim within batch."""
    mel0, mel1, ctxs, tgts = zip(*batch)
    # find max time dim
    max_t = max(m.shape[-1] for m in mel0)
    # pad
    mel0_p = torch.stack([torch.nn.functional.pad(m, (0, max_t - m.shape[-1])) for m in mel0])
    mel1_p = torch.stack([torch.nn.functional.pad(m, (0, max_t - m.shape[-1])) for m in mel1])
    return mel0_p, mel1_p, torch.stack(ctxs), torch.stack(tgts)


# ── model ──────────────────────────────────────────────────────────────────
class WhisperVAPLoRA(nn.Module):
    """Whisper encoder + LoRA → dual-channel cross-attn → context fusion → 5-class head."""

    def __init__(self, ctx_dim: int, encoder: nn.Module, wd: int = FEAT_DIM, d: int = 192):
        super().__init__()
        self.encoder = encoder
        # freeze BatchNorm-like layers in encoder (if any) — LoRA handles trainable parts
        self.proj = nn.Sequential(nn.Linear(wd, d), nn.LayerNorm(d), nn.GELU())
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cn = nn.BatchNorm1d(ctx_dim)
        fin = ctx_dim + 2 * d
        self.head = nn.Sequential(
            nn.LayerNorm(fin),
            nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

    def _encode_channel(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [B, 128, T] → encoder → downsample → proj → [B, 80, d]."""
        # match encoder dtype (bf16 on CUDA)
        enc_dtype = next(self.encoder.parameters()).dtype
        mel = mel.to(enc_dtype)
        enc = self.encoder(mel)
        h = enc.last_hidden_state  # [B, T_enc, 1280]
        # take tail CTX_SEC frames + downsample
        tail = h[:, -TAIL_FRAMES:, :]
        ds = torch.nn.functional.adaptive_avg_pool1d(
            tail.transpose(1, 2).float(), DS_FRAMES
        ).transpose(1, 2)  # [B, 80, 1280] fp32
        return self.proj(ds)  # [B, 80, d] fp32

    def forward(self, mel0: torch.Tensor, mel1: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        c = self.cn(ctx)
        a = self._encode_channel(mel0)   # [B, 80, d]
        b = self._encode_channel(mel1)   # [B, 80, d]
        B = a.shape[0]
        q = self.q.expand(B, -1, -1)
        ca, _ = self.cross(q, b, b)
        cb, _ = self.cross(q, a, a)
        return self.head(torch.cat([c, ca.squeeze(1), cb.squeeze(1)], -1))


# ── training loop ──────────────────────────────────────────────────────────
def train_fold(
    model: WhisperVAPLoRA,
    train_ds: TurnTakingSliceDataset,
    train_idx: list[int],
    epochs: int,
    pw: torch.Tensor,
    batch_size: int,
    lr_lora: float,
    lr_head: float,
) -> WhisperVAPLoRA:
    model.to(DEV)
    # separate param groups: LoRA slow, head fast
    lora_params = []
    head_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name or "encoder" in name:
            lora_params.append(p)
        else:
            head_params.append(p)

    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": lr_lora, "weight_decay": WEIGHT_DECAY_LORA},
        {"params": head_params, "lr": lr_head, "weight_decay": WEIGHT_DECAY_HEAD},
    ])
    total_steps = epochs * ((len(train_idx) + batch_size - 1) // batch_size)
    warmup = max(1, int(0.05 * total_steps))

    # manual cosine with warmup
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        collate_fn=collate_variable_mel, num_workers=2, pin_memory=True,
    )

    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for mel0, mel1, ctx, tgt in loader:
            mel0 = mel0.to(DEV, non_blocking=True)
            mel1 = mel1.to(DEV, non_blocking=True)
            ctx = ctx.to(DEV, non_blocking=True)
            tgt = tgt.to(DEV, non_blocking=True)

            opt.zero_grad()
            logits = model(mel0, mel1, ctx)
            loss = crit(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            epoch_loss += float(loss)
            n_batch += 1

        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            lr_now = sched.get_last_lr()[0]
            print(f"[lora]   epoch {ep+1}/{epochs} loss={epoch_loss/n_batch:.4f} lr={lr_now:.2e}",
                  file=sys.stderr)

    model.eval()
    return model


@torch.no_grad()
def predict_oof(
    model: WhisperVAPLoRA,
    dataset: TurnTakingSliceDataset,
    idx: list[int],
    batch_size: int = 256,
) -> np.ndarray:
    """OOF prediction on given indices."""
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=torch.utils.data.SequentialSampler.__new__(
            torch.utils.data.SequentialSampler
        ),
        collate_fn=collate_variable_mel, num_workers=2, pin_memory=True,
    )
    # Use subset
    sub = torch.utils.data.Subset(dataset, idx)
    loader = DataLoader(sub, batch_size=batch_size, collate_fn=collate_variable_mel,
                        num_workers=2, pin_memory=True)
    out = []
    for mel0, mel1, ctx, _ in loader:
        mel0 = mel0.to(DEV, non_blocking=True)
        mel1 = mel1.to(DEV, non_blocking=True)
        ctx = ctx.to(DEV, non_blocking=True)
        probs = torch.sigmoid(model(mel0, mel1, ctx))
        out.append(probs.cpu().numpy())
    return np.concatenate(out, axis=0)


# ── test prediction ────────────────────────────────────────────────────────
def predict_test(
    models: list[WhisperVAPLoRA],
    test_ds: TurnTakingSliceDataset,
    batch_size: int = 64,
) -> np.ndarray:
    """Ensemble predict on test set (prob average over folds)."""
    n = len(test_ds)
    probs = np.zeros((n, 5), dtype=np.float32)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_variable_mel, num_workers=2, pin_memory=True)
    for m in models:
        m.eval()
        fold_probs = []
        with torch.no_grad():
            for mel0, mel1, ctx, _ in loader:
                mel0 = mel0.to(DEV, non_blocking=True)
                mel1 = mel1.to(DEV, non_blocking=True)
                ctx = ctx.to(DEV, non_blocking=True)
                p = torch.sigmoid(m(mel0, mel1, ctx))
                fold_probs.append(p.cpu().numpy())
        probs += np.concatenate(fold_probs, axis=0)
    probs /= len(models)
    return probs


# ── main ───────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune whisper-large-v3 for turn-taking")
    ap.add_argument("--convs", type=int, default=0, help="0=full, >0=first N convs (smoke)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-cap", type=int, default=5, help="slices per conversation (cap5=1845 samples)")
    ap.add_argument("--batch-size", type=int, default=32, help="training batch size")
    ap.add_argument("--lr-lora", type=float, default=LR_LORA)
    ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    ap.add_argument("--run-dir", default="tools/runs/climb/lora-full")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    print(f"[lora] dev={DEV} model={MODEL_DIR}", file=sys.stderr)
    print(f"[lora] LoRA r={LORA_R} α={LORA_ALPHA} dropout={LORA_DROPOUT}", file=sys.stderr)
    print(f"[lora] lr_lora={args.lr_lora} lr_head={args.lr_head}", file=sys.stderr)
    print(f"[lora] epochs={args.epochs} folds={args.folds} slice_cap={args.slice_cap}", file=sys.stderr)

    # ── load conv ids ──
    import glob
    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs > 0:
        conv_ids = conv_ids[:args.convs]
    print(f"[lora] {len(conv_ids)} conversations, slice_cap={args.slice_cap} "
          f"≈ {len(conv_ids) * args.slice_cap} samples", file=sys.stderr)

    # ── build dataset ──
    print("[lora] building dataset...", file=sys.stderr)
    train_ds = TurnTakingSliceDataset(conv_ids, "train", slice_cap=args.slice_cap)
    print(f"[lora] dataset size: {len(train_ds)} samples", file=sys.stderr)

    # ── infer ctx_dim from data (ctxfeat output varies with label patterns) ──
    sample_ctx = train_ds[0][2]
    ctx_dim = sample_ctx.shape[0]
    print(f"[lora] ctx_dim = {ctx_dim} (inferred from data)", file=sys.stderr)

    # ── compute pos_weight from dataset ──
    targets = np.array([train_ds[i][3].numpy() for i in range(len(train_ds))])
    pw = torch.tensor(
        [(len(targets) - targets[:, k].sum()) / max(1, targets[:, k].sum())
         for k in range(5)]
    ).float().clamp(max=10).to(DEV)
    print(f"[lora] pos_weight: {[round(float(pw[k]), 2) for k in range(5)]}", file=sys.stderr)

    # ── fold split (conv-level) ──
    # map each sample to its conversation index for group-aware splitting
    conv_to_idx = {cid: i for i, cid in enumerate(conv_ids)}
    sample_groups = np.array([conv_to_idx[s[0]] for s in train_ds.samples])
    n_convs = len(conv_ids)
    rng = np.random.default_rng(SEED)
    conv_perm = rng.permutation(n_convs)
    fold_of_conv = {conv_perm[i]: i % args.folds for i in range(n_convs)}

    # ── 5-fold CV ──
    oof = np.zeros((len(train_ds), 5), dtype=np.float32)
    models = []

    t_total = time.time()
    for fi in range(args.folds):
        t_fold = time.time()
        # conv-level fold assignment
        val_convs = {conv_perm[i] for i in range(n_convs) if i % args.folds == fi}
        tr_idx = [i for i in range(len(train_ds)) if sample_groups[i] not in val_convs]
        va_idx = [i for i in range(len(train_ds)) if sample_groups[i] in val_convs]
        print(f"[lora] fold {fi+1}/{args.folds}: train={len(tr_idx)} val={len(va_idx)} "
              f"convs_train={n_convs - len(val_convs)} convs_val={len(val_convs)}",
              file=sys.stderr)

        # build encoder + LoRA (fresh per fold)
        enc = build_encoder_with_lora()
        model = WhisperVAPLoRA(ctx_dim=ctx_dim, encoder=enc)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[lora] params: trainable={trainable:,} / total={total_params:,} "
              f"({100*trainable/total_params:.2f}%)", file=sys.stderr)

        model = train_fold(
            model, train_ds, tr_idx, args.epochs, pw,
            args.batch_size, args.lr_lora, args.lr_head,
        )
        oof[va_idx] = predict_oof(model, train_ds, va_idx)
        models.append(model)
        dt = time.time() - t_fold
        print(f"[lora] fold {fi+1} done in {dt/60:.1f}min", file=sys.stderr)

    total_min = (time.time() - t_total) / 60
    print(f"[lora] all {args.folds} folds done in {total_min:.1f}min", file=sys.stderr)

    # ── cap1 slice CV evaluation ──
    # cap1 = first slice of each conversation (order==0 in sample list)
    cap1_idx = []
    seen_convs = set()
    for i, (cid, end, split) in enumerate(train_ds.samples):
        if cid not in seen_convs:
            cap1_idx.append(i)
            seen_convs.add(cid)

    # threshold sweep on cap1 OOF (for monitoring only — we use cycle1 fixed for submission)
    thr_cap1, f1_cap1 = {}, {}
    for k in range(5):
        bf, bt = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(
                targets[cap1_idx, k],
                (oof[cap1_idx, k] >= t).astype(int),
                zero_division=0,
            )
            if f > bf:
                bf, bt = f, float(t)
        thr_cap1[k], f1_cap1[k] = bt, bf
    macro_cap1 = float(np.mean(list(f1_cap1.values())))
    print(f"[lora] cap1 slice CV macro={macro_cap1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cap1[k]:.3f}@{thr_cap1[k]:.2f}" for k in range(5)),
          file=sys.stderr)
    print(f"[lora] ★BC={f1_cap1[2]:.3f} (frozen baseline=0.200, pure ctx LGBM=0.222)",
          file=sys.stderr)

    # also eval with cycle1 fixed thresholds
    f1_cycle1 = {}
    for k in range(5):
        f1_cycle1[k] = float(f1_score(
            targets[cap1_idx, k],
            (oof[cap1_idx, k] >= THR_CYCLE1[k]).astype(int),
            zero_division=0,
        ))
    macro_cycle1 = float(np.mean(list(f1_cycle1.values())))
    print(f"[lora] cap1 with cycle1 fixed thresholds: macro={macro_cycle1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cycle1[k]:.3f}" for k in range(5)),
          file=sys.stderr)

    # ── test prediction ──
    print("[lora] predicting on test set...", file=sys.stderr)
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    test_ds = TurnTakingSliceDataset(test_ids, "test", slice_cap=1)
    probs = predict_test(models, test_ds)
    print(f"[lora] test predictions: {probs.shape}", file=sys.stderr)

    # ── write CSVs ──
    SUBMIT = ["c", "na", "i", "bc", "t"]
    COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}

    # default submission = cycle1 fixed thresholds (阈值铁律)
    csv_path = run / "pred_test1.csv"
    pos_counts = {c: 0 for c in SUBMIT}
    with open(csv_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(probs[i, COL2K[c]] >= THR_CYCLE1[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items():
                pos_counts[c] += v
            row = [sid] + [str(vals[c]) for c in SUBMIT]
            f.write(",".join(row) + "\n")
    print(f"[lora] wrote {csv_path.name}: " +
          " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    # also write cap1-optimized thresholds CSV for comparison
    csv_cap1 = run / "pred_test1_cap1.csv"
    pos_c2 = {c: 0 for c in SUBMIT}
    with open(csv_cap1, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(probs[i, COL2K[c]] >= thr_cap1[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items():
                pos_c2[c] += v
            row = [sid] + [str(vals[c]) for c in SUBMIT]
            f.write(",".join(row) + "\n")
    print(f"[lora] wrote {csv_cap1.name}: " +
          " ".join(f"{c}={pos_c2[c]}" for c in SUBMIT), file=sys.stderr)

    # also save raw probabilities for potential ensemble use
    np.savez_compressed(run / "test_probs.npz", probs=probs, ids=np.array(test_ids))

    # ── save metrics ──
    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "lora-whisper-large-v3",
        "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                        "target_modules": ["q_proj", "v_proj"]},
        "lr": {"lora": args.lr_lora, "head": args.lr_head},
        "train_samples": len(train_ds),
        "slice_cap": args.slice_cap,
        "epochs": args.epochs,
        "folds": args.folds,
        "total_train_minutes": round(total_min, 1),
        "cap1_macro_f1": round(macro_cap1, 4),
        "per_sub_cap1": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)},
        "thresholds_cap1": {LAB[k]: round(thr_cap1[k], 2) for k in range(5)},
        "cap1_macro_cycle1_thr": round(macro_cycle1, 4),
        "per_sub_cycle1_thr": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "submission_thresholds": {LAB[k]: THR_CYCLE1[k] for k in range(5)},
        "_note": "pred_test1.csv uses cycle1 fixed thresholds (阈值铁律). "
                 "pred_test1_cap1.csv uses cap1-optimized thresholds (risky). "
                 "frozen baseline cap1=0.6521, SOTA variant-F=0.7124",
    }, ensure_ascii=False, indent=2))

    # stdout summary (machine-readable)
    print(json.dumps({
        "cap1_score": round(macro_cap1, 4),
        "cap1_cycle1_thr_score": round(macro_cycle1, 4),
        "per_sub_cap1": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)},
        "bc_f1": round(f1_cap1[2], 4),
        "train_minutes": round(total_min, 1),
    }))


if __name__ == "__main__":
    main()
