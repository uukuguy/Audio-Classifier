"""P1.5a — LoRA whisper-large-v3 + BC 音频增强 (cap5 cap5 数据多样性补救).

D-15 双轨 P1.5: 用户拍板"BC 正例稀少是分类常识必做数据增强". D-7 LoRA cap5 BC 0.267 = 项目最高。
本脚本 = train_lora.py + BC 音频增强 (变速/加噪/SpecAug 时间掩码), 每 BC 正例 3x.
val 只用原始 (防虚高). LoRA r=16 (减参提速 ~2x). epochs 20 (增强信号后早收敛).

D-15 红旗校验:
  ❌ 不"加第 N 个音频源" — whisper 是已有源 ✓
  ❌ 不"cap1 选 strat" — cap5 训练, cap1 仅评估 ✓
  ❌ 不"context 内同源算法集成" — encoder 微调 ✓
  ❌ 不"OOF 校准头无新源" — LoRA = 真新表征 ✓

5/30 audio-aug 失败不适用: 5/30 是冻结 VAP encoder + aug, encoder 不可学吸收不了多样性.
P1.5a 是 LoRA 可学 whisper + aug (D-7 证可学 encoder 能榨 BC 0.267).

Usage (云端):
  python cloud/train_lora_whisper_bcaug.py --convs 0 --epochs 20 --slice-cap 5 --bc-aug-n 3 \
    --run-dir tools/runs/climb/lora-whisper-bcaug-$(date +%Y%m%d-%H%M)
"""
from __future__ import annotations

import argparse
import glob
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
CTX_SEC = 8
TAIL_FRAMES = 400
DS_FRAMES = 80
FEAT_DIM = 1280
SEED = 42
BC_CLASS = 2
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = os.environ.get("WHISPER_DIR", str(Path.home() / ".cache/manual_models/whisper-large-v3"))
LORA_R = int(os.environ.get("LORA_R", "16"))  # 减参 32→16 提速 2x
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
LR_LORA = float(os.environ.get("LR_LORA", "2e-4"))
LR_HEAD = float(os.environ.get("LR_HEAD", "1e-3"))
WEIGHT_DECAY_LORA = float(os.environ.get("WD_LORA", "0.01"))
WEIGHT_DECAY_HEAD = float(os.environ.get("WD_HEAD", "1e-4"))

THR_CYCLE1 = {0: 0.05, 4: 0.25, 1: 0.50, 3: 0.65, 2: 0.75}  # variant-F SOTA 钙化阈值


# ── BC 音频增强 (基于 cloud/probe_vap_augment.py augment_wav) ──────────────────
def augment_wav_bc(wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """音频增强生成 BC 正例多样性变体. wav: [2, samples] float32.

    组合 (probe_vap_augment.py 实测安全): 加噪 + gain 扰 + 时间掩码 (SpecAug 时域).
    不变速 (保因果时序: turn-taking 任务对时机敏感).
    """
    x = wav.copy()
    # 1. 加高斯噪声 (SNR ~20-30dB)
    noise_std = x.std() * rng.uniform(0.03, 0.10)
    x = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)
    # 2. gain 扰 (±3dB)
    x = x * float(rng.uniform(0.7, 1.4))
    # 3. 时间掩码 (随机置零 2-8% 长度, SpecAug 时域版)
    if x.shape[1] > 2000:
        ml = int(rng.uniform(0.02, 0.08) * x.shape[1])
        st = int(rng.uniform(0, x.shape[1] - ml))
        x[:, st:st + ml] = 0.0
    return x.astype(np.float32)


# ── whisper encoder + LoRA wrapper (同 train_lora.py) ──────────────────────
def build_encoder_with_lora() -> nn.Module:
    from peft import LoraConfig, get_peft_model

    full = WhisperModel.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    base = full.encoder
    del full
    base.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    enc = get_peft_model(base, lora_cfg)
    enc.print_trainable_parameters()
    return enc


# ── audio I/O (同 train_lora.py) ──────────────────────────────────────────
_fe = WhisperFeatureExtractor.from_pretrained(MODEL_DIR)


def load_wav_8k_dual(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def audio_to_logmel(seg_8k: np.ndarray, sr_orig: int) -> torch.Tensor:
    t = torch.tensor(seg_8k)
    r16 = torchaudio.functional.resample(t, sr_orig, SR16).numpy()
    feats = _fe([r16], sampling_rate=SR16, return_tensors="pt")
    return feats.input_features


# ── dataset (改造: 加 BC 增强样本) ──────────────────────────────────────────
def pick_slice_ends(label_len: int, cap: int) -> list[int]:
    lo, hi = CTX, label_len - TGT
    if lo > hi:
        return []
    if cap <= 0:
        return list(range(lo, hi + 1, 5 * 8))
    step = max(1, (hi - lo) // cap)
    ends = list(range(lo, hi + 1, step))
    return ends[:cap]


class TurnTakingBCAugDataset(Dataset):
    """切片训练数据 + BC 正例 N 倍增强 (is_aug 标记, val 仅放原始).

    每个 sample = (cid, end_chunk, split, is_aug_seed)
      is_aug_seed = 0 表示原始, 1..N 表示第 N 个增强变体 (用 seed 控制可复现).
    """

    def __init__(self, conv_ids: list[str], split: str, slice_cap: int = 5, bc_aug_n: int = 0):
        self.samples: list[tuple[str, int, str, int]] = []
        self._labels: dict[tuple[str, str], np.ndarray] = {}
        self._wav_cache: dict[str, tuple[np.ndarray, int]] = {}
        self.bc_aug_n = bc_aug_n

        n_orig, n_aug = 0, 0
        for cid in conv_ids:
            if split == "train":
                labels = np.load(f"data/train/labels/{cid}.npy")
                ends = pick_slice_ends(len(labels), slice_cap)
            else:
                labels = None
                ends = [CTX]

            for e in ends:
                # 原始样本永远加
                self.samples.append((cid, e, split, 0))
                n_orig += 1

                # 训练 + BC 正例 → 加 N 个增强变体
                if split == "train" and bc_aug_n > 0 and labels is not None:
                    fut = set(int(x) for x in labels[e:e + TGT])
                    if BC_CLASS in fut:
                        for aug_seed in range(1, bc_aug_n + 1):
                            self.samples.append((cid, e, split, aug_seed))
                            n_aug += 1

        print(f"[bcaug-ds] split={split} 原始={n_orig} BC增强={n_aug} 总={n_orig+n_aug}",
              file=sys.stderr)

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
                self._labels[key] = np.load(f"data/test/context/{cid}.npy")
        return self._labels[key]

    def __len__(self) -> int:
        return len(self.samples)

    def get_target(self, idx: int) -> np.ndarray:
        cid, end, split, _aug = self.samples[idx]
        if split != "train":
            return np.zeros(5, dtype=np.float32)
        labels = self._get_labels(cid, split)
        fut = set(int(x) for x in labels[end:end + TGT])
        return np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)

    def get_is_aug(self, idx: int) -> int:
        return 1 if self.samples[idx][3] > 0 else 0

    def __getitem__(self, idx: int):
        cid, end, split, aug_seed = self.samples[idx]
        labels = self._get_labels(cid, split)
        wav, sr = self._get_wav(cid, split)

        if split == "train":
            ctx = ctxfeat(labels[end - CTX:end].astype(int))
            fut = set(int(x) for x in labels[end:end + TGT])
            target = np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)
        else:
            ctx = ctxfeat(labels.astype(int))
            target = np.zeros(5, dtype=np.float32)

        end_sample = int(end * CHUNK_MS / 1000 * sr) if split == "train" else wav.shape[1]
        start_sample = max(0, end_sample - CTX_SEC * sr)
        seg = wav[:, start_sample:end_sample]

        # BC 增强: 如果 aug_seed > 0, 用 (cid_hash, end, aug_seed) 派生 rng 增强 wav
        if aug_seed > 0:
            seed = abs(hash((cid, end, aug_seed))) % (2**32)
            rng = np.random.default_rng(seed)
            seg = augment_wav_bc(seg, sr, rng)

        mel_ch0 = audio_to_logmel(seg[0], sr)
        mel_ch1 = audio_to_logmel(seg[1], sr)
        return mel_ch0.squeeze(0), mel_ch1.squeeze(0), \
               torch.from_numpy(ctx), torch.from_numpy(target)


def collate_variable_mel(batch):
    mel0, mel1, ctxs, tgts = zip(*batch)
    max_t = max(m.shape[-1] for m in mel0)
    mel0_p = torch.stack([torch.nn.functional.pad(m, (0, max_t - m.shape[-1])) for m in mel0])
    mel1_p = torch.stack([torch.nn.functional.pad(m, (0, max_t - m.shape[-1])) for m in mel1])
    return mel0_p, mel1_p, torch.stack(ctxs), torch.stack(tgts)


class WhisperVAPLoRA(nn.Module):
    """复用 train_lora.py 同款架构."""

    def __init__(self, ctx_dim: int, encoder: nn.Module, wd: int = FEAT_DIM, d: int = 192):
        super().__init__()
        self.encoder = encoder
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
        enc_dtype = next(self.encoder.parameters()).dtype
        mel = mel.to(enc_dtype)
        enc = self.encoder(mel)
        h = enc.last_hidden_state
        tail = h[:, -TAIL_FRAMES:, :]
        ds = torch.nn.functional.adaptive_avg_pool1d(
            tail.transpose(1, 2).float(), DS_FRAMES
        ).transpose(1, 2)
        return self.proj(ds)

    def forward(self, mel0: torch.Tensor, mel1: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        c = self.cn(ctx)
        a = self._encode_channel(mel0)
        b = self._encode_channel(mel1)
        B = a.shape[0]
        q = self.q.expand(B, -1, -1)
        ca, _ = self.cross(q, b, b)
        cb, _ = self.cross(q, a, a)
        return self.head(torch.cat([c, ca.squeeze(1), cb.squeeze(1)], -1))


def train_fold(model, train_ds, train_idx, epochs, pw, batch_size, lr_lora, lr_head, grad_accum=2):
    model.to(DEV)
    lora_params, head_params = [], []
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
    steps_per_epoch = (len(train_idx) + batch_size - 1) // batch_size
    total_opt_steps = epochs * (steps_per_epoch // grad_accum + (1 if steps_per_epoch % grad_accum else 0))
    warmup = max(1, int(0.05 * total_opt_steps))
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_opt_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        collate_fn=collate_variable_mel, num_workers=2, pin_memory=True,
    )

    for ep in range(epochs):
        model.train()
        epoch_loss, n_batch = 0.0, 0
        opt.zero_grad()
        for mel0, mel1, ctx, tgt in loader:
            mel0 = mel0.to(DEV, non_blocking=True)
            mel1 = mel1.to(DEV, non_blocking=True)
            ctx = ctx.to(DEV, non_blocking=True)
            tgt = tgt.to(DEV, non_blocking=True)
            logits = model(mel0, mel1, ctx)
            loss = crit(logits, tgt) / grad_accum
            loss.backward()
            n_batch += 1
            epoch_loss += float(loss) * grad_accum
            if n_batch % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        if n_batch % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            vram = torch.cuda.memory_allocated() / 1024**3
            lr_now = sched.get_last_lr()[0]
            print(f"[lora-bcaug]   epoch {ep+1}/{epochs} loss={epoch_loss/n_batch:.4f} "
                  f"lr={lr_now:.2e} VRAM={vram:.1f}GB", file=sys.stderr)
    model.eval()
    return model


@torch.no_grad()
def predict_oof(model, dataset, idx, batch_size=64):
    sub = torch.utils.data.Subset(dataset, idx)
    loader = DataLoader(sub, batch_size=batch_size, collate_fn=collate_variable_mel,
                        num_workers=2, pin_memory=True)
    out = []
    for mel0, mel1, ctx, _ in loader:
        mel0 = mel0.to(DEV, non_blocking=True)
        mel1 = mel1.to(DEV, non_blocking=True)
        ctx = ctx.to(DEV, non_blocking=True)
        out.append(torch.sigmoid(model(mel0, mel1, ctx)).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_test(models, test_ds, batch_size=64):
    n = len(test_ds)
    probs = np.zeros((n, 5), dtype=np.float32)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_variable_mel, num_workers=2, pin_memory=True)
    for m in models:
        m.to(DEV)
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


def main():
    ap = argparse.ArgumentParser(description="P1.5a: LoRA whisper + BC 音频增强")
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-cap", type=int, default=5)
    ap.add_argument("--bc-aug-n", type=int, default=3, help="每 BC 正例生成 N 个增强变体")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=LR_LORA)
    ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    ap.add_argument("--run-dir", default="tools/runs/climb/lora-whisper-bcaug")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    print(f"[lora-bcaug] dev={DEV} model={MODEL_DIR}", file=sys.stderr)
    print(f"[lora-bcaug] LoRA r={LORA_R} α={LORA_ALPHA} BC aug N={args.bc_aug_n}", file=sys.stderr)

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs > 0:
        conv_ids = conv_ids[:args.convs]
    print(f"[lora-bcaug] {len(conv_ids)} convs slice_cap={args.slice_cap}", file=sys.stderr)

    train_ds = TurnTakingBCAugDataset(conv_ids, "train", args.slice_cap, args.bc_aug_n)
    sample_ctx = train_ds[0][2]
    ctx_dim = sample_ctx.shape[0]
    print(f"[lora-bcaug] ctx_dim={ctx_dim} samples={len(train_ds)}", file=sys.stderr)

    # pos_weight (含 BC 增强样本, BC 正例占比上升 ~3x 让 pos_weight 自动下降, 更稳)
    targets = np.array([train_ds.get_target(i) for i in range(len(train_ds))])
    pw = torch.tensor(
        [(len(targets) - targets[:, k].sum()) / max(1, targets[:, k].sum()) for k in range(5)]
    ).float().clamp(max=10).to(DEV)
    print(f"[lora-bcaug] pos_weight: {[round(float(pw[k]), 2) for k in range(5)]}", file=sys.stderr)

    conv_to_idx = {cid: i for i, cid in enumerate(conv_ids)}
    sample_groups = np.array([conv_to_idx[s[0]] for s in train_ds.samples])
    is_aug_arr = np.array([train_ds.get_is_aug(i) for i in range(len(train_ds))])
    n_convs = len(conv_ids)
    rng = np.random.default_rng(SEED)
    conv_perm = rng.permutation(n_convs)

    oof = np.zeros((len(train_ds), 5), dtype=np.float32)
    models = []
    t_total = time.time()

    for fi in range(args.folds):
        t_fold = time.time()
        val_convs = {conv_perm[i] for i in range(n_convs) if i % args.folds == fi}
        # train: 本折外所有样本 (含原始+增强); val: 本折内**仅原始** (防虚高)
        tr_idx = [i for i in range(len(train_ds)) if sample_groups[i] not in val_convs]
        va_idx = [i for i in range(len(train_ds)) if sample_groups[i] in val_convs and is_aug_arr[i] == 0]
        print(f"[lora-bcaug] fold {fi+1}/{args.folds}: train={len(tr_idx)} val={len(va_idx)} "
              f"convs_train={n_convs - len(val_convs)} convs_val={len(val_convs)} "
              f"(train含{int(is_aug_arr[tr_idx].sum())}增强)", file=sys.stderr)

        enc = build_encoder_with_lora()
        model = WhisperVAPLoRA(ctx_dim=ctx_dim, encoder=enc)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[lora-bcaug] params: trainable={trainable:,} / total={total_params:,} "
              f"({100*trainable/total_params:.2f}%)", file=sys.stderr)

        model = train_fold(
            model, train_ds, tr_idx, args.epochs, pw,
            args.batch_size, args.lr_lora, args.lr_head, args.grad_accum,
        )
        oof[va_idx] = predict_oof(model, train_ds, va_idx)
        models.append(model)
        dt = time.time() - t_fold
        print(f"[lora-bcaug] fold {fi+1} done in {dt/60:.1f}min", file=sys.stderr)

        ckpt_path = run / f"fold{fi}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if fi < args.folds - 1:
            models[-1] = models[-1].cpu()
            torch.cuda.empty_cache()

    total_min = (time.time() - t_total) / 60
    print(f"[lora-bcaug] all {args.folds} folds done in {total_min:.1f}min", file=sys.stderr)

    # cap1 评估 (只用原始样本, 不含增强 — 增强在 train 不在 cap1 eval)
    cap1_idx = []
    seen_convs = set()
    for i, (cid, _end, _split, aug_seed) in enumerate(train_ds.samples):
        if cid not in seen_convs and aug_seed == 0:  # 只首窗 + 原始
            cap1_idx.append(i)
            seen_convs.add(cid)
    cap1_idx = np.array(cap1_idx)
    print(f"[lora-bcaug] cap1 eval: {len(cap1_idx)} 窗 (仅原始首窗)", file=sys.stderr)

    f1_cycle1 = {}
    for k in range(5):
        f1_cycle1[k] = float(f1_score(
            targets[cap1_idx, k],
            (oof[cap1_idx, k] >= THR_CYCLE1[k]).astype(int),
            zero_division=0,
        ))
    macro_cycle1 = float(np.mean(list(f1_cycle1.values())))
    print(f"[lora-bcaug] cap1 cycle1-thr macro={macro_cycle1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cycle1[k]:.3f}" for k in range(5)), file=sys.stderr)
    print(f"[lora-bcaug] ★BC={f1_cycle1[2]:.3f} (frozen baseline=0.200, "
          f"LoRA cap5 D-7=0.267, SOTA orthofuse cap1=0.6410)", file=sys.stderr)

    # OOF 全量 (含原始 cap1 + 其它 order, 用于 orthofuse 融合输入)
    # 输出: train OOF [N_orig, 5] + test [1000, 5] + Y + G + order
    orig_idx = np.array([i for i in range(len(train_ds)) if is_aug_arr[i] == 0])
    oof_orig = oof[orig_idx]  # [N_orig, 5]
    Y_orig = targets[orig_idx]  # [N_orig, 5]
    G_orig = sample_groups[orig_idx]  # [N_orig]
    # order: 每通内的 (cid, end_chunk) 排序顺序, cap1 = order==0
    order_arr = np.zeros(len(orig_idx), dtype=np.int32)
    seen_count = {}
    for j, i in enumerate(orig_idx):
        cid = train_ds.samples[i][0]
        seen_count[cid] = seen_count.get(cid, -1) + 1
        order_arr[j] = seen_count[cid]
    print(f"[lora-bcaug] OOF orig 段 N={len(orig_idx)} cap1(order=0)={int((order_arr==0).sum())}",
          file=sys.stderr)

    # test 预测
    print("[lora-bcaug] predicting on test set...", file=sys.stderr)
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    test_ds = TurnTakingBCAugDataset(test_ids, "test", slice_cap=1, bc_aug_n=0)
    test_probs = predict_test(models, test_ds)
    print(f"[lora-bcaug] test: {test_probs.shape}", file=sys.stderr)

    # 落盘: 兼容 orthofuse 期望的 probs.npz 格式 (oof + test + Y + G + order)
    np.savez_compressed(
        run / "probs.npz",
        oof=oof_orig.astype(np.float32),
        test=test_probs.astype(np.float32),
        Y=Y_orig.astype(np.int8),
        G=G_orig.astype(np.int16),
        order=order_arr,
    )
    print(f"[lora-bcaug] saved {run}/probs.npz (oof+test+Y+G+order)", file=sys.stderr)

    # 提交 CSV (cycle1 固定阈值)
    SUBMIT = ["c", "na", "i", "bc", "t"]
    COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
    pos_counts = {c: 0 for c in SUBMIT}
    with open(run / "pred_test1.csv", "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(test_probs[i, COL2K[c]] >= THR_CYCLE1[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items():
                pos_counts[c] += v
            row = [sid] + [str(vals[c]) for c in SUBMIT]
            f.write(",".join(row) + "\n")
    print(f"[lora-bcaug] wrote pred_test1.csv: " +
          " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "lora-whisper-large-v3-bcaug",
        "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                        "target_modules": ["q_proj", "v_proj"]},
        "bc_aug_n": args.bc_aug_n,
        "lr": {"lora": args.lr_lora, "head": args.lr_head},
        "train_samples": len(train_ds),
        "train_samples_orig": int((is_aug_arr == 0).sum()),
        "train_samples_aug": int(is_aug_arr.sum()),
        "slice_cap": args.slice_cap,
        "epochs": args.epochs,
        "folds": args.folds,
        "total_train_minutes": round(total_min, 1),
        "cap1_macro_cycle1_thr": round(macro_cycle1, 4),
        "per_sub_cycle1_thr": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "submission_thresholds": {LAB[k]: THR_CYCLE1[k] for k in range(5)},
        "_note": "OOF probs.npz 可直接入 orthofuse 作 lora-whisper-bcaug 源. "
                 "D-7 LoRA cap5 BC=0.267, frozen=0.200. ★突破=BC>0.25 即真增益.",
    }, ensure_ascii=False, indent=2))

    print(json.dumps({
        "cap1_score": round(macro_cycle1, 4),
        "per_sub": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "bc_f1": round(f1_cycle1[2], 4),
        "train_minutes": round(total_min, 1),
    }))


if __name__ == "__main__":
    main()
