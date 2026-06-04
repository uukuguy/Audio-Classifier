"""P1.5b — LoRA chinese-hubert-large + BC 音频增强 (cap5).

D-15 双轨 P1.5: 用户拍板双路 (whisper + hubert). hubert 比 whisper 小 5x (315M vs 1.5B),
速度更快, 中文电话域更对口 (WenetSpeech pre-train), cycle 16 实测 head 训 4min.

承 train_lora_whisper_bcaug.py 结构, 改 whisper→hubert:
  - HubertModel (TencentGameMate/chinese-hubert-large) + LoRA
  - Wav2Vec2FeatureExtractor (raw 16k waveform 标准化, 非 mel)
  - 双声道帧序列 → cross-attn → 5 类 head (同款架构)

LoRA target_modules: hubert encoder.layers.*.attention.{q,k,v,out}_proj
BC 音频增强模板同 P1.5a (probe_vap_augment.py).

Usage (云端):
  python cloud/train_lora_hubert_bcaug.py --convs 0 --epochs 20 --slice-cap 5 --bc-aug-n 3 \
    --run-dir tools/runs/climb/lora-hubert-bcaug-$(date +%Y%m%d-%H%M)
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
from transformers import HubertModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
CTX, TGT, CHUNK_MS = 375, 25, 80
SR16 = 16000
CTX_SEC = 8
FEAT_DIM = 1024  # chinese-hubert-large hidden
DS_FRAMES = 80   # 降采样到 80 帧 (与 whisper 版对齐)
SEED = 42
BC_CLASS = 2
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = os.environ.get("HUBERT_DIR", str(Path.home() / ".cache/manual_models/chinese-hubert-large"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
LR_LORA = float(os.environ.get("LR_LORA", "2e-4"))
LR_HEAD = float(os.environ.get("LR_HEAD", "1e-3"))
WEIGHT_DECAY_LORA = float(os.environ.get("WD_LORA", "0.01"))
WEIGHT_DECAY_HEAD = float(os.environ.get("WD_HEAD", "1e-4"))

THR_CYCLE1 = {0: 0.05, 4: 0.25, 1: 0.50, 3: 0.65, 2: 0.75}


# ── BC 音频增强 (同 P1.5a) ─────────────────────────────────────────────────
def augment_wav_bc(wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    x = wav.copy()
    noise_std = x.std() * rng.uniform(0.03, 0.10)
    x = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)
    x = x * float(rng.uniform(0.7, 1.4))
    if x.shape[1] > 2000:
        ml = int(rng.uniform(0.02, 0.08) * x.shape[1])
        st = int(rng.uniform(0, x.shape[1] - ml))
        x[:, st:st + ml] = 0.0
    return x.astype(np.float32)


# ── hubert encoder + LoRA wrapper ─────────────────────────────────────────
def build_encoder_with_lora() -> nn.Module:
    from peft import LoraConfig, get_peft_model

    # hubert fp32 (避免 LoRA bf16 grad 数值问题)
    base = HubertModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    # 关闭 mask_time_prob (训练时 default 0.05, 推理时 0; 对 cap5 小数据可关)
    base.config.mask_time_prob = 0.0
    base.config.mask_feature_prob = 0.0

    base.gradient_checkpointing_enable()

    # hubert attention 层叫 attention.{q,k,v,out}_proj
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],  # 同 whisper 风格, 减少 trainable
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    enc = get_peft_model(base, lora_cfg)
    enc.print_trainable_parameters()
    return enc


# ── audio I/O ──────────────────────────────────────────────────────────────
_fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)


def load_wav_8k_dual(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def audio_to_hubert_input(seg_8k: np.ndarray, sr_orig: int) -> torch.Tensor:
    """8k → 16k → Wav2Vec2 FE 标准化 → [1, samples] tensor (raw waveform)."""
    t = torch.tensor(seg_8k)
    r16 = torchaudio.functional.resample(t, sr_orig, SR16).numpy()
    feats = _fe(r16, sampling_rate=SR16, return_tensors="pt")
    return feats.input_values  # [1, samples]


# ── dataset (BC 增强) ──────────────────────────────────────────────────────
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
                self.samples.append((cid, e, split, 0))
                n_orig += 1
                if split == "train" and bc_aug_n > 0 and labels is not None:
                    fut = set(int(x) for x in labels[e:e + TGT])
                    if BC_CLASS in fut:
                        for aug_seed in range(1, bc_aug_n + 1):
                            self.samples.append((cid, e, split, aug_seed))
                            n_aug += 1
        print(f"[hubert-bcaug-ds] split={split} 原始={n_orig} BC增强={n_aug} 总={n_orig+n_aug}",
              file=sys.stderr)

    def _get_wav(self, cid: str, split: str) -> tuple[np.ndarray, int]:
        if cid not in self._wav_cache:
            path = f"data/train/audio/{cid}.wav" if split == "train" else f"data/test/audio/{cid}.wav"
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

        if aug_seed > 0:
            seed = abs(hash((cid, end, aug_seed))) % (2**32)
            rng = np.random.default_rng(seed)
            seg = augment_wav_bc(seg, sr, rng)

        # 双声道分开过 hubert
        ch0 = audio_to_hubert_input(seg[0], sr)  # [1, samples]
        ch1 = audio_to_hubert_input(seg[1], sr)
        return ch0.squeeze(0), ch1.squeeze(0), \
               torch.from_numpy(ctx), torch.from_numpy(target)


def collate_variable_audio(batch):
    """Pad variable-length raw waveforms (samples dim)."""
    ch0_list, ch1_list, ctxs, tgts = zip(*batch)
    max_t = max(c.shape[0] for c in ch0_list)
    ch0_p = torch.stack([torch.nn.functional.pad(c, (0, max_t - c.shape[0])) for c in ch0_list])
    ch1_p = torch.stack([torch.nn.functional.pad(c, (0, max_t - c.shape[0])) for c in ch1_list])
    return ch0_p, ch1_p, torch.stack(ctxs), torch.stack(tgts)


# ── model ──────────────────────────────────────────────────────────────────
class HubertVAPLoRA(nn.Module):
    """Hubert encoder + LoRA → dual-channel cross-attn → 5-class head."""

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

    def _encode_channel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, samples] → hubert → downsample → proj → [B, 80, d]."""
        # hubert input is raw waveform [B, samples]
        out = self.encoder(wav)
        h = out.last_hidden_state  # [B, T_enc, 1024]
        ds = torch.nn.functional.adaptive_avg_pool1d(
            h.transpose(1, 2).float(), DS_FRAMES
        ).transpose(1, 2)  # [B, 80, 1024]
        return self.proj(ds)

    def forward(self, ch0: torch.Tensor, ch1: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        c = self.cn(ctx)
        a = self._encode_channel(ch0)
        b = self._encode_channel(ch1)
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
        collate_fn=collate_variable_audio, num_workers=2, pin_memory=True,
    )

    for ep in range(epochs):
        model.train()
        epoch_loss, n_batch = 0.0, 0
        opt.zero_grad()
        for ch0, ch1, ctx, tgt in loader:
            ch0 = ch0.to(DEV, non_blocking=True)
            ch1 = ch1.to(DEV, non_blocking=True)
            ctx = ctx.to(DEV, non_blocking=True)
            tgt = tgt.to(DEV, non_blocking=True)
            logits = model(ch0, ch1, ctx)
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
            print(f"[hubert-bcaug]   epoch {ep+1}/{epochs} loss={epoch_loss/n_batch:.4f} "
                  f"lr={lr_now:.2e} VRAM={vram:.1f}GB", file=sys.stderr)
    model.eval()
    return model


@torch.no_grad()
def predict_oof(model, dataset, idx, batch_size=64):
    sub = torch.utils.data.Subset(dataset, idx)
    loader = DataLoader(sub, batch_size=batch_size, collate_fn=collate_variable_audio,
                        num_workers=2, pin_memory=True)
    out = []
    for ch0, ch1, ctx, _ in loader:
        ch0 = ch0.to(DEV, non_blocking=True)
        ch1 = ch1.to(DEV, non_blocking=True)
        ctx = ctx.to(DEV, non_blocking=True)
        out.append(torch.sigmoid(model(ch0, ch1, ctx)).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_test(models, test_ds, batch_size=64):
    n = len(test_ds)
    probs = np.zeros((n, 5), dtype=np.float32)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_variable_audio, num_workers=2, pin_memory=True)
    for m in models:
        m.to(DEV)
        m.eval()
        fold_probs = []
        with torch.no_grad():
            for ch0, ch1, ctx, _ in loader:
                ch0 = ch0.to(DEV, non_blocking=True)
                ch1 = ch1.to(DEV, non_blocking=True)
                ctx = ctx.to(DEV, non_blocking=True)
                p = torch.sigmoid(m(ch0, ch1, ctx))
                fold_probs.append(p.cpu().numpy())
        probs += np.concatenate(fold_probs, axis=0)
    probs /= len(models)
    return probs


def main():
    ap = argparse.ArgumentParser(description="P1.5b: LoRA hubert + BC 音频增强")
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-cap", type=int, default=5)
    ap.add_argument("--bc-aug-n", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-lora", type=float, default=LR_LORA)
    ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    ap.add_argument("--run-dir", default="tools/runs/climb/lora-hubert-bcaug")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    print(f"[hubert-bcaug] dev={DEV} model={MODEL_DIR}", file=sys.stderr)
    print(f"[hubert-bcaug] LoRA r={LORA_R} α={LORA_ALPHA} BC aug N={args.bc_aug_n}", file=sys.stderr)

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs > 0:
        conv_ids = conv_ids[:args.convs]
    print(f"[hubert-bcaug] {len(conv_ids)} convs slice_cap={args.slice_cap}", file=sys.stderr)

    train_ds = TurnTakingBCAugDataset(conv_ids, "train", args.slice_cap, args.bc_aug_n)
    sample_ctx = train_ds[0][2]
    ctx_dim = sample_ctx.shape[0]
    print(f"[hubert-bcaug] ctx_dim={ctx_dim} samples={len(train_ds)}", file=sys.stderr)

    targets = np.array([train_ds.get_target(i) for i in range(len(train_ds))])
    pw = torch.tensor(
        [(len(targets) - targets[:, k].sum()) / max(1, targets[:, k].sum()) for k in range(5)]
    ).float().clamp(max=10).to(DEV)
    print(f"[hubert-bcaug] pos_weight: {[round(float(pw[k]), 2) for k in range(5)]}", file=sys.stderr)

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
        tr_idx = [i for i in range(len(train_ds)) if sample_groups[i] not in val_convs]
        va_idx = [i for i in range(len(train_ds)) if sample_groups[i] in val_convs and is_aug_arr[i] == 0]
        print(f"[hubert-bcaug] fold {fi+1}/{args.folds}: train={len(tr_idx)} val={len(va_idx)} "
              f"convs_train={n_convs - len(val_convs)} convs_val={len(val_convs)} "
              f"(train含{int(is_aug_arr[tr_idx].sum())}增强)", file=sys.stderr)

        enc = build_encoder_with_lora()
        model = HubertVAPLoRA(ctx_dim=ctx_dim, encoder=enc)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[hubert-bcaug] params: trainable={trainable:,} / total={total_params:,} "
              f"({100*trainable/total_params:.2f}%)", file=sys.stderr)

        model = train_fold(
            model, train_ds, tr_idx, args.epochs, pw,
            args.batch_size, args.lr_lora, args.lr_head, args.grad_accum,
        )
        oof[va_idx] = predict_oof(model, train_ds, va_idx)
        models.append(model)
        dt = time.time() - t_fold
        print(f"[hubert-bcaug] fold {fi+1} done in {dt/60:.1f}min", file=sys.stderr)
        ckpt_path = run / f"fold{fi}.pt"
        torch.save(model.state_dict(), ckpt_path)
        if fi < args.folds - 1:
            models[-1] = models[-1].cpu()
            torch.cuda.empty_cache()

    total_min = (time.time() - t_total) / 60
    print(f"[hubert-bcaug] all {args.folds} folds done in {total_min:.1f}min", file=sys.stderr)

    # cap1 评估
    cap1_idx = []
    seen_convs = set()
    for i, (cid, _end, _split, aug_seed) in enumerate(train_ds.samples):
        if cid not in seen_convs and aug_seed == 0:
            cap1_idx.append(i)
            seen_convs.add(cid)
    cap1_idx = np.array(cap1_idx)
    print(f"[hubert-bcaug] cap1 eval: {len(cap1_idx)} 窗 (仅原始首窗)", file=sys.stderr)

    f1_cycle1 = {}
    for k in range(5):
        f1_cycle1[k] = float(f1_score(
            targets[cap1_idx, k],
            (oof[cap1_idx, k] >= THR_CYCLE1[k]).astype(int),
            zero_division=0,
        ))
    macro_cycle1 = float(np.mean(list(f1_cycle1.values())))
    print(f"[hubert-bcaug] cap1 cycle1-thr macro={macro_cycle1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cycle1[k]:.3f}" for k in range(5)), file=sys.stderr)
    print(f"[hubert-bcaug] ★BC={f1_cycle1[2]:.3f} (frozen hubert head=0.000 cycle 16, "
          f"frozen baseline=0.200)", file=sys.stderr)

    # OOF 全量输出 (同 P1.5a 格式, 供 orthofuse 用)
    orig_idx = np.array([i for i in range(len(train_ds)) if is_aug_arr[i] == 0])
    oof_orig = oof[orig_idx]
    Y_orig = targets[orig_idx]
    G_orig = sample_groups[orig_idx]
    order_arr = np.zeros(len(orig_idx), dtype=np.int32)
    seen_count = {}
    for j, i in enumerate(orig_idx):
        cid = train_ds.samples[i][0]
        seen_count[cid] = seen_count.get(cid, -1) + 1
        order_arr[j] = seen_count[cid]
    print(f"[hubert-bcaug] OOF orig N={len(orig_idx)} cap1(order=0)={int((order_arr==0).sum())}",
          file=sys.stderr)

    print("[hubert-bcaug] predicting on test set...", file=sys.stderr)
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    test_ds = TurnTakingBCAugDataset(test_ids, "test", slice_cap=1, bc_aug_n=0)
    test_probs = predict_test(models, test_ds)
    print(f"[hubert-bcaug] test: {test_probs.shape}", file=sys.stderr)

    np.savez_compressed(
        run / "probs.npz",
        oof=oof_orig.astype(np.float32),
        test=test_probs.astype(np.float32),
        Y=Y_orig.astype(np.int8),
        G=G_orig.astype(np.int16),
        order=order_arr,
    )
    print(f"[hubert-bcaug] saved {run}/probs.npz", file=sys.stderr)

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
    print(f"[hubert-bcaug] wrote pred_test1.csv: " +
          " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "lora-hubert-large-bcaug",
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
        "_note": "OOF probs.npz 可直接入 orthofuse 作 lora-hubert-bcaug 源. "
                 "对比 frozen hubert head cycle 16 BC=0.000.",
    }, ensure_ascii=False, indent=2))

    print(json.dumps({
        "cap1_score": round(macro_cycle1, 4),
        "per_sub": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "bc_f1": round(f1_cycle1[2], 4),
        "train_minutes": round(total_min, 1),
    }))


if __name__ == "__main__":
    main()
