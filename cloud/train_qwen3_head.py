"""N1+ (P1) — Qwen3-0.6B 端到端 LoRA 微调 ASR 文本头.

D-15 P1 主路径: Wang ICASSP 2024 (arxiv 2401.14717) 同结构 LLM late fusion turn-taking
macro F1 +0.03~0.05 文献铁证. D-3 否决的是 sklearn 词袋+LGBM, 不否 LLM 端到端微调.

输入: 30s 切片对应的 ASR 文本 (history 标签序列 + ASR utterances) → Qwen3 LoRA → 5 类 head
评估: cap1 首窗 + 5fold conv-level GroupKFold
输出: probs.npz (OOF + test) 同 orthofuse 期望格式 → 入 N 源融合

LoRA target: q_proj/v_proj (Qwen 系列标准), r=16, α=16
合规: Qwen3 是白名单 (CLAUDE.md), 不需独立报备

Usage (云端, 等 P1.5 跑完后):
  python cloud/train_qwen3_head.py --convs 0 --epochs 5 --slice-cap 5 \\
    --run-dir tools/runs/climb/qwen3-head-$(date +%Y%m%d-%H%M)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
CTX, TGT, CHUNK_MS = 375, 25, 80
def _read_hidden_size(qwen_dir: str) -> int:
    """从 Qwen3 config.json 动态读 hidden_size (0.6B=1024, 1.7B=2048, 4B=2560)."""
    import json as _json
    return int(_json.loads((Path(qwen_dir) / "config.json").read_text())["hidden_size"])
QWEN_DIM = None  # 延迟初始化, main 内根据 QWEN_DIR 设
MAXLEN = 256
SPK = {1: "[SPK1]", 2: "[SPK2]"}
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_DIR = os.environ.get("QWEN_DIR", str(Path.home() / ".cache/manual_models/Qwen3-0.6B"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
LR_LORA = float(os.environ.get("LR_LORA", "2e-4"))
LR_HEAD = float(os.environ.get("LR_HEAD", "1e-3"))

THR_CYCLE1 = {0: 0.05, 4: 0.25, 1: 0.50, 3: 0.65, 2: 0.75}


# ── Qwen3 encoder + LoRA wrapper ──────────────────────────────────────────
def build_qwen_with_lora() -> nn.Module:
    from peft import LoraConfig, get_peft_model

    # Qwen3 是 LM, 用 AutoModel (encoder-only output, masked mean pool)
    enc = AutoModel.from_pretrained(QWEN_DIR, torch_dtype=torch.bfloat16)
    enc.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    peft_enc = get_peft_model(enc, lora_cfg)
    peft_enc.print_trainable_parameters()
    return peft_enc


# ── text building ──────────────────────────────────────────────────────────
def build_window_text(utts, end_ms, hist_labels):
    """Build text prompt for a window:
    [HIST] C C T BC NA ... [SEP] [SPK1] xxx [SPK2] xxx ...

    hist_labels: 最近 50 chunk 的 label 序列 (转 token), 上下文末段
    utts: ASR utterances 列表, 取末 end_ms 之前的句子 (因果)
    """
    # 历史标签 token (最近 50 chunks)
    LAB_TOKEN = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
    hist_str = " ".join(LAB_TOKEN.get(int(x), "?") for x in hist_labels[-50:])

    # ASR 文本 (因果, end_ms 前的句子)
    parts = []
    for u in utts:
        if int(u.get("end_ms", 0)) <= end_ms:
            t = str(u.get("text", "")).strip()
            if t:
                parts.append(f"{SPK.get(int(u.get('channel_id', 1)), '[SPK1]')} {t}")
    asr_str = " ".join(parts[-30:]) if parts else "<silence>"

    return f"[HIST] {hist_str} [SEP] {asr_str}"


# ── dataset ────────────────────────────────────────────────────────────────
def pick_slice_ends(label_len: int, cap: int) -> list[int]:
    lo, hi = CTX, label_len - TGT
    if lo > hi:
        return []
    if cap <= 0:
        return list(range(lo, hi + 1, 5 * 8))
    step = max(1, (hi - lo) // cap)
    ends = list(range(lo, hi + 1, step))
    return ends[:cap]


class TextTurnTakingDataset(Dataset):
    """切片训练数据 + 文本 prompt → token ids."""

    def __init__(self, conv_ids: list[str], split: str, slice_cap: int, tokenizer):
        self.samples: list[tuple[str, int, str]] = []
        self._labels: dict[tuple[str, str], np.ndarray] = {}
        self._utts: dict[str, list] = {}
        self.tokenizer = tokenizer

        for cid in conv_ids:
            if split == "train":
                labels = np.load(f"data/train/labels/{cid}.npy")
                ends = pick_slice_ends(len(labels), slice_cap)
            else:
                ends = [CTX]
            for e in ends:
                self.samples.append((cid, e, split))
        print(f"[qwen3-ds] split={split} samples={len(self.samples)}", file=sys.stderr)

    def _get_labels(self, cid: str, split: str) -> np.ndarray:
        key = (cid, split)
        if key not in self._labels:
            if split == "train":
                self._labels[key] = np.load(f"data/train/labels/{cid}.npy")
            else:
                self._labels[key] = np.load(f"data/test/context/{cid}.npy")
        return self._labels[key]

    def _get_utts(self, cid: str, split: str) -> list:
        if cid not in self._utts:
            text_dir = "data/train/text" if split == "train" else "data/test/text"
            try:
                with open(f"{text_dir}/{cid}.json") as f:
                    self._utts[cid] = json.load(f).get("utterances", [])
            except FileNotFoundError:
                self._utts[cid] = []
        return self._utts[cid]

    def __len__(self) -> int:
        return len(self.samples)

    def get_target(self, idx: int) -> np.ndarray:
        cid, end, split = self.samples[idx]
        if split != "train":
            return np.zeros(5, dtype=np.float32)
        labels = self._get_labels(cid, split)
        fut = set(int(x) for x in labels[end:end + TGT])
        return np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)

    def __getitem__(self, idx: int):
        cid, end, split = self.samples[idx]
        labels = self._get_labels(cid, split)
        utts = self._get_utts(cid, split)

        if split == "train":
            hist_labels = labels[end - CTX:end]
            end_ms = end * CHUNK_MS
            ctx = ctxfeat(hist_labels.astype(int))
            fut = set(int(x) for x in labels[end:end + TGT])
            target = np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)
        else:
            # test: context.npy 长度 == 375 (恒定), end_ms 从 utts 推
            hist_labels = labels
            # test 的 30s 切片 end_ms 是切片末时刻 = 30000ms
            end_ms = 30000
            ctx = ctxfeat(labels.astype(int))
            target = np.zeros(5, dtype=np.float32)

        text = build_window_text(utts, end_ms, hist_labels)
        ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=MAXLEN, padding=False)
        return {
            "input_ids": ids["input_ids"].squeeze(0),
            "attention_mask": ids["attention_mask"].squeeze(0),
            "ctx": torch.from_numpy(ctx),
            "target": torch.from_numpy(target),
        }


def collate_padded(batch):
    """Pad input_ids/attention_mask to max len in batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn = torch.zeros(len(batch), max_len, dtype=torch.long)
    ctx = torch.stack([b["ctx"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attn[i, :n] = b["attention_mask"]
    return input_ids, attn, ctx, target


# ── model ──────────────────────────────────────────────────────────────────
class Qwen3HeadLoRA(nn.Module):
    """Qwen3-0.6B encoder + LoRA → masked mean pool → ctx fusion → 5 class head."""

    def __init__(self, ctx_dim: int, encoder: nn.Module, qd: int = None, d: int = 192):
        if qd is None:
            qd = QWEN_DIM
        assert qd is not None, "QWEN_DIM 未初始化"
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(nn.Linear(qd, d), nn.LayerNorm(d), nn.GELU())
        self.cn = nn.BatchNorm1d(ctx_dim)
        fin = ctx_dim + d
        self.head = nn.Sequential(
            nn.LayerNorm(fin),
            nn.Linear(fin, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

    def forward(self, input_ids, attn, ctx):
        # 同 extract_text_feats.py: masked mean pool 末层
        out = self.encoder(input_ids=input_ids, attention_mask=attn)
        h = out.last_hidden_state.float()  # [B, T, 1024]
        mask = attn.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, 1024]
        c = self.cn(ctx)
        a = self.proj(pooled)  # [B, d]
        return self.head(torch.cat([c, a], -1))


def train_fold(model, dataset, tr_idx, epochs, pw, batch_size, lr_lora, lr_head, grad_accum=2):
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
        {"params": lora_params, "lr": lr_lora, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-4},
    ])
    steps_per_epoch = (len(tr_idx) + batch_size - 1) // batch_size
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
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(tr_idx),
        collate_fn=collate_padded, num_workers=2, pin_memory=True,
    )

    for ep in range(epochs):
        model.train()
        epoch_loss, n_batch = 0.0, 0
        opt.zero_grad()
        for ids, attn, ctx, tgt in loader:
            ids = ids.to(DEV, non_blocking=True)
            attn = attn.to(DEV, non_blocking=True)
            ctx = ctx.to(DEV, non_blocking=True)
            tgt = tgt.to(DEV, non_blocking=True)
            logits = model(ids, attn, ctx)
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
            print(f"[qwen3]   epoch {ep+1}/{epochs} loss={epoch_loss/n_batch:.4f} "
                  f"lr={lr_now:.2e} VRAM={vram:.1f}GB", file=sys.stderr)
    model.eval()
    return model


@torch.no_grad()
def predict_oof(model, dataset, idx, batch_size=64):
    sub = torch.utils.data.Subset(dataset, idx)
    loader = DataLoader(sub, batch_size=batch_size, collate_fn=collate_padded,
                        num_workers=2, pin_memory=True)
    out = []
    for ids, attn, ctx, _ in loader:
        ids = ids.to(DEV, non_blocking=True)
        attn = attn.to(DEV, non_blocking=True)
        ctx = ctx.to(DEV, non_blocking=True)
        out.append(torch.sigmoid(model(ids, attn, ctx)).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_test(models, test_ds, batch_size=64):
    n = len(test_ds)
    probs = np.zeros((n, 5), dtype=np.float32)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_padded, num_workers=2, pin_memory=True)
    for m in models:
        m.to(DEV)
        m.eval()
        fold_probs = []
        with torch.no_grad():
            for ids, attn, ctx, _ in loader:
                ids = ids.to(DEV, non_blocking=True)
                attn = attn.to(DEV, non_blocking=True)
                ctx = ctx.to(DEV, non_blocking=True)
                p = torch.sigmoid(m(ids, attn, ctx))
                fold_probs.append(p.cpu().numpy())
        probs += np.concatenate(fold_probs, axis=0)
    probs /= len(models)
    return probs


def main():
    ap = argparse.ArgumentParser(description="N1+ Qwen3-0.6B LoRA 文本头")
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-cap", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr-lora", type=float, default=LR_LORA)
    ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    ap.add_argument("--run-dir", default="tools/runs/climb/qwen3-head")
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    global QWEN_DIM
    QWEN_DIM = _read_hidden_size(QWEN_DIR)
    print(f"[qwen3] dev={DEV} model={QWEN_DIR} hidden_size={QWEN_DIM}", file=sys.stderr)
    print(f"[qwen3] LoRA r={LORA_R} α={LORA_ALPHA}", file=sys.stderr)

    print("[qwen3] loading tokenizer...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR)

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs > 0:
        conv_ids = conv_ids[:args.convs]
    print(f"[qwen3] {len(conv_ids)} convs slice_cap={args.slice_cap}", file=sys.stderr)

    train_ds = TextTurnTakingDataset(conv_ids, "train", args.slice_cap, tokenizer)
    sample = train_ds[0]
    ctx_dim = sample["ctx"].shape[0]
    print(f"[qwen3] ctx_dim={ctx_dim} samples={len(train_ds)}", file=sys.stderr)
    print(f"[qwen3] sample text len: {sample['input_ids'].shape[0]} tokens", file=sys.stderr)

    targets = np.array([train_ds.get_target(i) for i in range(len(train_ds))])
    pw = torch.tensor(
        [(len(targets) - targets[:, k].sum()) / max(1, targets[:, k].sum()) for k in range(5)]
    ).float().clamp(max=10).to(DEV)
    print(f"[qwen3] pos_weight: {[round(float(pw[k]), 2) for k in range(5)]}", file=sys.stderr)

    conv_to_idx = {cid: i for i, cid in enumerate(conv_ids)}
    sample_groups = np.array([conv_to_idx[s[0]] for s in train_ds.samples])
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
        va_idx = [i for i in range(len(train_ds)) if sample_groups[i] in val_convs]
        print(f"[qwen3] fold {fi+1}/{args.folds}: train={len(tr_idx)} val={len(va_idx)}",
              file=sys.stderr)

        enc = build_qwen_with_lora()
        model = Qwen3HeadLoRA(ctx_dim=ctx_dim, encoder=enc)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[qwen3] params: trainable={trainable:,} / total={total_params:,} "
              f"({100*trainable/total_params:.2f}%)", file=sys.stderr)

        model = train_fold(
            model, train_ds, tr_idx, args.epochs, pw,
            args.batch_size, args.lr_lora, args.lr_head, args.grad_accum,
        )
        oof[va_idx] = predict_oof(model, train_ds, va_idx)
        models.append(model)
        dt = time.time() - t_fold
        print(f"[qwen3] fold {fi+1} done in {dt/60:.1f}min", file=sys.stderr)
        ckpt_path = run / f"fold{fi}.pt"
        # 只 save LoRA adapter + head, 不 save frozen Qwen3 base (避免 3.5G/7.6G 占盘, 修磁盘满 bug)
        save_state = {k: v.cpu() for k, v in model.state_dict().items()
                      if "lora_" in k or k.startswith(("proj.", "cn.", "head."))}
        torch.save(save_state, ckpt_path)
        print(f"[qwen3] saved fold{fi}.pt ({sum(v.numel() for v in save_state.values())*2/1e6:.1f}MB)",
              file=sys.stderr)
        if fi < args.folds - 1:
            models[-1] = models[-1].cpu()
            torch.cuda.empty_cache()

    total_min = (time.time() - t_total) / 60
    print(f"[qwen3] all {args.folds} folds done in {total_min:.1f}min", file=sys.stderr)

    # cap1 评估
    cap1_idx = []
    seen_convs = set()
    for i, (cid, _end, _split) in enumerate(train_ds.samples):
        if cid not in seen_convs:
            cap1_idx.append(i)
            seen_convs.add(cid)
    cap1_idx = np.array(cap1_idx)
    print(f"[qwen3] cap1 eval: {len(cap1_idx)} 窗", file=sys.stderr)

    f1_cycle1 = {}
    for k in range(5):
        f1_cycle1[k] = float(f1_score(
            targets[cap1_idx, k],
            (oof[cap1_idx, k] >= THR_CYCLE1[k]).astype(int),
            zero_division=0,
        ))
    macro_cycle1 = float(np.mean(list(f1_cycle1.values())))
    print(f"[qwen3] cap1 cycle1-thr macro={macro_cycle1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cycle1[k]:.3f}" for k in range(5)), file=sys.stderr)

    # OOF 全量 (用于 orthofuse)
    G_arr = sample_groups
    order_arr = np.zeros(len(train_ds), dtype=np.int32)
    seen_count = {}
    for i, (cid, _end, _split) in enumerate(train_ds.samples):
        seen_count[cid] = seen_count.get(cid, -1) + 1
        order_arr[i] = seen_count[cid]
    print(f"[qwen3] OOF N={len(train_ds)} cap1(order=0)={int((order_arr==0).sum())}",
          file=sys.stderr)

    print("[qwen3] predicting on test set...", file=sys.stderr)
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    test_ds = TextTurnTakingDataset(test_ids, "test", slice_cap=1, tokenizer=tokenizer)
    test_probs = predict_test(models, test_ds)
    print(f"[qwen3] test: {test_probs.shape}", file=sys.stderr)

    np.savez_compressed(
        run / "probs.npz",
        oof=oof.astype(np.float32),
        test=test_probs.astype(np.float32),
        Y=targets.astype(np.int8),
        G=G_arr.astype(np.int16),
        order=order_arr,
    )
    print(f"[qwen3] saved {run}/probs.npz", file=sys.stderr)

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
    print(f"[qwen3] wrote pred_test1.csv: " +
          " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "qwen3-0.6b-lora-text-head",
        "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT,
                        "target_modules": ["q_proj", "v_proj"]},
        "lr": {"lora": args.lr_lora, "head": args.lr_head},
        "train_samples": len(train_ds),
        "slice_cap": args.slice_cap,
        "epochs": args.epochs,
        "folds": args.folds,
        "total_train_minutes": round(total_min, 1),
        "cap1_macro_cycle1_thr": round(macro_cycle1, 4),
        "per_sub_cycle1_thr": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "submission_thresholds": {LAB[k]: THR_CYCLE1[k] for k in range(5)},
        "_note": "Wang ICASSP 2024 同结构. OOF probs.npz 入 orthofuse 作 qwen3-text 源.",
    }, ensure_ascii=False, indent=2))

    print(json.dumps({
        "cap1_score": round(macro_cycle1, 4),
        "per_sub": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "train_minutes": round(total_min, 1),
    }))


if __name__ == "__main__":
    main()
