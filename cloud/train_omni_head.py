"""P1.5d — Qwen2.5-Omni-7B Thinker LoRA + BC 音频增强 (多模态主路径).

D-15 后用户加方向: 文本端 Qwen3-0.6B 单源已证伪 (cap1=0.5823 ≈ ctx, D-15 路线否).
Omni 是 Qwen 系列多模态主路, audio+text 融合在 LLM 序列内, 期望突破冻结 head 0.71755 SOTA 天花板.

架构 (RESEARCH §Option A):
  - 加载 Qwen2_5OmniThinkerForConditionalGeneration (8.5B, 跳 Talker/Token2Wav)
  - LoRA r=16 q_proj/v_proj 全覆盖 (text + audio encoder)
  - 输入: ASR 文本 (含历史标签序列) + 8s 末段 audio (8kHz→16kHz resample)
  - last_hidden_state mask-aware mean pool 3584d → ctx_proj 拼 5-class head
  - 5fold conv-level + BCE + pos_weight + BC 3x 音频增强 (复用 augment_wav_bc)
  - val 仅原始 (防虚高) + cap1 评估 + OOF probs.npz 入 orthofuse

避坑 (CLAUDE.md + RESUME 累积):
  - 离线 resample 8k→16k 一次性算好, 不在 __getitem__ 实时算 (避 5/30 卡死)
  - device_map="cuda" 强制 (不 "auto")
  - bf16 model + fp32 LoRA (peft 默认), 不 .half() (避 nan)
  - 守 varF 阈值算法, 绝不 BC 阈值 cherry-pick (D-18 教训)

Push 门 (D-17 红旗):
  - cap1 macro ≥ 0.6410 (SOTA cap1) 且 per-class 至少一类 ≥ strat winner +0.005
  - 否则 SKIP, 不烧配额

Usage (云端):
  python cloud/train_omni_head.py --convs 0 --epochs 5 --slice-cap 5 --bc-aug-n 3 \\
    --run-dir tools/runs/climb/omni-lora-$(date +%Y%m%d-%H%M)
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

# ★ stderr line-buffered — nohup 下 block-buffered 会让 print 攒在 buffer 看不见
# (上次全量训练卡死诊断不出, 因为 53min 没一行日志 flush)
sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
CTX, TGT, CHUNK_MS = 375, 25, 80
SR16 = 16000
CTX_SEC = 8                 # 末 8s 音频 (跟 whisper_bcaug 一致, 公平比较)
def _read_hidden_size(omni_dir: str) -> int:
    """从 Thinker config.json 动态读 hidden_size (Omni-7B=3584, Omni-3B=2048)."""
    import json as _json
    cfg = _json.loads((Path(omni_dir) / "config.json").read_text())
    # Omni 顶层 config 含 thinker_config / talker_config / token2wav_config
    thinker = cfg.get("thinker_config", cfg)
    text_cfg = thinker.get("text_config", thinker)
    return int(text_cfg["hidden_size"])
HIDDEN_SIZE = None  # 延迟初始化, main() 内根据 OMNI_DIR 设
MAXLEN_TEXT = 384           # text prefix token 上限 (留 ~600 给 audio token, 总 ~1000 安全)
SPK = {1: "[SPK1]", 2: "[SPK2]"}
SEED = 42
BC_CLASS = 2
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

OMNI_DIR = os.environ.get("OMNI_DIR", str(Path.home() / ".cache/manual_models/Qwen2.5-Omni-7B"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.1"))
LR_LORA = float(os.environ.get("LR_LORA", "1e-4"))      # Omni 7B 比 whisper-large-v3 大, lr 稍低
LR_HEAD = float(os.environ.get("LR_HEAD", "5e-4"))
WD_LORA = float(os.environ.get("WD_LORA", "0.01"))
WD_HEAD = float(os.environ.get("WD_HEAD", "1e-4"))

# 跟项目 SOTA 同一组 cycle1 钙化阈值 (variant-F SOTA), 守 varF BC=0.75 (D-18 铁律)
THR_CYCLE1 = {0: 0.05, 4: 0.25, 1: 0.50, 3: 0.65, 2: 0.75}


# ── BC 音频增强 (跟 train_lora_whisper_bcaug.py 完全同, hubert_bcaug 破 SOTA 的同款配方) ──
def augment_wav_bc(wav: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """音频增强生成 BC 正例多样性变体. wav: [2, samples] float32.

    组合: 加噪 + gain 扰 + 时间掩码 (SpecAug 时域). 不变速 (保因果时序).
    """
    x = wav.copy()
    noise_std = x.std() * rng.uniform(0.03, 0.10)
    x = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)
    x = x * float(rng.uniform(0.7, 1.4))
    if x.shape[1] > 2000:
        ml = int(rng.uniform(0.02, 0.08) * x.shape[1])
        st = int(rng.uniform(0, x.shape[1] - ml))
        x[:, st:st + ml] = 0.0
    return x.astype(np.float32)


# ── Omni Thinker + LoRA wrapper ──────────────────────────────────────────
def build_thinker_with_lora() -> nn.Module:
    """加载 Thinker only (跳 Talker/Token2Wav 省一半显存), 注入 LoRA."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        OMNI_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        # attn_implementation 让 transformers 自选 (sdpa / flash_attn 视环境)
    )
    thinker.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT, bias="none",
    )
    peft_thinker = get_peft_model(thinker, lora_cfg)
    peft_thinker.print_trainable_parameters()
    return peft_thinker


# ── audio I/O (跟 train_lora_whisper_bcaug.py 同) ──────────────────────────
def load_wav_8k_dual(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def to_mono_16k(seg_8k_dual: np.ndarray, sr_orig: int) -> np.ndarray:
    """双声道 8kHz → 单声道 16kHz (Omni audio 输入: 单通道 16k float32, 跟 Whisper FE 一致)."""
    mono = seg_8k_dual.mean(axis=0)
    t = torch.from_numpy(mono)
    r16 = torchaudio.functional.resample(t, sr_orig, SR16).numpy()
    return r16.astype(np.float32)


# ── text 构造 (跟 qwen3_head.py 同样的 [HIST] + [SPK1/2] 模板, 给 Omni Thinker text 部分) ──
def build_text_prompt(utts: list, end_ms: int, hist_labels: np.ndarray) -> str:
    """模板: 'history: C C T BC ... | dialogue: [SPK1] 你好 [SPK2] 嗯'.

    Omni Thinker 是 Qwen2 LLM, 对自然语言 prompt 风格更友好 (不用 qwen3 的 [HIST]/[SEP] 特殊 token).
    """
    LAB_TOKEN = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
    hist_str = " ".join(LAB_TOKEN.get(int(x), "?") for x in hist_labels[-50:])
    parts = []
    for u in utts:
        if int(u.get("end_ms", 0)) <= end_ms:
            t = str(u.get("text", "")).strip()
            if t:
                parts.append(f"{SPK.get(int(u.get('channel_id', 1)), '[SPK1]')} {t}")
    asr_str = " ".join(parts[-20:]) if parts else "<silence>"
    return f"history: {hist_str} | dialogue: {asr_str}"


# ── dataset (audio + text 双模态, BC 增强同 whisper_bcaug) ──────────────────
def pick_slice_ends(label_len: int, cap: int) -> list[int]:
    lo, hi = CTX, label_len - TGT
    if lo > hi:
        return []
    if cap <= 0:
        return list(range(lo, hi + 1, 5 * 8))
    step = max(1, (hi - lo) // cap)
    ends = list(range(lo, hi + 1, step))
    return ends[:cap]


class OmniMultimodalDataset(Dataset):
    """audio (mono 16k) + text (history+ASR) + ctx_oof + target. BC 正例 N 倍增强 (val 仅原始)."""

    def __init__(self, conv_ids: list[str], split: str, slice_cap: int = 5, bc_aug_n: int = 0,
                 processor=None):
        assert processor is not None, "需传入 Qwen2_5OmniProcessor"
        self.processor = processor
        self.samples: list[tuple[str, int, str, int]] = []
        self._labels: dict[tuple[str, str], np.ndarray] = {}
        self._utts: dict[str, list] = {}
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

        print(f"[omni-ds] split={split} 原始={n_orig} BC增强={n_aug} 总={n_orig+n_aug}",
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
        cid, end, split, _ = self.samples[idx]
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
        utts = self._get_utts(cid, split)
        wav, sr = self._get_wav(cid, split)

        if split == "train":
            hist_labels = labels[end - CTX:end]
            end_ms = end * CHUNK_MS
            ctx = ctxfeat(hist_labels.astype(int))
            fut = set(int(x) for x in labels[end:end + TGT])
            target = np.array([1 if k in fut else 0 for k in range(5)], dtype=np.float32)
        else:
            hist_labels = labels
            end_ms = 30000
            ctx = ctxfeat(labels.astype(int))
            target = np.zeros(5, dtype=np.float32)

        # 末 CTX_SEC 秒音频
        end_sample = int(end * CHUNK_MS / 1000 * sr) if split == "train" else wav.shape[1]
        start_sample = max(0, end_sample - CTX_SEC * sr)
        seg = wav[:, start_sample:end_sample]

        if aug_seed > 0:
            seed = abs(hash((cid, end, aug_seed))) % (2**32)
            rng = np.random.default_rng(seed)
            seg = augment_wav_bc(seg, sr, rng)

        # mono 16k for Omni processor
        audio_16k = to_mono_16k(seg, sr)

        # text prompt
        text = build_text_prompt(utts, end_ms, hist_labels)

        # ★ 改: processor 在 __getitem__ 单样本调 (避 collate 内 batched processor 第一次调用大延迟)
        proc_out = self.processor(
            text=[text], audio=[audio_16k],
            sampling_rate=SR16,
            return_tensors="pt", padding=False, truncation=True,
            max_length=MAXLEN_TEXT + 800,
        )
        # 去掉 batch 维 (单样本)
        return {
            "input_ids": proc_out["input_ids"].squeeze(0),
            "attention_mask": proc_out["attention_mask"].squeeze(0),
            "input_features": proc_out.get("input_features", torch.empty(0)).squeeze(0)
                if "input_features" in proc_out else None,
            "feature_attention_mask": proc_out.get("feature_attention_mask", torch.empty(0)).squeeze(0)
                if "feature_attention_mask" in proc_out else None,
            "ctx": torch.from_numpy(ctx),
            "target": torch.from_numpy(target),
        }


def pad_stack_collate(batch):
    """简单 pad+stack collate. 不调 processor (已在 __getitem__ 做完)."""
    max_text = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_text, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_text, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]

    # input_features (audio mel): (n_mels, frames), pad frames 维
    has_audio = batch[0]["input_features"] is not None
    if has_audio:
        max_frames = max(b["input_features"].shape[-1] for b in batch)
        first_feat = batch[0]["input_features"]
        assert first_feat is not None
        n_mels = first_feat.shape[0]
        input_features = torch.zeros(len(batch), n_mels, max_frames, dtype=first_feat.dtype)
        feature_attention_mask = torch.zeros(len(batch), max_frames, dtype=torch.long)
        for i, b in enumerate(batch):
            bf = b["input_features"]
            assert bf is not None
            f = bf.shape[-1]
            input_features[i, :, :f] = bf
            feature_attention_mask[i, :f] = 1
    else:
        input_features = None
        feature_attention_mask = None

    ctxs = torch.stack([b["ctx"] for b in batch])
    tgts = torch.stack([b["target"] for b in batch])

    proc_out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if has_audio:
        proc_out["input_features"] = input_features
        proc_out["feature_attention_mask"] = feature_attention_mask
    return proc_out, ctxs, tgts


# ── model ────────────────────────────────────────────────────────────────
class OmniHeadLoRA(nn.Module):
    """Omni Thinker (LoRA) → masked mean pool last_hidden_state → ctx_proj 拼 5-class head."""

    def __init__(self, ctx_dim: int, thinker: nn.Module, hd: int = None, d: int = 192):
        if hd is None:
            hd = HIDDEN_SIZE
        assert hd is not None, "HIDDEN_SIZE 未初始化 (main 里要先 set)"
        super().__init__()
        self.thinker = thinker
        self.proj = nn.Sequential(nn.Linear(hd, d), nn.LayerNorm(d), nn.GELU())
        self.cn = nn.BatchNorm1d(ctx_dim)
        fin = ctx_dim + d
        self.head = nn.Sequential(
            nn.LayerNorm(fin),
            nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

    def forward(self, proc_out, ctx):
        """proc_out: dict from Qwen2_5OmniProcessor (input_ids/attention_mask/input_features/feature_attention_mask).

        Qwen2_5OmniThinker**ForConditionalGeneration** 是 CausalLM 类, output 默认是 logits,
        要拿 hidden state 必须 output_hidden_states=True 然后取 hidden_states[-1].
        """
        out = self.thinker(
            input_ids=proc_out["input_ids"],
            attention_mask=proc_out["attention_mask"],
            input_features=proc_out.get("input_features"),
            feature_attention_mask=proc_out.get("feature_attention_mask"),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        # hidden_states: tuple(L_layers+1) of (B, L_total, 3584). [-1] = 最后一层 (text + audio 已拼接为单序列)
        h = out.hidden_states[-1].float()
        # mask-aware mean pool (跟 qwen3_head 同款 — Omni Thinker 用 text attention_mask 作 mask 足够)
        mask = proc_out["attention_mask"].unsqueeze(-1).float()
        # 若 audio 序列扩展超出 text mask 长度, 用 1 补全 (audio token 都参与 pool)
        if h.shape[1] > mask.shape[1]:
            pad = torch.ones(mask.shape[0], h.shape[1] - mask.shape[1], 1, device=mask.device)
            mask = torch.cat([mask, pad], dim=1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, 3584)
        c = self.cn(ctx)
        a = self.proj(pooled)
        return self.head(torch.cat([c, a], -1))


def train_fold(model, dataset, tr_idx, epochs, pw, batch_size, lr_lora, lr_head,
               grad_accum, processor):
    model.to(DEV)
    lora_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name or "thinker" in name:
            lora_params.append(p)
        else:
            head_params.append(p)
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": lr_lora, "weight_decay": WD_LORA},
        {"params": head_params, "lr": lr_head, "weight_decay": WD_HEAD},
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
    # num_workers=0 — processor 已在 __getitem__ 内单样本调, 不需 worker. 避 fork issues + 可见进度
    # (processor 引用未用, 但保留 signature 兼容)
    _ = processor
    # ★ drop_last=True 防 BatchNorm batch=1 ValueError (fold 5 crash 教训)
    loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(tr_idx),
        collate_fn=pad_stack_collate, num_workers=0, pin_memory=True, drop_last=True,
    )

    total_steps = len(tr_idx) // batch_size
    print(f"[omni]   train loop start: epochs={epochs} steps/ep={total_steps}",
          file=sys.stderr, flush=True)
    for ep in range(epochs):
        model.train()
        epoch_loss, n_batch = 0.0, 0
        t_ep = time.time()
        opt.zero_grad()
        for proc_out, ctx, tgt in loader:
            proc_out = {k: v.to(DEV, non_blocking=True) for k, v in proc_out.items()}
            ctx = ctx.to(DEV, non_blocking=True)
            tgt = tgt.to(DEV, non_blocking=True)
            logits = model(proc_out, ctx)
            loss = crit(logits, tgt) / grad_accum
            loss.backward()
            n_batch += 1
            epoch_loss += float(loss) * grad_accum
            if n_batch % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            # 进度心跳: 每 50 batch 打一行 + flush
            if n_batch == 1 or n_batch % 50 == 0:
                vram = torch.cuda.memory_allocated() / 1024**3
                elapsed = time.time() - t_ep
                print(f"[omni]   ep{ep+1} batch {n_batch}/{total_steps} "
                      f"loss={epoch_loss/max(1,n_batch):.4f} "
                      f"VRAM={vram:.1f}GB elapsed={elapsed:.0f}s",
                      file=sys.stderr, flush=True)
        if n_batch % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        vram = torch.cuda.memory_allocated() / 1024**3
        lr_now = sched.get_last_lr()[0]
        dt = time.time() - t_ep
        print(f"[omni]   ★ epoch {ep+1}/{epochs} done loss={epoch_loss/n_batch:.4f} "
              f"lr={lr_now:.2e} VRAM={vram:.1f}GB dt={dt/60:.1f}min",
              file=sys.stderr, flush=True)
    model.eval()
    return model


@torch.no_grad()
def predict_oof(model, dataset, idx, batch_size, processor):
    sub = torch.utils.data.Subset(dataset, idx)
    _ = processor
    loader = DataLoader(sub, batch_size=batch_size, collate_fn=pad_stack_collate,
                        num_workers=0, pin_memory=True)
    out = []
    for proc_out, ctx, _ in loader:
        proc_out = {k: v.to(DEV, non_blocking=True) for k, v in proc_out.items()}
        ctx = ctx.to(DEV, non_blocking=True)
        out.append(torch.sigmoid(model(proc_out, ctx)).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_test(fold_ckpts, ctx_dim, test_ds, batch_size, processor):
    """test 推理: 按 ckpt 路径逐 fold build_thinker → load_state_dict (LoRA+head) → forward → del.

    避 5 个 17GB Thinker 同时占 CPU/GPU. 每 fold ~1min Thinker 加载 + 1000段 forward.
    """
    n = len(test_ds)
    probs = np.zeros((n, 5), dtype=np.float32)
    _ = processor
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=pad_stack_collate, num_workers=0, pin_memory=True)
    for fi, ckpt_path in enumerate(fold_ckpts):
        print(f"[omni-test] fold {fi+1}/{len(fold_ckpts)} loading thinker...",
              file=sys.stderr, flush=True)
        thinker = build_thinker_with_lora()
        m = OmniHeadLoRA(ctx_dim=ctx_dim, thinker=thinker)
        # 只 load LoRA + head 权重 (strict=False 跳过 base thinker)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[omni-test]   loaded ckpt: {len(sd)} keys, missing={len(missing)} unexpected={len(unexpected)}",
              file=sys.stderr, flush=True)
        m.to(DEV)
        m.eval()
        fold_probs = []
        with torch.no_grad():
            for bi, (proc_out, ctx, _) in enumerate(loader):
                proc_out = {k: v.to(DEV, non_blocking=True) for k, v in proc_out.items()}
                ctx = ctx.to(DEV, non_blocking=True)
                p = torch.sigmoid(m(proc_out, ctx))
                fold_probs.append(p.cpu().numpy())
                if bi == 0 or (bi + 1) % 50 == 0:
                    print(f"[omni-test]   fold {fi+1} batch {bi+1}", file=sys.stderr, flush=True)
        probs += np.concatenate(fold_probs, axis=0)
        del m, thinker
        torch.cuda.empty_cache()
    probs /= len(fold_ckpts)
    return probs


def main():
    ap = argparse.ArgumentParser(description="P1.5d: Omni-7B Thinker LoRA + BC 音频增强 + ctx 拼接")
    ap.add_argument("--convs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--slice-cap", type=int, default=5)
    ap.add_argument("--bc-aug-n", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Omni 7B + audio 长序列, batch 不宜大. 4090 48GB 估 batch=4 安全, batch=6 试")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=LR_LORA)
    ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    ap.add_argument("--run-dir", default="tools/runs/climb/omni-lora")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式: 2 通 1 epoch 1 fold, 验通就退出")
    ap.add_argument("--resume-from-fold", type=int, default=None,
                    help="从第 N (0-indexed) fold 开始训练, 前面 fold 加载已存 ckpt 重抽 OOF (fold 5 crash 救场用)")
    args = ap.parse_args()

    if args.smoke:
        args.convs = 2
        args.epochs = 1
        args.folds = 2
        args.slice_cap = 2
        args.bc_aug_n = 0
        args.batch_size = 2
        args.grad_accum = 1

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    global HIDDEN_SIZE
    HIDDEN_SIZE = _read_hidden_size(OMNI_DIR)
    print(f"[omni] dev={DEV} model={OMNI_DIR} hidden_size={HIDDEN_SIZE}", file=sys.stderr)
    print(f"[omni] LoRA r={LORA_R} α={LORA_ALPHA} BC aug N={args.bc_aug_n}", file=sys.stderr)

    # processor 一次加载, dataset+collate 共享
    print("[omni] loading processor...", file=sys.stderr)
    from transformers import Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained(OMNI_DIR)

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    if args.convs > 0:
        conv_ids = conv_ids[:args.convs]
    print(f"[omni] {len(conv_ids)} convs slice_cap={args.slice_cap}", file=sys.stderr)

    train_ds = OmniMultimodalDataset(conv_ids, "train", args.slice_cap, args.bc_aug_n,
                                     processor=processor)
    sample_ctx = train_ds[0]["ctx"]
    ctx_dim = sample_ctx.shape[0]
    print(f"[omni] ctx_dim={ctx_dim} samples={len(train_ds)}", file=sys.stderr)

    targets = np.array([train_ds.get_target(i) for i in range(len(train_ds))])
    pw = torch.tensor(
        [(len(targets) - targets[:, k].sum()) / max(1, targets[:, k].sum()) for k in range(5)]
    ).float().clamp(max=10).to(DEV)
    print(f"[omni] pos_weight: {[round(float(pw[k]), 2) for k in range(5)]}", file=sys.stderr)

    conv_to_idx = {cid: i for i, cid in enumerate(conv_ids)}
    sample_groups = np.array([conv_to_idx[s[0]] for s in train_ds.samples])
    is_aug_arr = np.array([train_ds.get_is_aug(i) for i in range(len(train_ds))])
    n_convs = len(conv_ids)
    rng = np.random.default_rng(SEED)
    conv_perm = rng.permutation(n_convs)

    oof = np.zeros((len(train_ds), 5), dtype=np.float32)
    fold_ckpts = []  # ★ 只存 ckpt 路径, 不保留 GPU/CPU 上的 model 对象 (避 5×17GB Thinker 堆 CPU)
    t_total = time.time()

    for fi in range(args.folds):
        t_fold = time.time()
        val_convs = {conv_perm[i] for i in range(n_convs) if i % args.folds == fi}
        tr_idx = [i for i in range(len(train_ds)) if sample_groups[i] not in val_convs]
        va_idx = [i for i in range(len(train_ds)) if sample_groups[i] in val_convs and is_aug_arr[i] == 0]
        print(f"[omni] fold {fi+1}/{args.folds}: train={len(tr_idx)} val={len(va_idx)} "
              f"convs_train={n_convs - len(val_convs)} convs_val={len(val_convs)} "
              f"(train含{int(is_aug_arr[tr_idx].sum())}增强)", file=sys.stderr)

        ckpt_path = run / f"fold{fi}.pt"

        # ★ resume 模式: 已有 ckpt 跳过 train, 只加载重抽 OOF
        if args.resume_from_fold is not None and fi < args.resume_from_fold and ckpt_path.exists():
            print(f"[omni] fold {fi+1} RESUME: 加载已有 ckpt 跳过 train", file=sys.stderr, flush=True)
            thinker = build_thinker_with_lora()
            model = OmniHeadLoRA(ctx_dim=ctx_dim, thinker=thinker).to(DEV)
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[omni]   loaded {len(sd)} keys, missing={len(missing)} unexpected={len(unexpected)}",
                  file=sys.stderr, flush=True)
            oof[va_idx] = predict_oof(model, train_ds, va_idx, args.batch_size, processor)
            dt = time.time() - t_fold
            print(f"[omni] fold {fi+1} OOF done in {dt/60:.1f}min", file=sys.stderr)
            fold_ckpts.append(ckpt_path)
            del model, thinker
            torch.cuda.empty_cache()
            continue

        thinker = build_thinker_with_lora()
        model = OmniHeadLoRA(ctx_dim=ctx_dim, thinker=thinker)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[omni] params: trainable={trainable:,} / total={total_params:,} "
              f"({100*trainable/total_params:.2f}%)", file=sys.stderr)

        model = train_fold(
            model, train_ds, tr_idx, args.epochs, pw,
            args.batch_size, args.lr_lora, args.lr_head, args.grad_accum, processor,
        )
        oof[va_idx] = predict_oof(model, train_ds, va_idx, args.batch_size, processor)
        dt = time.time() - t_fold
        print(f"[omni] fold {fi+1} done in {dt/60:.1f}min", file=sys.stderr)

        # 只存 LoRA + head (不存 base thinker, 8GB 太大)
        save_state = {k: v.cpu() for k, v in model.state_dict().items()
                      if "lora_" in k or k.startswith(("proj.", "cn.", "head."))}
        torch.save(save_state, ckpt_path)
        print(f"[omni] saved fold{fi}.pt ({sum(v.numel() for v in save_state.values())*2/1e6:.1f}MB)",
              file=sys.stderr)
        fold_ckpts.append(ckpt_path)
        # ★ 不留 model 在 CPU/GPU (释放 17GB Thinker), test 时按 ckpt 路径重新 build
        del model
        torch.cuda.empty_cache()

    total_min = (time.time() - t_total) / 60
    print(f"[omni] all {args.folds} folds done in {total_min:.1f}min", file=sys.stderr)

    if args.smoke:
        print("[omni] SMOKE OK — 训练通了, 退出 (跳 cap1/test/落盘)", file=sys.stderr)
        return

    # cap1 评估 (首窗 + 原始)
    cap1_idx = []
    seen_convs = set()
    for i, (cid, _end, _split, aug_seed) in enumerate(train_ds.samples):
        if cid not in seen_convs and aug_seed == 0:
            cap1_idx.append(i)
            seen_convs.add(cid)
    cap1_idx = np.array(cap1_idx)
    print(f"[omni] cap1 eval: {len(cap1_idx)} 窗 (仅原始首窗)", file=sys.stderr)

    f1_cycle1 = {}
    for k in range(5):
        f1_cycle1[k] = float(f1_score(
            targets[cap1_idx, k],
            (oof[cap1_idx, k] >= THR_CYCLE1[k]).astype(int),
            zero_division=0,
        ))
    macro_cycle1 = float(np.mean(list(f1_cycle1.values())))
    print(f"[omni] cap1 cycle1-thr macro={macro_cycle1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cycle1[k]:.3f}" for k in range(5)), file=sys.stderr)
    print(f"[omni] PUSH 门 (D-17): cap1≥0.6410 且 per-class ≥strat winner+0.005 才 push. "
          f"当前 cap1={macro_cycle1:.4f} (SOTA cap1=0.6410)", file=sys.stderr)

    # OOF 原始段输出 (跟 train_lora_whisper_bcaug.py 同格式, 入 orthofuse)
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
    print(f"[omni] OOF orig N={len(orig_idx)} cap1(order=0)={int((order_arr==0).sum())}",
          file=sys.stderr)

    # test 预测
    print("[omni] predicting on test set...", file=sys.stderr)
    test_ids = sorted(Path(p).stem for p in glob.glob("data/test/audio/*.wav"))
    test_ds = OmniMultimodalDataset(test_ids, "test", slice_cap=1, bc_aug_n=0,
                                    processor=processor)
    test_probs = predict_test(fold_ckpts, ctx_dim, test_ds, args.batch_size, processor)
    print(f"[omni] test: {test_probs.shape}", file=sys.stderr)

    np.savez_compressed(
        run / "probs.npz",
        oof=oof_orig.astype(np.float32),
        test=test_probs.astype(np.float32),
        Y=Y_orig.astype(np.int8),
        G=G_orig.astype(np.int16),
        order=order_arr,
    )
    print(f"[omni] saved {run}/probs.npz (oof+test+Y+G+order)", file=sys.stderr)

    # 提交 CSV (cycle1 固定阈值, 守 varF — 不调 BC 阈值, D-18 铁律)
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
    print(f"[omni] wrote pred_test1.csv: " +
          " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "qwen2.5-omni-7b-thinker-lora-bcaug",
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
        "ctx_sec": CTX_SEC,
        "hidden_size": HIDDEN_SIZE,
        "total_train_minutes": round(total_min, 1),
        "cap1_macro_cycle1_thr": round(macro_cycle1, 4),
        "per_sub_cycle1_thr": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "submission_thresholds": {LAB[k]: THR_CYCLE1[k] for k in range(5)},
        "push_gate": "D-17: cap1≥0.6410 AND per-class ≥strat winner+0.005",
        "_note": "Omni Thinker only (跳 Talker/Token2Wav). audio (mono 16k 8s) + text (history+ASR) → "
                 "multimodal LLM 序列融合 → mask-aware mean pool → ctx 拼接 → 5 class head. "
                 "守 varF 阈值, BC 严禁 cherry-pick (D-18). OOF probs.npz 入 orthofuse 作 omni 源.",
    }, ensure_ascii=False, indent=2))

    print(json.dumps({
        "cap1_score": round(macro_cycle1, 4),
        "per_sub": {LAB[k]: round(f1_cycle1[k], 4) for k in range(5)},
        "bc_f1": round(f1_cycle1[2], 4),
        "train_minutes": round(total_min, 1),
    }))


if __name__ == "__main__":
    main()
