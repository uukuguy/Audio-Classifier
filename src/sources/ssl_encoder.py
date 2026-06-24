"""源 2/3: SSL encoder (whisper / hubert) + VAP head。

训练-推理一致性契约 (cloud/train_head_bcaug.py): head 吃 [2, 80, FDIM] 帧特征
(2 声道 × 80 帧 × encoder 维度)。现场提取必须严格复刻训练侧:
  - 8kHz 双声道 → 取末 8s → resample 16k → 各声道独立过 encoder → adaptive_pool 到 80 帧。
  - whisper: WhisperFeatureExtractor → mel → encoder; wd=1280
  - hubert : Wav2Vec2FeatureExtractor → input_values → encoder; wd=1024
head ckpt = 6/22 复赛重训单 seed (model.pt)。
"""
from __future__ import annotations

import glob
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from src.common import CTX_SEC, DEV, DS_FRAMES, DTYPE, MODELS, NUM, SR16, TEST_ROOT
from src.sources.context import featurize, normalize_ctx_to_375

CTX_DIM = 46  # featurize 输出维度 (feature_spec.json)
# whisper 固定 30s 输入→1500 帧, 有效音频仅末 8s → 取末 400 帧再 pool (训练侧 extract_whisper_cuda.py:32,63)。
# 不取 tail = 对全 1500 帧 (含 22s 静音 pad) pool → 特征被静音稀释 → wsp probs 全乱 (实测 T/I/BC 崩)。
WHISPER_TAIL_FRAMES = 400


class WhisperVAP(nn.Module):
    """双声道 cross-attention head (训练时同结构, cloud/train_head_bcaug.py:153)。"""

    def __init__(self, ctx_dim: int, wd: int, d: int = 192):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(wd, d), nn.LayerNorm(d), nn.GELU())
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cn = nn.BatchNorm1d(ctx_dim)
        fin = 2 * d + ctx_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fin), nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, NUM))

    def forward(self, ctx: torch.Tensor, aud: torch.Tensor) -> torch.Tensor:
        ctx = self.cn(ctx)
        aud = self.proj(aud)
        B, C2, T, D = aud.shape
        aud = aud.reshape(B * C2, T, D)
        q = self.q.expand(B * 2, -1, -1)
        aud, _ = self.cross(q, aud, aud)
        aud = aud.squeeze(1).reshape(B, -1)
        return self.head(torch.cat([aud, ctx], dim=1))


def _load_encoder(encoder_type: str):
    from transformers import (
        HubertModel,
        Wav2Vec2FeatureExtractor,
        WhisperFeatureExtractor,
        WhisperModel,
    )

    if encoder_type == "whisper":
        enc = WhisperModel.from_pretrained(str(MODELS / "whisper-large-v3"), dtype=DTYPE).encoder
        extractor = WhisperFeatureExtractor.from_pretrained(str(MODELS / "whisper-large-v3"))
        return enc.to(DEV).eval(), extractor, 1280
    if encoder_type == "hubert":
        enc = HubertModel.from_pretrained(str(MODELS / "chinese-hubert-large"), dtype=DTYPE)
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODELS / "chinese-hubert-large"))
        return enc.to(DEV).eval(), extractor, 1024
    raise ValueError(f"Unknown SSL encoder: {encoder_type}")


def _read_stereo(wf: str) -> tuple[np.ndarray, int]:
    """读 wav → [2, T] float32 (单声道复制成双)。"""
    with wave.open(wf, "rb") as w:
        sr, nch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    d = np.frombuffer(raw, dtype=np.int16).reshape(-1, nch).T.astype(np.float32) / 32768.0
    if d.ndim == 1:
        d = np.stack([d, d])
    elif d.shape[0] == 1:
        d = np.concatenate([d, d])
    return d, sr


@torch.no_grad()
def infer_ssl(encoder_type: str, head_dir: Path) -> np.ndarray:
    """encoder + head 推理。返回 [N, 5] 概率 (N = test 段数, 与 context 对齐)。"""
    test_files = sorted(glob.glob(str(TEST_ROOT / "audio/*.wav")))
    head_pt = Path(head_dir) / "model.pt"
    if not head_pt.exists():
        print(f"WARNING: {head_pt} missing, zero probs", flush=True)
        return np.zeros((len(test_files), NUM), dtype=np.float32)

    enc, extractor, wd = _load_encoder(encoder_type)
    model = WhisperVAP(ctx_dim=CTX_DIM, wd=wd).to(DEV)
    # weights_only=True: 仅加载张量 (我方自训 head, 防 pickle 任意代码)
    model.load_state_dict(torch.load(head_pt, map_location="cpu", weights_only=True), strict=False)
    model.eval()

    probs = np.zeros((len(test_files), NUM), dtype=np.float32)
    for i, wf in enumerate(test_files):
        try:
            d, sr = _read_stereo(wf)
            raw_ctx = np.load(str(TEST_ROOT / f"context/{Path(wf).stem}.npy")).astype(int)
            ctx_feat = featurize(normalize_ctx_to_375(raw_ctx))  # ★ 变长 ctx 先 normalize (同 context.py)
            aud_feats = []
            for ch in range(2):
                end = d.shape[1]
                # ★ 在【原始采样率 sr=8000】上切末 8s (训练侧 extract_*_cuda.py: CTX_SEC*sr), 再 resample。
                # bug: 之前用 CTX_SEC*SR16(=128000)在8kHz数据上切 → 实切末16s, 音频内容错 → probs全乱。
                seg = d[ch, max(0, end - CTX_SEC * sr):end]
                if len(seg) < CTX_SEC * sr:
                    seg = np.pad(seg, (CTX_SEC * sr - len(seg), 0))
                w16 = torchaudio.functional.resample(torch.tensor(seg), sr, SR16).numpy()
                if encoder_type == "whisper":
                    mel = extractor(w16, sampling_rate=SR16, return_tensors="pt").input_features.to(DEV, DTYPE)
                    h = enc(mel).last_hidden_state          # [1, 1500, 1280] (whisper 固定 30s)
                    h = h[:, -WHISPER_TAIL_FRAMES:, :]      # ★ 取末 400 帧 = 末 8s 有效音频 (训练侧一致)
                else:
                    feat = extractor(w16, sampling_rate=SR16, return_tensors="pt", padding=True).input_values.to(DEV, DTYPE)
                    h = enc(feat).last_hidden_state         # [1, ~400, 1024] (hubert 输入即 8s, 无需 tail)
                ds = torch.nn.functional.adaptive_avg_pool1d(h.transpose(1, 2).float(), DS_FRAMES).transpose(1, 2)
                aud_feats.append(ds.squeeze(0).cpu().numpy().astype(np.float16))
            aud = np.stack(aud_feats, axis=0)
            ctx_t = torch.tensor(ctx_feat, dtype=torch.float32, device=DEV).unsqueeze(0)
            aud_t = torch.tensor(aud.astype(np.float32), dtype=torch.float32, device=DEV).unsqueeze(0)
            probs[i] = torch.sigmoid(model(ctx_t, aud_t)).cpu().numpy()
        except Exception as e:  # noqa: BLE001 — 单段失败不崩全局, 出 safe default
            print(f"  [{encoder_type}] {Path(wf).name} ERROR: {e}", flush=True)
            probs[i] = [0.95, 0.5, 0.1, 0.3, 0.95]
    return probs
