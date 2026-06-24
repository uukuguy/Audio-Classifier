"""S5 全栈推理 — 复赛 Docker 镜像入口。

加载所有模型权重, 对 /data/test 做端到端推理, 输出 /output/pred_test1.csv。
评测环境无网络, 所有模型路径均为 Docker 内 /app/models/ 子目录。
"""
from __future__ import annotations
import glob, json, os, sys, time, wave
from pathlib import Path
import numpy as np
import joblib, torch, torchaudio, torch.nn as nn
import os as _os
# torch 2.5 compatibility with newer transformers
_os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# === Config ===
TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/xydata"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", "/app/submit/submit.csv"))
MODELS = Path(os.environ.get("MODELS", "/app/models"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32
SR16, CTX_SEC, DS_FRAMES = 16000, 8, 80
CHUNK_MS, CTX, TGT = 80, 375, 25

LAB = ["C", "T", "BC", "I", "NA"]
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
THR = [0.05, 0.50, 0.75, 0.65, 0.25]

os.environ["OMNI_DIR"] = str(MODELS / "Qwen2.5-Omni-3B")
os.environ["HUBERT_DIR"] = str(MODELS / "chinese-hubert-large")
sys.path.insert(0, "/app")


# === 1. Context LGBM ===
def featurize(ctx):
    """featurize_variable_length — 任意长度 ctx 自适应, 短上下文不退化."""
    NA_LABEL = 4; NUM = 5
    L_eff = int((ctx != NA_LABEL).sum())
    if L_eff == 0: L_eff = len(ctx)
    oh = np.eye(NUM)[ctx]; feats = []
    for w in (10, 25, 50, 100, 200, 375):
        eff_w = min(w, L_eff)
        feats.extend(oh[-eff_w:].mean(axis=0))
    for i in range(1, 6): feats.append(ctx[-i] if len(ctx) >= i else -1)
    L = len(ctx)
    for k in range(NUM):
        pos = np.where(ctx == k)[0]; feats.append((L-1-pos[-1])/max(L,1) if len(pos) else 1.0)
    for k in range(NUM): feats.append((ctx==k).sum()/max(L,1))
    feats.append((ctx[1:]!=ctx[:-1]).mean() if L>1 else 0.0)
    return np.array(feats, dtype=np.float32)

def normalize_ctx_to_375(ctx):
    """Keep original length — varlen featurize handles any length."""
    return ctx.astype(np.int32)

def infer_context():
    ctx_dir = MODELS / "ctx_only"
    clfs = {k: joblib.load(ctx_dir / f"lgbm_{LAB[k].lower()}.joblib") for k in range(5)}
    thr = json.loads((ctx_dir / "thresholds.json").read_text())
    thrs = [thr[LAB[k].lower()] for k in range(5)]

    test_files = sorted(glob.glob(str(TEST_ROOT / "context/*.npy")))
    if not test_files:
        print(f"WARNING: no test context at {TEST_ROOT}/context/", flush=True)
        return np.zeros((0, 5), dtype=np.float32), []
    seg_ids = [Path(p).stem for p in test_files]
    Xte = np.array([featurize(normalize_ctx_to_375(np.load(p).astype(int))) for p in test_files])
    # Ensure 2D
    if Xte.ndim == 1: Xte = Xte.reshape(1, -1)
    probs = np.zeros((len(Xte), 5), dtype=np.float32)
    for k in range(5):
        try:
            probs[:, k] = clfs[k].predict_proba(Xte)[:, 1]
        except Exception as e:
            print(f"LGBM predict failed for {LAB[k]}: {e}, using raw...", flush=True)
            probs[:, k] = clfs[k].predict(Xte, raw_score=True)
    return probs, seg_ids


# === 2. SSL Encoder Inference ===
class WhisperVAP(nn.Module):
    def __init__(self, ctx_dim, wd, d=192):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(wd, d), nn.LayerNorm(d), nn.GELU())
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cn = nn.BatchNorm1d(ctx_dim)
        fin = 2 * d + ctx_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fin), nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, ctx, aud):
        ctx = self.cn(ctx)
        aud = self.proj(aud)
        B, C2, T, D = aud.shape
        aud = aud.reshape(B * C2, T, D)
        q = self.q.expand(B * 2, -1, -1)
        aud, _ = self.cross(q, aud, aud)
        aud = aud.squeeze(1)
        aud = aud.reshape(B, -1)
        return self.head(torch.cat([aud, ctx], dim=1))


@torch.no_grad()
def infer_encoder(encoder_type, head_ckpt_dir):
    """Run encoder + head inference. Load single model.pt (not multi-seed fold)."""
    from transformers import (
        WhisperModel, WhisperFeatureExtractor,
        HubertModel, Wav2Vec2FeatureExtractor,
        WavLMModel,
    )

    if encoder_type == "whisper":
        enc = WhisperModel.from_pretrained(str(MODELS / "whisper-large-v3"), dtype=DTYPE).encoder.to(DEV).eval()
        extractor = WhisperFeatureExtractor.from_pretrained(str(MODELS / "whisper-large-v3"))
        wd = 1280
    elif encoder_type == "hubert":
        enc = HubertModel.from_pretrained(str(MODELS / "chinese-hubert-large"), dtype=DTYPE).to(DEV).eval()
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODELS / "chinese-hubert-large"))
        wd = 1024
    elif encoder_type == "emotion2vec":
        enc = WavLMModel.from_pretrained(str(MODELS / "emotion2vec_base"), dtype=DTYPE).to(DEV).eval()
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODELS / "emotion2vec_base"))
        wd = 768
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")

    head_pt = Path(head_ckpt_dir) / "model.pt"
    if not head_pt.exists():
        print(f"WARNING: {head_pt} not found, returning zero probs", flush=True)
        test_files = sorted(glob.glob(str(TEST_ROOT / "audio/*.wav")))
        return np.zeros((len(test_files), 5), dtype=np.float32)

    model = WhisperVAP(ctx_dim=46, wd=wd).to(DEV)
    sd = torch.load(head_pt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()

    test_files = sorted(glob.glob(str(TEST_ROOT / "audio/*.wav")))
    probs = np.zeros((len(test_files), 5), dtype=np.float32)
    for i, wf in enumerate(test_files):
        try:
            with wave.open(wf, "rb") as w:
                sr = w.getframerate(); full = w.readframes(w.getnframes()); nch = w.getnchannels()
            d = np.frombuffer(full, dtype=np.int16).reshape(-1, nch).T.astype(np.float32) / 32768.0
            # Handle mono by duplicating to stereo
            if d.ndim == 1: d = np.stack([d, d])
            elif d.shape[0] == 1: d = np.concatenate([d, d])
            ctx_feat = featurize(normalize_ctx_to_375(
                np.load(str(TEST_ROOT / f"context/{Path(wf).stem}.npy")).astype(int)))

            aud_feats = []
            for ch in range(2):
                end = d.shape[1]; start = max(0, end - CTX_SEC * SR16)
                seg = d[ch, start:end]
                if len(seg) < CTX_SEC * SR16: seg = np.pad(seg, (CTX_SEC * SR16 - len(seg), 0))
                w16 = torchaudio.functional.resample(torch.tensor(seg), sr, SR16).numpy()
                if encoder_type == "whisper":
                    mel = extractor(w16, sampling_rate=SR16, return_tensors="pt").input_features.to(DEV, DTYPE)
                    h = enc(mel).last_hidden_state
                else:
                    feat = extractor(w16, sampling_rate=SR16, return_tensors="pt", padding=True).input_values.to(DEV, DTYPE)
                    h = enc(feat).last_hidden_state
                ds = torch.nn.functional.adaptive_avg_pool1d(h.transpose(1,2).float(), DS_FRAMES).transpose(1,2)
                aud_feats.append(ds.squeeze(0).cpu().numpy().astype(np.float16))
            aud = np.stack(aud_feats, axis=0)
            ctx_t = torch.tensor(ctx_feat, dtype=torch.float32, device=DEV).unsqueeze(0)
            aud_t = torch.tensor(aud.astype(np.float32), dtype=torch.float32, device=DEV).unsqueeze(0)
            logits = model(ctx_t, aud_t)
            probs[i] = torch.sigmoid(logits).detach().cpu().numpy()
        except Exception as e:
            print(f"  [{encoder_type}] file {i} {Path(wf).name} ERROR: {e}", flush=True)
            probs[i] = [0.95, 0.5, 0.1, 0.3, 0.95]  # safe defaults
    return probs


# === 3. Omni-3B Inference ===
@torch.no_grad()
def infer_omni():
    sys.path.insert(0, "/app")
    from cloud.train_omni_head import (
        build_thinker_with_lora, OmniHeadLoRA, OmniMultimodalDataset
    )
    from transformers import Qwen2_5OmniProcessor

    omni_dir = str(MODELS / "Qwen2.5-Omni-3B")
    processor = Qwen2_5OmniProcessor.from_pretrained(omni_dir)

    test_ids = sorted(Path(p).stem for p in glob.glob(str(TEST_ROOT / "audio/*.wav")))
    test_ds = OmniMultimodalDataset(test_ids, "test", slice_cap=1, bc_aug_n=0, processor=processor)
    if len(test_ds) == 0:
        return np.zeros((len(test_ids), 5), dtype=np.float32)
    ctx_dim = test_ds[0]["ctx"].shape[0]

    lora_dir = MODELS / "omni3b-lora"
    lora_ckpts = sorted(lora_dir.glob("*/fold*.pt"))  # seed subdirs
    if not lora_ckpts:
        print("WARNING: no Omni LoRA ckpt found", flush=True)
        return np.zeros((len(test_ids), 5), dtype=np.float32)

    all_probs = []
    for ckpt in lora_ckpts[:5]:  # use 1 seed (5 folds) for speed
        thinker = build_thinker_with_lora()
        model = OmniHeadLoRA(ctx_dim=ctx_dim, thinker=thinker)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
        model.to(DEV).eval()

        probs = np.zeros((len(test_ds), 5), dtype=np.float32)
        for idx, (proc_out, ctx, _) in enumerate(test_ds):
            proc_out2 = {k: v.unsqueeze(0).to(DEV) if isinstance(v, torch.Tensor) else v
                         for k, v in proc_out.items()}
            ctx2 = ctx.unsqueeze(0).to(DEV)
            logits = model(proc_out2, ctx2)
            probs[idx] = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)

    return np.mean(all_probs, axis=0) if all_probs else np.zeros((len(test_ids), 5), dtype=np.float32)


# === 4. Fusion ===
def fuse_s5(ctx_p, wsp_p, hub_p, e2v_p, omni_p=None):
    """R4/S5 orthofuse — weights from env vars."""
    WSP_W = float(os.environ.get("WSP_W", "0.07"))
    E2V_W = float(os.environ.get("E2V_W", "0.03"))
    HUB_W = float(os.environ.get("HUB_W", "0.03"))
    OMNI_W = float(os.environ.get("OMNI_W", "0.05"))
    
    p = np.zeros_like(ctx_p)
    p[:, 0] = ctx_p[:, 0]
    p[:, 4] = ctx_p[:, 4]
    p[:, 1] = 0.7 * wsp_p[:, 1] + 0.3 * hub_p[:, 1]
    p[:, 2] = ctx_p[:, 2]
    p[:, 3] = (ctx_p[:, 3] + wsp_p[:, 3] + hub_p[:, 3]) / 3

    if WSP_W > 0:
        for k in [1, 2, 3]: p[:, k] = (1-WSP_W)*p[:, k] + WSP_W*wsp_p[:, k]
    if E2V_W > 0:
        for k in [1, 2, 3]: p[:, k] = (1-E2V_W)*p[:, k] + E2V_W*e2v_p[:, k]
    if HUB_W > 0:
        for k in [1, 2, 3]: p[:, k] = (1-HUB_W)*p[:, k] + HUB_W*hub_p[:, k]
    if omni_p is not None and OMNI_W > 0:
        for k in [1, 2, 3]: p[:, k] = (1-OMNI_W)*p[:, k] + OMNI_W*omni_p[:, k]
    return p


def main():
    import sys as _sys, traceback as _tb
    print(f"[S5] DEV={DEV} TEST_ROOT={TEST_ROOT} CUDA={torch.cuda.is_available()}", flush=True)
    try:
        # Step 1: Context LGBM
        print("[S5] Step 1/4: Context LGBM...", flush=True)
        ctx_probs, seg_ids = infer_context()
        print(f"  ctx probs: {ctx_probs.shape}", flush=True)

        # Step 2: Whisper + HuBERT + Emotion2vec
        print("[S5] Step 2/3: Whisper...", flush=True)
        wsp_probs = infer_encoder("whisper", MODELS / "wsp_head")
        print(f"  wsp: {wsp_probs.shape}", flush=True)

        print("[S5] Step 2/3: HuBERT...", flush=True)
        hub_probs = infer_encoder("hubert", MODELS / "hub_head")
        print(f"  hub: {hub_probs.shape}", flush=True)

        print("[S5] Step 2/3: Emotion2vec...", flush=True)
        e2v_probs = infer_encoder("emotion2vec", MODELS / "e2v_head")
        print(f"  e2v: {e2v_probs.shape}", flush=True)

        # Step 3: Fusion
        print("[S5] Step 3/3: Orthofuse...", flush=True)
        s5_probs = fuse_s5(ctx_probs, wsp_probs, hub_probs, e2v_probs)

        n_test = len(seg_ids)
        if n_test == 0:
            print("[S5] No test data found. Creating empty output.", flush=True)
            OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_CSV, "w") as f:
                f.write("segment_id," + ",".join(SUBMIT) + "\n")
            print(f"[S5] Done (empty).", flush=True)
            return
        s5_probs = s5_probs[:n_test]  # ensure alignment

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(seg_ids):
                vals = [str(int(s5_probs[i, COL2K[c]] >= THR[COL2K[c]])) for c in SUBMIT]
                f.write(f"{sid}," + ",".join(vals) + "\n")

        pos = {c: int((s5_probs[:, COL2K[c]] >= THR[COL2K[c]]).sum()) for c in SUBMIT}
        print(f"[S5] Done. {OUTPUT_CSV} pos={pos}", flush=True)
    except Exception:
        _tb.print_exc()
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_CSV, "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
        _sys.exit(1)


if __name__ == "__main__":
    main()
