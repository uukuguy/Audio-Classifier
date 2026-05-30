"""H-V4 信号探针 (不训练, 纯前向): VAP 预训练 head 的 p_now/p_future/vad/熵
对 BC 段 vs 非BC段 有无区分度。

之前 train_vap.py 错在 mean-pool out["x"] 喂自定义MLP, 绕过预训练 vap_head。
这里用回 VAP 原生 model.probs(): 返回 p_now/p_future(下一说话人) + 256类probs + vad + 熵H。
若这些原生信号在 BC 段有区分度 → H-V4(用projection概率当特征)值得训练。
若无 → 印证 VAP 原文 BC=TODO, head 对 BC 也无能, 转方向。

纯前向, 几分钟。GROUND TRUTH 用 train labels (有真值才能算区分度)。
"""
import glob
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio

VAP_ROOT = os.environ.get("VAP_ROOT", str(Path(__file__).parent.parent / "baselines/VAP"))
sys.path.insert(0, VAP_ROOT)

CTX, TGT, CHUNK_MS = 375, 25, 80
SR16 = 16000
VAP_CKPT = os.environ.get("VAP_CKPT", str(Path(VAP_ROOT) / "example/checkpoints/VAP_state_dict.pt"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
WIN_SEC = int(os.environ.get("WIN_SEC", "10"))
NCONV = int(os.environ.get("NCONV", "40"))


def build_vap():
    from vap.modules.encoder import EncoderCPC
    from vap.modules.modules import TransformerStereo
    from vap.modules.VAP import VAP
    enc = EncoderCPC(load_pretrained=False)
    tr = TransformerStereo(dim=enc.dim)
    m = VAP(enc, tr)
    sd = torch.load(VAP_CKPT, map_location="cpu", weights_only=True)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[probe] loaded missing={len(miss)} unexpected={len(unexp)}", file=sys.stderr)
    return m.to(DEV).eval()


def load_wav(cid):
    with wave.open(f"data/train/audio/{cid}.wav", "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def main():
    m = build_vap()
    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))[:NCONV]
    print(f"[probe] {len(conv_ids)} convs, win={WIN_SEC}s dev={DEV}", file=sys.stderr)

    # 对每通取多个切片点, 记录: VAP原生信号 (末帧的 p_future/p_now/vad熵) + 未来2s是否有BC
    rows = []  # (has_bc, p_now_last, p_fut_last, vad_act_last, entropy_last, vad_ch_imbalance)
    BC = 2
    for ci, cid in enumerate(conv_ids):
        labels = np.load(f"data/train/labels/{cid}.npy")
        wav, sr = load_wav(cid)
        lo, hi = CTX, len(labels) - TGT
        if lo >= hi:
            continue
        ends = list(range(lo, hi, max(1, (hi - lo) // 20)))[:20]  # 20 切片/通
        for end in ends:
            fut = set(int(x) for x in labels[end:end + TGT])
            has_bc = 1 if BC in fut else 0
            # 末 win_sec 双声道 8k → 16k
            end8 = int(end * CHUNK_MS / 1000 * sr)
            seg = wav[:, max(0, end8 - WIN_SEC * sr):end8]
            seg16 = torchaudio.functional.resample(torch.from_numpy(seg), sr, SR16)
            need = WIN_SEC * SR16
            if seg16.shape[1] < need:
                seg16 = torch.nn.functional.pad(seg16, (need - seg16.shape[1], 0))
            else:
                seg16 = seg16[:, -need:]
            x = seg16.unsqueeze(0).to(DEV)
            with torch.inference_mode():
                out = m.probs(x)  # p_now/p_future/probs/vad/H
            # 取末帧 (预测点)
            p_now = float(out["p_now"][0, -1])       # 下一说话人=speaker0 概率
            p_fut = float(out["p_future"][0, -1])
            H = float(out["H"][0, -1])               # 熵 (低=模型确信)
            vad = out["vad"][0, -5:].mean(dim=0).cpu().numpy()  # 末5帧两声道VAD均值
            vad_imb = abs(float(vad[0]) - float(vad[1]))  # 声道不平衡(一方在说=可能将BC)
            vad_tot = float(vad[0] + vad[1])
            rows.append((has_bc, p_now, p_fut, H, vad_imb, vad_tot))
        if (ci + 1) % 10 == 0:
            print(f"[probe] {ci+1}/{len(conv_ids)} convs, {len(rows)} slices", file=sys.stderr)

    arr = np.array(rows)
    n = len(arr)
    bc_mask = arr[:, 0] == 1
    print(f"\n[probe] {n} slices, BC正例={int(bc_mask.sum())} ({bc_mask.mean()*100:.1f}%)")
    names = ["p_now", "p_future", "entropy_H", "vad_imbalance", "vad_total"]
    print(f"{'signal':<16}{'BC=1 mean':>12}{'BC=0 mean':>12}{'abs_diff':>10}{'pointbiser_r':>14}")
    from scipy.stats import pointbiserialr
    for j, nm in enumerate(names, start=1):
        v = arr[:, j]
        m1, m0 = v[bc_mask].mean(), v[~bc_mask].mean()
        try:
            r, _ = pointbiserialr(arr[:, 0], v)
        except Exception:
            r = float("nan")
        print(f"{nm:<16}{m1:>12.4f}{m0:>12.4f}{abs(m1-m0):>10.4f}{r:>14.3f}")
    print("\n[probe] 判读: |r|>0.15 = 该原生信号对BC有区分度 → H-V4值得训练")
    print("[probe]       全部|r|<0.1 = VAP原生head对BC无信息(印证原文TODO) → 转方向")


if __name__ == "__main__":
    main()
