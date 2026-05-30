"""音频韵律分支探针: 哪个显式韵律维度对 BC 信号强 (用户问).

之前只整体测音频塔(VAP/mel/whisper)vs context, never拆音频内部分支。
VAP/mel/whisper 是 DL 频谱表征(学语义音色), 可能漏掉最朴素韵律(F0下降/能量/停顿)
=turn-taking文献 backchannel 经典线索。本探针提显式韵律, 逐分支测对 BC 的 point-biserial r。

分支 (预测点前窗的韵律, 因果):
  能量: 末窗 RMS 均值/末段斜率(下降?)/停顿比例(低能量帧占比)
  F0/pitch: torchaudio.detect_pitch_frequency 末窗均值/末段斜率(下降=turn-yield)
  双声道: 能量比/主导方/连续单声道时长
  动态: 能量变化率(短窗vs长窗)

本地, 8kHz 原始音频, 不经 VAP. 限线程. 40通快验.
"""
import glob
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio

CTX, TGT, CHUNK_MS, SR = 375, 25, 80, 8000
BC = 2
WIN_SEC = 5
NCONV = 40


def load_wav(cid):
    with wave.open(f"data/train/audio/{cid}.wav", "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2).T.astype(np.float32) / 32768.0
    return arr, sr


def prosody_feats(seg2):
    """seg2: [2, samples] 8kHz 双声道窗. 返回显式韵律特征 dict."""
    f = {}
    # 帧化 (25ms 窗, 10ms hop)
    fl, hop = int(0.025 * SR), int(0.010 * SR)
    def frames(x):
        n = max(0, (len(x) - fl) // hop + 1)
        return np.array([x[i * hop:i * hop + fl] for i in range(n)]) if n > 0 else np.zeros((1, fl))
    mono = seg2.mean(0)
    fr = frames(mono)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-8)            # 每帧能量
    # 能量分支
    f["energy_mean"] = float(rms.mean())
    f["energy_last_slope"] = float(rms[-10:].mean() - rms[-30:-10].mean()) if len(rms) >= 30 else 0.0
    f["pause_ratio"] = float((rms < rms.mean() * 0.3).mean())   # 低能量(停顿)帧占比
    f["energy_var"] = float(rms.var())
    # F0/pitch 分支 (torchaudio)
    try:
        t = torch.from_numpy(mono).unsqueeze(0)
        pf = torchaudio.functional.detect_pitch_frequency(t, SR).squeeze(0).numpy()
        pf = pf[pf > 0]                                  # 有声帧
        if len(pf) >= 20:
            f["f0_mean"] = float(pf.mean())
            f["f0_last_slope"] = float(pf[-len(pf)//4:].mean() - pf[:len(pf)//4].mean())  # 末段-初段(下降?)
            f["f0_var"] = float(pf.var())
        else:
            f["f0_mean"] = f["f0_last_slope"] = f["f0_var"] = 0.0
    except Exception:
        f["f0_mean"] = f["f0_last_slope"] = f["f0_var"] = 0.0
    # 双声道分支
    rms0 = np.sqrt((frames(seg2[0]) ** 2).mean(1) + 1e-8)
    rms1 = np.sqrt((frames(seg2[1]) ** 2).mean(1) + 1e-8)
    f["ch_energy_ratio"] = float(rms0.mean() / (rms1.mean() + 1e-6))
    f["ch_dominance"] = float(abs(rms0.mean() - rms1.mean()) / (rms0.mean() + rms1.mean() + 1e-6))
    # 末段哪方主导(听者沉默=BC前兆)
    f["last_ch0_active"] = float((rms0[-20:] > rms0.mean()).mean()) if len(rms0) >= 20 else 0.0
    f["last_ch1_active"] = float((rms1[-20:] > rms1.mean()).mean()) if len(rms1) >= 20 else 0.0
    # 动态: 能量变化率(短vs长窗)
    f["energy_recent_vs_old"] = float(rms[-len(rms)//4:].mean() - rms.mean()) if len(rms) >= 4 else 0.0
    return f


def main():
    torch.set_num_threads(4)
    conv = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))[:NCONV]
    rows, has_bc = [], []
    for ci, cid in enumerate(conv):
        labels = np.load(f"data/train/labels/{cid}.npy")
        wav, sr = load_wav(cid)
        lo, hi = CTX, len(labels) - TGT
        if lo >= hi:
            continue
        for e in list(range(lo, hi, max(1, (hi - lo) // 20)))[:20]:
            fut = set(int(x) for x in labels[e:e + TGT])
            end8 = int(e * CHUNK_MS / 1000 * sr)
            seg = wav[:, max(0, end8 - WIN_SEC * sr):end8]
            if seg.shape[1] < SR:
                continue
            rows.append(prosody_feats(seg))
            has_bc.append(1 if BC in fut else 0)
        if (ci + 1) % 10 == 0:
            print(f"[prosody] {ci+1}/{len(conv)} convs, {len(rows)} slices", file=sys.stderr)

    has_bc = np.array(has_bc)
    keys = list(rows[0].keys())
    from scipy.stats import pointbiserialr
    print(f"\n[prosody] {len(rows)} slices, BC正例={int(has_bc.sum())} ({has_bc.mean()*100:.1f}%)")
    print(f"{'韵律分支':<22}{'BC=1':>10}{'BC=0':>10}{'|r|':>8}{'分支':>10}")
    results = []
    for k in keys:
        v = np.array([r[k] for r in rows])
        try:
            r, _ = pointbiserialr(has_bc, v)
        except Exception:
            r = 0.0
        results.append((k, abs(r), v[has_bc == 1].mean(), v[has_bc == 0].mean(), r))
    # 按 |r| 排序, 标分支
    branch = {"energy": "能量", "pause": "能量", "f0": "F0", "ch": "双声道",
              "last_ch": "双声道", "energy_recent": "动态"}
    for k, ar, m1, m0, r in sorted(results, key=lambda x: -x[1]):
        b = next((v for kk, v in branch.items() if k.startswith(kk)), "其他")
        flag = "★强" if ar > 0.15 else ("弱+" if ar > 0.08 else "")
        print(f"{k:<22}{m1:>10.3f}{m0:>10.3f}{ar:>8.3f}{b:>8} {flag}")
    print("\n[prosody] 对比: VAP现成信号|r|<0.04, context时序r~0.13, LGBM-BC基座0.222")
    print("[prosody] |r|>0.15分支=该显式韵律对BC强→值得专门提此分支特征(可能补DL频谱漏掉的)")


if __name__ == "__main__":
    main()
