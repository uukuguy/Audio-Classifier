"""上云神经头训练（消费 whisper-large-v3 帧特征缓存）→ 攻 BC → 出 pred_test1.csv。

架构(承 cycle_vap_whisper 的 WhisperVAP，升 large-v3 1280维):
  双声道 whisper 帧[2,80,1280](冻结特征) → proj → cross-attn(query 聚合) → 音频2向量
  + context 手工特征(80d) → MLP head → 5类 sigmoid。

评估用 cap1 切片 CV(可信协议, 见 docs/status/2026-05-28-sliced-cv-audit.md), 不用滑窗。
阈值在 cap1 OOF 上调(接近 0.5, 阈值铁律)。

依赖缓存: data/whisper_cache/{train,test}/<cid>.npz (extract_whisper_cuda.py 产出)。

Usage（云终端）:
  python cloud/train_head_cuda.py --epochs 15 --run-dir tools/runs/climb/cloud-whisper-h001
输出: <run-dir>/pred_test1.csv, <run-dir>/cv_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat  # noqa: E402

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
CTX, TGT = 375, 25
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CACHE = os.environ.get("WCACHE", "data/whisper_cache")


def load_train():
    """读 train 缓存 + 对齐 label → ctx 特征 / 音频帧 / 标签 / 会话分组 / 每通片段序号。"""
    Xc, Xa, Y, G, order = [], [], [], [], []
    files = sorted(glob.glob(f"{CACHE}/train/*.npz"))
    for gi, f in enumerate(files):
        cid = Path(f).stem
        z = np.load(f)
        frames, ends = z["frames"], z["ends"]  # [W,2,80,1280], [W]
        a = np.load(f"data/train/labels/{cid}.npy")
        for j, e in enumerate(ends):
            Xc.append(ctxfeat(a[e - CTX:e].astype(int)))
            Xa.append(frames[j])
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(5)])
            G.append(gi)
            order.append(j)
    return (np.array(Xc, np.float32), np.array(Xa, np.float16),
            np.array(Y, np.float32), np.array(G), np.array(order, np.int32))


def load_test():
    Xc, Xa, ids = [], [], []
    for f in sorted(glob.glob(f"{CACHE}/test/*.npz")):
        cid = Path(f).stem
        z = np.load(f)
        frames = z["frames"]  # [1,2,80,1280]
        ctx = np.load(f"data/test/context/{cid}.npy")
        Xc.append(ctxfeat(ctx.astype(int)))
        Xa.append(frames[0])
        ids.append(cid)
    return np.array(Xc, np.float32), np.array(Xa, np.float16), ids


class WhisperVAP(nn.Module):
    def __init__(self, ctx_dim: int, wd: int = 1280, d: int = 192):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(wd, d), nn.LayerNorm(d), nn.GELU())
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.cn = nn.BatchNorm1d(ctx_dim)
        fin = ctx_dim + 2 * d
        self.head = nn.Sequential(
            nn.LayerNorm(fin), nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, ctx, aud):
        c = self.cn(ctx)
        a = self.proj(aud[:, 0]); b = self.proj(aud[:, 1])
        B = a.shape[0]; q = self.q.expand(B, -1, -1)
        ca, _ = self.cross(q, b, b); cb, _ = self.cross(q, a, a)
        return self.head(torch.cat([c, ca.squeeze(1), cb.squeeze(1)], -1))


def train_fold(Xc, Xa, Y, tr, epochs, pw):
    m = WhisperVAP(ctx_dim=Xc.shape[1], wd=Xa.shape[-1]).to(DEV)  # wd 从特征维推断(small 768/large 1280)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    ct = torch.tensor(Xc[tr]).to(DEV); at = torch.tensor(Xa[tr]).float().to(DEV)
    yt = torch.tensor(Y[tr]).to(DEV)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            idx = perm[i:i + 256]; opt.zero_grad()
            loss = crit(m(ct[idx], at[idx]), yt[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--run-dir", default="tools/runs/climb/cloud-whisper-h001")
    args = ap.parse_args()
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)

    print(f"[head] dev={DEV} 读 train 缓存...", file=sys.stderr)
    Xc, Xa, Y, G, order = load_train()
    print(f"[head] {len(Xc)}窗 ctx{Xc.shape[1]} aud{Xa.shape}", file=sys.stderr)
    pw = torch.tensor([(len(Y) - Y[:, k].sum()) / max(1, Y[:, k].sum())
                       for k in range(5)]).float().clamp(max=10).to(DEV)

    # 会话级 fold(防泄漏)
    uniq = np.unique(G)
    rng = np.random.default_rng(SEED); rng.shuffle(uniq)
    fold_of = {g: i % args.folds for i, g in enumerate(uniq)}
    fold_id = np.array([fold_of[g] for g in G])

    oof = np.zeros((len(Xc), 5)); models = []
    for fi in range(args.folds):
        va = np.where(fold_id == fi)[0]; tr = np.where(fold_id != fi)[0]
        m = train_fold(Xc, Xa, Y, tr, args.epochs, pw)
        with torch.no_grad():
            oof[va] = torch.sigmoid(m(torch.tensor(Xc[va]).to(DEV),
                                      torch.tensor(Xa[va]).float().to(DEV))).cpu().numpy()
        models.append(m)
        print(f"[head] fold {fi + 1}/{args.folds}", file=sys.stderr)

    # cap1 切片 CV: 每通取片段序号 0 的窗(模拟 test 独立片段)
    cap1 = order == 0
    thr, f1s = {}, {}
    for k in range(5):
        bf, bt = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(Y[cap1, k], (oof[cap1, k] >= t).astype(int), zero_division=0)
            if f > bf: bf, bt = f, float(t)
        thr[k], f1s[k] = bt, bf
    macro = float(np.mean(list(f1s.values())))
    print(f"[head] cap1切片CV macro={macro:.4f} | " +
          " ".join(f"{LAB[k]}={f1s[k]:.3f}@{thr[k]:.2f}" for k in range(5)), file=sys.stderr)
    print(f"[head] ★BC={f1s[2]:.3f} (纯ctx基线0.222, 看是否破)", file=sys.stderr)

    # test 预测(5 fold 模型 rank 平均)
    Xtc, Xta, ids = load_test()
    probs = np.zeros((len(ids), 5))
    for m in models:
        with torch.no_grad():
            probs += torch.sigmoid(m(torch.tensor(Xtc).to(DEV),
                                     torch.tensor(Xta).float().to(DEV))).cpu().numpy()
    probs /= len(models)

    SUBMIT = ["c", "na", "i", "bc", "t"]; COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
    with open(run / "pred_test1.csv", "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(ids):
            row = [sid] + [str(int(probs[i, COL2K[c]] >= thr[COL2K[c]])) for c in SUBMIT]
            f.write(",".join(row) + "\n")

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "cloud-whisper-large-v3-vap", "cap1_macro_f1": round(macro, 4),
        "per_sub": {LAB[k]: round(f1s[k], 4) for k in range(5)},
        "thresholds": {LAB[k]: round(thr[k], 2) for k in range(5)},
    }, ensure_ascii=False, indent=2))
    print(f"[head] wrote {run}/pred_test1.csv", file=sys.stderr)
    print(json.dumps({"score": round(macro, 4),
                      "per_sub": {LAB[k]: round(f1s[k], 4) for k in range(5)}}))


if __name__ == "__main__":
    main()
