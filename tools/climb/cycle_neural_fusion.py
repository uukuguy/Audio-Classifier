"""H-N1: 神经融合小头验证 Qwen3 文本 embedding 能否救 T/I.

纠正 LGBM 错配：稠密 embedding 喂神经层(线性投影+MLP)，不喂树。
冻结编码器(用已缓存文本特征)，只训小头，MPS 秒级。

架构: context手工特征(80d) + Qwen3文本(1024d→线性投影128d) → concat → MLP → 5类 sigmoid
损失: BCE + per-label pos_weight (温和)。

先用已缓存的 ~110 通小规模验证。看到 T/I 真涨再全量提取剩余通。

Usage: python tools/climb/cycle_neural_fusion.py [--epochs N]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "tools/climb")
from cycle_context_v2 import featurize as ctxfeat

CTX, TGT, STRIDE, CHUNK_MS = 375, 25, 5, 80
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SEED = 42


def load_data(cached_ids):
    Xc, Xt, Y, G = [], [], [], []
    for gi, cid in enumerate(cached_ids):
        a = np.load(f"data/train/labels/{cid}.npy")
        utts = json.load(open(f"data/train/text/{cid}.json")).get("utterances", [])
        ends = sorted(int(u.get("end_ms", 0)) for u in utts)
        tf = np.load(f"data/cache/qwen_text/{cid}.npz")
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE):
            nvis = sum(1 for x in ends if x <= e * CHUNK_MS)
            if str(nvis) not in tf:
                continue
            ctx = a[e - CTX:e].astype(int)
            fut = set(int(x) for x in a[e:e + TGT])
            Xc.append(ctxfeat(ctx)); Xt.append(tf[str(nvis)])
            Y.append([1 if k in fut else 0 for k in range(5)]); G.append(gi)
    return (np.array(Xc, dtype=np.float32), np.array(Xt, dtype=np.float32),
            np.array(Y, dtype=np.float32), np.array(G))


class FusionHead(nn.Module):
    def __init__(self, ctx_dim, txt_dim, proj=128, hid=256):
        super().__init__()
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, proj), nn.LayerNorm(proj), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim + proj, hid), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hid // 2, 5))

    def forward(self, xc, xt):
        return self.mlp(torch.cat([xc, self.txt_proj(xt)], dim=-1))


def train_eval(Xc, Xt, Y, tr, va, pos_weight, device, epochs):
    model = FusionHead(Xc.shape[1], Xt.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xc_tr = torch.tensor(Xc[tr]).to(device); xt_tr = torch.tensor(Xt[tr]).to(device)
    y_tr = torch.tensor(Y[tr]).to(device)
    bs = 4096
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(xc_tr[idx], xt_tr[idx]), y_tr[idx])
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xc[va]).to(device),
                                 torch.tensor(Xt[va]).to(device))).cpu().numpy()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--ctx_only", action="store_true", help="消融：只 context 不文本")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    cached = sorted(Path(p).stem for p in glob.glob("data/cache/qwen_text/*.npz"))
    print(f"[neural] {len(cached)} 通缓存可用，device={device}", file=sys.stderr)
    Xc, Xt, Y, G = load_data(cached)
    print(f"[neural] {len(Xc)} 窗, ctx{Xc.shape[1]}d + qwen{Xt.shape[1]}d", file=sys.stderr)
    if args.ctx_only:
        Xt = np.zeros_like(Xt)  # 消融

    pw = torch.tensor([(len(Y) - Y[:, k].sum()) / max(1, Y[:, k].sum()) for k in range(5)],
                      dtype=torch.float32).to(device).clamp(max=10)  # 温和封顶
    gkf = GroupKFold(3)
    oof = np.zeros((len(Xc), 5))
    for tr, va in gkf.split(Xc, Y[:, 0], groups=G):
        oof[va] = train_eval(Xc, Xt, Y, tr, va, pw, device, args.epochs)

    f1s = {}
    for k in range(5):
        bt, bf = 0.5, -1
        lo, hi = (0.05, 0.25) if k == 0 else (0.35, 0.65)
        for t in np.linspace(lo, hi, 13):
            f = f1_score(Y[:, k], (oof[:, k] >= t).astype(int), zero_division=0)
            if f > bf: bf, bt = f, t
        f1s[k] = bf
    macro = np.mean(list(f1s.values()))
    tag = "ctx-only(消融)" if args.ctx_only else "ctx+qwen神经融合"
    print(f"[neural] {tag}: macro={macro:.4f} | " + " ".join(f"{LAB[k]}={f1s[k]:.3f}" for k in range(5)))


if __name__ == "__main__":
    main()
