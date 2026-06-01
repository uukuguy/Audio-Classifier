"""N1 真实现 - 派生 train_head_hubert.py + DB-Loss + SupCon (B4 Knowledge Layer 发现).

D-14 教训: 单纯校准 ctx+whisper 不涨 cap1 (校准头无新源 = 重复 SOTA orthofuse).
N1 真实现 = 在 train_head 神经头基础上替换 BCE 为 DB-Loss + α·SupCon, 配合 hubert/whisper 新源。

设计:
- 不动 WhisperVAP 架构 (已验证, hubert head cap1 0.6239)
- 替换 BCE → Distribution-Balanced BCE (negative-tolerant + class freq re-weight)
- 加 α × SupCon on BC (最长尾, 0.5%) 强迫 BC 簇集
- 5fold OOF + cap1 评估 + 3 阈值预设出 csv (跟 hubert head 一致)

Push 门 (D-13):
- cap1 macro >= hubert head baseline 0.6239 + 0.005 = 0.6289
- 单源不破 SOTA, 但要跟 SOTA 跨源融合后看真分

云端运行:
- 在 4090, 用 stride40 hubert/whisper cache (11G+~14G stride40 whisper 部分), 系统盘 200G 扩容够
- 估算时间: train_head_hubert 4min/fold × 5fold = 20min (跟 D-8 cycle 16 一致)

Usage (云端):
  WCACHE=/root/autodl-fs/hubert_cache OMP_NUM_THREADS=4 \\
    python cloud/train_head_n1.py --epochs 15 --alpha 0.3 --run-dir tools/runs/climb/n1-hubert-dbloss
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

# 复用 train_head_hubert 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_head_hubert as base  # type: ignore

LAB = base.LAB
SEED = base.SEED
DEV = base.DEV
WhisperVAP = base.WhisperVAP
load_train = base.load_train
load_test = base.load_test
predict_batched = base.predict_batched


def distribution_balanced_bce(logits, targets, class_freq, neg_scale=2.0):
    """DB-Loss (Wu et al ECCV 2020 简化版).

    1. pos_weight = log(1 / class_freq).clamp(max=5) — 罕见类正样本权重更高
    2. 负样本 logit / neg_scale → 缓解 negative over-suppression
    """
    pos_weight = torch.log(1.0 / class_freq.clamp(min=1e-4)).clamp(max=5.0).to(logits.device)
    logits_for_bce = torch.where(targets > 0.5, logits, logits / neg_scale)
    return F.binary_cross_entropy_with_logits(
        logits_for_bce, targets, pos_weight=pos_weight, reduction='mean'
    )


def supcon_loss(features, labels, target_class=2, temperature=0.1):
    """SupCon for one target class (BC=2 default).

    强制同类样本在 contrast feature space 聚类, 远离其它. 不动主 head.
    """
    target = labels[:, target_class]
    if target.sum() < 2:
        return torch.tensor(0.0, device=features.device)

    sim = features @ features.t() / temperature
    mask = (target.unsqueeze(0) == target.unsqueeze(1)).float()
    mask = (mask - torch.eye(len(mask), device=mask.device)).clamp(min=0)

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = sim.exp() - torch.eye(len(sim), device=sim.device)
    log_prob = sim - exp_sim.sum(1, keepdim=True).log()

    pos_count = mask.sum(1).clamp(min=1)
    return -((mask * log_prob).sum(1) / pos_count).mean()


class WhisperVAP_N1(nn.Module):
    """派生 WhisperVAP + 加 contrast projection head 用于 SupCon."""

    def __init__(self, ctx_dim: int, wd: int = 1280, d: int = 192):
        super().__init__()
        self.base = WhisperVAP(ctx_dim=ctx_dim, wd=wd, d=d)
        # 通过 hook 截 head 的倒数第二层 (128d) 做 contrast projection
        # 简化: 直接用 head 输出前一层 (128) 接一个新 projection
        fin = ctx_dim + 2 * d
        self.contrast = nn.Linear(128, 32)  # contrast projection
        # 拆 head 让我们能拿到中间表征
        self.head_part1 = nn.Sequential(
            nn.LayerNorm(fin), nn.Linear(fin, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3))
        self.head_part2 = nn.Linear(128, 5)

    def forward(self, ctx, aud, return_feats=False):
        c = self.base.cn(ctx)
        a = self.base.proj(aud[:, 0]); b = self.base.proj(aud[:, 1])
        B = a.shape[0]; q = self.base.q.expand(B, -1, -1)
        ca, _ = self.base.cross(q, b, b); cb, _ = self.base.cross(q, a, a)
        h = self.head_part1(torch.cat([c, ca.squeeze(1), cb.squeeze(1)], -1))  # [B, 128]
        logits = self.head_part2(h)
        if return_feats:
            return logits, F.normalize(self.contrast(h), dim=-1)
        return logits


def train_fold_n1(Xc, Xa, Y, tr_idx, epochs, class_freq, alpha=0.3, batch_size=256):
    """派生 train_fold: 加 DB-Loss + α·SupCon."""
    m = WhisperVAP_N1(ctx_dim=Xc.shape[1], wd=Xa.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(tr_idx)
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(tr_idx)
        ep_loss_db, ep_loss_sc, nbatch = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            cb = torch.from_numpy(np.asarray(Xc[batch_idx])).to(DEV, non_blocking=True)
            ab = torch.from_numpy(np.asarray(Xa[batch_idx])).to(DEV, non_blocking=True).float()
            yb = torch.from_numpy(np.asarray(Y[batch_idx])).to(DEV, non_blocking=True)
            opt.zero_grad()
            logits, feats = m(cb, ab, return_feats=True)
            l_db = distribution_balanced_bce(logits, yb, class_freq)
            l_sc = supcon_loss(feats, yb, target_class=2)  # BC = 最长尾
            loss = l_db + alpha * l_sc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            ep_loss_db += float(l_db); ep_loss_sc += float(l_sc); nbatch += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"[N1]   ep {ep+1}/{epochs} db={ep_loss_db/nbatch:.4f} sc={ep_loss_sc/nbatch:.4f}", file=sys.stderr)
    m.eval()
    return m


@torch.no_grad()
def predict_n1(model, Xc, Xa, idx, batch_size=512):
    """N1 推理: 仅取 logits, 不要 contrast feats."""
    n = len(idx)
    out = np.zeros((n, 5), dtype=np.float32)
    for i in range(0, n, batch_size):
        sub = idx[i:i + batch_size]
        cb = torch.from_numpy(np.asarray(Xc[sub])).to(DEV)
        ab = torch.from_numpy(np.asarray(Xa[sub])).to(DEV).float()
        logits = model(cb, ab, return_feats=False)
        out[i:i + batch_size] = torch.sigmoid(logits).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.3, help="SupCon 权重")
    ap.add_argument("--run-dir", default="tools/runs/climb/n1-hubert-dbloss")
    args = ap.parse_args()
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)

    print(f"[N1] dev={DEV} 读 train 缓存...", file=sys.stderr)
    Xc, Xa, Y, G, order = load_train()
    print(f"[N1] {len(Xc)}窗 ctx{Xc.shape[1]} aud{Xa.shape}", file=sys.stderr)

    # 类频率 (DB-Loss 用)
    class_freq = torch.tensor(Y.mean(0), dtype=torch.float32)
    print(f"[N1] class freq: {dict(zip(LAB, [f'{x:.4f}' for x in class_freq]))}", file=sys.stderr)

    # 会话级 fold
    uniq = np.unique(G)
    rng = np.random.default_rng(SEED); rng.shuffle(uniq)
    fold_of = {g: i % args.folds for i, g in enumerate(uniq)}
    fold_id = np.array([fold_of[g] for g in G])

    oof = np.zeros((len(Xc), 5)); models = []
    for fi in range(args.folds):
        va = np.where(fold_id == fi)[0]; tr = np.where(fold_id != fi)[0]
        m = train_fold_n1(Xc, Xa, Y, tr, args.epochs, class_freq, alpha=args.alpha)
        oof[va] = predict_n1(m, Xc, Xa, va)
        models.append(m)
        print(f"[N1] fold {fi+1}/{args.folds} done", file=sys.stderr)

    # cap1 + 阈值 (与 train_head_hubert 一致)
    cap1 = order == 0
    thr_cap1, f1_cap1 = {}, {}
    for k in range(5):
        bf, bt = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(Y[cap1, k], (oof[cap1, k] >= t).astype(int), zero_division=0)
            if f > bf: bf, bt = f, float(t)
        thr_cap1[k], f1_cap1[k] = bt, bf
    macro_cap1 = float(np.mean(list(f1_cap1.values())))
    print(f"[N1] cap1 macro={macro_cap1:.4f} (baseline hubert head 0.6239)", file=sys.stderr)
    print(f"[N1] per-class: {dict(zip(LAB, [f'{f1_cap1[k]:.3f}' for k in range(5)]))}", file=sys.stderr)

    # test
    Xtc, Xta, ids = load_test()
    test_idx = np.arange(len(ids))
    probs = np.zeros((len(ids), 5), dtype=np.float32)
    for m in models:
        probs += predict_n1(m, Xtc, Xta, test_idx)
    probs /= len(models)

    # 出 csv (与 train_head_hubert 一致)
    SUBMIT = ["c", "na", "i", "bc", "t"]; COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
    THR_PRESETS = {
        "cap1": thr_cap1,
        "cycle1": {0: 0.05, 4: 0.05, 1: 0.50, 3: 0.50, 2: 0.50},
        "orthofuse": {0: 0.05, 4: 0.25, 1: 0.50, 3: 0.65, 2: 0.75},
    }
    csv_summary = {}
    for name, thr in THR_PRESETS.items():
        csv_path = run / f"pred_test1_{name}.csv"
        pos_counts = {c: 0 for c in SUBMIT}
        with open(csv_path, "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(ids):
                vals = {c: int(probs[i, COL2K[c]] >= thr[COL2K[c]]) for c in SUBMIT}
                for c, v in vals.items(): pos_counts[c] += v
                f.write(f"{sid}," + ",".join(str(vals[c]) for c in SUBMIT) + "\n")
        csv_summary[name] = {"thresholds": {LAB[k]: round(thr[k], 2) for k in range(5)},
                              "pos": pos_counts}
        print(f"[N1] wrote {csv_path.name}: " + " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    # 存连续概率 npz
    np.savez_compressed(
        run / "probs.npz",
        oof=oof.astype(np.float32), test=probs.astype(np.float32),
        Y=Y.astype(np.int8), G=G.astype(np.int32),
        order=order.astype(np.int32), test_ids=np.array(ids),
    )
    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "n1-dbloss-supcon",
        "cap1_macro_f1": round(macro_cap1, 4),
        "per_sub_cap1": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)},
        "thresholds_cap1": {LAB[k]: round(thr_cap1[k], 2) for k in range(5)},
        "alpha": args.alpha,
        "csv_variants": csv_summary,
        "_note": "N1 = DB-Loss + α·SupCon on hubert head. SOTA orthofuse cap1=0.6410. Push 门 cap1 ≥0.6460.",
    }, ensure_ascii=False, indent=2))
    print(f"[N1] artifacts: {run}/", file=sys.stderr)
    print(json.dumps({"cap1_score": round(macro_cap1, 4),
                      "per_sub": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)}}))


if __name__ == "__main__":
    main()
