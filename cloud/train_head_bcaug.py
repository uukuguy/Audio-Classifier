"""P1.5 v2 训 head — 消费 BC 增强 cache + 原 cache → 5fold OOF + test → probs.npz.

承 train_head_hubert.py 架构 (WhisperVAP):
  双声道帧 [2, 80, FDIM] (冻结) → proj → cross-attn → context fusion → 5 类 head

差别: load_train 合并原 stride40 cache + BC 增强 cache (BC 正例多 3x), val 只用原始 (防虚高).

Usage (云端):
  WCACHE=/autodl-fs/data/whisper_cache_full WCACHE_BCAUG=/autodl-fs/data/whisper_bcaug \\
    python cloud/train_head_bcaug.py --epochs 15 \\
      --run-dir tools/runs/climb/whisper-bcaug-head-$(date +%Y%m%d-%H%M)

  WCACHE=/root/autodl-fs/hubert_cache WCACHE_BCAUG=/autodl-fs/data/hubert_bcaug \\
    python cloud/train_head_bcaug.py --epochs 15 \\
      --run-dir tools/runs/climb/hubert-bcaug-head-$(date +%Y%m%d-%H%M)
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = os.environ.get("WCACHE", "data/whisper_cache")
CACHE_BCAUG = os.environ.get("WCACHE_BCAUG", "data/whisper_bcaug")


def load_train_with_bcaug():
    """两遍扫描:
    第一遍数原 cache + BC 增强 cache 总 N, 分配 Xa.
    第二遍填 Xa.

    输出:
      Xc [N, ctx_dim], Xa [N, 2, 80, FDIM] fp16,
      Y [N, 5], G [N] conv id, order [N] (原 cache 内为窗序 0,1,2..; 增强为 -1)
      is_aug [N] (0=原始, 1=增强)
    """
    orig_files = sorted(glob.glob(f"{CACHE}/train/*.npz"))
    if not orig_files:
        sys.exit(f"[head-bcaug] 原 cache {CACHE}/train 不存在 (空)")
    bcaug_files = sorted(glob.glob(f"{CACHE_BCAUG}/train/*.npz"))
    print(f"[head-bcaug] 原 cache: {len(orig_files)} 通", file=sys.stderr)
    print(f"[head-bcaug] BC 增强 cache: {len(bcaug_files)} 通", file=sys.stderr)

    # 探测 FDIM
    _probe = np.load(orig_files[0])["frames"]
    FDIM = _probe.shape[-1]
    print(f"[head-bcaug] FDIM={FDIM}", file=sys.stderr)

    # 第一遍: 数 N, 累 ctx/Y/G/order/is_aug
    Xc, Y, G, order_arr, is_aug_arr = [], [], [], [], []
    sizes_orig, sizes_aug = [], []
    cid_to_gi = {}

    for gi, f in enumerate(orig_files):
        cid = Path(f).stem
        cid_to_gi[cid] = gi
        z = np.load(f)
        ends = z["ends"]
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        sizes_orig.append(len(ends))
        for j, e in enumerate(ends):
            Xc.append(ctxfeat(a[e - CTX:e].astype(int)))
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(5)])
            G.append(gi)
            order_arr.append(int(j))
            is_aug_arr.append(0)

    # BC 增强 cache
    bcaug_map = {Path(f).stem: f for f in bcaug_files}
    for cid, gi in cid_to_gi.items():
        if cid not in bcaug_map:
            sizes_aug.append(0)
            continue
        z = np.load(bcaug_map[cid])
        orig_ends = z["orig_end"]  # [N_aug]
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        sizes_aug.append(len(orig_ends))
        for k_aug, e in enumerate(orig_ends):
            Xc.append(ctxfeat(a[e - CTX:e].astype(int)))
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(5)])
            G.append(gi)
            order_arr.append(-1)  # 增强样本无原始窗序
            is_aug_arr.append(1)

    N_orig = sum(sizes_orig)
    N_aug = sum(sizes_aug)
    N = N_orig + N_aug
    gb_need = N * 2 * 80 * FDIM * 2 / 1024**3
    print(f"[head-bcaug] 原 {N_orig}窗 + 增强 {N_aug}窗 = {N}, "
          f"Xa fp16 = {gb_need:.1f} GB", file=sys.stderr)

    # 第二遍: 一次性分配 Xa, 逐通填 (原 + 增强 都按 conv 顺序)
    Xa = np.empty((N, 2, 80, FDIM), dtype=np.float16)
    pos = 0
    # 原 cache
    for f in orig_files:
        z = np.load(f)
        frames = z["frames"]
        Xa[pos:pos + len(frames)] = frames
        pos += len(frames)
        if pos % 50000 < len(frames) and pos > 50000:
            print(f"[head-bcaug] orig load pos={pos}/{N}", file=sys.stderr)
    # BC 增强 cache (按相同 conv 顺序)
    for cid, gi in cid_to_gi.items():
        if cid not in bcaug_map:
            continue
        z = np.load(bcaug_map[cid])
        frames_bcaug = z["frames_bcaug"]  # [N_aug, 2, 80, FDIM]
        if len(frames_bcaug) == 0:
            continue
        Xa[pos:pos + len(frames_bcaug)] = frames_bcaug
        pos += len(frames_bcaug)
    print(f"[head-bcaug] Xa 装载完成 pos={pos}/{N}", file=sys.stderr)

    return (np.array(Xc, np.float32), Xa,
            np.array(Y, np.float32), np.array(G), np.array(order_arr, np.int32),
            np.array(is_aug_arr, np.int8))


def load_test():
    """test 1000 段 × 1 窗 = 1000 窗. 不要 BC 增强 (test 集不动)."""
    Xc, Xa, ids = [], [], []
    for f in sorted(glob.glob(f"{CACHE}/test/*.npz")):
        cid = Path(f).stem
        z = np.load(f)
        frames = z["frames"]
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


def train_fold(Xc, Xa, Y, tr_idx, epochs, pw, batch_size=1024, seed=42):
    """seed 控制 model init + batch perm, 用于 multi-seed 训练."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = WhisperVAP(ctx_dim=Xc.shape[1], wd=Xa.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(tr_idx)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            cb = torch.from_numpy(np.asarray(Xc[batch_idx])).to(DEV, non_blocking=True)
            ab = torch.from_numpy(np.asarray(Xa[batch_idx])).to(DEV, non_blocking=True).float()
            yb = torch.from_numpy(np.asarray(Y[batch_idx])).to(DEV, non_blocking=True)
            opt.zero_grad()
            loss = crit(m(cb, ab), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"[head-bcaug]   ep{ep+1}/{epochs} loss={float(loss):.4f}", file=sys.stderr, flush=True)
    m.eval()
    return m


@torch.no_grad()
def predict_batched(model, Xc, Xa, idx, batch_size=512):
    n = len(idx)
    out = np.zeros((n, 5), dtype=np.float32)
    for i in range(0, n, batch_size):
        sub = idx[i:i + batch_size]
        cb = torch.from_numpy(np.asarray(Xc[sub])).to(DEV)
        ab = torch.from_numpy(np.asarray(Xa[sub])).to(DEV).float()
        out[i:i + batch_size] = torch.sigmoid(model(cb, ab)).cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=str, default="42",
                    help="逗号分隔 seed 列表, 如 '42,1,7' 训 3 seed × 5 fold = 15 ckpt")
    ap.add_argument("--save-fold-ckpt", action="store_true", help="保存每 fold ckpt + per-fold test probs")
    ap.add_argument("--run-dir", default="tools/runs/climb/head-bcaug")
    args = ap.parse_args()
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"[head-bcaug] seeds={seed_list} save_ckpt={args.save_fold_ckpt}", file=sys.stderr, flush=True)

    print(f"[head-bcaug] dev={DEV} CACHE={CACHE} CACHE_BCAUG={CACHE_BCAUG}", file=sys.stderr)
    Xc, Xa, Y, G, order, is_aug = load_train_with_bcaug()
    print(f"[head-bcaug] {len(Xc)}窗 ctx{Xc.shape[1]} aud{Xa.shape} "
          f"原始{int((is_aug==0).sum())} 增强{int(is_aug.sum())}", file=sys.stderr)

    pw = torch.tensor([(len(Y) - Y[:, k].sum()) / max(1, Y[:, k].sum())
                       for k in range(5)]).float().clamp(max=10).to(DEV)
    print(f"[head-bcaug] pos_weight: {[round(float(pw[k]), 2) for k in range(5)]}", file=sys.stderr)

    # 会话级 fold
    uniq = np.unique(G)
    rng = np.random.default_rng(SEED); rng.shuffle(uniq)
    fold_of = {g: i % args.folds for i, g in enumerate(uniq)}
    fold_id = np.array([fold_of[g] for g in G])

    oof = np.zeros((len(Xc), 5)); models = []
    fold_seed_meta = []  # [(seed, fi)] 对应每个 ckpt
    per_seed_oof = []  # 每 seed 一个独立 OOF (n_seeds × N × 5), 不平均防多样性丢失
    for seed in seed_list:
        print(f"\n[head-bcaug] === seed {seed} ===", file=sys.stderr, flush=True)
        oof_seed = np.zeros((len(Xc), 5))
        for fi in range(args.folds):
            va_mask = fold_id == fi
            tr_idx = np.where(~va_mask)[0]
            va_idx = np.where(va_mask & (is_aug == 0))[0]
            print(f"[head-bcaug] seed={seed} fold {fi+1}/{args.folds}: train={len(tr_idx)} val={len(va_idx)} "
                  f"(train 含 {int(is_aug[tr_idx].sum())} 增强)", file=sys.stderr, flush=True)
            m = train_fold(Xc, Xa, Y, tr_idx, args.epochs, pw, seed=seed)
            oof_seed[va_idx] = predict_batched(m, Xc, Xa, va_idx)
            models.append(m)
            fold_seed_meta.append((seed, fi))
            if args.save_fold_ckpt:
                torch.save(m.state_dict(), run / f"ckpt_seed{seed}_fold{fi}.pt")
        oof += oof_seed / len(seed_list)
        per_seed_oof.append(oof_seed.astype(np.float32))
    print(f"[head-bcaug] total {len(models)} ckpts ({len(seed_list)} seeds × {args.folds} folds)",
          file=sys.stderr, flush=True)
    # 保存 per-seed OOF (每 seed 独立的 OOF, 用于多 seed 软加权重搜索)
    np.savez_compressed(run / "per_seed_oof.npz",
                        per_seed=np.stack(per_seed_oof),  # (n_seeds, N, 5)
                        seeds=np.array(seed_list, dtype=np.int32))
    print(f"[head-bcaug] saved per_seed_oof.npz: {np.stack(per_seed_oof).shape}",
          file=sys.stderr, flush=True)

    # cap1 评估 (原始窗 order==0)
    cap1_mask = (order == 0) & (is_aug == 0)
    print(f"[head-bcaug] cap1 eval: {int(cap1_mask.sum())} 窗 (仅原始首窗)", file=sys.stderr)

    THR_PRESETS = {
        "cycle1": {0: 0.05, 4: 0.05, 1: 0.50, 3: 0.50, 2: 0.50},
        "varF":   {0: 0.05, 4: 0.25, 1: 0.50, 3: 0.65, 2: 0.75},
    }

    # cap1 自适应阈值 (sweep)
    thr_cap1, f1_cap1 = {}, {}
    for k in range(5):
        bf, bt = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(Y[cap1_mask, k], (oof[cap1_mask, k] >= t).astype(int), zero_division=0)
            if f > bf: bf, bt = f, float(t)
        thr_cap1[k], f1_cap1[k] = bt, bf
    macro_cap1 = float(np.mean(list(f1_cap1.values())))
    print(f"[head-bcaug] cap1自适应阈值 macro={macro_cap1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cap1[k]:.3f}@{thr_cap1[k]:.2f}" for k in range(5)), file=sys.stderr)
    print(f"[head-bcaug] ★BC={f1_cap1[2]:.3f} (cycle 16 frozen hubert head=0.000, 纯ctx基线 0.222)",
          file=sys.stderr)

    # 各阈值预设的 cap1 macro
    for name, thr in THR_PRESETS.items():
        f1s = []
        for k in range(5):
            f1s.append(f1_score(Y[cap1_mask, k], (oof[cap1_mask, k] >= thr[k]).astype(int), zero_division=0))
        macro = np.mean(f1s)
        print(f"[head-bcaug]   {name:<10s} macro={macro:.4f} | " +
              " ".join(f"{LAB[k]}={f1s[k]:.3f}@{thr[k]:.2f}" for k in range(5)), file=sys.stderr)

    # test 预测 — 每 ckpt 单独保存 per-fold test probs 用于 后续多 ckpt 软加融合
    Xtc, Xta, ids = load_test()
    test_idx = np.arange(len(ids))
    per_ckpt_test = np.zeros((len(models), len(ids), 5), dtype=np.float32)
    for ci, m in enumerate(models):
        per_ckpt_test[ci] = predict_batched(m, Xtc, Xta, test_idx)
    probs = per_ckpt_test.mean(axis=0)
    # 落盘 per-ckpt test (即使没 --save-fold-ckpt 也存, 体积小)
    np.savez_compressed(
        run / "per_ckpt_test.npz",
        per_ckpt=per_ckpt_test,
        meta=np.array(fold_seed_meta, dtype=np.int32),  # (n_ckpts, 2) = [(seed, fi), ...]
    )
    print(f"[head-bcaug] saved per_ckpt_test.npz: {per_ckpt_test.shape}", file=sys.stderr, flush=True)

    SUBMIT = ["c", "na", "i", "bc", "t"]; COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
    THR_PRESETS["cap1"] = thr_cap1

    csv_summary = {}
    for name, thr in THR_PRESETS.items():
        csv_path = run / f"pred_test1_{name}.csv"
        pos_counts = {c: 0 for c in SUBMIT}
        with open(csv_path, "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(ids):
                vals = {c: int(probs[i, COL2K[c]] >= thr[COL2K[c]]) for c in SUBMIT}
                for c, v in vals.items(): pos_counts[c] += v
                row = [sid] + [str(vals[c]) for c in SUBMIT]
                f.write(",".join(row) + "\n")
        csv_summary[name] = {"thresholds": {LAB[k]: round(thr[k], 2) for k in range(5)},
                              "pos": pos_counts}
        print(f"[head-bcaug] wrote {csv_path.name}: " +
              " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    # 默认 = varF (D-15 推荐阈值)
    import shutil as _sh
    _sh.copy(run / "pred_test1_varF.csv", run / "pred_test1.csv")
    print("[head-bcaug] 默认提交档 pred_test1.csv = pred_test1_varF.csv", file=sys.stderr)

    # 存 probs.npz (orthofuse 用)
    orig_mask = is_aug == 0
    np.savez_compressed(
        run / "probs.npz",
        oof=oof[orig_mask].astype(np.float32),
        test=probs.astype(np.float32),
        Y=Y[orig_mask].astype(np.int8),
        G=G[orig_mask].astype(np.int32),
        order=order[orig_mask].astype(np.int32),
        test_ids=np.array(ids),
    )
    print(f"[head-bcaug] saved probs.npz (oof[{orig_mask.sum()}]+test[{len(ids)}])",
          file=sys.stderr)

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "head-bcaug-frozen-encoder",
        "cap1_macro_f1": round(macro_cap1, 4),
        "per_sub_cap1": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)},
        "thresholds_cap1": {LAB[k]: round(thr_cap1[k], 2) for k in range(5)},
        "csv_variants": csv_summary,
        "n_orig": int(orig_mask.sum()),
        "n_aug": int(is_aug.sum()),
        "_note": "BC 增强 cache + 原 cache 合并训 head. val 仅原始. SOTA orthofuse cap1=0.6410.",
    }, ensure_ascii=False, indent=2))

    print(json.dumps({"cap1_score": round(macro_cap1, 4),
                      "per_sub": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)}}))


if __name__ == "__main__":
    main()
