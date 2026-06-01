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
    """两遍扫描 + 预分配单一 ndarray, 避免 `np.array(列表)` 触发临时拷贝.

    根因: 之前 `np.array(Xa列表, np.float16)` 会先把列表里每个 ndarray 累计, 内部
    需要临时空间, 触发 64G+ 临时分配 → page cache 风暴 → IO 死锁.

    修法: 第一遍扫描算出总 N, 第二遍预分配 [N,2,80,1280] fp16 数组 (单次 64G 直接 mmap
    style allocation, 不需临时拷贝), 然后逐通往里填. 1TB 内存稳容.
    """
    files = sorted(glob.glob(f"{CACHE}/train/*.npz"))
    _probe = np.load(files[0])["frames"]; FDIM = _probe.shape[-1]  # 动态探测(whisper1280/hubert1024)
    # 第一遍: 累计 N + 读 ctx/Y/G/order (这些小, 全 RAM ~MB 量级)
    Xc, Y, G, order = [], [], [], []
    sizes = []
    print(f"[head] 第一遍扫描 {len(files)} 通...", file=sys.stderr)
    for gi, f in enumerate(files):
        cid = Path(f).stem
        z = np.load(f)
        ends = z["ends"]
        a = np.load(f"data/train/labels/{cid}.npy")
        sizes.append(len(ends))
        for j, e in enumerate(ends):
            Xc.append(ctxfeat(a[e - CTX:e].astype(int)))
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(5)])
            G.append(gi)
            order.append(j)
    N = sum(sizes)
    gb_need = N * 2 * 80 * FDIM * 2 / 1024**3
    print(f"[head] 共 {N} 窗, 预分配 Xa [{N},2,80,{FDIM}] fp16 = {gb_need:.1f} GB", file=sys.stderr)

    # 第二遍: 一次性预分配 + 逐通填. 不走 append + np.array() 路径 (后者会触发临时拷贝)
    Xa = np.empty((N, 2, 80, FDIM), dtype=np.float16)
    pos = 0
    for gi, f in enumerate(files):
        z = np.load(f)
        frames = z["frames"]  # [W,2,80,1280] fp16
        Xa[pos:pos + len(frames)] = frames
        pos += len(frames)
        if (gi + 1) % 50 == 0:
            print(f"[head] load {gi+1}/{len(files)} (pos={pos}/{N})", file=sys.stderr)
    print(f"[head] Xa 装载完成, 内存常驻 {gb_need:.1f} GB", file=sys.stderr)

    return (np.array(Xc, np.float32), Xa,
            np.array(Y, np.float32), np.array(G), np.array(order, np.int32))


def load_test():
    """test 1000 段 × 1 窗 = 1000 窗, 100 MB 量级, 不必 memmap."""
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


def train_fold(Xc, Xa, Y, tr_idx, epochs, pw, batch_size=256):
    """Xa 是 numpy memmap (磁盘 backed), batch fancy index 只拷贝 batch_size 行到 RAM 再到 GPU.
    RAM 峰值 = batch_size × 单窗大小 = 256 × 400KB = 100MB. 训完释放, 远不会撑爆内存."""
    m = WhisperVAP(ctx_dim=Xc.shape[1], wd=Xa.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(tr_idx)
    # tr_idx 是绝对索引到 Xa/Xc 全集; 每 epoch 内 shuffle 这些索引
    for ep in range(epochs):
        m.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i + batch_size]
            # memmap fancy index → 实际 batch 拷到 RAM (100MB) → 上 GPU 转 fp32
            cb = torch.from_numpy(np.asarray(Xc[batch_idx])).to(DEV, non_blocking=True)
            ab = torch.from_numpy(np.asarray(Xa[batch_idx])).to(DEV, non_blocking=True).float()
            yb = torch.from_numpy(np.asarray(Y[batch_idx])).to(DEV, non_blocking=True)
            opt.zero_grad()
            loss = crit(m(cb, ab), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"[head]   epoch {ep+1}/{epochs} loss={float(loss):.4f}", file=sys.stderr)
    m.eval()
    return m


@torch.no_grad()
def predict_batched(model, Xc, Xa, idx, batch_size=512):
    """OOF / test 推理: idx 是 Xa/Xc 中的绝对索引列表 (memmap 友好)."""
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
        oof[va] = predict_batched(m, Xc, Xa, va)  # OOF: 传 va 索引,内部 memmap fancy index
        models.append(m)
        print(f"[head] fold {fi + 1}/{args.folds}", file=sys.stderr)

    # cap1 切片 CV: 每通取片段序号 0 的窗(模拟 test 独立片段)
    cap1 = order == 0
    thr_cap1, f1_cap1 = {}, {}
    for k in range(5):
        bf, bt = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f = f1_score(Y[cap1, k], (oof[cap1, k] >= t).astype(int), zero_division=0)
            if f > bf: bf, bt = f, float(t)
        thr_cap1[k], f1_cap1[k] = bt, bf
    macro_cap1 = float(np.mean(list(f1_cap1.values())))
    print(f"[head] cap1切片CV macro={macro_cap1:.4f} | " +
          " ".join(f"{LAB[k]}={f1_cap1[k]:.3f}@{thr_cap1[k]:.2f}" for k in range(5)), file=sys.stderr)
    print(f"[head] ★BC={f1_cap1[2]:.3f} (纯ctx基线0.222)", file=sys.stderr)

    # 阈值铁律 + 冒烟教训: cap1 调出来的阈值在 whisper paradigm 上线上反挫 (smoke 0.641 CV → 0.634 LB).
    # 对照基线 (variant-F SOTA 0.7124 使用) 给 3 套候选, 全量出 3 份 CSV 让用户比较选最稳一份提交.
    # variant-F SOTA 阈值快照 (来自 cycle1, 接近 0.5 钙化): C=0.05, NA=0.05, T=0.50, I=0.50, BC=0.50
    THR_PRESETS = {
        "cap1":      thr_cap1,                              # cap1 切片 CV 自适应 (冒烟用此, 信号假阳)
        "cycle1":    {0: 0.05, 4: 0.05, 1: 0.50, 3: 0.50, 2: 0.50},  # variant-F SOTA 阈值钙化
        "balanced":  {0: 0.05, 4: 0.05, 1: 0.50, 3: 0.50, 2: 0.40},  # BC 稍激进, 其他 cycle1
    }

    # 也用 cap1 OOF 算 cycle1/balanced 这两套预设的 cap1 macro (校准参考)
    print("[head] === cap1 OOF 上各阈值预设的 macro F1 (校准参考, 非线上预测) ===", file=sys.stderr)
    for name, thr in THR_PRESETS.items():
        f1s = []
        for k in range(5):
            f1s.append(f1_score(Y[cap1, k], (oof[cap1, k] >= thr[k]).astype(int), zero_division=0))
        print(f"[head]   {name:<10s} macro={np.mean(f1s):.4f} | " +
              " ".join(f"{LAB[k]}={f1s[k]:.3f}@{thr[k]:.2f}" for k in range(5)), file=sys.stderr)

    # test 预测(5 fold 模型 prob 平均, 分批不 OOM)
    Xtc, Xta, ids = load_test()
    test_idx = np.arange(len(ids))
    probs = np.zeros((len(ids), 5), dtype=np.float32)
    for m in models:
        probs += predict_batched(m, Xtc, Xta, test_idx)  # 内部 batch_size=512
    probs /= len(models)

    SUBMIT = ["c", "na", "i", "bc", "t"]; COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}

    # 出 3 份 CSV: pred_test1_cap1.csv / pred_test1_cycle1.csv / pred_test1_balanced.csv
    # 同时把 cycle1 命名为 pred_test1.csv (默认提交档)
    csv_summary = {}
    for name, thr in THR_PRESETS.items():
        csv_path = run / f"pred_test1_{name}.csv"
        pos_counts = {c: 0 for c in SUBMIT}
        with open(csv_path, "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(ids):
                vals = {c: int(probs[i, COL2K[c]] >= thr[COL2K[c]]) for c in SUBMIT}
                for c, v in vals.items():
                    pos_counts[c] += v
                row = [sid] + [str(vals[c]) for c in SUBMIT]
                f.write(",".join(row) + "\n")
        csv_summary[name] = {
            "path": str(csv_path.relative_to(run.parent.parent.parent) if csv_path.is_relative_to(Path.cwd()) else csv_path),
            "thresholds": {LAB[k]: round(thr[k], 2) for k in range(5)},
            "pos_pct": {c: round(pos_counts[c] / len(ids) * 100, 1) for c in SUBMIT},
        }
        print(f"[head] wrote {csv_path.name}: " + " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT), file=sys.stderr)

    # 默认提交档 = cycle1 (保守, 阈值铁律)
    import shutil as _sh
    _sh.copy(run / "pred_test1_cycle1.csv", run / "pred_test1.csv")
    print("[head] 默认提交档 pred_test1.csv = pred_test1_cycle1.csv (保守, 阈值铁律最稳)", file=sys.stderr)

    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "cloud-whisper-large-v3-vap",
        "cap1_macro_f1": round(macro_cap1, 4),
        "per_sub_cap1": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)},
        "thresholds_cap1": {LAB[k]: round(thr_cap1[k], 2) for k in range(5)},
        "csv_variants": csv_summary,
        "_note": "出 3 份 CSV. 默认 pred_test1.csv = cycle1 阈值 (保守). cap1 阈值在冒烟阶段已证伪 (0.641 CV → 0.634 LB).",
    }, ensure_ascii=False, indent=2))
    print(f"[head] cv_metrics.json saved", file=sys.stderr)

    # ★存连续概率 npz (产物保存铁律 + 跨源融合必需): OOF[N,5] + test[M,5] + 对齐元数据.
    # 后续本机 per-class 正交融合(借 whisper T/I + context C/NA/BC)+ nested 验证读这个.
    np.savez_compressed(
        run / "probs.npz",
        oof=oof.astype(np.float32),          # [N,5] cap1 OOF 连续概率 (会话级 fold, 无泄漏)
        test=probs.astype(np.float32),       # [M,5] test 连续概率 (5fold 平均)
        Y=Y.astype(np.int8),                 # [N,5] 标签 (对齐 oof)
        G=G.astype(np.int32),                # [N] 会话 id (cap1/fold 划分)
        order=order.astype(np.int32),        # [N] 通内片段序号 (order==0 = cap1)
        test_ids=np.array(ids),              # [M] test segment id (对齐 test)
    )
    print(f"[head] ★probs.npz saved (oof{oof.shape}+test{probs.shape}) — 融合用连续概率", file=sys.stderr)
    print(json.dumps({"cap1_score": round(macro_cap1, 4),
                      "per_sub": {LAB[k]: round(f1_cap1[k], 4) for k in range(5)},
                      "csv_count": len(THR_PRESETS)}))


if __name__ == "__main__":
    main()
