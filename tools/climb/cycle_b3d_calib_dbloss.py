"""climb cycle B3d — calibration MLP on [ctx_lgbm_v1, whisper] probs with DB-Loss + SupCon.

设计 (B4 N1 改 chain-first pivot 后):
- 输入: 2 信号源对齐 (stride5 179867 OOF): ctx_lgbm_v1 (5d) + whisper (5d) = 10d
- 输出: 5d logits → sigmoid → 概率
- 损失: Distribution-Balanced BCE + α × Supervised Contrastive (BC/T 长尾)
- 评估: 严格 5fold groupKFold OOF (G=group_id 369 通) → 不踩 cap1 cherry-pick

成立条件 (D-13 push 门 + D-9 noise floor):
- 5fold OOF macro F1 ≥ 0.5701 + 0.005 = 0.5751 (vs ctx_lgbm_v1 OOF baseline)
- cap1 macro F1 ≥ 0.6228 + 0.005 = 0.6280 (vs ctx_lgbm_v1 cap1 baseline)
- 任一不达 → SKIP, 不浪费提交配额

Falsified 路径 (D-1~D-12 不复活):
- 不在 cap1 369 上选 strat / 调阈值 / 加 grid
- 不 BC 单类替换
- 不 LGBM 超参 sweep
- 使用 stride5 全集 + 严格 GroupKFold 避 cap1 cherry-pick

Usage:
  OMP_NUM_THREADS=4 python tools/climb/cycle_b3d_calib_dbloss.py [--alpha 0.3] [--epochs 30]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LAB = ['C', 'T', 'BC', 'I', 'NA']
NUM = 5
THR_VARF = [0.05, 0.5, 0.5, 0.5, 0.05]  # variant-F SOTA 阈值

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


def load_data():
    """加载 stride5 OOF + test."""
    stack = np.load(ROOT / 'tools/runs/climb/_stack_cache_s40.npz')
    whisper = np.load(ROOT / 'tools/runs/climb/whisper-fusion-20260531-0143/probs.npz')

    # 对齐验证: stack_cache Y 和 whisper Y 必须一致 (都来自 stride5 全集)
    Y = stack['Y']
    G = stack['G']
    assert Y.shape == whisper['Y'].shape, f"shape mismatch {Y.shape} vs {whisper['Y'].shape}"
    assert np.array_equal(Y, whisper['Y']), "Y 标签不一致 — 数据未对齐"

    # OOF 拼接: [ctx_lgbm_v1 (5d), whisper (5d)]
    X_oof = np.concatenate([stack['oof_lgbm_v1'], whisper['oof']], axis=1).astype(np.float32)
    X_te = np.concatenate([stack['te_lgbm_v1'], whisper['test']], axis=1).astype(np.float32)

    return X_oof, Y.astype(np.float32), G.astype(np.int64), X_te


class CalibMLP(nn.Module):
    """轻量校准头: 10d → 5d logits."""

    def __init__(self, in_dim: int = 10, hidden: int = 64, n_classes: int = 5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(hidden, n_classes)
        # SupCon projection head (训练用, 推理弃)
        self.contrast_proj = nn.Linear(hidden, 32)

    def forward(self, x):
        z = self.proj(x)
        return self.head(z), F.normalize(self.contrast_proj(z), dim=-1)


def distribution_balanced_loss(logits, targets, class_freq, neg_scale=2.0):
    """Distribution-Balanced BCE (Wu et al ECCV 2020 简化版).

    - re-balance weights: 给罕见类正样本更高权重 (1 / class_freq)
    - negative-tolerant regularization: 负样本 logit 缩 neg_scale → 缓解 over-suppression
    """
    # re-balance weight: log(1/freq), clip 防爆
    pos_weight = torch.log(1.0 / class_freq.clamp(min=1e-4))
    pos_weight = pos_weight.clamp(max=5.0).to(logits.device)

    # 负样本 logit scaled down (tolerate)
    logits_neg = logits / neg_scale
    logits_for_bce = torch.where(targets > 0.5, logits, logits_neg)

    bce = F.binary_cross_entropy_with_logits(
        logits_for_bce, targets, pos_weight=pos_weight, reduction='mean'
    )
    return bce


def supcon_loss(features, labels, temperature=0.1, target_class=2):
    """SupCon for one target class (BC=2 default) — 简化 single-class 版本.

    把 target_class 当 anchor: 同类样本互为正 pair, 其它为负.
    """
    target = labels[:, target_class]  # [B]
    if target.sum() < 2:  # 没足够正样本
        return torch.tensor(0.0, device=features.device)

    sim = features @ features.t() / temperature  # [B,B]
    mask = (target.unsqueeze(0) == target.unsqueeze(1)).float()  # 同类 1, 不同 0
    # 不对角 (自身)
    mask = mask - torch.eye(len(mask), device=mask.device)
    mask = mask.clamp(min=0)

    # log_sum_exp over neg + pos
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()  # numerical stability
    exp_sim = sim.exp() - torch.eye(len(sim), device=sim.device)  # exclude self
    log_prob = sim - exp_sim.sum(1, keepdim=True).log()

    pos_count = mask.sum(1)
    pos_count = pos_count.clamp(min=1)  # 防 0
    mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count
    loss = -mean_log_prob_pos.mean()
    return loss


def train_fold(X_tr, Y_tr, epochs=30, alpha=0.3, batch=2048):
    model = CalibMLP(in_dim=X_tr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 计算类频率 (用于 DB-Loss)
    class_freq = torch.tensor(Y_tr.mean(0), dtype=torch.float32)

    Xt = torch.from_numpy(X_tr)
    Yt = torch.from_numpy(Y_tr)
    n = len(Xt)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        nbatch = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx].to(DEV)
            yb = Yt[idx].to(DEV)

            opt.zero_grad()
            logits, feats = model(xb)
            l_db = distribution_balanced_loss(logits, yb, class_freq)
            # SupCon 在 BC=2 类做 (最长尾)
            l_sc = supcon_loss(feats, yb, target_class=2)
            loss = l_db + alpha * l_sc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss)
            nbatch += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{epochs} loss={ep_loss/nbatch:.4f}", file=sys.stderr)

    model.eval()
    return model


@torch.no_grad()
def predict(model, X, batch=4096):
    model.eval()
    Xt = torch.from_numpy(X)
    n = len(Xt)
    out = np.zeros((n, NUM), dtype=np.float32)
    for i in range(0, n, batch):
        xb = Xt[i:i + batch].to(DEV)
        logits, _ = model(xb)
        out[i:i + batch] = torch.sigmoid(logits).cpu().numpy()
    return out


def eval_macro(probs, Y, thr=THR_VARF):
    pred = np.stack([(probs[:, k] >= thr[k]).astype(int) for k in range(NUM)], axis=1)
    macro = f1_score(Y, pred, average='macro', zero_division=0)
    per = f1_score(Y, pred, average=None, zero_division=0)
    return macro, per


def eval_cap1(probs, Y, G, thr=THR_VARF):
    """cap1: 每通 1 切片 (取最后窗 = max order, 但这里没 order, 取每通中位)."""
    cap1_idx = []
    for g in np.unique(G):
        mask = G == g
        # cap1 用每通最后窗 (与 baseline orthofuse 对齐)
        idx_in_group = np.where(mask)[0]
        cap1_idx.append(idx_in_group[len(idx_in_group) // 2])  # 中位窗 (代理 cap1, 无 order 字段)
    cap1_idx = np.array(cap1_idx)
    return eval_macro(probs[cap1_idx], Y[cap1_idx], thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.3, help='SupCon 权重')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--seeds', type=int, default=1, help='多 seed 集成 (1=快, 5=稳)')
    args = ap.parse_args()

    print(f"[B3d] device={DEV}, alpha={args.alpha}, epochs={args.epochs}, seeds={args.seeds}", file=sys.stderr)
    X_oof, Y, G, X_te = load_data()
    print(f"[B3d] X_oof={X_oof.shape} Y={Y.shape} G={G.shape} unique groups={len(np.unique(G))}", file=sys.stderr)
    print(f"[B3d] class freq: {dict(zip(LAB, Y.mean(0)))}", file=sys.stderr)

    # baseline (ctx_lgbm_v1 only)
    stack = np.load(ROOT / 'tools/runs/climb/_stack_cache_s40.npz')
    base_macro, base_per = eval_macro(stack['oof_lgbm_v1'], Y)
    print(f"[B3d] BASELINE ctx_lgbm_v1 OOF macro={base_macro:.4f} per={[f'{x:.3f}' for x in base_per]}", file=sys.stderr)
    base_cap1, base_cap1_per = eval_cap1(stack['oof_lgbm_v1'], Y, G)
    print(f"[B3d] BASELINE ctx_lgbm_v1 cap1 macro={base_cap1:.4f} per={[f'{x:.3f}' for x in base_cap1_per]}", file=sys.stderr)

    # 5fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros_like(Y)
    test_preds = []
    for fold, (tr, va) in enumerate(gkf.split(X_oof, Y, groups=G)):
        print(f"\n[B3d] === fold {fold+1}/5 (tr={len(tr)} va={len(va)}) ===", file=sys.stderr)
        seed_preds_te = []
        for s in range(args.seeds):
            torch.manual_seed(42 + s + fold * 100)
            np.random.seed(42 + s + fold * 100)
            model = train_fold(X_oof[tr], Y[tr], epochs=args.epochs, alpha=args.alpha)
            oof_pred[va] += predict(model, X_oof[va]) / args.seeds
            seed_preds_te.append(predict(model, X_te))
        test_preds.append(np.mean(seed_preds_te, axis=0))

    # OOF 评估
    oof_macro, oof_per = eval_macro(oof_pred, Y)
    print(f"\n[B3d] === RESULT ===", file=sys.stderr)
    print(f"[B3d] OOF macro={oof_macro:.4f} (baseline {base_macro:.4f}, Δ={oof_macro-base_macro:+.4f})", file=sys.stderr)
    print(f"[B3d] OOF per: {dict(zip(LAB, [f'{x:.3f}' for x in oof_per]))}", file=sys.stderr)

    cap1_macro, cap1_per = eval_cap1(oof_pred, Y, G)
    print(f"[B3d] cap1 macro={cap1_macro:.4f} (baseline {base_cap1:.4f}, Δ={cap1_macro-base_cap1:+.4f})", file=sys.stderr)
    print(f"[B3d] cap1 per: {dict(zip(LAB, [f'{x:.3f}' for x in cap1_per]))}", file=sys.stderr)

    # 决策门 (D-13)
    oof_gate = oof_macro >= base_macro + 0.005
    cap1_gate = cap1_macro >= base_cap1 + 0.005
    push = oof_gate and cap1_gate
    print(f"\n[B3d] D-13 push 门: OOF gate (+0.005) {'✓' if oof_gate else '✗'}, cap1 gate (+0.005) {'✓' if cap1_gate else '✗'} → {'PUSH' if push else 'SKIP-advance'}", file=sys.stderr)

    # 保存产物
    out_dir = ROOT / f'tools/runs/climb/b3d-calib-{__import__("datetime").datetime.now().strftime("%Y%m%d-%H%M")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    test_mean = np.mean(test_preds, axis=0)
    np.savez(out_dir / 'probs.npz', oof=oof_pred, test=test_mean, Y=Y, G=G)

    import json
    metrics = {
        'cycle': 'B3d-calib-dbloss-supcon',
        'alpha': args.alpha,
        'epochs': args.epochs,
        'seeds': args.seeds,
        'oof_macro': float(oof_macro),
        'oof_per': {k: float(v) for k, v in zip(LAB, oof_per)},
        'cap1_macro': float(cap1_macro),
        'cap1_per': {k: float(v) for k, v in zip(LAB, cap1_per)},
        'baseline_oof_macro': float(base_macro),
        'baseline_cap1_macro': float(base_cap1),
        'oof_delta': float(oof_macro - base_macro),
        'cap1_delta': float(cap1_macro - base_cap1),
        'oof_gate_pass': oof_gate,
        'cap1_gate_pass': cap1_gate,
        'push_decision': 'PUSH' if push else 'SKIP',
    }
    (out_dir / 'cv_metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n[B3d] artifacts: {out_dir}/", file=sys.stderr)
    print(f"[B3d] decision: {metrics['push_decision']}", file=sys.stderr)

    return push


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
