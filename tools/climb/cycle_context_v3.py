"""climb cycle B1 — context-only LGBM v3 (46d→76d 加 30 个 EDA 强特征).

B1 EDA (2026-06-01-b1-eda-v3-features.json) 找到 30+ 个 info>0.15 特征:
- runlen_*_mean/max (per-class run-length stats, 同类连续 chunk 长度)
- burst_*_w{50,100,375} (per-class run density 簇集度)
- trans_{CC,CNA,TC,CT,...} (跨类转移概率, last 100 chunks)
- diff1_w{10..375} (1 阶差分 = chunk-to-chunk 变化率)
- diff2_w{10..375} (2 阶差分)

设计 (chain-first, 不踩 D-1~D-13 红旗):
- 不动 LGBM 超参 (D-12 已证 baseline 最优)
- 严格 GroupKFold 5fold (跟 B3d 一致), 不用 cycle_context 的 15% 单 split
- 评估指标: OOF macro F1 (variant-F 阈值 [0.05,0.5,0.5,0.5,0.05]) + cap1 macro (order=0 首窗, orthofuse 阈值 [0.05,0.5,0.75,0.65,0.25])
- baseline = ctx_lgbm_v1 (现 46d), 在 stack_cache_s40.npz 里有

Push 门 (D-13):
- OOF macro >= ctx_lgbm_v1 OOF baseline + 0.005
- AND cap1 macro >= ctx_lgbm_v1 cap1 baseline + 0.005
- 两 gate 通过 → 写 stack_cache_v3.npz, 进 orthofuse 重做

Usage:
  OMP_NUM_THREADS=4 python tools/climb/cycle_context_v3.py [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
LAB = ['C', 'T', 'BC', 'I', 'NA']
NUM = 5
CTX, TGT, STRIDE = 375, 25, 5
SEED = 42

# variant-F SOTA 训练阈值 (for OOF eval, 与 cycle_context.py 一致)
THR_VARF_OOF = [0.05, 0.5, 0.5, 0.5, 0.05]
# orthofuse cap1 评估阈值 (for cap1 eval)
THR_ORTH = [0.05, 0.5, 0.75, 0.65, 0.25]


def featurize_v1(ctx: np.ndarray) -> np.ndarray:
    """现 46d (复制 cycle_context.featurize)."""
    oh = np.eye(NUM)[ctx]
    feats = []
    for w in (10, 25, 50, 100, 200, 375):
        feats.extend(oh[-w:].mean(axis=0))
    for i in range(1, 6):
        feats.append(ctx[-i] if len(ctx) >= i else -1)
    L = len(ctx)
    for k in range(NUM):
        pos = np.where(ctx == k)[0]
        feats.append((L - 1 - pos[-1]) / L if len(pos) else 1.0)
    for k in range(NUM):
        feats.append((ctx == k).sum() / L)
    feats.append((ctx[1:] != ctx[:-1]).mean())
    return np.array(feats, dtype=np.float32)


def featurize_v3(ctx: np.ndarray) -> np.ndarray:
    """v1 46d + 30 个 EDA 强特征 = ~76d."""
    v1 = featurize_v1(ctx)
    extra = []
    L = len(ctx)
    oh = np.eye(NUM)[ctx]

    # 1. diff1 (1 阶差分, 6 窗): 6d
    diff = (ctx[1:] != ctx[:-1]).astype(np.float32) if L > 1 else np.zeros(1, dtype=np.float32)
    for w in (10, 25, 50, 100, 200, 375):
        extra.append(float(diff[-w:].mean()) if len(diff) >= w else float(diff.mean()))

    # 2. diff2 (2 阶差分, 6 窗): 6d
    diff2 = (diff[1:] != diff[:-1]).astype(np.float32) if len(diff) > 1 else np.zeros(1, dtype=np.float32)
    for w in (10, 25, 50, 100, 200, 375):
        extra.append(float(diff2[-w:].mean()) if len(diff2) >= w else float(diff2.mean()))

    # 3. burst (per-class density, 3 窗 × 5 类): 15d
    for w in (50, 100, 375):
        win = ctx[-w:] if L >= w else ctx
        win_len = max(1, len(win))
        for k in range(NUM):
            mask = (win == k).astype(int)
            if mask.sum() == 0:
                extra.append(0.0)
            else:
                diffs = np.diff(mask)
                runs = int((diffs == 1).sum()) + (1 if mask[0] == 1 else 0)
                density = float(mask.mean())
                extra.append(runs / max(1e-6, density * win_len))

    # 4. trans (跨类转移, last 100): 10d
    if L >= 100:
        win = ctx[-100:]
        trans = {}
        for i in range(len(win) - 1):
            key = (int(win[i]), int(win[i+1]))
            trans[key] = trans.get(key, 0) + 1
        total = max(1, sum(trans.values()))
        key_trans = [(1,2), (0,1), (0,4), (2,1), (3,0), (4,0), (1,0), (4,1), (2,0), (0,0)]
        for src, dst in key_trans:
            extra.append(trans.get((src, dst), 0) / total)
    else:
        extra.extend([0.0] * 10)

    # 5. runlen (per-class mean/max in last 100): 10d
    if L >= 100:
        win = ctx[-100:]
        for k in range(NUM):
            mask = (win == k).astype(int)
            runs = []
            cur = 0
            for v in mask:
                if v == 1:
                    cur += 1
                else:
                    if cur > 0: runs.append(cur)
                    cur = 0
            if cur > 0: runs.append(cur)
            extra.append(float(np.mean(runs)) if runs else 0.0)
            extra.append(float(max(runs)) if runs else 0.0)
    else:
        extra.extend([0.0] * 10)

    extra_arr = np.array(extra, dtype=np.float32)
    return np.concatenate([v1, extra_arr])


def build_data(conv_files, label_files, version='v3'):
    """全 stride5 OOF 数据 (对齐 stack_cache_s40 = stride5 179867)."""
    feat_fn = featurize_v3 if version == 'v3' else featurize_v1
    X, Y, G, order_list = [], [], [], []
    for gi, cid in enumerate(conv_files):
        a = np.load(label_files[cid])
        order_in_conv = 0
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE):
            X.append(feat_fn(a[e - CTX:e].astype(int)))
            fut = set(int(x) for x in a[e:e + TGT])
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
            order_list.append(order_in_conv)
            order_in_conv += 1
        if (gi + 1) % 50 == 0:
            print(f"[B1-v3] conv {gi+1}/{len(conv_files)} done, rows so far={len(X)}", file=sys.stderr)
    return np.array(X), np.array(Y, dtype=np.int32), np.array(G, dtype=np.int32), np.array(order_list, dtype=np.int32)


def fit_lgbm(X, y, spw):
    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED,
    )
    clf.fit(X, y)
    return clf


def eval_macro(probs, Y, thr):
    pred = np.stack([(probs[:, k] >= thr[k]).astype(int) for k in range(NUM)], axis=1)
    per = [f1_score(Y[:, k], pred[:, k], zero_division=0) for k in range(NUM)]
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='只 30 通通 smoke')
    args = ap.parse_args()

    # 加载所有 train label files
    all_files = sorted(glob.glob(str(ROOT / 'data/train/labels/*.npy')))
    label_files = {Path(f).stem: f for f in all_files}
    conv_ids = sorted(label_files.keys())
    if args.smoke:
        conv_ids = conv_ids[:30]
        print(f"[B1-v3] SMOKE mode: {len(conv_ids)} convs", file=sys.stderr)

    print(f"[B1-v3] 构造 v3 特征 ({len(conv_ids)} convs)...", file=sys.stderr)
    X, Y, G, order = build_data(conv_ids, label_files, version='v3')
    print(f"[B1-v3] X={X.shape} Y={Y.shape} G={G.shape}, unique groups={len(np.unique(G))}", file=sys.stderr)

    # baseline v1 (现 46d)
    print(f"[B1-v3] 构造 v1 特征对比 baseline...", file=sys.stderr)
    Xv1, _, _, _ = build_data(conv_ids, label_files, version='v1')
    print(f"[B1-v3] X_v1={Xv1.shape}", file=sys.stderr)

    # 5fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof_v3 = np.zeros_like(Y, dtype=np.float32)
    oof_v1 = np.zeros_like(Y, dtype=np.float32)

    for fold, (tr, va) in enumerate(gkf.split(X, Y, groups=G)):
        print(f"\n[B1-v3] === fold {fold+1}/5 (tr={len(tr)} va={len(va)}) ===", file=sys.stderr)
        for k in range(NUM):
            spw = (len(tr) - Y[tr, k].sum()) / max(1, Y[tr, k].sum())
            # v3
            clf = fit_lgbm(X[tr], Y[tr, k], spw)
            oof_v3[va, k] = clf.predict_proba(X[va])[:, 1]
            # v1
            clf = fit_lgbm(Xv1[tr], Y[tr, k], spw)
            oof_v1[va, k] = clf.predict_proba(Xv1[va])[:, 1]
            print(f"  k={k} {LAB[k]}: v3 fold mean prob={oof_v3[va,k].mean():.3f}, v1={oof_v1[va,k].mean():.3f}", file=sys.stderr)

    # 评估
    v3_oof_macro, v3_oof_per = eval_macro(oof_v3, Y, THR_VARF_OOF)
    v1_oof_macro, v1_oof_per = eval_macro(oof_v1, Y, THR_VARF_OOF)

    # cap1 (order=0)
    cap1_idx = np.where(order == 0)[0]
    v3_cap1_macro, v3_cap1_per = eval_macro(oof_v3[cap1_idx], Y[cap1_idx], THR_ORTH)
    v1_cap1_macro, v1_cap1_per = eval_macro(oof_v1[cap1_idx], Y[cap1_idx], THR_ORTH)

    print(f"\n[B1-v3] === RESULT ===", file=sys.stderr)
    print(f"[B1-v3] OOF macro: v1={v1_oof_macro:.4f}, v3={v3_oof_macro:.4f} (Δ={v3_oof_macro-v1_oof_macro:+.4f})", file=sys.stderr)
    print(f"[B1-v3] OOF per v3: {dict(zip(LAB, [f'{x:.3f}' for x in v3_oof_per]))}", file=sys.stderr)
    print(f"[B1-v3] OOF per v1: {dict(zip(LAB, [f'{x:.3f}' for x in v1_oof_per]))}", file=sys.stderr)
    print(f"[B1-v3] cap1 macro: v1={v1_cap1_macro:.4f}, v3={v3_cap1_macro:.4f} (Δ={v3_cap1_macro-v1_cap1_macro:+.4f})", file=sys.stderr)
    print(f"[B1-v3] cap1 per v3: {dict(zip(LAB, [f'{x:.3f}' for x in v3_cap1_per]))}", file=sys.stderr)
    print(f"[B1-v3] cap1 per v1: {dict(zip(LAB, [f'{x:.3f}' for x in v1_cap1_per]))}", file=sys.stderr)

    # 决策门 (D-13)
    oof_gate = (v3_oof_macro - v1_oof_macro) >= 0.005
    cap1_gate = (v3_cap1_macro - v1_cap1_macro) >= 0.005
    push = oof_gate and cap1_gate
    print(f"\n[B1-v3] D-13 push 门: OOF +0.005 {'✓' if oof_gate else '✗'}, cap1 +0.005 {'✓' if cap1_gate else '✗'} → {'PUSH' if push else 'SKIP-advance'}", file=sys.stderr)

    # 保存
    out_dir = ROOT / f'tools/runs/climb/ctx-v3-{datetime.now().strftime("%Y%m%d-%H%M")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'oof.npz', oof_v3=oof_v3, oof_v1=oof_v1, Y=Y, G=G, order=order, X_v3=X, X_v1=Xv1)
    (out_dir / 'cv_metrics.json').write_text(json.dumps({
        'cycle': 'B1-context-v3',
        'v1_oof_macro': v1_oof_macro, 'v3_oof_macro': v3_oof_macro,
        'v1_cap1_macro': v1_cap1_macro, 'v3_cap1_macro': v3_cap1_macro,
        'v3_oof_per': dict(zip(LAB, [float(x) for x in v3_oof_per])),
        'v3_cap1_per': dict(zip(LAB, [float(x) for x in v3_cap1_per])),
        'oof_delta': v3_oof_macro - v1_oof_macro,
        'cap1_delta': v3_cap1_macro - v1_cap1_macro,
        'oof_gate_pass': oof_gate, 'cap1_gate_pass': cap1_gate,
        'push_decision': 'PUSH' if push else 'SKIP',
        'smoke': args.smoke,
    }, indent=2, ensure_ascii=False))
    print(f"[B1-v3] artifacts: {out_dir}/", file=sys.stderr)
    return push


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
