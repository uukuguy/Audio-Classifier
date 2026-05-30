"""BC 诊断: 当前 context-only LGBM 的 BC 是 recall 还是 precision 拖后腿?
+ 探查 BC 在 context 标签序列里的可预测结构(自相关/最近BC距离 vs 未来BC)。
本机秒级, 用真值 OOF (会话级 5-fold)。
"""
import glob
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, "tools/climb")
from cycle_context import featurize
from sklearn.metrics import precision_recall_fscore_support, f1_score
from lightgbm import LGBMClassifier

CTX, TGT, NUM, STRIDE, SEED = 375, 25, 5, 40, 42
BC = 2

conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
print(f"{len(conv_ids)} convs", file=sys.stderr)

# build with conv grouping
X, Y, G = [], [], []
extra = []  # BC专属探查: (最近BC距离/L, 过去BC频率, 过去2s内有无BC)
for gi, cid in enumerate(conv_ids):
    a = np.load(f"data/train/labels/{cid}.npy").astype(int)
    for e in range(CTX, len(a) - TGT + 1, STRIDE):
        ctx = a[e - CTX:e]
        fut = set(int(x) for x in a[e:e + TGT])
        X.append(featurize(ctx))
        Y.append([1 if k in fut else 0 for k in range(NUM)])
        G.append(gi)
        bc_pos = np.where(ctx == BC)[0]
        last_bc = (len(ctx) - 1 - bc_pos[-1]) if len(bc_pos) else len(ctx)
        recent_bc = 1 if (BC in set(ctx[-25:])) else 0  # 过去2s有BC
        extra.append([last_bc / len(ctx), len(bc_pos) / len(ctx), recent_bc])
X, Y, G, extra = np.array(X), np.array(Y), np.array(G), np.array(extra)
print(f"samples={len(X)} BC正例率={Y[:,BC].mean():.3f}", file=sys.stderr)

# 5-fold conv-grouped OOF for BC
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(conv_ids))
oof = np.zeros(len(X))
for fi in range(5):
    val_g = {perm[i] for i in range(len(conv_ids)) if i % 5 == fi}
    tr = [i for i in range(len(X)) if G[i] not in val_g]
    va = [i for i in range(len(X)) if G[i] in val_g]
    y = Y[tr, BC]
    spw = (len(y) - y.sum()) / max(1, y.sum())
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
    clf.fit(X[tr], y)
    oof[va] = clf.predict_proba(X[va])[:, 1]

yt = Y[:, BC]
print(f"\n=== BC OOF 诊断 (会话级5fold, {len(X)}样本, BC正例率{yt.mean():.3f}) ===")
print(f"{'thr':>5}{'P':>8}{'R':>8}{'F1':>8}{'pred_pos':>10}")
for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
    pred = (oof >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(yt, pred, average='binary', zero_division=0)
    print(f"{t:>5.2f}{p:>8.3f}{r:>8.3f}{f:>8.3f}{int(pred.sum()):>10}")

# 最优阈值
best_t, best_f = 0.5, -1
for t in np.linspace(0.05, 0.95, 37):
    f = f1_score(yt, (oof >= t).astype(int), zero_division=0)
    if f > best_f:
        best_f, best_t = f, t
pred = (oof >= best_t).astype(int)
p, r, f, _ = precision_recall_fscore_support(yt, pred, average='binary', zero_division=0)
print(f"\n最优 thr={best_t:.2f}: P={p:.3f} R={r:.3f} F1={f:.3f}")
print(f"→ {'RECALL拖后腿(漏报多,需更敏感特征)' if r < p else 'PRECISION拖后腿(误报多,需更准判别)'}")

# BC 时序结构探查
print(f"\n=== BC 时序可预测性探查 ===")
from scipy.stats import pointbiserialr
for j, nm in enumerate(["最近BC距离/L", "过去BC频率", "过去2s有BC"]):
    rr, _ = pointbiserialr(yt, extra[:, j])
    m1, m0 = extra[yt==1, j].mean(), extra[yt==0, j].mean()
    print(f"  {nm:<14} BC=1:{m1:.3f} BC=0:{m0:.3f} r={rr:+.3f}")
print("  → r 高的特征 = BC 在 context 序列里的可预测结构, 当前若没充分利用=改进空间")
