"""VAP-BC vs LGBM-BC 互补性分析 (本地, 不烧云不占配额).

数据:
  - VAP: test_probs.npz [1000,5] 概率, col order [C,T,BC,I,NA]
  - LGBM SOTA: variant-F / cycle1 的 pred_test1.csv 硬 0/1 (col: segment_id,c,na,i,bc,t)
分析 (test 集, 无真值也能看分歧):
  1. VAP-BC 概率 vs LGBM-BC 硬预测的一致性 (VAP 在 LGBM=1 vs LGBM=0 上的概率分布)
  2. BC 正例集合重叠 (Jaccard) — 重叠高=同样本=无互补; 低=分歧=可能互补
  3. 若用不同阈值让 VAP 也出 ~30-44 正例, 与 LGBM 正例的 overlap
  4. 简单融合模拟: VAP-BC 概率替换/平均进 LGBM, 看正例集合怎么变
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VAP_NPZ = ROOT / "tools/runs/climb/vap-full/test_probs.npz"
LGBM_CSV = ROOT / "tools/runs/climb/variant-F-20260528-0559/pred_test1.csv"

# --- load VAP probs (col order C,T,BC,I,NA = idx 0,1,2,3,4) ---
d = np.load(VAP_NPZ, allow_pickle=True)
vap_p = d["probs"]            # [1000,5]
vap_ids = [str(x) for x in d["ids"]]
vap_bc = vap_p[:, 2]          # BC column
id2vapbc = {i: float(p) for i, p in zip(vap_ids, vap_bc)}

# --- load LGBM hard preds ---
rows = list(csv.DictReader(open(LGBM_CSV)))
lgbm_bc = {r["segment_id"]: int(r["bc"]) for r in rows}

# align
ids = [i for i in vap_ids if i in lgbm_bc]
assert len(ids) == 1000, f"only {len(ids)} aligned"
v = np.array([id2vapbc[i] for i in ids])           # VAP BC prob
l = np.array([lgbm_bc[i] for i in ids])            # LGBM BC hard 0/1

print(f"=== 对齐 {len(ids)} 段 ===")
print(f"LGBM BC 正例数: {int(l.sum())}")
print(f"VAP BC 概率: mean={v.mean():.3f} | >0.5: {int((v>=0.5).sum())} | >0.55(VAP thr): {int((v>=0.55).sum())}")

# 1. VAP 概率在 LGBM 分组上的分布
print("\n=== 1. VAP-BC 概率 ~ LGBM-BC 硬预测 ===")
print(f"  LGBM=1 ({int(l.sum())}段): VAP-BC prob mean={v[l==1].mean():.3f} median={np.median(v[l==1]):.3f}")
print(f"  LGBM=0 ({int((l==0).sum())}段): VAP-BC prob mean={v[l==0].mean():.3f} median={np.median(v[l==0]):.3f}")
print("  → 若 LGBM=1 组 VAP 概率明显更高 = 两者一致(无互补); 差不多 = VAP 没复现 LGBM 的判断")

# point-biserial correlation (VAP continuous vs LGBM binary)
from scipy.stats import pointbiserialr
r, pval = pointbiserialr(l, v)
print(f"  point-biserial corr(LGBM_BC, VAP_BC_prob) = {r:.3f} (p={pval:.2e})")
print("  → |r| 高=强相关(无互补); 接近0=独立(可能互补但也可能是噪声)")

# 2. 让 VAP 在 top-K 出正例(匹配 LGBM 正例数), 比 overlap
print("\n=== 2. BC 正例集合重叠 (VAP top-K vs LGBM) ===")
K = int(l.sum())  # 30
vap_topk_idx = set(np.argsort(-v)[:K])
lgbm_pos_idx = set(np.where(l == 1)[0])
inter = vap_topk_idx & lgbm_pos_idx
union = vap_topk_idx | lgbm_pos_idx
print(f"  VAP top-{K} BC 正例 ∩ LGBM {K} 正例 = {len(inter)} 重叠")
print(f"  Jaccard = {len(inter)/len(union):.3f}")
print(f"  VAP 独有(LGBM没标): {len(vap_topk_idx - lgbm_pos_idx)} | LGBM独有: {len(lgbm_pos_idx - vap_topk_idx)}")
print("  → 重叠高(Jaccard>0.5)=抓同样本无互补; 低=分歧(互补需OOF真值验证才知谁对)")

# 3. VAP 在 0.55 阈值(它自己cap1调的)的正例 vs LGBM
print("\n=== 3. VAP@0.55(自身阈值) 正例 vs LGBM ===")
vap_pos_055 = set(np.where(v >= 0.55)[0])
print(f"  VAP@0.55 正例数={len(vap_pos_055)} | ∩ LGBM={len(vap_pos_055 & lgbm_pos_idx)} | VAP独有={len(vap_pos_055 - lgbm_pos_idx)}")

# 4. 融合模拟(概率层面无法直接做因LGBM无概率, 报告硬层面 union/intersection 大小)
print("\n=== 4. 集合运算 (硬层面, 无真值只看规模) ===")
print(f"  union(VAP@0.55 ∪ LGBM) = {len(vap_pos_055 | lgbm_pos_idx)} (融合若取或, 召回↑precision可能↓)")
print(f"  intersect = {len(vap_pos_055 & lgbm_pos_idx)} (融合若取与, precision↑召回↓)")

print("\n=== 结论框架 ===")
print("  - |corr| > 0.3 且 Jaccard > 0.4 → VAP 与 LGBM 抓同样本 = 无互补, 融合无增量 → 判死音频路线")
print("  - |corr| < 0.15 且 Jaccard < 0.25 → 高度分歧 = 可能互补, 但需云端 OOF 真值验证谁对(VAP抓对LGBM错的吗)")
print("  - 注意: test 无真值, 分歧≠互补(可能只是VAP在乱猜). 强结论需OOF.")
