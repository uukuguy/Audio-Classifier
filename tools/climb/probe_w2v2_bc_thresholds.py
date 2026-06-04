"""探测 w2v2_bcaug BC 在中等阈值下的 cap1 macro (不烧配额, 本机 30s).

用户问: w2v2_bcaug BC OOF F1=0.261 项目最高, 单独打 BC 行不行?
D-18 教训: thr=0.10 真分 -0.048 (cherry-pick). varF thr=0.75 砍到 0 (没用).

本脚本扫 BC 阈值 {0.30, 0.40, 0.50, varF=0.75 baseline}, cap1 上看 macro
+ pos count, 决定能不能进 orthofuse 4src.

复用现有 OOF (不重训), 30s 跑完.

Usage:
  OMP_NUM_THREADS=4 python tools/climb/probe_w2v2_bc_thresholds.py
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}  # 项目钙化阈值

# SOTA orthofuse-3src 用的 3 源 probs 路径
PATHS = {
    "ctx_cache": "tools/runs/climb/_stack_cache_s40.npz",  # ctx oof 基座
    "whisper": "tools/runs/climb/whisper-fusion-20260531-0143/probs.npz",
    "hubert": "tools/runs/climb/hubert-bcaug-head-20260601-1533/probs.npz",
    "w2v2": "tools/runs/climb/w2v2-bcaug-head-20260601-1926/probs.npz",
}


def cap1_idx(G):
    """每通取首窗作为 cap1 评估集 = 369 通."""
    seen, cap1 = set(), []
    for i, g in enumerate(G):
        if int(g) not in seen:
            cap1.append(i); seen.add(int(g))
    return np.array(cap1)


def main():
    # 1. 加载 ctx 基座 (lgbm_v1 = cycle1 SOTA cap1=0.6228)
    zc = np.load(PATHS["ctx_cache"])
    Yc = zc["Y"]
    Gc = zc["G"]
    cap1 = cap1_idx(Gc)
    ctx_oof = zc["oof_lgbm_v1"][cap1]   # [369, 5]
    y_cap1 = Yc[cap1].astype(int)
    print(f"[probe] ctx cap1: {ctx_oof.shape}, y: {y_cap1.shape}")

    # 2. 加载 whisper + hubert + w2v2 (确认 align)
    def load_cap1(path, name):
        z = np.load(path)
        oof = z["oof"]; Y = z["Y"]; order = z["order"]
        mask = order == 0
        oof_c = oof[mask]
        y_c = Y[mask].astype(int)
        align = (y_c == y_cap1).all()
        print(f"[probe] {name}: cap1 {oof_c.shape}, align={align}")
        return oof_c

    wsp_oof = load_cap1(PATHS["whisper"], "whisper")
    hub_oof = load_cap1(PATHS["hubert"], "hubert")
    w2v_oof = load_cap1(PATHS["w2v2"], "w2v2")

    # 3. SOTA orthofuse-3src 策略 (从 cv_metrics.json):
    #    C=ctx  T=whisper_hubert_70(0.7w+0.3h)  BC=ctx  I=ctx_whisper_hubert_eq  NA=ctx
    def make_sota_probs():
        p = np.zeros_like(ctx_oof)
        p[:, 0] = ctx_oof[:, 0]                              # C
        p[:, 1] = 0.7 * wsp_oof[:, 1] + 0.3 * hub_oof[:, 1]  # T
        p[:, 2] = ctx_oof[:, 2]                              # BC (SOTA 用 ctx)
        p[:, 3] = (ctx_oof[:, 3] + wsp_oof[:, 3] + hub_oof[:, 3]) / 3  # I
        p[:, 4] = ctx_oof[:, 4]                              # NA
        return p

    # 4. 对比方案 = SOTA BC 替换为 w2v2@thr
    def eval_f1(probs, thresholds, label=""):
        f1s = {}
        pos_counts = {}
        for k in range(5):
            pred = (probs[:, k] >= thresholds[k]).astype(int)
            f1s[k] = f1_score(y_cap1[:, k], pred, zero_division=0)
            pos_counts[k] = int(pred.sum())
        macro = np.mean(list(f1s.values()))
        per_str = " ".join(f"{LAB[k]}={f1s[k]:.3f}(p={pos_counts[k]})" for k in range(5))
        print(f"  {label}: macro={macro:.4f} | {per_str}")
        return macro

    print("\n=== Baseline ===")
    sota_probs = make_sota_probs()
    sota_macro = eval_f1(sota_probs, THR_VARF, "SOTA orthofuse-3src (BC=ctx@0.75)")

    # 5. 替换 BC 列用 w2v2, 扫不同阈值
    print("\n=== BC 替换为 w2v2_bcaug, 扫阈值 ===")
    print(f"  varF baseline BC thr=0.75 (SOTA, ctx-based above):")
    for bc_thr in [0.75, 0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
        p = sota_probs.copy()
        p[:, 2] = w2v_oof[:, 2]
        thrs = dict(THR_VARF); thrs[2] = bc_thr
        eval_f1(p, thrs, f"BC=w2v2@thr={bc_thr:.2f}")

    # 6. 看 w2v2 BC 输出分布
    print("\n=== w2v2 BC 输出分布 (369 cap1) ===")
    bc_probs = w2v_oof[:, 2]
    print(f"  min={bc_probs.min():.3f} max={bc_probs.max():.3f} mean={bc_probs.mean():.3f}")
    print(f"  median={np.median(bc_probs):.3f} q90={np.quantile(bc_probs, 0.9):.3f}")
    bc_y = y_cap1[:, 2]
    pos_idx = bc_y == 1
    neg_idx = bc_y == 0
    print(f"  BC 真正例 N={pos_idx.sum()}, prob mean={bc_probs[pos_idx].mean():.3f}")
    print(f"  BC 真负例 N={neg_idx.sum()}, prob mean={bc_probs[neg_idx].mean():.3f}")

    # 7. ensemble: ctx_BC + w2v2_BC (Platt-style soft fusion)
    print("\n=== BC = avg(ctx, w2v2) 软融合 ===")
    for w in [0.3, 0.5, 0.7]:
        p = sota_probs.copy()
        p[:, 2] = w * ctx_oof[:, 2] + (1 - w) * w2v_oof[:, 2]
        for bc_thr in [0.75, 0.50, 0.30]:
            thrs = dict(THR_VARF); thrs[2] = bc_thr
            eval_f1(p, thrs, f"BC=(ctx*{w}+w2v2*{1-w:.1f})@thr={bc_thr:.2f}")


if __name__ == "__main__":
    main()
