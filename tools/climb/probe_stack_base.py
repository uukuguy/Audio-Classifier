"""探测 _stack_cache 里 4 个 ctx base (lgbm_v1/xgb_v1/lgbm_v2/mlp_v1) 的算法多样性融合.

用户洞察 (2026-06-02): 排行榜上 ≥0.72 都是多源融合, 单源见顶. 但本项目花 6 天训新 encoder
都比 SOTA 差 0.03+, 该转向"算法 × 数据多样性" stacking.

意外发现 cache 已有 lgbm_v1/xgb_v1/lgbm_v2/mlp_v1 4 个 ctx base, SOTA orthofuse 只用 lgbm_v1.
本脚本 0 训练成本探测: 4 个 base 单源 cap1 + 多种融合策略 cap1, 30s 本机.

如果 4 base 平均 / 投票 / per-class 选最强 能涨, 立即扩 5 base × 5 seed = 25 模型 (cycle_context 已经能跑).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
CACHE = "tools/runs/climb/_stack_cache_s40.npz"


def cap1_idx(G):
    seen, cap1 = set(), []
    for i, g in enumerate(G):
        if int(g) not in seen:
            cap1.append(i); seen.add(int(g))
    return np.array(cap1)


def eval_f1(probs, Y, label=""):
    f1s = {}
    for k in range(5):
        pred = (probs[:, k] >= THR_VARF[k]).astype(int)
        f1s[k] = f1_score(Y[:, k], pred, zero_division=0)
    macro = float(np.mean(list(f1s.values())))
    per_str = " ".join(f"{LAB[k]}={f1s[k]:.3f}" for k in range(5))
    print(f"  {label:42s}: macro={macro:.4f} | {per_str}")
    return macro, f1s


def main():
    print("\n=== Load cache ===")
    d = np.load(CACHE)
    Y = d["Y"].astype(int); G = d["G"]
    cap1 = cap1_idx(G)
    Y_c1 = Y[cap1]
    print(f"cap1 N={len(cap1)}, Y shape {Y_c1.shape}")

    bases = {
        "lgbm_v1": d["oof_lgbm_v1"][cap1],
        "xgb_v1":  d["oof_xgb_v1"][cap1],
        "lgbm_v2": d["oof_lgbm_v2"][cap1],
        "mlp_v1":  d["oof_mlp_v1"][cap1],
    }

    print("\n=== 4 个 base 单源 cap1 ===")
    base_f1s = {}
    for name, p in bases.items():
        _, f1s = eval_f1(p, Y_c1, name)
        base_f1s[name] = f1s

    print("\n=== 跨算法融合 (mean) ===")
    # 4 个全平均
    p_all = np.stack(list(bases.values())).mean(0)
    eval_f1(p_all, Y_c1, "all 4 mean")
    # lgbm_v1 + xgb_v1 (2 不同算法)
    p_lx = (bases["lgbm_v1"] + bases["xgb_v1"]) / 2
    eval_f1(p_lx, Y_c1, "lgbm_v1 + xgb_v1 (2-algo)")
    # lgbm_v1 + xgb_v1 + lgbm_v2 (3 strong)
    p_lxl = (bases["lgbm_v1"] + bases["xgb_v1"] + bases["lgbm_v2"]) / 3
    eval_f1(p_lxl, Y_c1, "lgbm_v1 + xgb_v1 + lgbm_v2 (3-base)")
    # 加 mlp
    p_lxlm = (bases["lgbm_v1"] + bases["xgb_v1"] + bases["lgbm_v2"] + bases["mlp_v1"]) / 4
    eval_f1(p_lxlm, Y_c1, "+ mlp_v1 (4-algo all)")

    print("\n=== Per-class 选最强 (greedy) ===")
    # 每类挑表现最好的 base
    p_greedy = np.zeros_like(Y_c1, dtype=float)
    strat = {}
    for k in range(5):
        best_name = max(base_f1s.keys(), key=lambda n: base_f1s[n][k])
        p_greedy[:, k] = bases[best_name][:, k]
        strat[k] = best_name
    eval_f1(p_greedy, Y_c1, f"per-class greedy {strat}")

    print("\n=== Per-class 跨算法平均 (前 N 强) ===")
    for n_topk in [2, 3, 4]:
        p_topk = np.zeros_like(Y_c1, dtype=float)
        for k in range(5):
            ranked = sorted(base_f1s.keys(), key=lambda n: base_f1s[n][k], reverse=True)[:n_topk]
            stacked = np.stack([bases[name][:, k] for name in ranked])
            p_topk[:, k] = stacked.mean(0)
        eval_f1(p_topk, Y_c1, f"per-class top-{n_topk} mean")


if __name__ == "__main__":
    main()
