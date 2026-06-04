"""P2 阈值 ±0.05 sweep on new SOTA orthofuse-3src-20260601-1607.

D-15 P2 + D-16 SOTA 后续: 新 SOTA cap1=0.6532 → 真分 0.71755. 阈值 cycle1 钙化值
THR_VARF={C:0.05, T:0.50, BC:0.75, I:0.65, NA:0.25}. 跨源融合后概率分布跟单源不同,
阈值微调 ±0.05 (5档) 可能有 +0.001~0.003 真分增益.

策略 (Claude self review 盲点 2 修正):
  - 阈值搜索空间 5 档/类 × 5 类 = 25 候选, sample/candidate=369/25=14.7, 不到过拟合阈值
  - 守 cycle1 附近 ±0.05 微调 (T[0.45-0.55] BC[0.70-0.80] I[0.60-0.70] NA[0.20-0.30])
  - C 类不动 (近全正 0.05 已是钙化)

判读:
  - 重建 fused cap1 OOF (用 ctx/whisper/hubert cap1 OOF + 已知 strat)
  - per-class 在 cap1 上找最优阈值
  - 出新 csv + 报 pos diff vs 0.71755 SOTA
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
THR_BASE = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}  # cycle1 钙化阈值
NUM = 5

# 新 SOTA 配置 (orthofuse-3src-20260601-1607)
SOTA_DIR = Path("tools/runs/climb/orthofuse-3src-20260601-1607")
STACK_CACHE = "tools/runs/climb/_stack_cache_s40.npz"
WHISPER_NPZ = "tools/runs/climb/whisper-fusion-20260531-0143/probs.npz"
HUBERT_BCAUG_NPZ = "tools/runs/climb/hubert-bcaug-head-20260601-1533/probs.npz"

# strats from SOTA 3src run:
# C=ctx T=whisper_hubert_70 BC=ctx I=ctx_whisper_hubert_eq NA=ctx
SOTA_STRATS = {
    0: ("ctx", None),                          # C
    1: ("whisper_hubert_70", ["whisper", "hubert"]),  # T = 0.7*whisper + 0.3*hubert
    2: ("ctx", None),                          # BC
    3: ("ctx_whisper_hubert_eq", ["ctx", "whisper", "hubert"]),  # I = 等权
    4: ("ctx", None),                          # NA
}


def cap1_idx(G):
    seen, cap1 = set(), []
    for i in range(len(G)):
        if int(G[i]) not in seen:
            cap1.append(i); seen.add(int(G[i]))
    return np.array(cap1)


def load_src(npz_path, y_cap1_ctx):
    """Load whisper/hubert cap1 OOF + test."""
    z = np.load(npz_path)
    oof = z["oof"]; y_full = z["Y"]; order = z["order"]
    mask = order == 0
    oof_cap1 = oof[mask]
    test = z["test"]
    return oof_cap1, test


def compute_fused_prob(probs_by_src, k, strat_name):
    """Compute fused probs for class k using SOTA strat."""
    if strat_name == "ctx":
        return probs_by_src["ctx"][:, k]
    elif strat_name == "whisper_hubert_70":
        return 0.7 * probs_by_src["whisper"][:, k] + 0.3 * probs_by_src["hubert"][:, k]
    elif strat_name == "ctx_whisper_hubert_eq":
        return (probs_by_src["ctx"][:, k] + probs_by_src["whisper"][:, k] +
                probs_by_src["hubert"][:, k]) / 3.0
    raise ValueError(f"unknown strat: {strat_name}")


def main():
    # === 1. 重建 cap1 fused OOF ===
    zc = np.load(STACK_CACHE)
    Gc = zc["G"]; Yc = zc["Y"]
    cap1 = cap1_idx(Gc)
    y_cap1 = Yc[cap1]
    probs_cap1 = {"ctx": zc["oof_lgbm_v1"][cap1]}
    probs_test = {"ctx": zc["te_lgbm_v1"]}

    for src, npz in [("whisper", WHISPER_NPZ), ("hubert", HUBERT_BCAUG_NPZ)]:
        oof_cap1, te = load_src(npz, y_cap1)
        probs_cap1[src] = oof_cap1
        probs_test[src] = te

    # 重建 cap1 fused per-class (用 SOTA strat)
    fused_cap1 = np.zeros((len(y_cap1), NUM))
    fused_test = np.zeros((1000, NUM))
    for k in range(NUM):
        strat_name, _ = SOTA_STRATS[k]
        fused_cap1[:, k] = compute_fused_prob(probs_cap1, k, strat_name)
        fused_test[:, k] = compute_fused_prob(probs_test, k, strat_name)

    # === 2. 验证 baseline cap1 macro (应该 = 0.6532) ===
    base_macro_per = {}
    for k in range(NUM):
        base_macro_per[k] = f1_score(y_cap1[:, k], (fused_cap1[:, k] >= THR_BASE[k]).astype(int),
                                       zero_division=0)
    base_macro = float(np.mean(list(base_macro_per.values())))
    print(f"=== Baseline (SOTA strats + cycle1 阈值): cap1 macro = {base_macro:.4f} ===")
    for k in range(NUM):
        print(f"  {LAB[k]}: {base_macro_per[k]:.4f} @ thr={THR_BASE[k]}")

    if abs(base_macro - 0.6532) > 0.002:
        print(f"⚠️ 期望 0.6532, 实际 {base_macro:.4f}, 偏差大. 可能 strat 重建有误")

    # === 3. per-class 阈值 sweep (cycle1 ±0.05, 5 档) ===
    sweep_range = {
        0: [0.05],  # C 不动
        1: np.linspace(0.45, 0.55, 5),  # T
        2: np.linspace(0.70, 0.80, 5),  # BC
        3: np.linspace(0.60, 0.70, 5),  # I
        4: np.linspace(0.20, 0.30, 5),  # NA
    }

    best_thr = {}
    best_per = {}
    print(f"\n=== Per-class threshold sweep ({sum(len(v) for v in sweep_range.values())} 候选) ===")
    for k in range(NUM):
        best_f1, best_t = -1, THR_BASE[k]
        for t in sweep_range[k]:
            f = f1_score(y_cap1[:, k], (fused_cap1[:, k] >= float(t)).astype(int),
                         zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        best_thr[k] = best_t
        best_per[k] = best_f1
        delta_thr = best_t - THR_BASE[k]
        delta_f1 = best_f1 - base_macro_per[k]
        print(f"  {LAB[k]}: f1={best_f1:.4f} @ thr={best_t:.2f} "
              f"(base={base_macro_per[k]:.4f}@{THR_BASE[k]:.2f}, Δthr={delta_thr:+.2f} Δf1={delta_f1:+.4f})")

    best_macro = float(np.mean(list(best_per.values())))
    print(f"\n=== Sweep best macro = {best_macro:.4f} (Δ {best_macro - base_macro:+.4f}) ===")

    # === 4. 出 csv (新阈值) ===
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(f"tools/runs/climb/orthofuse-3src-sweep-{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = [Path(p).stem for p in sorted(Path("data/test/context").glob("*.npy"))]
    assert len(test_ids) == 1000

    pos_counts = {c: 0 for c in SUBMIT}
    with open(out_dir / "pred_test1.csv", "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(fused_test[i, COL2K[c]] >= best_thr[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items(): pos_counts[c] += v
            row = [sid] + [str(vals[c]) for c in SUBMIT]
            f.write(",".join(row) + "\n")

    # 同时出 SOTA 阈值的 csv 对照 (验证 = 0.71755 reproduce)
    sota_pos = {c: 0 for c in SUBMIT}
    with open(out_dir / "pred_test1_baseline.csv", "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(fused_test[i, COL2K[c]] >= THR_BASE[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items(): sota_pos[c] += v
            row = [sid] + [str(vals[c]) for c in SUBMIT]
            f.write(",".join(row) + "\n")

    # SOTA reference pos: c=975 na=947 i=81 bc=27 t=522 (from orthofuse-3src-20260601-1607)
    SOTA_POS = {"c": 975, "na": 947, "i": 81, "bc": 27, "t": 522}

    print(f"\n=== POS counts ===")
    print(f"  SOTA  (orthofuse-3src-20260601-1607): {SOTA_POS}")
    print(f"  reproduced baseline:                  {sota_pos}")
    print(f"  sweep new thr:                        {pos_counts}")
    diff = {c: pos_counts[c] - SOTA_POS[c] for c in SUBMIT}
    print(f"  diff (sweep - SOTA):                  {diff}")

    summary = {
        "cycle": "P2-THRESHOLD-SWEEP",
        "sota_run": str(SOTA_DIR),
        "sota_real_score": 0.71755,
        "base_macro_cap1": round(base_macro, 4),
        "sweep_macro_cap1": round(best_macro, 4),
        "delta_macro": round(best_macro - base_macro, 4),
        "best_thresholds": {LAB[k]: round(best_thr[k], 2) for k in range(NUM)},
        "base_thresholds": {LAB[k]: THR_BASE[k] for k in range(NUM)},
        "per_class_base_f1": {LAB[k]: round(base_macro_per[k], 4) for k in range(NUM)},
        "per_class_sweep_f1": {LAB[k]: round(best_per[k], 4) for k in range(NUM)},
        "pos_sota": SOTA_POS,
        "pos_reproduced": sota_pos,
        "pos_sweep": pos_counts,
        "pos_diff": diff,
        "_note": "阈值 sweep 空间 ratio 369/25=14.7 < 过拟合阈值. 若 Δmacro>+0.003 值得 push.",
    }
    (out_dir / "cv_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[saved] {out_dir}/")
    print(f"  pred_test1.csv (新阈值 sweep)")
    print(f"  pred_test1_baseline.csv (cycle1 阈值, 验证 0.71755)")
    print(f"\n判定: Δmacro = {best_macro - base_macro:+.4f}, 阈值铁律: Δ>+0.003 才 push")
    if best_macro - base_macro > 0.003:
        print(f"  ★ 值得 push, 期望线上 {best_macro + 0.072:+.4f}")
    else:
        print(f"  noise floor 内, 不 push 浪费配额")


if __name__ == "__main__":
    main()
