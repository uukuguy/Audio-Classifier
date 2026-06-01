"""climb cycle 17 — N源跨源融合 (ctx + whisper + hubert + w2v2 + emotion2vec).

派生自 cycle_orthofuse_3src.py. 通用化 N 源 (1-5 源, 自动选择策略集).

设计原则 (基于 D-8/D-9 教训):
  1. 不暴增策略空间 (5^5=3125 grid会过拟合, 故守 fixed-weights 凸组合 + 先验导向选)
  2. 策略集 = N源所有"单源" + 所有"主源+次源 70/30 / 50/50 / 30/70" + "全等权"
  3. 收紧采纳门: 3源 +0.008, 4源 +0.010, 5源 +0.012 (源越多越要更确凿增益才采纳)
  4. nested 报告 N源 vs (N-1)源 vs ctx-only 三档 margin

Usage:
  # 双源 (回退 = orthofuse 原版)
  python tools/climb/cycle_orthofuse_nsrc.py --whisper-npz <p>

  # 三源 (cycle 16 兼容)
  python tools/climb/cycle_orthofuse_nsrc.py --whisper-npz <p> --hubert-npz <p>

  # 四源 (cycle 17 加 w2v2)
  python tools/climb/cycle_orthofuse_nsrc.py --whisper-npz <p> --hubert-npz <p> --w2v2-npz <p>

  # 五源 (cycle 17 完整)
  python tools/climb/cycle_orthofuse_nsrc.py --whisper-npz <p> --hubert-npz <p> --w2v2-npz <p> --e2v-npz <p> --submit
"""
from __future__ import annotations
import argparse
import glob
import json
import itertools
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

NUM = 5
LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
CACHE = "tools/runs/climb/_stack_cache_s40.npz"
SEED = 42

# 源 ID (顺序固定: ctx 永远第0, 其它按命令行加载顺序)
SOURCE_IDS = ["ctx", "whisper", "hubert", "w2v2", "e2v"]


def cap1_idx(G):
    seen, cap1 = set(), []
    for i in range(len(G)):
        if int(G[i]) not in seen:
            cap1.append(i); seen.add(int(G[i]))
    return np.array(cap1)


def f1k(p, y, thr):
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


def build_strats(active_srcs: list[str]) -> dict:
    """active_srcs = ['ctx', 'whisper', ...] 当前有数据的源 (按 SOURCE_IDS 顺序).
    生成策略集: 单源 + 双源各种凸 + 全等权."""
    strats = {}
    n = len(active_srcs)

    # 1. 所有单源 (N 个)
    for s in active_srcs:
        strats[s] = (s,)

    # 2. 所有双源凸组合 (C(N,2) × 3 = 3 N(N-1)/2 个)
    #    eq / s1_70 / s1_30 (其中 s1 是字典序小的源)
    for s1, s2 in itertools.combinations(active_srcs, 2):
        strats[f"{s1}_{s2}_eq"] = (s1, s2, 0.5, 0.5)
        strats[f"{s1}_{s2}_70"] = (s1, s2, 0.7, 0.3)
        strats[f"{s1}_{s2}_30"] = (s1, s2, 0.3, 0.7)

    # 3. 三源/四源/五源等权 (各 1 个)
    if n >= 3:
        for k in range(3, n + 1):
            for combo in itertools.combinations(active_srcs, k):
                strats[f"{'_'.join(combo)}_eq"] = (*combo, *[1.0 / k] * k)

    return strats


def eval_strat(strat_spec, probs_by_src: dict, k: int, y, thr) -> float:
    """strat_spec = ('whisper',) 或 ('whisper', 'hubert', 0.5, 0.5) 或 ('w','h','e2v', 0.33, 0.33, 0.34)"""
    if len(strat_spec) == 1:
        p = probs_by_src[strat_spec[0]][:, k]
    else:
        # 前半源名, 后半权重
        half = len(strat_spec) // 2
        srcs = strat_spec[:half]
        ws = strat_spec[half:]
        p = sum(w * probs_by_src[s][:, k] for s, w in zip(srcs, ws))
    return f1k(p, y, thr)


def compute_strat_probs(strat_spec, test_probs_by_src: dict, k: int) -> np.ndarray:
    if len(strat_spec) == 1:
        return test_probs_by_src[strat_spec[0]][:, k]
    half = len(strat_spec) // 2
    srcs = strat_spec[:half]
    ws = strat_spec[half:]
    return sum(w * test_probs_by_src[s][:, k] for s, w in zip(srcs, ws))


def load_oof_test(npz_path: str, y_cap1_ctx: np.ndarray):
    """从 whisper/hubert/w2v2/e2v head 训出的 probs.npz 读 cap1 OOF + test, 校验标签对齐."""
    z = np.load(npz_path)
    oof = z["oof"]; y_full = z["Y"]; order = z["order"]
    test = z["test"]
    cap1_mask = order == 0
    oof_cap1 = oof[cap1_mask]
    y_cap1_npz = y_full[cap1_mask]
    align_ok = bool((y_cap1_ctx == y_cap1_npz).all())
    return oof_cap1, test, align_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-npz", default=None)
    ap.add_argument("--hubert-npz", default=None)
    ap.add_argument("--w2v2-npz", default=None)
    ap.add_argument("--e2v-npz", default=None)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--gate-base", type=float, default=0.008,
                    help="基础采纳门, 源数+1 自动 +0.002")
    ap.add_argument("--ctx-base", default="lgbm_v1",
                    choices=["lgbm_v1", "xgb_v1", "lgbm_v2", "mlp_v1"],
                    help="cycle 18: 换 ctx 基座算法 (D-5 已测同源融合不正交, 本次只换 base 不融合)")
    args = ap.parse_args()

    # === 1. context 基座 (变体F lgbm_v1, cap1=0.6228) ===
    zc = np.load(CACHE)
    Gc = zc["G"]; Yc = zc["Y"]
    cap1 = cap1_idx(Gc)
    y_cap1 = Yc[cap1]

    # cycle 18: ctx base 可选 (lgbm_v1 默认, xgb_v1/lgbm_v2/mlp_v1 备)
    probs_by_src = {"ctx": zc[f"oof_{args.ctx_base}"][cap1]}
    test_by_src = {"ctx": zc[f"te_{args.ctx_base}"]}
    print(f"[ortho-nsrc] ctx base = {args.ctx_base}")

    # === 2. 各音频源 (按命令行参数注入) ===
    src_paths = {"whisper": args.whisper_npz, "hubert": args.hubert_npz,
                 "w2v2": args.w2v2_npz, "e2v": args.e2v_npz}
    align_status = {}
    for src, p in src_paths.items():
        if p is None:
            continue
        oof, te, aok = load_oof_test(p, y_cap1)
        probs_by_src[src] = oof
        test_by_src[src] = te
        align_status[src] = aok

    active_srcs = [s for s in SOURCE_IDS if s in probs_by_src]
    n_srcs = len(active_srcs)
    gate = args.gate_base + 0.002 * max(0, n_srcs - 3)  # 3源 +0.008, 4源 +0.010, 5源 +0.012

    print(f"[ortho-nsrc] {n_srcs}源 模式: {active_srcs}, 采纳门 +{gate:.4f}")
    print(f"[ortho-nsrc] cap1 align: {align_status}")

    # === 3. 各源单独 per-class F1 (变体F固定阈值) ===
    print(f"\n[ortho-nsrc] 各源 per-class F1 (cap1 369, 变体F阈值):")
    for src in active_srcs:
        line = f"    {src:10s}: " + " ".join(
            f"{LAB[k]}={f1k(probs_by_src[src][:,k], y_cap1[:,k], THR_VARF[k]):.3f}" for k in range(NUM))
        print(line)

    base_macro = float(np.mean([f1k(probs_by_src["ctx"][:, k], y_cap1[:, k], THR_VARF[k]) for k in range(NUM)]))
    print(f"\n[ortho-nsrc] ctx-only baseline macro = {base_macro:.4f}")

    # === 4. 构建策略集 + per-class 选最优 ===
    strats = build_strats(active_srcs)
    print(f"\n[ortho-nsrc] 策略集 |S|={len(strats)}: {list(strats.keys())[:8]}...")

    per = {}; test_prob = np.zeros((1000, NUM))
    for k in range(NUM):
        thr = THR_VARF[k]
        scores = {name: eval_strat(spec, probs_by_src, k, y_cap1[:, k], thr)
                  for name, spec in strats.items()}
        best = max(scores, key=lambda n: scores[n])
        # 守门: best 必须比 ctx 高至少 gate
        if scores[best] < scores["ctx"] + gate:
            best = "ctx"
        per[k] = (best, scores[best], scores["ctx"])
        test_prob[:, k] = compute_strat_probs(strats[best], test_by_src, k)
        print(f"[fuse {LAB[k]:>3}] best={best} ({scores[best]:.3f}) "
              f"vs ctx ({scores['ctx']:.3f})  margin {scores[best]-scores['ctx']:+.4f}")

    fused_macro = float(np.mean([per[k][1] for k in range(NUM)]))
    print(f"\n=== ORTHOFUSE N-SRC ({n_srcs} sources, cap1, gate +{gate:.4f}) ===")
    print(f"  ctx-only baseline    {base_macro:.4f}")
    print(f"  {n_srcs}源 per-class-best  {fused_macro:.4f} ({fused_macro-base_macro:+.4f} vs ctx)")
    print(f"  strat: " + " ".join(f"{LAB[k]}={per[k][0]}" for k in range(NUM)))
    # ctx gap +0.072 from variant F (历史校准, 接近性 caveat 来自 D-9 noise floor)
    print(f"  ctx gap+0.072 → 线上估 {fused_macro+0.072:.4f}")

    out = {"cycle": f"H-ORTHOFUSE-{n_srcs}SRC",
           "n_srcs": n_srcs, "active_srcs": active_srcs, "gate": round(gate, 4),
           "base_cap1": round(base_macro, 4),
           "fused_cap1": round(fused_macro, 4),
           "gain_vs_ctx": round(fused_macro - base_macro, 4),
           "strat": {LAB[k]: per[k][0] for k in range(NUM)},
           "per_class": {LAB[k]: round(per[k][1], 4) for k in range(NUM)},
           "align_status": align_status}

    if args.submit:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        run = Path(f"tools/runs/climb/orthofuse-{n_srcs}src-{ts}")
        run.mkdir(parents=True, exist_ok=True)
        test_ids = [Path(p).stem for p in sorted(glob.glob("data/test/context/*.npy"))]
        with open(run / "pred_test1.csv", "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(test_ids):
                row = [sid] + [str(int(test_prob[i, COL2K[c]] >= THR_VARF[COL2K[c]])) for c in SUBMIT]
                f.write(",".join(row) + "\n")
        cnts = {c: int((test_prob[:, COL2K[c]] >= THR_VARF[COL2K[c]]).sum()) for c in SUBMIT}
        out["pos_counts"] = cnts
        out["csv"] = str(run / "pred_test1.csv")
        np.savez_compressed(run / "fused_probs.npz",
                            test=test_prob.astype(np.float32),
                            **{f"{s}_te": test_by_src[s].astype(np.float32) for s in active_srcs})
        (run / "cv_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[submit] wrote {run}/pred_test1.csv pos={cnts}")

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
