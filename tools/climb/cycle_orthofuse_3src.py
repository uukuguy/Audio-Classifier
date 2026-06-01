"""climb cycle 16 — context × whisper × hubert 三源正交融合 (冲 0.7285).

派生自 cycle_orthofuse.py (双源 context+whisper). 加 chinese-hubert 第三独立源.

关键变化:
  1. 加 --hubert-npz 参数 (None 时退回双源行为)
  2. STRATS 扩展: 三源各种凸组合 (5 + 双源饱和/三源主辅3种 = ~10 候选, 不暴涨防 cap1 过拟合)
  3. 采纳门槛: +0.003 → +0.008 (三源策略空间大, 选择噪声增加, 守更严)
  4. 双源对照: 输出 ctx+whisper baseline 让我们看 hubert 边际增量是否真实

判读:
  - hubert per-class F1 比 whisper/ctx 强 → 三源采纳, 报告每类选哪源
  - hubert 各类持平/弱 → 退回双源, 报告"hubert 不正交"
  - nested 选择算法: 同 cycle_orthofuse 固定权重凸组合无搜索 (cap1 即泛化估计)

Usage:
  OMP_NUM_THREADS=4 python tools/climb/cycle_orthofuse_3src.py \\
    --whisper-npz tools/runs/climb/whisper-fusion-20260531-0143/probs.npz \\
    --hubert-npz tools/runs/climb/hubert-fusion-20260531-0750/probs.npz \\
    --folds 5 --submit
"""
from __future__ import annotations
import argparse
import glob
import json
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

# 三源 per-class 候选策略 (固定权重, 无搜索, 防 cap1 过拟合)
# 双源策略 (h=None 时也能用)
STRATS_2SRC = {
    "ctx": lambda c, w, h: c,
    "whisper": lambda c, w, h: w,
    "ctx_w_eq": lambda c, w, h: 0.5 * c + 0.5 * w,
    "ctx_w_70": lambda c, w, h: 0.7 * c + 0.3 * w,
    "ctx_w_30": lambda c, w, h: 0.3 * c + 0.7 * w,
}
# 三源策略 (h is not None)
STRATS_3SRC_EXT = {
    "hubert": lambda c, w, h: h,
    "ctx_h_eq": lambda c, w, h: 0.5 * c + 0.5 * h,
    "ctx_h_70": lambda c, w, h: 0.7 * c + 0.3 * h,
    "all_eq": lambda c, w, h: (c + w + h) / 3.0,
    "all_60_20_20": lambda c, w, h: 0.6 * c + 0.2 * w + 0.2 * h,
    "w_h_eq": lambda c, w, h: 0.5 * w + 0.5 * h,
}


def cap1_idx(G):
    seen, cap1 = set(), []
    for i in range(len(G)):
        if int(G[i]) not in seen:
            cap1.append(i); seen.add(int(G[i]))
    return np.array(cap1)


def f1k(p, y, thr):
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-npz", required=True)
    ap.add_argument("--hubert-npz", default=None,
                    help="如未提供则退回双源 (=cycle_orthofuse.py 等价行为)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--gate", type=float, default=0.008,
                    help="单类策略采纳门槛 (3src 守 +0.008 防过拟合, 2src 可降到 +0.003)")
    args = ap.parse_args()

    three_src = args.hubert_npz is not None
    strats = dict(STRATS_2SRC)
    if three_src:
        strats.update(STRATS_3SRC_EXT)
    print(f"[ortho-3src] {'三源' if three_src else '双源'} 模式, 候选策略 {len(strats)} 种, 采纳门槛 +{args.gate}")

    # context 成员 (lgbm_v1 = 变体F 基座 cap1=0.6228, stride40)
    zc = np.load(CACHE)
    Gc = zc["G"]; Yc = zc["Y"]
    cap1 = cap1_idx(Gc)
    ctx_oof = zc["oof_lgbm_v1"][cap1]   # [369,5]
    ctx_te = zc["te_lgbm_v1"]           # [1000,5]
    y_cap1 = Yc[cap1]                   # [369,5]

    # whisper probs (训练产物 oof/Y/order/test)
    zw = np.load(args.whisper_npz)
    w_oof_full = zw["oof"]; w_Y = zw["Y"]; w_order = zw["order"]
    w_cap1_mask = w_order == 0
    wsp_oof = w_oof_full[w_cap1_mask]   # [369,5]
    wsp_te = zw["test"]                 # [1000,5]
    w_y_cap1 = w_Y[w_cap1_mask]

    # hubert probs (同 whisper 结构, 我们用 train_head_hubert.py 产物)
    if three_src:
        zh = np.load(args.hubert_npz)
        h_oof_full = zh["oof"]; h_Y = zh["Y"]; h_order = zh["order"]
        h_cap1_mask = h_order == 0
        hub_oof = h_oof_full[h_cap1_mask]   # [369,5]
        hub_te = zh["test"]                  # [1000,5]
        h_y_cap1 = h_Y[h_cap1_mask]
        # 对齐三源
        align_wh = int((y_cap1 == w_y_cap1).all() and (y_cap1 == h_y_cap1).all())
        print(f"[ortho-3src] cap1 ctx{ctx_oof.shape} wsp{wsp_oof.shape} hub{hub_oof.shape} "
              f"test ctx{ctx_te.shape} wsp{wsp_te.shape} hub{hub_te.shape}")
        print(f"[ortho-3src] cap1 标签三源对齐: {'✓' if align_wh else '✗'}")
    else:
        hub_oof = np.zeros_like(wsp_oof)  # placeholder, 双源模式不读
        hub_te = np.zeros_like(wsp_te)
        align_wh = int((y_cap1 == w_y_cap1).all())
        print(f"[ortho-3src] cap1 ctx{ctx_oof.shape} wsp{wsp_oof.shape} 双源模式")
        print(f"[ortho-3src] cap1 标签双源对齐: {'✓' if align_wh else '✗'}")

    base_macro = float(np.mean([f1k(ctx_oof[:, k], y_cap1[:, k], THR_VARF[k]) for k in range(NUM)]))
    print(f"\n[ortho-3src] context-only baseline cap1 macro = {base_macro:.4f}")
    print(f"[ortho-3src] 各源 per-class F1 (变体F固定阈值):")
    for k in range(NUM):
        line = f"    {LAB[k]}: ctx={f1k(ctx_oof[:,k],y_cap1[:,k],THR_VARF[k]):.3f} " \
               f"wsp={f1k(wsp_oof[:,k],y_cap1[:,k],THR_VARF[k]):.3f}"
        if three_src:
            line += f" hub={f1k(hub_oof[:,k],y_cap1[:,k],THR_VARF[k]):.3f}"
        print(line)

    # 双源对照 (报告 hubert 边际增量, 即使三源模式)
    base_2src_per = {}
    base_2src_macro = base_macro  # 双源模式 fallback = ctx-only baseline
    if three_src:
        for k in range(NUM):
            thr = THR_VARF[k]
            scores2 = {n: f1k(fn(ctx_oof[:,k], wsp_oof[:,k], None), y_cap1[:,k], thr)
                       for n, fn in STRATS_2SRC.items()}
            best2 = max(scores2, key=lambda n: scores2[n])
            if scores2[best2] < scores2["ctx"] + 0.003:
                best2 = "ctx"
            base_2src_per[k] = (best2, scores2[best2])
        base_2src_macro = float(np.mean([base_2src_per[k][1] for k in range(NUM)]))
        print(f"\n[ortho-3src] 双源 (ctx+whisper) baseline = {base_2src_macro:.4f} (vs ctx +{base_2src_macro-base_macro:.4f})")

    # per-class 选策略 (三源/双源)
    per = {}; test_prob = np.zeros((1000, NUM))
    for k in range(NUM):
        thr = THR_VARF[k]
        scores = {n: f1k(fn(ctx_oof[:,k], wsp_oof[:,k], hub_oof[:,k]), y_cap1[:,k], thr)
                  for n, fn in strats.items()}
        best = max(scores, key=lambda n: scores[n])
        # 守门: 比 ctx 高至少 args.gate 才采纳
        if scores[best] < scores["ctx"] + args.gate:
            best = "ctx"
        per[k] = (best, scores[best], scores["ctx"])
        test_prob[:, k] = strats[best](ctx_te[:, k], wsp_te[:, k], hub_te[:, k])
        line = f"[fuse {LAB[k]:>3}] best={best} ({scores[best]:.3f})"
        if three_src:
            # 三源模式打印所有 hubert 涉及策略 + best 历史
            h_strats = ["hubert", "ctx_h_eq", "ctx_h_70", "all_eq", "all_60_20_20", "w_h_eq"]
            hub_part = " ".join(f"{n}={scores[n]:.3f}" for n in h_strats if n in scores)
            line += f"  | hub-strats: {hub_part}"
        print(line)

    fused_macro = float(np.mean([per[k][1] for k in range(NUM)]))
    print(f"\n=== ORTHOFUSE 3SRC (cap1, 变体F固定阈值, gate +{args.gate}) ===")
    print(f"  context-only baseline   {base_macro:.4f}")
    if three_src:
        print(f"  双源 ctx+whisper        {base_2src_macro:.4f} ({base_2src_macro-base_macro:+.4f})")
        print(f"  三源 per-class-best     {fused_macro:.4f} ({fused_macro-base_macro:+.4f} vs ctx, "
              f"{fused_macro-base_2src_macro:+.4f} vs 双源)")
    else:
        print(f"  双源 per-class-best     {fused_macro:.4f} ({fused_macro-base_macro:+.4f})")
    print(f"  strat: " + " ".join(f"{LAB[k]}={per[k][0]}" for k in range(NUM)))
    print(f"  cap1 gap +0.072(变体F类) → 线上估 {fused_macro+0.072:.4f}")

    if three_src:
        # 关键判读
        hub_selected = sum(1 for k in range(NUM) if "h" in per[k][0] or "hub" in per[k][0] or "all" in per[k][0])
        margin_3vs2 = fused_macro - base_2src_macro
        print(f"\n判读 (cycle 16 决策门):")
        print(f"  hubert 被选中类数 = {hub_selected}/{NUM}")
        print(f"  三源 vs 双源 margin = +{margin_3vs2:.4f} (> +0.003 = hubert 真有正交价值)")
        if margin_3vs2 > 0.003 and hub_selected >= 1:
            print(f"  ★ hubert 正交成立 → 推荐扩容跑 stride8 密集 hubert + 加 w2v2 第四源")
        else:
            print(f"  ✗ hubert 不正交 → 撤 w2v2 第四源, 转 emotion2vec/后处理")

    out = {"cycle": "H-ORTHOFUSE-3SRC" if three_src else "H-ORTHOFUSE-2SRC",
           "base_cap1": round(base_macro, 4),
           "fused_cap1": round(fused_macro, 4),
           "gain_vs_ctx": round(fused_macro - base_macro, 4),
           "strat": {LAB[k]: per[k][0] for k in range(NUM)},
           "per_class": {LAB[k]: round(per[k][1], 4) for k in range(NUM)},
           "align_ok": bool(align_wh), "gate": args.gate, "three_src": three_src}
    if three_src:
        out["base_2src_cap1"] = round(base_2src_macro, 4)
        out["gain_vs_2src"] = round(fused_macro - base_2src_macro, 4)

    if args.submit:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        suffix = "3src" if three_src else "2src"
        run = Path(f"tools/runs/climb/orthofuse-{suffix}-{ts}")
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
        # 存融合用概率 (产物铁律)
        np.savez_compressed(run / "fused_probs.npz",
                            test=test_prob.astype(np.float32),
                            ctx_te=ctx_te.astype(np.float32),
                            wsp_te=wsp_te.astype(np.float32),
                            **({"hub_te": hub_te.astype(np.float32)} if three_src else {}))
        (run / "cv_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[submit] wrote {run}/pred_test1.csv pos={cnts}")

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
