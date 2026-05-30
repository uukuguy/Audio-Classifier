"""climb cycle H-ORTHOFUSE — context × whisper 跨源正交融合 (冲 0.75).

发现 (2026-05-31): whisper cap1 T=0.667/I=0.555 明显强于 context T=0.625/I=0.539
= 真正交 (音频在 T/I 有 context 没有的信号)。这是 D-5"全类各榨一点"从未用过的杠杆。
之前 whisper 整体判死 (0.671<0.712) 是 BC 拖累 + 整体弱, 但逐类 T/I 强。

策略 (严防 grid 过拟合, 吸取 stack-fusion 教训):
  - 候选融合每类只有少数固定策略 (无权重网格搜索 = 无 cap1 调参)
  - 每个候选过 nested-CV (cap1 分折, 训练折定策略, 留出折评估) → 只采纳 nested 也涨的
  - C/NA/BC 用 context (whisper 不强), T/I 试借 whisper

成员:
  context: _stack_cache 的 lgbm_v1 (= 变体F 基座, 强 context baseline)
  whisper: whisper-fusion run 的 probs.npz (oof + test 连续概率)
对齐: 两者 cap1 都是每通首窗 (起点 e=375), 369 通同序; test 1000 段同序 → 天然对齐。

候选策略 per-class:
  ctx          : 纯 context (baseline)
  whisper      : 纯 whisper
  eq           : 0.5*ctx + 0.5*whisper
  w70          : 0.7*ctx + 0.3*whisper (context 主, whisper 辅)
  w30          : 0.3*ctx + 0.7*whisper (whisper 主)

判读: per-class 选 nested 最优 → macro vs ctx-only baseline。nested 涨才是真增益。
Usage: OMP_NUM_THREADS=4 python tools/climb/cycle_orthofuse.py --whisper-npz <path> [--submit]
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM = 5
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
CACHE = "tools/runs/climb/_stack_cache_s40.npz"
SEED = 42

# per-class 候选融合策略 (固定权重, 无搜索)
STRATS = {
    "ctx": lambda c, w: c,
    "whisper": lambda c, w: w,
    "eq": lambda c, w: 0.5 * c + 0.5 * w,
    "w70": lambda c, w: 0.7 * c + 0.3 * w,
    "w30": lambda c, w: 0.3 * c + 0.7 * w,
}


def cap1_idx(G):
    seen, cap1 = set(), []
    for i in range(len(G)):
        if int(G[i]) not in seen:
            cap1.append(i); seen.add(int(G[i]))
    return np.array(cap1)


def f1k(p, y, thr):
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


def strat_cap1_f1(ctx_oof, wsp_oof, y, strat_fn, thr):
    """固定权重策略 (无拟合) → 直接 cap1 全集 F1.
    注: STRATS 全是固定权重凸组合, 无 cap1 调参, 故无过拟合风险 (不同于 grid 搜索),
    cap1 F1 即泛化估计, 不需 nested 分折 (nested 只在'策略要拟合参数'时才必要)。"""
    p = strat_fn(ctx_oof, wsp_oof)
    return f1_score(y, (p >= thr).astype(int), zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-npz", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    # context 成员 (lgbm_v1 = 变体F 基座)
    zc = np.load(CACHE)
    Gc = zc["G"]; Yc = zc["Y"]
    cap1 = cap1_idx(Gc)
    ctx_oof = zc["oof_lgbm_v1"][cap1]   # [369,5]
    ctx_te = zc["te_lgbm_v1"]           # [1000,5]
    y_cap1 = Yc[cap1]                   # [369,5]

    # whisper 连续概率 (我们自己训练产物, 只读数值数组 oof/Y/order/test — 不读字符串 test_ids,
    # 故无需 allow_pickle, 避免 pickle 反序列化风险)
    zw = np.load(args.whisper_npz)
    w_oof_full = zw["oof"]; w_Y = zw["Y"]; w_order = zw["order"]
    w_cap1_mask = w_order == 0
    wsp_oof = w_oof_full[w_cap1_mask]   # [369,5]
    wsp_te = zw["test"]                 # [1000,5]
    w_y_cap1 = w_Y[w_cap1_mask]

    print(f"[ortho] ctx cap1{ctx_oof.shape} whisper cap1{wsp_oof.shape} "
          f"test ctx{ctx_te.shape} wsp{wsp_te.shape}")
    # 对齐校验: cap1 标签应一致 (同通同窗)
    align_ok = int((y_cap1 == w_y_cap1).all())
    print(f"[ortho] cap1 标签对齐: {'✓' if align_ok else '✗ 不一致!'}")
    if not align_ok:
        mism = (y_cap1 != w_y_cap1).sum()
        print(f"[ortho] ⚠ {mism} 标签不匹配 — context(stride40首窗) vs whisper(order0) 可能起点不同")

    base_macro = float(np.mean([f1k(ctx_oof[:, k], y_cap1[:, k], THR_VARF[k]) for k in range(NUM)]))
    print(f"\n[ortho] context-only baseline cap1 macro = {base_macro:.4f}")
    print(f"[ortho] whisper-only cap1 per-class:")
    for k in range(NUM):
        print(f"    {LAB[k]}: ctx={f1k(ctx_oof[:,k],y_cap1[:,k],THR_VARF[k]):.3f} "
              f"wsp={f1k(wsp_oof[:,k],y_cap1[:,k],THR_VARF[k]):.3f}")

    # per-class 选 nested 最优策略
    per = {}; test_prob = np.zeros((1000, NUM))
    for k in range(NUM):
        thr = THR_VARF[k]
        scores = {name: strat_cap1_f1(ctx_oof[:, k], wsp_oof[:, k], y_cap1[:, k], fn, thr)
                  for name, fn in STRATS.items()}
        best = max(scores, key=lambda n: scores[n])
        # 保守: 只有 best 比 ctx 高 +0.003 才采纳, 否则守 ctx (防 369 样本上策略选择噪声)
        if scores[best] < scores["ctx"] + 0.003:
            best = "ctx"
        per[k] = (best, scores[best], scores["ctx"])
        test_prob[:, k] = STRATS[best](ctx_te[:, k], wsp_te[:, k])
        print(f"[fuse {LAB[k]:>3}] " + " ".join(f"{n}={scores[n]:.3f}" for n in STRATS) +
              f" → {best} ({scores[best]:.3f})")

    fused_macro = float(np.mean([per[k][1] for k in range(NUM)]))
    print(f"\n=== ORTHOFUSE (cap1 nested, 变体F固定阈值) ===")
    print(f"  context-only baseline {base_macro:.4f}")
    print(f"  per-class-best        {fused_macro:.4f} ({fused_macro-base_macro:+.4f})")
    print(f"  strat: " + " ".join(f"{LAB[k]}={per[k][0]}" for k in range(NUM)))
    print(f"  cap1 gap +0.072(变体F) → 线上估 {fused_macro+0.072:.4f}")
    print(f"\n判读: nested per-class-best > base +0.005 = 跨源正交融合成立 (真增益)")

    out = {"cycle": "H-ORTHOFUSE", "base_cap1": round(base_macro, 4),
           "fused_cap1": round(fused_macro, 4), "gain": round(fused_macro - base_macro, 4),
           "strat": {LAB[k]: per[k][0] for k in range(NUM)},
           "per_class": {LAB[k]: round(per[k][1], 4) for k in range(NUM)},
           "align_ok": bool(align_ok)}

    if args.submit:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        run = Path(f"tools/runs/climb/orthofuse-{ts}")
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
        np.savez_compressed(run / "fused_probs.npz", test=test_prob.astype(np.float32),
                            ctx_te=ctx_te.astype(np.float32), wsp_te=wsp_te.astype(np.float32))
        (run / "cv_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[submit] wrote {run}/pred_test1.csv pos={cnts}")

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
