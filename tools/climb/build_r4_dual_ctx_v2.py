"""R4 全栈 + dual-ckpt ctx 路由 csv 生成 V2 — 同纲版.

V1 版本错误: 用 single-seed ctx ckpt (models/ctx_only/), 跟 R4 baseline 0.7458 不同源
            (后者用 variant-F 5 seed avg = _stack_cache_s40.npz 的 te_lgbm_v1).

V2 修复: ctx_long = _stack_cache_s40.npz.te_lgbm_v1 (variant-F 5 seed baseline)
        ctx_short = variant-F-mask050-<ts>/probs.npz.test (variant-F 5 seed + mask=0.5)
        两者**算法同纲**, 只差"训练数据有无 mask".

dual-route 仿照 V2 (ctx-only 验证):
  seg_id mod 2 == 0 → 截短到 10s, 用 ctx_short 的对应 prob (前提: short 模型也是按 30s test 跑的,
                                                          但 short 模型见过 mask 训练, "懂"短 ctx 的语义)
  seg_id mod 2 == 1 → 全 30s, 用 ctx_long 的对应 prob

严格同纲: gen_variant_f_mask050 v2 同时算 test (全 30s) 和 test_v2 (V2 截短规则).
D1 ctx_short 用 test_v2 → 跟 V2 ctx-only 完全同纲, 只差 R4 软加.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "stack_cache": ROOT / "tools/runs/climb/_stack_cache_s40.npz",   # te_lgbm_v1 = variant-F 5 seed baseline
    "fused_3src": ROOT / "tools/runs/climb/orthofuse-3src-20260601-1607/fused_probs.npz",
    "wsp_ms": ROOT / "tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/probs.npz",
    "e2v_ms": ROOT / "tools/runs/climb/e2v-bcaug-multiseed-20260602-1632/probs.npz",
    "hub_ms": ROOT / "tools/runs/climb/hubert-bcaug-multiseed-20260602-1506/probs.npz",
    "omni3b_ms2": ROOT / "tools/runs/climb/omni-3b-ms2-mean-3seed/probs.npz",
}

SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}


def softadd(base, src, w, cols):
    out = base.copy()
    for c in cols:
        out[:, c] = (1 - w) * base[:, c] + w * src[:, c]
    return out


def make_sota_3src(ctx, wsp, hub):
    p = np.zeros_like(ctx)
    p[:, 0] = ctx[:, 0]
    p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * hub[:, 1]
    p[:, 2] = ctx[:, 2]
    p[:, 3] = (ctx[:, 3] + wsp[:, 3] + hub[:, 3]) / 3
    p[:, 4] = ctx[:, 4]
    return p


def build_s5(ctx_te, wsp_te, hub_te, wsp_ms_te, e2v_ms_te, hub_ms_te, omni3b_te):
    sota = make_sota_3src(ctx_te, wsp_te, hub_te)
    nsota = softadd(sota, wsp_ms_te, 0.07, (1, 2, 3))
    r4 = softadd(softadd(nsota, e2v_ms_te, 0.03, (1, 2, 3)), hub_ms_te, 0.03, (1, 2, 3))
    return softadd(r4, omni3b_te, 0.05, (1, 2, 3))


def write_csv(probs, seg_ids, out_path: Path, desc: str) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.zeros_like(probs, dtype=int)
    for k in range(5):
        pred[:, k] = (probs[:, k] >= THR_VARF[k]).astype(int)
    with open(out_path, "w", newline="\n") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(seg_ids):
            row = ",".join(str(pred[i, COL2K[c]]) for c in SUBMIT)
            f.write(f"{sid},{row}\n")
    pos = {c: int(pred[:, COL2K[c]].sum()) for c in SUBMIT}
    print(f"  written {out_path.parent.name}/{out_path.name}: pos={pos}  ({desc})", file=sys.stderr)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask050-probs", required=True,
                    help="path to variant-F-mask050 probs.npz (含 test (1000,5))")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    print("[r4-dual-v2] loading sources...", file=sys.stderr)

    # ctx_long: variant-F 5 seed baseline (= R4 baseline 用的)
    cache = np.load(PATHS["stack_cache"])
    ctx_long = cache["te_lgbm_v1"]   # (1000, 5)
    print(f"[r4-dual-v2] ctx_long (variant-F baseline) shape={ctx_long.shape}", file=sys.stderr)

    # ctx_short: variant-F-mask050 5 seed 输出
    # 注意: 用 test_v2 字段 — 这是按 V2 截短规则实际截短再 normalize 后跑出来的 prob
    # (跟 V2 ctx-only 完全同纲), 不是 test 字段 (test 是全 30s 推理)
    mask050 = np.load(args.mask050_probs)
    if "test_v2" not in mask050.files:
        raise SystemExit(
            f"[r4-dual-v2] {args.mask050_probs} 缺 'test_v2' 字段 "
            f"(只有 {mask050.files}). 用旧版 gen_variant_f_mask050 跑的, 需重训."
        )
    ctx_short_v2 = mask050["test_v2"]   # (1000, 5)
    ctx_short_full = mask050["test"]    # (1000, 5) 全 30s 推理, sanity 用
    print(f"[r4-dual-v2] ctx_short_v2 (V2 截短) shape={ctx_short_v2.shape}", file=sys.stderr)
    print(f"[r4-dual-v2] ctx_short_full (sanity 全 30s) shape={ctx_short_full.shape}", file=sys.stderr)

    # R4 软加源 (固定)
    fused = np.load(PATHS["fused_3src"])
    wsp_te = fused["whisper_te"]
    hub_te = fused["hubert_te"]
    wsp_ms_te = np.load(PATHS["wsp_ms"])["test"]
    e2v_ms_te = np.load(PATHS["e2v_ms"])["test"]
    hub_ms_te = np.load(PATHS["hub_ms"])["test"]
    omni3b_te = np.load(PATHS["omni3b_ms2"])["test"]

    # seg_ids: 从 fused.ctx_te 顺序 (= 6/4 R4 baseline 一致)
    # 取 wsp_ms 的 seg_ids 兜底 (跟 day7/8/9 同源)
    wsp_npz = np.load(PATHS["wsp_ms"])
    if "test_ids" in wsp_npz.files:
        seg_ids = [str(x) for x in wsp_npz["test_ids"]]
    elif "seg_ids" in wsp_npz.files:
        seg_ids = [str(x) for x in wsp_npz["seg_ids"]]
    else:
        sample = ROOT / "submission/probe-day7-20260604-1005/S5_R4+omni3b_ms2_005/pred_test1.csv"
        with open(sample) as f:
            next(f)
            seg_ids = [line.split(",")[0] for line in f.read().strip().split("\n")]

    print(f"[r4-dual-v2] seg_ids n={len(seg_ids)} first 3 = {seg_ids[:3]}", file=sys.stderr)

    # ====== Sanity D2: ctx 全用 ctx_long → R4 全栈 = R4 baseline 0.7458 应精确复现 (S5 pos = 975/947/80/15/528) ======
    s5_sanity = build_s5(ctx_long, wsp_te, hub_te, wsp_ms_te, e2v_ms_te, hub_ms_te, omni3b_te)
    s5_pred = np.zeros_like(s5_sanity, dtype=int)
    for k in range(5):
        s5_pred[:, k] = (s5_sanity[:, k] >= THR_VARF[k]).astype(int)
    s5_pos = {c: int(s5_pred[:, COL2K[c]].sum()) for c in SUBMIT}
    expected = {"c": 975, "na": 947, "i": 80, "bc": 15, "t": 528}
    print(f"[r4-dual-v2 sanity] S5 复现 pos: {s5_pos}", file=sys.stderr)
    assert s5_pos == expected, f"S5 sanity FAILED: got {s5_pos}, expected {expected}"
    print(f"[r4-dual-v2 sanity] ✓ S5 精确复现 = R4 baseline 0.7458", file=sys.stderr)

    # ====== D1: dual-route ctx, V2 同截短规则 (seg_id mod 2 == 0 → 用 ctx_short, else ctx_long) ======
    ctx_dual = np.empty_like(ctx_long)
    route = {"long": 0, "short": 0}
    for i, sid in enumerate(seg_ids):
        if int(sid) % 2 == 0:
            ctx_dual[i] = ctx_short_v2[i]   # V2 截短 + mask050 推理
            route["short"] += 1
        else:
            ctx_dual[i] = ctx_long[i]       # 全 30s + baseline 推理
            route["long"] += 1
    print(f"[r4-dual-v2] D1 dual route: {route}", file=sys.stderr)

    s5_dual = build_s5(ctx_dual, wsp_te, hub_te, wsp_ms_te, e2v_ms_te, hub_ms_te, omni3b_te)

    # ====== D2 也写出 (sanity csv, 等同 R4 baseline) ======
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_root = Path(args.out_dir) if args.out_dir else ROOT / f"submission/probe-day9-r4dualv2-{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    pos_D2 = write_csv(s5_sanity, seg_ids,
                       out_root / "D2_R4_dual_30s_sanity/pred_test1.csv",
                       "R4 全栈 sanity (ctx 全用 long) = R4 baseline 0.7458")
    pos_D1 = write_csv(s5_dual, seg_ids,
                       out_root / "D1_R4_dual_half_truncated/pred_test1.csv",
                       "R4 全栈 + dual-route ctx (V2 截短规则, ctx_long/ctx_short_v2 混合)")

    # D3: ctx 全用 mask050 (全 30s 推理) — 不路由, 跟 V1=0.7108 (cycle1 baseline) 对照看 mask050 在 30s 是涨是跌
    s5_all_mask = build_s5(ctx_short_full, wsp_te, hub_te, wsp_ms_te, e2v_ms_te, hub_ms_te, omni3b_te)
    pos_D3 = write_csv(s5_all_mask, seg_ids,
                       out_root / "D3_R4_all_mask050_30s/pred_test1.csv",
                       "R4 全栈 + 全用 mask050 ctx (全 30s 推理, 不路由, 测 mask050 在 30s 上 ctx 全栈下表现)")

    manifest = {
        "_note": "A3 先验 V2: R4 全栈 + variant-F mask050 dual-route ctx (严格同纲).",
        "_chain": "V1 ctx-only single = 0.710789 → V2 ctx-only dual = 0.720935 (+0.010) → ?",
        "_strict_align": "ctx_long = variant-F 5 seed baseline (te_lgbm_v1, R4 baseline 用的), ctx_short = variant-F 5 seed mask050 按 V2 截短规则推理 (test_v2 字段). 同纲.",
        "_anchor": {
            "S5_baseline_real": 0.747131,
            "V1_ctx_only_real": 0.710789,
            "V2_ctx_only_real": 0.720935,
        },
        "candidates": [
            {
                "name": "D2_R4_dual_30s_sanity",
                "description": "R4 全栈 sanity (ctx 全 long, 不路由)",
                "expected": "≈ R4 baseline 0.7458 (验工件一致性)",
                "pos": pos_D2,
            },
            {
                "name": "D1_R4_dual_half_truncated",
                "description": "R4 全栈 + dual-route ctx (V2 截短规则)",
                "hypothesis": "ctx-only V2 涨 +0.010, 经 softadd 进 R4 全栈估算 +0.005 ~ +0.015",
                "key_signal": "若 ≥ 0.7458 → dual-route 进复赛镜像; 若 < 0.740 → 反向, 弃 dual-route",
                "expected": "0.737 ~ 0.760 区间过宽, 公榜定",
                "route": route,
                "pos": pos_D1,
            },
            {
                "name": "D3_R4_all_mask050_30s",
                "description": "R4 全栈 + 全用 mask050 ctx (全 30s 推理, 不路由)",
                "hypothesis": "测 mask050 在不路由情况下对 R4 全栈的影响 (单 ckpt 替换 baseline)",
                "key_signal": "跟 D1 比 — 若 D3 ≈ D1 路由价值 ≈ 0 (mask050 直接换 baseline 够); 若 D3 < D1 路由价值真实存在",
                "pos": pos_D3,
            },
        ],
        "_reference": {
            "R4_baseline_10s_anchor": 0.721787,
            "M2_R4_mask050_10s_real": 0.737580,
        },
        "_created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[r4-dual-v2] MANIFEST → {out_root}/MANIFEST.json", file=sys.stderr)


if __name__ == "__main__":
    main()
