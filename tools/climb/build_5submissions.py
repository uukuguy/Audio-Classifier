"""6/2 5 push 真分校验脚本 — 构造 5 个不同融合策略的 submission csv.

用户指示: 不要再用 cap1 红旗筛选, 直接 push 出真分数据, 让真分校准 D-17/D-19/D-20 规则.

5 个候选 (按多样性):

  1. orthofuse-3src + Omni (4src 等权融合 BC/T/I) — 验 Omni 加进融合是否反挫 (D-20 预测会)
  2. orthofuse-3src + Omni 加权小 (w_omni=0.2) — 软加 Omni
  3. orthofuse-3src + w2v2_bcaug (4src 等权, BC=ctx 守 varF) — 验 w2v2 进融合不取 BC 列
  4. Omni 单源直接 push — 验 cap1=0.5649 单源到底真分多少
  5. 4 个 BC 增强 head 等权 (whisper_bcaug + hubert_bcaug + w2v2_bcaug + e2v_bcaug) — 全 BC 增强源融合

每个候选输出 pred_test1.csv 到 tools/runs/climb/probe-5push-YYYYMMDD-HHMM/cand_N/.
用户手动选 1-5 push, 贴回真分让我们校准.

Usage:
  python3 tools/climb/build_5submissions.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}

# 已有 probs.npz 路径
PATHS = {
    "ctx_cache": "tools/runs/climb/_stack_cache_s40.npz",
    "whisper":   "tools/runs/climb/whisper-fusion-20260531-0143/probs.npz",
    "hubert":    "tools/runs/climb/hubert-bcaug-head-20260601-1533/probs.npz",
    "w2v2":      "tools/runs/climb/w2v2-bcaug-head-20260601-1926/probs.npz",
    "e2v":       "tools/runs/climb/e2v-bcaug-head-20260601-1755/probs.npz",
    "whisper_bcaug": "tools/runs/climb/whisper-bcaug-head-20260601-1730/probs.npz",
    "omni":      "tools/runs/climb/omni-lora-20260602-1002/probs.npz",
}

RUN_DIR = Path(f"tools/runs/climb/probe-5push-{time.strftime('%Y%m%d-%H%M')}")


def load_test(path):
    """加载 test probs, 形状 (1000, 5)."""
    z = np.load(path)
    return z["test"]


def load_ctx_test():
    """ctx_cache 没有 test 字段直接给, 用 te_lgbm_v1."""
    z = np.load(PATHS["ctx_cache"])
    return z["te_lgbm_v1"]


def get_test_ids():
    """test_ids 从 hubert_bcaug 拿 (跟所有 bcaug head 同源)."""
    z = np.load(PATHS["hubert"])
    return z["test_ids"]


def make_csv(probs, test_ids, thresholds, out_path: Path, desc: str):
    """probs: (1000, 5) labels: C/T/BC/I/NA."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos_counts = {c: 0 for c in SUBMIT}
    with open(out_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(probs[i, COL2K[c]] >= thresholds[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items():
                pos_counts[c] += v
            row = [sid] + [str(vals[c]) for c in SUBMIT]
            f.write(",".join(row) + "\n")
    print(f"  {out_path}: " + " ".join(f"{c}={pos_counts[c]}" for c in SUBMIT))
    return pos_counts


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[push] 6/2 5 push 候选构造 → {RUN_DIR}", file=sys.stderr)

    test_ids = get_test_ids()
    ctx_t = load_ctx_test()
    wsp_t = load_test(PATHS["whisper"])
    hub_t = load_test(PATHS["hubert"])
    w2v_t = load_test(PATHS["w2v2"])
    e2v_t = load_test(PATHS["e2v"])
    wsb_t = load_test(PATHS["whisper_bcaug"])
    omni_t = load_test(PATHS["omni"])
    print(f"[push] 加载完成. test 1000 段, 7 个源.", file=sys.stderr)

    # SOTA 策略 (从 orthofuse-3src cv_metrics):
    # C=ctx  T=whisper_hubert_70 (0.7w+0.3h)  BC=ctx  I=ctx_whisper_hubert_eq  NA=ctx
    def make_sota_probs(ctx, wsp, hub):
        p = np.zeros_like(ctx)
        p[:, 0] = ctx[:, 0]                              # C
        p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * hub[:, 1]      # T
        p[:, 2] = ctx[:, 2]                              # BC
        p[:, 3] = (ctx[:, 3] + wsp[:, 3] + hub[:, 3]) / 3  # I
        p[:, 4] = ctx[:, 4]                              # NA
        return p

    candidates = {}
    print("\n=== 候选 1: SOTA + Omni 软加 (BC/T/I 加 Omni 等权, NA/C 保持 SOTA) ===")
    sota = make_sota_probs(ctx_t, wsp_t, hub_t)
    p1 = sota.copy()
    p1[:, 1] = (sota[:, 1] + omni_t[:, 1]) / 2  # T 加 omni
    p1[:, 2] = (sota[:, 2] + omni_t[:, 2]) / 2  # BC 加 omni
    p1[:, 3] = (sota[:, 3] + omni_t[:, 3]) / 2  # I 加 omni
    pos = make_csv(p1, test_ids, THR_VARF, RUN_DIR / "cand1_sota_plus_omni_half" / "pred_test1.csv",
                   "SOTA+Omni T/BC/I 0.5/0.5 软融")
    candidates["cand1_sota_plus_omni_half"] = {"strat": "SOTA T/BC/I 0.5+omni 0.5", "pos": pos}

    print("\n=== 候选 2: SOTA + Omni 极小权重 (0.8 SOTA + 0.2 Omni, T/BC/I) ===")
    p2 = sota.copy()
    p2[:, 1] = 0.8 * sota[:, 1] + 0.2 * omni_t[:, 1]
    p2[:, 2] = 0.8 * sota[:, 2] + 0.2 * omni_t[:, 2]
    p2[:, 3] = 0.8 * sota[:, 3] + 0.2 * omni_t[:, 3]
    pos = make_csv(p2, test_ids, THR_VARF, RUN_DIR / "cand2_sota_omni_02" / "pred_test1.csv",
                   "SOTA+Omni 0.8/0.2 软加")
    candidates["cand2_sota_omni_02"] = {"strat": "SOTA T/BC/I 0.8+omni 0.2", "pos": pos}

    print("\n=== 候选 3: SOTA + w2v2_bcaug 进 T/I (BC 守 SOTA ctx) ===")
    p3 = sota.copy()
    p3[:, 1] = (sota[:, 1] + w2v_t[:, 1]) / 2  # T 加 w2v2
    p3[:, 3] = (sota[:, 3] + w2v_t[:, 3]) / 2  # I 加 w2v2
    # BC 仍是 ctx (守 D-18/D-19)
    pos = make_csv(p3, test_ids, THR_VARF, RUN_DIR / "cand3_sota_plus_w2v2_TI" / "pred_test1.csv",
                   "SOTA + w2v2 T/I 0.5+0.5")
    candidates["cand3_sota_plus_w2v2_TI"] = {"strat": "SOTA + w2v2 T/I 0.5", "pos": pos}

    print("\n=== 候选 4: Omni 单源直接 push ===")
    pos = make_csv(omni_t, test_ids, THR_VARF, RUN_DIR / "cand4_omni_only" / "pred_test1.csv",
                   "Omni 单源 varF")
    candidates["cand4_omni_only"] = {"strat": "Omni single", "pos": pos}

    print("\n=== 候选 5: 4 个 BC 增强源等权融合 (no ctx) ===")
    p5 = (hub_t + w2v_t + e2v_t + wsb_t) / 4
    pos = make_csv(p5, test_ids, THR_VARF, RUN_DIR / "cand5_4bcaug_eq" / "pred_test1.csv",
                   "4 BC-aug heads 等权 (hub+w2v2+e2v+wsb)")
    candidates["cand5_4bcaug_eq"] = {"strat": "(hub+w2v2+e2v+wsb)/4", "pos": pos}

    # 落 manifest
    sota_pos = {"c": 975, "na": 947, "i": 81, "bc": 27, "t": 522}
    (RUN_DIR / "MANIFEST.json").write_text(json.dumps({
        "sota_csv_baseline_pos": sota_pos,
        "sota_real_score": 0.71755,
        "candidates": candidates,
        "_note": "用户指示: 不再用 cap1 红旗筛, 直接 push 真分校准规则. "
                 "选 1-5 push, 贴回真分.",
    }, ensure_ascii=False, indent=2))
    print(f"\n=== MANIFEST ===")
    print(f"  {RUN_DIR}/MANIFEST.json")
    print(f"\n=== 5 个候选 csv 已生成, 路径 ===")
    for cand in sorted(candidates.keys()):
        print(f"  {RUN_DIR}/{cand}/pred_test1.csv")


if __name__ == "__main__":
    main()
