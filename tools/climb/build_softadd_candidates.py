"""6/3 软加融合候选大扫 — 基于 cand2 (SOTA+Omni 0.2 = 0.728524) 真分.

用户指示 (2026-06-02):
  - 不再硬筛 cap1 红旗, 多推几个 push 拿真分
  - base 不一定是 orthofuse-3src 0.71755, 任何子集都可能更好
  - 8B 参数总和软约束 (复赛镜像装 Omni 实际超 = 心理有数)

5 push (明日 6/3 配额) 候选方向:

  A. Omni 权重扫: 0.10/0.15/0.20★/0.25/0.30 (找最优软加权)
  B. 其他源软加 0.2: qwen3/w2v2/e2v/whisper_bcaug/hubert_bcaug 各一
  C. 多源叠加: SOTA + Omni 0.2 + qwen3 0.1 + e2v 0.1 等
  D. 替换 base: orthofuse-4src+Omni 0.2 (验 base 多样)

本脚本生成 ~20 个候选 csv, 用户选 5 个 push.
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

PATHS = {
    "ctx_cache":     "tools/runs/climb/_stack_cache_s40.npz",
    "whisper":       "tools/runs/climb/whisper-fusion-20260531-0143/probs.npz",
    "hubert":        "tools/runs/climb/hubert-bcaug-head-20260601-1533/probs.npz",
    "w2v2":          "tools/runs/climb/w2v2-bcaug-head-20260601-1926/probs.npz",
    "e2v":           "tools/runs/climb/e2v-bcaug-head-20260601-1755/probs.npz",
    "whisper_bcaug": "tools/runs/climb/whisper-bcaug-head-20260601-1730/probs.npz",
    "qwen3":         "tools/runs/climb/qwen3-head-20260601-1514/probs.npz",
    "omni":          "tools/runs/climb/omni-lora-20260602-1002/probs.npz",
}

RUN_DIR = Path(f"tools/runs/climb/probe-softadd-{time.strftime('%Y%m%d-%H%M')}")


def load_test(path):
    z = np.load(path)
    return z["test"]


def load_ctx_test():
    z = np.load(PATHS["ctx_cache"])
    return z["te_lgbm_v1"]


def get_test_ids():
    z = np.load(PATHS["hubert"])
    return z["test_ids"]


def make_csv(probs, test_ids, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = {c: 0 for c in SUBMIT}
    with open(out_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(probs[i, COL2K[c]] >= THR_VARF[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items():
                pos[c] += v
            f.write(",".join([sid] + [str(vals[c]) for c in SUBMIT]) + "\n")
    return pos


def make_sota_3src(ctx, wsp, hub):
    """orthofuse-3src 真分 0.71755 配置."""
    p = np.zeros_like(ctx)
    p[:, 0] = ctx[:, 0]
    p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * hub[:, 1]
    p[:, 2] = ctx[:, 2]
    p[:, 3] = (ctx[:, 3] + wsp[:, 3] + hub[:, 3]) / 3
    p[:, 4] = ctx[:, 4]
    return p


def softadd(base, extra, w=0.2, cols=(1, 2, 3)):
    """base * (1-w) + extra * w on specified cols."""
    p = base.copy()
    for k in cols:
        p[:, k] = (1 - w) * base[:, k] + w * extra[:, k]
    return p


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[softadd] 候选构造 → {RUN_DIR}", file=sys.stderr)

    test_ids = get_test_ids()
    ctx_t = load_ctx_test()
    wsp_t = load_test(PATHS["whisper"])
    hub_t = load_test(PATHS["hubert"])
    w2v_t = load_test(PATHS["w2v2"])
    e2v_t = load_test(PATHS["e2v"])
    wsb_t = load_test(PATHS["whisper_bcaug"])
    qwen_t = load_test(PATHS["qwen3"])
    omni_t = load_test(PATHS["omni"])

    sota = make_sota_3src(ctx_t, wsp_t, hub_t)

    candidates = []

    # === A. Omni 权重扫 (寻找最优软加权区间) ===
    for w in [0.10, 0.15, 0.25, 0.30]:  # 0.20 已知 = 0.728524
        p = softadd(sota, omni_t, w, cols=(1, 2, 3))  # T/BC/I
        name = f"A_sota+omni{int(w*100):03d}"
        pos = make_csv(p, test_ids, RUN_DIR / name / "pred_test1.csv")
        candidates.append((name, f"SOTA + Omni {w} on T/BC/I", pos))

    # === B. 其他单源 0.2 软加 (同 cand2 模式换源) ===
    for name_short, src in [("qwen3", qwen_t), ("w2v2", w2v_t), ("e2v", e2v_t),
                            ("wsb", wsb_t)]:
        p = softadd(sota, src, 0.2, cols=(1, 2, 3))
        name = f"B_sota+{name_short}020"
        pos = make_csv(p, test_ids, RUN_DIR / name / "pred_test1.csv")
        candidates.append((name, f"SOTA + {name_short} 0.2 on T/BC/I", pos))

    # === C. 多源软加 (Omni 0.2 + 其他 0.1) ===
    # C1: SOTA + Omni 0.2 + qwen3 0.1
    p = softadd(sota, omni_t, 0.2, cols=(1, 2, 3))
    p = softadd(p, qwen_t, 0.1, cols=(1, 3))  # qwen3 BC=0, 只融 T/I
    pos = make_csv(p, test_ids, RUN_DIR / "C1_omni020_qwen010" / "pred_test1.csv")
    candidates.append(("C1_omni020_qwen010", "SOTA + Omni 0.2 + qwen3 0.1 (T/I)", pos))

    # C2: SOTA + Omni 0.2 + e2v 0.1
    p = softadd(sota, omni_t, 0.2, cols=(1, 2, 3))
    p = softadd(p, e2v_t, 0.1, cols=(1, 2, 3))
    pos = make_csv(p, test_ids, RUN_DIR / "C2_omni020_e2v010" / "pred_test1.csv")
    candidates.append(("C2_omni020_e2v010", "SOTA + Omni 0.2 + e2v 0.1", pos))

    # C3: SOTA + Omni 0.2 + qwen3 0.1 + e2v 0.1 (三新源)
    p = softadd(sota, omni_t, 0.2, cols=(1, 2, 3))
    p = softadd(p, qwen_t, 0.1, cols=(1, 3))
    p = softadd(p, e2v_t, 0.1, cols=(1, 2, 3))
    pos = make_csv(p, test_ids, RUN_DIR / "C3_omni020_qwen010_e2v010" / "pred_test1.csv")
    candidates.append(("C3_omni020_qwen010_e2v010", "SOTA + 3 新源软加 (Omni+qwen+e2v)", pos))

    # C4: SOTA + Omni 0.2 + w2v2 0.1 (T/I only, BC 守 SOTA)
    p = softadd(sota, omni_t, 0.2, cols=(1, 2, 3))
    p = softadd(p, w2v_t, 0.1, cols=(1, 3))
    pos = make_csv(p, test_ids, RUN_DIR / "C4_omni020_w2v010_TI" / "pred_test1.csv")
    candidates.append(("C4_omni020_w2v010_TI", "SOTA + Omni 0.2 + w2v2 0.1 (T/I only)", pos))

    # === D. 替换 base (验非 0.71755 base 是否更好) ===
    # D1: ctx+whisper 双源 base + Omni 0.2 (去掉 hubert)
    sota_2src = ctx_t.copy()
    sota_2src[:, 1] = 0.7 * wsp_t[:, 1] + 0.3 * sota_2src[:, 1]  # T 用 ctx*0.3+whisper*0.7
    sota_2src[:, 3] = (ctx_t[:, 3] + wsp_t[:, 3]) / 2  # I 双源均值
    p = softadd(sota_2src, omni_t, 0.2, cols=(1, 2, 3))
    pos = make_csv(p, test_ids, RUN_DIR / "D1_ctx_whisper_base_omni020" / "pred_test1.csv")
    candidates.append(("D1_ctx_whisper_base_omni020", "ctx+whisper base + Omni 0.2", pos))

    # D2: ctx 单源 base + Omni 0.3 重加 (因为 base 弱所以软加权要大)
    p = softadd(ctx_t, omni_t, 0.3, cols=(1, 2, 3))
    pos = make_csv(p, test_ids, RUN_DIR / "D2_ctx_base_omni030" / "pred_test1.csv")
    candidates.append(("D2_ctx_base_omni030", "ctx base + Omni 0.3", pos))

    # === 打印 + manifest ===
    print(f"\n=== 候选 csv 生成 ({len(candidates)} 个) ===")
    sota_pos = {"c": 975, "na": 947, "i": 81, "bc": 27, "t": 522}
    print(f"参考: SOTA orthofuse-3src pos = {sota_pos}")
    print(f"参考: cand2 (yesterday 0.728524) pos = c=975 na=947 i=77 bc=20 t=531")
    print()
    for name, desc, pos in candidates:
        print(f"  {name}: {desc}")
        print(f"    pos = {pos}")

    (RUN_DIR / "MANIFEST.json").write_text(json.dumps({
        "_note": "明日 6/3 5 push 候选. 不再 cap1 筛选, 直接拿真分校准.",
        "yesterday_results": {
            "cand2_sota+omni020": 0.72852,
            "cand1_sota+omni050": 0.69094,
            "cand3_sota+w2v2_TI": 0.71452,
            "cand4_omni_only": 0.61305,
            "cand5_4bcaug_eq": 0.60734,
        },
        "candidates": [
            {"name": n, "desc": d, "pos": pos} for n, d, pos in candidates
        ],
    }, ensure_ascii=False, indent=2))
    print(f"\nMANIFEST: {RUN_DIR}/MANIFEST.json")


if __name__ == "__main__":
    main()
