"""6/6 day8 候选 — D-27 战略后第一日, 1-2 push 拿信息.

S5 (R4 + omni3b_ms2 0.05) = 0.7471 NEW SOTA 后, 探两条复赛准备相关假设:

  P1 = S5 + wsp_ms 0.10  → 验证 Omni-3B 与 wsp_ms 是否跨范式正交协同
                            (S4 wsp_ms 0.10 单独 = 0.7395 vs SOTA-3src;
                             S5 已含 wsp_ms 0.07. 这是 0.07→0.10 升 + Omni 协同)
                            ↑ 期望: 0.747-0.751, 若涨说明 wsp 与 Omni-3B 不同质

  P2 = R4 + omni3b_ms2 0.10  → Omni-3B 权重曲线右探 (S5=0.05 是否是峰)
                                 ↑ 期望: 0.745-0.749, 若降说明 S5 是峰, 若涨说明还能扩

依赖 build_truncated_r4 的 R4 配方 (NSOTA07 + e2v 0.03 + hub 0.03), 但 ctx 用 full 375 chunk
(非截短, 这是公榜冲分不是变长验证).

Usage:
  python3 tools/climb/build_day8_candidates.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}

PATHS = {
    "orthofuse_3src": "tools/runs/climb/orthofuse-3src-20260601-1607/fused_probs.npz",
    "wsp_ms": "tools/runs/climb/whisper-bcaug-multiseed-20260602-1704/probs.npz",
    "e2v_ms": "tools/runs/climb/e2v-bcaug-multiseed-20260602-1632/probs.npz",
    "hub_ms": "tools/runs/climb/hubert-bcaug-multiseed-20260602-1506/probs.npz",
    "omni3b_ms2": "tools/runs/climb/omni-3b-ms2-mean-3seed/probs.npz",
    "omni7b_ms2": "tools/runs/climb/omni-7b-ms2-mean-3seed/probs.npz",
}


def softadd(base: np.ndarray, src: np.ndarray, w: float, cols: tuple) -> np.ndarray:
    out = base.copy()
    for c in cols:
        out[:, c] = (1 - w) * base[:, c] + w * src[:, c]
    return out


def make_sota_3src(ctx, wsp, hub):
    """orthofuse-3src 配置 — 跟 cycle_orthofuse 一致."""
    p = np.zeros_like(ctx)
    p[:, 0] = ctx[:, 0]
    p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * hub[:, 1]
    p[:, 2] = ctx[:, 2]
    p[:, 3] = (ctx[:, 3] + wsp[:, 3] + hub[:, 3]) / 3
    p[:, 4] = ctx[:, 4]
    return p


def build_r4_test(ctx_te, wsp_te, hub_te, wsp_ms_te, e2v_ms_te, hub_ms_te):
    """R4 配方: SOTA-3src → NSOTA07 → +e2v_ms 0.03 → +hub_ms 0.03."""
    sota = make_sota_3src(ctx_te, wsp_te, hub_te)
    nsota_07 = softadd(sota, wsp_ms_te, 0.07, (1, 2, 3))
    r4 = softadd(softadd(nsota_07, e2v_ms_te, 0.03, (1, 2, 3)), hub_ms_te, 0.03, (1, 2, 3))
    return r4


def write_csv(probs, seg_ids, out_path: Path, desc: str) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.zeros((len(seg_ids), 5), dtype=int)
    for k in range(5):
        pred[:, k] = (probs[:, k] >= THR_VARF[k]).astype(int)
    df = pd.DataFrame({"segment_id": seg_ids})
    for c in SUBMIT_COLS:
        df[c] = pred[:, COL2K[c]]
    df.to_csv(out_path, index=False)
    pos = {c: int(df[c].sum()) for c in SUBMIT_COLS}
    print(f"  ✓ {out_path}")
    print(f"    pos: {pos}")
    return pos


def main():
    print("=== day8 候选生成 (D-27 战略, 1-2 push) ===\n", file=sys.stderr)

    # 1. 加载 SOTA-3src fused probs (含 ctx_te wsp_te hub_te; seg_ids 不在这里)
    z3 = np.load(PATHS["orthofuse_3src"])
    ctx_te = z3["ctx_te"]
    wsp_te = z3["whisper_te"]
    hub_te = z3["hubert_te"]
    print(f"loaded sota-3src: ctx={ctx_te.shape} wsp={wsp_te.shape} hub={hub_te.shape}",
          file=sys.stderr)

    # 2. 加载 multi-seed test probs + seg_ids (从 wsp_ms 拿)
    z_wsp = np.load(PATHS["wsp_ms"])
    wsp_ms_te = z_wsp["test"]
    seg_ids = z_wsp["test_ids"]
    e2v_ms_te = np.load(PATHS["e2v_ms"])["test"]
    hub_ms_te = np.load(PATHS["hub_ms"])["test"]
    omni3b_ms2_te = np.load(PATHS["omni3b_ms2"])["test"]
    omni7b_ms2_te = np.load(PATHS["omni7b_ms2"])["test"]
    print(f"loaded ms sources: wsp_ms={wsp_ms_te.shape} e2v_ms={e2v_ms_te.shape} "
          f"hub_ms={hub_ms_te.shape} omni3b_ms2={omni3b_ms2_te.shape}", file=sys.stderr)

    # 3. R4 (NSOTA07 + e2v 0.03 + hub 0.03)
    r4 = build_r4_test(ctx_te, wsp_te, hub_te, wsp_ms_te, e2v_ms_te, hub_ms_te)
    print(f"\nR4 base built ✓", file=sys.stderr)

    # 4. S5 = R4 + omni3b_ms2 0.05 (重生, 验证跟昨天 csv 一致)
    s5 = softadd(r4, omni3b_ms2_te, 0.05, (1, 2, 3))

    # 5. 出 day8 候选
    out_root = Path(f"submission/probe-day8-{time.strftime('%Y%m%d-%H%M')}")
    candidates = []

    # ===== P1: S5 + wsp_ms 0.10 =====
    # 假设: Omni-3B 与 wsp_ms 跨范式正交, 协同 +0.002-0.004
    # 风险: wsp_ms 在 R4 中已是 0.07, 再叠 0.10 可能过载
    s5_plus_wsp_010 = softadd(s5, wsp_ms_te, 0.10, (1, 2, 3))
    pos_p1 = write_csv(s5_plus_wsp_010, seg_ids,
                       out_root / "P1_S5+wsp_ms_010" / "pred_test1.csv",
                       "S5 + wsp_ms 0.10 (Omni × wsp 协同测试)")
    candidates.append({
        "name": "P1_S5+wsp_ms_010",
        "description": "S5 (R4+omni3b_ms2 0.05) + wsp_ms 0.10",
        "hypothesis": "Omni-3B-ms2 与 wsp_ms 跨范式正交协同",
        "expected": "0.747-0.751",
        "validates": "若涨 +0.002-0.004 → 多模态 LLM 路线与 SSL ms 路线不同质, 复赛镜像应叠加",
        "if_fail": "若 < 0.747 → wsp_ms 0.07 已饱和, 复赛镜像锁 S5 配方",
        "pos": pos_p1,
    })

    # ===== P2: R4 + omni3b_ms2 0.10 =====
    # 假设: Omni-3B 权重 0.05→0.10 = D-23 wsp_ms 0.05→0.10 同向单调?
    # 风险: 0.10 可能过载, 跟 wsp_ms 0.10 vs 0.07 几乎平台对应 (S4 vs R5 +0.0006)
    r4_omni3b_010 = softadd(r4, omni3b_ms2_te, 0.10, (1, 2, 3))
    pos_p2 = write_csv(r4_omni3b_010, seg_ids,
                       out_root / "P2_R4+omni3b_ms2_010" / "pred_test1.csv",
                       "R4 + omni3b_ms2 0.10 (omni3b 权重曲线右探)")
    candidates.append({
        "name": "P2_R4+omni3b_ms2_010",
        "description": "R4 + omni3b_ms2 0.10",
        "hypothesis": "Omni-3B 权重 0.05→0.10 仍升 (跟 D-23 wsp_ms 同模式)",
        "expected": "0.745-0.749",
        "validates": "若涨 → omni3b 0.05 不是峰, 0.10 是; 若降 → S5 0.05 是峰",
        "if_fail": "复赛镜像锁 omni3b_ms2 0.05 不试更高",
        "pos": pos_p2,
    })

    # ===== P3: S5 + e2v_ms 0.05 (验 Omni 是否已覆盖 e2v 信号) =====
    # R4 已含 e2v_ms 0.03, S5 在 R4 上叠 Omni-3B. 再加 e2v_ms 0.05 = e2v 总权 0.08
    # 假设: Omni-3B 已经把 e2v 信号覆盖了 (多模态 LLM 通常含语音情绪/句法), 加 e2v 微降
    # 反假设: Omni 不覆盖 e2v 局部信号, 协同 +0.001-0.003
    s5_plus_e2v_005 = softadd(s5, e2v_ms_te, 0.05, (1, 2, 3))
    pos_p3 = write_csv(s5_plus_e2v_005, seg_ids,
                       out_root / "P3_S5+e2v_ms_005" / "pred_test1.csv",
                       "S5 + e2v_ms 0.05 (Omni 是否覆盖 e2v 信号)")
    candidates.append({
        "name": "P3_S5+e2v_ms_005",
        "description": "S5 + e2v_ms 0.05 (e2v 总权 0.08)",
        "hypothesis": "Omni-3B 与 e2v_ms 信号正交 vs 同质",
        "expected": "0.744-0.749",
        "validates": "若涨 → 复赛镜像可叠 e2v 升权; 若降 → Omni 已覆盖 e2v, e2v_ms 0.03 是天花板",
        "if_fail": "锁 R4 e2v 0.03 不上探",
        "pos": pos_p3,
    })

    # ===== P4: NSOTA07 + omni3b_ms2 0.05 (跳过双 SSL_ms 直接 Omni) =====
    # NSOTA07 = SOTA-3src + wsp_ms 0.07, 没有 e2v_ms 也没有 hub_ms (R4 比它多两层)
    # 假设: 双 SSL_ms 0.03+0.03 = +0.0069 (R4 vs NSOTA07), Omni-3B 0.05 = +0.0013 (S5 vs R4)
    #       若 NSOTA07+omni 直接打到 0.74+, 说明 Omni 可以替代双 SSL 的部分工作
    #       → 复赛镜像可简化 (跳过 e2v/hub multi-seed 训练, 只训 Omni-3B)
    sota = make_sota_3src(ctx_te, wsp_te, hub_te)
    nsota_07 = softadd(sota, wsp_ms_te, 0.07, (1, 2, 3))
    nsota07_plus_omni3b = softadd(nsota_07, omni3b_ms2_te, 0.05, (1, 2, 3))
    pos_p4 = write_csv(nsota07_plus_omni3b, seg_ids,
                       out_root / "P4_NSOTA07+omni3b_ms2_005" / "pred_test1.csv",
                       "NSOTA07 + omni3b_ms2 0.05 (跳过双 SSL_ms 直 Omni)")
    candidates.append({
        "name": "P4_NSOTA07+omni3b_ms2_005",
        "description": "NSOTA07 + omni3b_ms2 0.05 (无 e2v/hub multi-seed)",
        "hypothesis": "Omni-3B 是否能替代双 SSL_ms 0.03+0.03 的部分工作",
        "expected": "0.740-0.747",
        "validates": "若打到 0.744+ → 复赛镜像可简化 (跳过 e2v/hub multi-seed 训练 60h)",
        "if_fail": "若 < 0.740 → Omni 不可替代 SSL_ms, 复赛镜像必须保留双 SSL_ms",
        "pos": pos_p4,
    })

    # ===== P5: R4 + omni7b_ms2 0.05 (8B 超额但拿信息: 7B vs 3B 多模态能力差距) =====
    # ⚠ 8B 合规超额 (R4 1.7B + Omni-7B 7B = 8.7B), 真分仅作数据点, 不用于复赛
    # 假设: Omni-7B 多模态能力 > Omni-3B, S5(3B)=0.7471, 7B 版本应更高
    # 用途: 验证"复赛镜像选 3B 而非 7B 损失了多少分", 答辩素材
    r4_omni7b_005 = softadd(r4, omni7b_ms2_te, 0.05, (1, 2, 3))
    pos_p5 = write_csv(r4_omni7b_005, seg_ids,
                       out_root / "P5_R4+omni7b_ms2_005" / "pred_test1.csv",
                       "R4 + omni7b_ms2 0.05 (8B 超额, 7B vs 3B 对照)")
    candidates.append({
        "name": "P5_R4+omni7b_ms2_005",
        "description": "R4 + omni7b_ms2 0.05 (8B 超额 ~8.7B)",
        "hypothesis": "Omni-7B 多模态能力 > Omni-3B, 真分 > S5 0.7471",
        "expected": "0.748-0.753",
        "validates": "拿 7B vs 3B 差距数据点, 答辩可讲 '我们选 3B 牺牲 X 分换合规'",
        "if_fail": "复赛镜像方案不变 (8B 合规硬约束 > 边际涨分)",
        "pos": pos_p5,
        "note": "⚠ 8B 超额, 仅作信息收集, 不用于复赛镜像",
    })

    # 6. MANIFEST
    manifest = {
        "_note": "6/6 day8 D-27 战略后第 1 日. 1-2 push, 拿信息为主.",
        "_sota": "S5 R4+omni3b_ms2_005 真分 0.747131 (6/5 NEW SOTA, 8B 合规 5B 总参)",
        "_strategy": "复赛准备压倒, 公榜投只为验证假设; 不投纯权重叠加",
        "candidates": candidates,
    }
    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"\n=== day8 候选 ({len(candidates)}) → {out_root} ===")
    for c in candidates:
        print(f"\n  {c['name']}: {c['description']}")
        print(f"    期望: {c['expected']}")
        print(f"    验证: {c['validates']}")
        print(f"    pos:  {c['pos']}")


if __name__ == "__main__":
    main()
