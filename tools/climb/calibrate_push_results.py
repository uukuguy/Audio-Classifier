"""6/3 5 push 真分回来后跑: OOF vs 真分 calibration 分析.

输入: 用户贴的真分 (apply-lb-score 后 runs.csv 已落)
输出:
  - 5 push 的 (OOF cap1, 真分, gap) 矩阵
  - 维度级 gap (A Omni 权重 / B multi-seed / C 多源叠加)
  - 哪些 OOF 维度跟真分同向 / 反向
  - 明日 6/4 5 push 策略建议 (按真分排序而非 OOF)

Usage:
  # 先 apply-lb-score 把真分写进 runs.csv, 然后:
  python3 tools/climb/calibrate_push_results.py --date 2026-06-03
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

# 6/3 用户实际上传顺序 (vs 我原推荐, Push 3 被替换为 A_omni_w015, Push 4 替为 PF_omni_max, Push 5 替为 Omni-3B 合规)
PUSH_MAP = {
    1: ("A_omni_w010_TBCI", "A: Omni-7B 0.10"),
    2: ("C_omni020+e2v_ms_w010", "C: 多源叠加 Omni0.2+e2v_ms0.1"),
    3: ("A_omni_w015_TBCI", "A: Omni-7B 0.15"),
    4: ("PF_omni_max_w020", "D: per-fold max"),
    5: ("A3B_omni3b_w015_TBCI", "A3B: Omni-3B 0.15 (合规)"),
}

# 历史已知锚点 (6/2 实测 + 5/31 baseline)
HISTORICAL = [
    {"name": "orthofuse_3src", "real": 0.71755, "oof": 0.6532, "config": "SOTA-3src baseline"},
    {"name": "cand2_omni_0.20", "real": 0.72852, "oof": 0.6503, "config": "SOTA+Omni 0.2 T/BC/I"},
    {"name": "cand1_omni_0.50", "real": 0.69094, "oof": None, "config": "SOTA+Omni 0.5 (过载)"},
    {"name": "cand3_w2v2_TI_0.5", "real": 0.71452, "oof": None, "config": "SOTA+w2v2 0.5 T/I"},
    {"name": "cand4_omni_only", "real": 0.61305, "oof": None, "config": "Omni 单源"},
    {"name": "cand5_4bcaug_eq", "real": 0.60734, "oof": None, "config": "4 BC-aug 等权"},
]


def find_latest_v2():
    runs = sorted(Path("tools/runs/climb").glob("probe-softadd-v2-*"))
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-03", help="push 日期 (用于 log)")
    ap.add_argument("--scores", nargs="*", default=[],
                    help="真分 list 顺序对应 push 1-5, 如 --scores 0.7285 0.7290 ...")
    args = ap.parse_args()

    rd = find_latest_v2()
    if rd is None:
        print("ERROR: 找不到 probe-softadd-v2-* 目录"); return
    mf = json.load(open(rd / "MANIFEST.json"))
    cands = {c["name"]: c for c in mf["candidates"]}
    sota_oof = mf["sota_3src_oof_macro"]

    print(f"=== Calibration 报告 ({args.date}) ===")
    print(f"SOTA-3src OOF baseline: {sota_oof:.4f}")
    print(f"历史 cand2 真分锚点: 0.72852 (OOF -0.0028, gap=+0.0782)\n")

    # 解析用户贴的真分 (或 demo 占位)
    scores = {}
    if args.scores:
        for i, s in enumerate(args.scores, 1):
            scores[i] = float(s)
    else:
        print("⚠️  --scores 未给, 仅打印 OOF 表 (真分待回填)")

    # 5 push 表
    print(f"{'#':<3} {'name':<35} {'OOF':>7} {'真分':>8} {'gap':>7} | {'T':>5} {'BC':>5} {'I':>5} | 维度")
    print("-" * 110)
    rows = []
    for pid, (name, axis) in PUSH_MAP.items():
        # 找候选 — v2 目录 + perfold-scan 目录
        c = cands.get(name)
        if c is None:
            for pf_dir in Path("tools/runs/climb").glob("perfold-scan-*"):
                pf_mf = json.load(open(pf_dir / "MANIFEST.json"))
                pf_cands = {cc["name"]: cc for cc in pf_mf["candidates"]}
                if name in pf_cands:
                    c = pf_cands[name]
                    c["oof_cap1_macro"] = None  # perfold 候选无 OOF
                    c["oof_cap1_per_class"] = {"C": None, "T": None, "BC": None, "I": None, "NA": None}
                    break
        if c is None:
            print(f"{pid:<3} {name:<35} CANDIDATE NOT FOUND")
            continue
        pc = c["oof_cap1_per_class"]
        real = scores.get(pid)
        oof = c["oof_cap1_macro"]
        gap = (real - oof) if (real and oof is not None) else None
        rows.append({"push": pid, "name": name, "axis": axis,
                     "oof": oof, "real": real, "gap": gap,
                     "T": pc["T"], "BC": pc["BC"], "I": pc["I"]})
        real_s = f"{real:.4f}" if real else "  ?  "
        gap_s = f"{gap:+.4f}" if gap is not None else "  ?  "
        oof_s = f"{oof:.4f}" if oof is not None else "  ?  "
        T_s = f"{pc['T']:.3f}" if pc['T'] is not None else "  ?  "
        BC_s = f"{pc['BC']:.3f}" if pc['BC'] is not None else "  ?  "
        I_s = f"{pc['I']:.3f}" if pc['I'] is not None else "  ?  "
        print(f"{pid:<3} {name:<35} {oof_s:>7} {real_s:>8} {gap_s:>7} | "
              f"{T_s} {BC_s} {I_s} | {axis}")

    if not scores:
        return

    # 真分排序
    print(f"\n=== 按真分排序 (vs OOF 排序对照) ===")
    by_real = sorted(rows, key=lambda r: -r["real"] if r["real"] else 0)
    by_oof = sorted(rows, key=lambda r: -r["oof"])
    for i, r in enumerate(by_real, 1):
        oof_rank = next(j for j, rr in enumerate(by_oof, 1) if rr["push"] == r["push"])
        print(f"真分 Top {i}: Push {r['push']} {r['name']:<35} 真分={r['real']:.4f} (OOF 排第 {oof_rank})")

    # 维度 gap 分布
    print(f"\n=== 维度 gap 分析 ===")
    dim_gaps = {}
    for r in rows:
        if r["gap"] is None: continue
        dim = r["axis"].split(":")[0]
        dim_gaps.setdefault(dim, []).append(r["gap"])
    for dim, gs in sorted(dim_gaps.items()):
        mean_g = sum(gs) / len(gs)
        print(f"  维度 {dim} ({len(gs)} 候选): mean gap = {mean_g:+.4f}, range [{min(gs):+.4f}, {max(gs):+.4f}]")

    # 落盘
    out = rd / f"CALIBRATION-{args.date}.json"
    out.write_text(json.dumps({
        "date": args.date,
        "sota_oof_baseline": sota_oof,
        "historical": HISTORICAL,
        "push_results": rows,
        "dim_gaps": {d: {"n": len(gs), "mean": sum(gs)/len(gs),
                        "min": min(gs), "max": max(gs)} for d, gs in dim_gaps.items()},
    }, ensure_ascii=False, indent=2))
    print(f"\n落盘: {out}")

    # 明日策略建议
    print(f"\n=== 明日 6/4 5 push 策略建议 ===")
    if by_real[0]["real"] > 0.72852:
        print(f"✓ NEW SOTA: Push {by_real[0]['push']} {by_real[0]['name']} = {by_real[0]['real']:.5f}")
        print(f"  下一步: 沿 {by_real[0]['axis']} 维度细粒度探索, 加 fold 数 / seed 数 / 替换 base 实验")
    else:
        print(f"⚠️  5 push 没破 cand2 (0.72852), 守现 SOTA")
        print(f"  下一步: 把 6/3 真分校准结果反馈到 v3 候选生成, 排除 D-22 后仍是反向的维度")


if __name__ == "__main__":
    main()
