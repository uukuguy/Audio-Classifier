"""扫多源软加 OOF cap1, 不烧 push 配额预筛.

用户洞察 (2026-06-02):
  - 0.2 软加不是唯一最优, 权重要按源校准 (避超参过拟合)
  - 单变量扫: 固定 base, 一次只换一个 (源 / 权重)
  - 所有源 (包括 D-X 历史"证伪") 都进候选, 让 OOF + 真分自己选

输入: 所有现有 probs.npz (含 per-seed OOF if available)
输出: ranked OOF cap1 表 + top-K 候选 csv (备 push)

注意 cap1 OOF 红旗有过 D-22 系统性错误, 仅做粗筛, 不硬筛.

Usage:
  python3 tools/climb/sweep_softadd_oof.py [--top 20]
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}

# 软加权重扫描 (单变量轴)
WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


def cap1_idx_from_G_order(G, order):
    """每通取首窗 (order=0) 作为 cap1."""
    seen, cap1 = set(), []
    for i, (g, o) in enumerate(zip(G, order)):
        if int(o) == 0 and int(g) not in seen:
            cap1.append(i); seen.add(int(g))
    return np.array(cap1)


def cap1_idx_simple(G):
    """每通取第一次出现的 idx."""
    seen, cap1 = set(), []
    for i, g in enumerate(G):
        if int(g) not in seen:
            cap1.append(i); seen.add(int(g))
    return np.array(cap1)


def macro_f1(probs, Y, thresholds=None):
    thresholds = thresholds or THR_VARF
    f1s = []
    for k in range(5):
        pred = (probs[:, k] >= thresholds[k]).astype(int)
        f1s.append(f1_score(Y[:, k], pred, zero_division=0))
    return float(np.mean(f1s)), f1s


def load_source_oof_cap1(path, y_cap1):
    """读 probs.npz, 返回 cap1 OOF (369, 5) 跟标签 align.

    None 表示加载失败.
    """
    try:
        z = np.load(path)
        if "oof" not in z:
            return None
        oof = z["oof"]
        Y = z.get("Y", None)
        order = z.get("order", None)
        if Y is None or order is None:
            return None
        Y = Y.astype(int)
        cap1_mask = order == 0
        oof_c = oof[cap1_mask]
        y_c = Y[cap1_mask]
        # 长度对齐
        if len(oof_c) != len(y_cap1):
            return None
        if not (y_c == y_cap1).all():
            return None
        return oof_c.astype(np.float32)
    except Exception:
        return None


def make_sota_3src_oof(ctx_oof, wsp_oof, hub_oof):
    p = np.zeros_like(ctx_oof)
    p[:, 0] = ctx_oof[:, 0]
    p[:, 1] = 0.7 * wsp_oof[:, 1] + 0.3 * hub_oof[:, 1]
    p[:, 2] = ctx_oof[:, 2]
    p[:, 3] = (ctx_oof[:, 3] + wsp_oof[:, 3] + hub_oof[:, 3]) / 3
    p[:, 4] = ctx_oof[:, 4]
    return p


def softadd(base, extra, w, cols):
    p = base.copy()
    for k in cols:
        p[:, k] = (1 - w) * base[:, k] + w * extra[:, k]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30, help="输出 top K OOF cap1 候选")
    args = ap.parse_args()

    # 1. 加载 ctx (基座, 含 4 base + Y/G)
    print("=== 加载源 ===")
    zc = np.load("tools/runs/climb/_stack_cache_s40.npz")
    Yc_full = zc["Y"].astype(int); Gc = zc["G"]
    cap1_c = cap1_idx_simple(Gc)
    Y_c1 = Yc_full[cap1_c]
    print(f"ctx cap1: N={len(cap1_c)}, Y {Y_c1.shape}")

    ctx_oof = zc["oof_lgbm_v1"][cap1_c]

    # 2. 加载所有 head probs 的 cap1 OOF
    sources_to_load = {
        "whisper":        "tools/runs/climb/whisper-fusion-20260531-0143/probs.npz",
        "hubert":         "tools/runs/climb/hubert-bcaug-head-20260601-1533/probs.npz",
        "hubert_ms":      "tools/runs/climb/hubert-bcaug-multiseed-*/probs.npz",  # 多 seed 版
        "w2v2":           "tools/runs/climb/w2v2-bcaug-head-20260601-1926/probs.npz",
        "w2v2_ms":        "tools/runs/climb/w2v2-bcaug-multiseed-*/probs.npz",
        "e2v":            "tools/runs/climb/e2v-bcaug-head-20260601-1755/probs.npz",
        "e2v_ms":         "tools/runs/climb/e2v-bcaug-multiseed-*/probs.npz",
        "whisper_bcaug":  "tools/runs/climb/whisper-bcaug-head-20260601-1730/probs.npz",
        "whisper_bcaug_ms": "tools/runs/climb/whisper-bcaug-multiseed-*/probs.npz",
        "qwen3":          "tools/runs/climb/qwen3-head-20260601-1514/probs.npz",
        "omni":           "tools/runs/climb/omni-lora-20260602-1002/probs.npz",
        "omni3b":         "tools/runs/climb/omni3b-lora-20260602-2147/probs.npz",
        "omni_ms2":       "tools/runs/climb/omni-7b-ms2-mean-3seed/probs.npz",
        "omni3b_ms2":     "tools/runs/climb/omni-3b-ms2-mean-3seed/probs.npz",
        "qwen17b_ms2":    "tools/runs/climb/qwen17b-ms2-mean-3seed/probs.npz",
    }

    sources = {}
    for name, path_pattern in sources_to_load.items():
        if "*" in path_pattern:
            matches = sorted(glob.glob(path_pattern))
            if not matches:
                print(f"  {name}: SKIP (no match for {path_pattern})")
                continue
            path = matches[-1]  # 最新一个
        else:
            path = path_pattern
        oof_c = load_source_oof_cap1(path, Y_c1)
        if oof_c is None:
            print(f"  {name}: SKIP (load failed)")
            continue
        sources[name] = oof_c
        print(f"  {name}: ✓ from {path}")
    print(f"\n共加载 {len(sources)} 个源 (除 ctx)")

    if "whisper" not in sources or "hubert" not in sources:
        print("缺关键源 whisper/hubert, 无法构 SOTA, 退出")
        return

    # 3. 构 SOTA 3src OOF
    sota = make_sota_3src_oof(ctx_oof, sources["whisper"], sources["hubert"])
    sota_macro, sota_per = macro_f1(sota, Y_c1)
    print(f"\n=== SOTA orthofuse-3src OOF cap1 ===")
    print(f"macro={sota_macro:.4f} | " + " ".join(f"{LAB[k]}={sota_per[k]:.3f}" for k in range(5)))

    # 4. 单源软加扫 (单变量轴, 各源 × 各权重)
    print(f"\n=== 单源软加 cap1 扫 (vs SOTA {sota_macro:.4f}) ===")
    results = []  # (name, w, macro, per_class_list, key_metric_for_sort)
    for name, src_oof in sources.items():
        if name in ("whisper", "hubert"):
            continue  # SOTA 已用
        for w in WEIGHTS:
            p = softadd(sota, src_oof, w, cols=(1, 2, 3))  # T/BC/I 软加
            macro, per = macro_f1(p, Y_c1)
            delta = macro - sota_macro
            results.append({
                "config": f"sota+{name}_{w:.2f}_TBCI",
                "macro": macro, "delta_oof": delta,
                "per_class": per, "src": name, "weight": w, "cols": "T/BC/I",
            })
        # 同时扫 T/I only (不动 BC)
        for w in WEIGHTS:
            p = softadd(sota, src_oof, w, cols=(1, 3))
            macro, per = macro_f1(p, Y_c1)
            delta = macro - sota_macro
            results.append({
                "config": f"sota+{name}_{w:.2f}_TI",
                "macro": macro, "delta_oof": delta,
                "per_class": per, "src": name, "weight": w, "cols": "T/I",
            })

    # 排序 + 输出 top K
    results.sort(key=lambda r: -r["macro"])
    print(f"\n=== Top {args.top} 单源软加候选 (OOF cap1) ===")
    print(f"{'config':<45s} {'macro':>7s} {'ΔOOF':>7s} | C  T  BC  I  NA")
    for r in results[:args.top]:
        pcs = " ".join(f"{r['per_class'][k]:.3f}" for k in range(5))
        sign = "+" if r["delta_oof"] >= 0 else ""
        print(f"{r['config']:<45s} {r['macro']:.4f}  {sign}{r['delta_oof']:.4f} | {pcs}")

    # 5. 写 manifest 备明日 push 选
    out_dir = Path(f"tools/runs/climb/sweep-{__import__('time').strftime('%Y%m%d-%H%M')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranked.json").write_text(json.dumps({
        "sota_oof_macro": sota_macro,
        "sota_per_class": sota_per,
        "n_sources_loaded": len(sources),
        "sources": sorted(sources.keys()),
        "top": results[:args.top],
        "_note": "OOF cap1 单变量扫. 不硬筛, 仅排序参考. 真分校准必走 push.",
    }, ensure_ascii=False, indent=2))
    print(f"\n落盘: {out_dir}/ranked.json")


if __name__ == "__main__":
    main()
