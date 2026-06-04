"""v2: 扩展软加候选 + 全量 OOF 打分.

vs v1 (build_softadd_candidates.py):
  - 加入 multi-seed 新源 (hubert_ms / w2v2_ms / e2v_ms, 待 whisper_ms)
  - 同时算 OOF cap1 + per-class, 写 meta.json (v1 只生 csv 盲选)
  - 候选维度扩到 9 类 ~30+ 个 (单变量轴覆盖):
      A. Omni 权重档 (0.05/0.10/0.15/0.20/0.25/0.30)
      B. 单源 0.2 (qwen3/w2v2/e2v/wsb/hubert_ms/w2v2_ms/e2v_ms)
      B'. multi-seed 替换 single-seed (hubert→hubert_ms 同位)
      C. 多源叠加 (Omni 0.2 + 各 0.05/0.1)
      C'. multi-seed 叠加 (Omni + w2v2_ms 0.1)
      D. 替换 base (orthofuse-2src / ctx-only / ctx+whisper)
      E. cols 范围 (T/BC/I vs T/I only vs BC only)
      F. 极端权重 (0.03 vs 0.40 探边界)
      G. 同源双重叠加 (w2v2 + w2v2_ms 不同 seed 信号叠加)

Usage:
  python3 tools/climb/build_softadd_candidates_v2.py
"""
from __future__ import annotations
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}

PATHS = {
    "ctx_cache":       "tools/runs/climb/_stack_cache_s40.npz",
    "whisper":         "tools/runs/climb/whisper-fusion-20260531-0143/probs.npz",
    "hubert":          "tools/runs/climb/hubert-bcaug-head-20260601-1533/probs.npz",
    "hubert_ms":       "tools/runs/climb/hubert-bcaug-multiseed-20260602-1506/probs.npz",
    "w2v2":            "tools/runs/climb/w2v2-bcaug-head-20260601-1926/probs.npz",
    "w2v2_ms":         "tools/runs/climb/w2v2-bcaug-multiseed-20260602-1549/probs.npz",
    "e2v":             "tools/runs/climb/e2v-bcaug-head-20260601-1755/probs.npz",
    "e2v_ms":          "tools/runs/climb/e2v-bcaug-multiseed-20260602-1632/probs.npz",
    "whisper_bcaug":   "tools/runs/climb/whisper-bcaug-head-20260601-1730/probs.npz",
    "qwen3":           "tools/runs/climb/qwen3-head-20260601-1514/probs.npz",
    "omni":            "tools/runs/climb/omni-lora-20260602-1002/probs.npz",
    "omni3b":          "tools/runs/climb/omni3b-lora-20260602-2147/probs.npz",
}

# 试加 whisper_ms (若已完成)
WSP_MS_GLOB = sorted(glob.glob("tools/runs/climb/whisper-bcaug-multiseed-*/probs.npz"))
if WSP_MS_GLOB:
    PATHS["whisper_bcaug_ms"] = WSP_MS_GLOB[-1]

RUN_DIR = Path(f"tools/runs/climb/probe-softadd-v2-{time.strftime('%Y%m%d-%H%M')}")


def cap1_idx(G):
    seen, idx = set(), []
    for i, g in enumerate(G):
        if int(g) not in seen:
            idx.append(i); seen.add(int(g))
    return np.array(idx)


def macro_f1(probs, Y):
    pred = np.zeros_like(probs, dtype=int)
    for k in range(5):
        pred[:, k] = (probs[:, k] >= THR_VARF[k]).astype(int)
    per = [f1_score(Y[:, k], pred[:, k], zero_division=0) for k in range(5)]
    return float(np.mean(per)), per


def load_source(path, Y_cap1_ref):
    """加载源, 返回 (oof_cap1, test) 或 None."""
    z = np.load(path)
    keys = set(z.files)
    if "test" not in keys:
        return None
    test = z["test"].astype(np.float32)
    # OOF cap1 align
    if "oof" in keys and "Y" in keys and "order" in keys:
        Y = z["Y"].astype(int); order = z["order"]
        mask = order == 0
        oof_c = z["oof"][mask].astype(np.float32)
        y_c = Y[mask]
        if len(y_c) != len(Y_cap1_ref):
            return None
        if not (y_c == Y_cap1_ref).all():
            return None
        return oof_c, test
    return None


def softadd(base, extra, w, cols):
    p = base.copy()
    for k in cols:
        p[:, k] = (1 - w) * base[:, k] + w * extra[:, k]
    return p


def make_sota_3src(ctx, wsp, hub):
    """orthofuse-3src 真分 0.71755 配置."""
    p = np.zeros_like(ctx)
    p[:, 0] = ctx[:, 0]
    p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * hub[:, 1]
    p[:, 2] = ctx[:, 2]
    p[:, 3] = (ctx[:, 3] + wsp[:, 3] + hub[:, 3]) / 3
    p[:, 4] = ctx[:, 4]
    return p


def make_sota_2src(ctx, wsp):
    """orthofuse-2src 真分 0.71523."""
    p = ctx.copy()
    p[:, 1] = 0.7 * wsp[:, 1] + 0.3 * ctx[:, 1]
    p[:, 3] = (ctx[:, 3] + wsp[:, 3]) / 2
    return p


def make_csv(probs_test, test_ids, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = {c: 0 for c in SUBMIT}
    with open(out_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT) + "\n")
        for i, sid in enumerate(test_ids):
            vals = {c: int(probs_test[i, COL2K[c]] >= THR_VARF[COL2K[c]]) for c in SUBMIT}
            for c, v in vals.items():
                pos[c] += v
            f.write(",".join([sid] + [str(vals[c]) for c in SUBMIT]) + "\n")
    return pos


def write_candidate(name, desc, dimension, oof_probs, test_probs, Y_cap1, test_ids, sota_oof_macro):
    """写 csv + meta.json."""
    out_dir = RUN_DIR / name
    pos = make_csv(test_probs, test_ids, out_dir / "pred_test1.csv")
    macro, per = macro_f1(oof_probs, Y_cap1)
    delta = macro - sota_oof_macro
    meta = {
        "name": name,
        "desc": desc,
        "dimension": dimension,
        "oof_cap1_macro": macro,
        "oof_cap1_per_class": {LAB[k]: per[k] for k in range(5)},
        "delta_oof_vs_sota": delta,
        "test_pos": pos,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[v2] 候选生成 → {RUN_DIR}", file=sys.stderr)

    # 1. ctx 基座 (含 4 base + Y/G)
    zc = np.load(PATHS["ctx_cache"])
    Yc_full = zc["Y"].astype(int); Gc = zc["G"]
    cap1_c = cap1_idx(Gc)
    Y_cap1 = Yc_full[cap1_c]
    ctx_oof = zc["oof_lgbm_v1"][cap1_c].astype(np.float32)
    ctx_test = zc["te_lgbm_v1"].astype(np.float32)
    print(f"ctx cap1: N={len(cap1_c)} (Y shape={Y_cap1.shape})")

    # 2. 加载所有源 OOF cap1 + test (cap1 标签对齐验证)
    sources_oof = {}
    sources_test = {}
    for name, path in PATHS.items():
        if name == "ctx_cache":
            continue
        if not Path(path).exists():
            print(f"  {name}: SKIP (missing {path})")
            continue
        loaded = load_source(path, Y_cap1)
        if loaded is None:
            print(f"  {name}: SKIP (load/align failed)")
            continue
        sources_oof[name], sources_test[name] = loaded
        print(f"  {name}: ✓")
    print(f"\n共加载 {len(sources_oof)} 源 + ctx\n")

    # 3. 构 SOTA-3src OOF + test
    sota_oof = make_sota_3src(ctx_oof, sources_oof["whisper"], sources_oof["hubert"])
    sota_test = make_sota_3src(ctx_test, sources_test["whisper"], sources_test["hubert"])
    sota_oof_macro, sota_oof_per = macro_f1(sota_oof, Y_cap1)
    print(f"=== SOTA-3src OOF cap1 ===")
    print(f"macro={sota_oof_macro:.4f} | " + " ".join(f"{LAB[k]}={sota_oof_per[k]:.3f}" for k in range(5)))
    print()

    sota_2src_oof = make_sota_2src(ctx_oof, sources_oof["whisper"])
    sota_2src_test = make_sota_2src(ctx_test, sources_test["whisper"])
    sota_2src_macro, _ = macro_f1(sota_2src_oof, Y_cap1)
    print(f"=== SOTA-2src OOF cap1 ===")
    print(f"macro={sota_2src_macro:.4f}\n")

    # 拿 test_ids (任一含 test_ids 的源)
    z = np.load(PATHS["whisper"])
    test_ids = z["test_ids"]

    candidates = []

    def add(name, desc, dim, oof, test):
        m = write_candidate(name, desc, dim, oof, test, Y_cap1, test_ids, sota_oof_macro)
        candidates.append(m)

    # === A. Omni 权重档 (单变量轴: w ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}) ===
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        oof_p = softadd(sota_oof, sources_oof["omni"], w, (1, 2, 3))
        tst_p = softadd(sota_test, sources_test["omni"], w, (1, 2, 3))
        add(f"A_omni_w{int(w*100):03d}_TBCI", f"SOTA + Omni {w} on T/BC/I", "A_omni_weight", oof_p, tst_p)

    # === A'. Omni-3B 权重档 (合规版, 跟 Omni-7B 对照) ===
    if "omni3b" in sources_oof:
        for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
            oof_p = softadd(sota_oof, sources_oof["omni3b"], w, (1, 2, 3))
            tst_p = softadd(sota_test, sources_test["omni3b"], w, (1, 2, 3))
            add(f"A3B_omni3b_w{int(w*100):03d}_TBCI", f"SOTA + Omni-3B {w} on T/BC/I", "A3B_omni3b_weight", oof_p, tst_p)

    # === B. 单源 0.2 (其他源跟 cand2 模式) ===
    for src_name in ["qwen3", "w2v2", "e2v", "whisper_bcaug",
                     "hubert_ms", "w2v2_ms", "e2v_ms"]:
        if src_name not in sources_oof:
            continue
        oof_p = softadd(sota_oof, sources_oof[src_name], 0.2, (1, 2, 3))
        tst_p = softadd(sota_test, sources_test[src_name], 0.2, (1, 2, 3))
        add(f"B_{src_name}_w020_TBCI", f"SOTA + {src_name} 0.2 on T/BC/I", "B_single_source", oof_p, tst_p)

    # whisper_ms (若已加载)
    if "whisper_bcaug_ms" in sources_oof:
        oof_p = softadd(sota_oof, sources_oof["whisper_bcaug_ms"], 0.2, (1, 2, 3))
        tst_p = softadd(sota_test, sources_test["whisper_bcaug_ms"], 0.2, (1, 2, 3))
        add("B_whisper_bcaug_ms_w020_TBCI", "SOTA + whisper_bcaug_ms 0.2 on T/BC/I", "B_single_source", oof_p, tst_p)

    # === C'. Omni-3B 0.2 + 其他 0.05/0.10 (合规多源叠加) ===
    if "omni3b" in sources_oof:
        base_omni3b = softadd(sota_oof, sources_oof["omni3b"], 0.2, (1, 2, 3))
        base_omni3b_t = softadd(sota_test, sources_test["omni3b"], 0.2, (1, 2, 3))
        for src_name, w in [("e2v_ms", 0.10), ("whisper_bcaug_ms", 0.10)]:
            if src_name not in sources_oof:
                continue
            oof_p = softadd(base_omni3b, sources_oof[src_name], w, (1, 2, 3))
            tst_p = softadd(base_omni3b_t, sources_test[src_name], w, (1, 2, 3))
            add(f"C3B_omni3b020+{src_name}_w{int(w*100):03d}", f"SOTA+Omni3B0.2 + {src_name} {w}", "C3B_multi_compliant", oof_p, tst_p)

    # === C. 多源叠加 (Omni 0.2 + 各 0.05/0.10) ===
    base_omni = softadd(sota_oof, sources_oof["omni"], 0.2, (1, 2, 3))
    base_omni_t = softadd(sota_test, sources_test["omni"], 0.2, (1, 2, 3))
    for src_name, w in [("qwen3", 0.05), ("qwen3", 0.10),
                        ("e2v_ms", 0.05), ("e2v_ms", 0.10),
                        ("w2v2_ms", 0.05), ("w2v2_ms", 0.10)]:
        if src_name not in sources_oof:
            continue
        oof_p = softadd(base_omni, sources_oof[src_name], w, (1, 2, 3))
        tst_p = softadd(base_omni_t, sources_test[src_name], w, (1, 2, 3))
        add(f"C_omni020+{src_name}_w{int(w*100):03d}", f"SOTA+Omni0.2 + {src_name} {w}", "C_multi_source", oof_p, tst_p)

    # === D. 替换 base ===
    # D1. 2src base + Omni 0.2
    oof_p = softadd(sota_2src_oof, sources_oof["omni"], 0.2, (1, 2, 3))
    tst_p = softadd(sota_2src_test, sources_test["omni"], 0.2, (1, 2, 3))
    add("D1_sota2src+omni020", "SOTA-2src (ctx+whisper) + Omni 0.2", "D_replace_base", oof_p, tst_p)

    # D2. ctx-only + Omni 0.3 (无 audio base, 大权)
    oof_p = softadd(ctx_oof, sources_oof["omni"], 0.30, (1, 2, 3))
    tst_p = softadd(ctx_test, sources_test["omni"], 0.30, (1, 2, 3))
    add("D2_ctx+omni030", "ctx-only base + Omni 0.3", "D_replace_base", oof_p, tst_p)

    # D3. 4src (ctx + whisper + hubert + w2v2_ms 平均 T) + Omni 0.2
    if "w2v2_ms" in sources_oof:
        sota_4src_oof = ctx_oof.copy()
        sota_4src_oof[:, 1] = 0.5*sources_oof["whisper"][:, 1] + 0.3*sources_oof["hubert"][:, 1] + 0.2*sources_oof["w2v2_ms"][:, 1]
        sota_4src_oof[:, 3] = (ctx_oof[:, 3] + sources_oof["whisper"][:, 3] + sources_oof["hubert"][:, 3] + sources_oof["w2v2_ms"][:, 3]) / 4
        sota_4src_t = ctx_test.copy()
        sota_4src_t[:, 1] = 0.5*sources_test["whisper"][:, 1] + 0.3*sources_test["hubert"][:, 1] + 0.2*sources_test["w2v2_ms"][:, 1]
        sota_4src_t[:, 3] = (ctx_test[:, 3] + sources_test["whisper"][:, 3] + sources_test["hubert"][:, 3] + sources_test["w2v2_ms"][:, 3]) / 4
        oof_p = softadd(sota_4src_oof, sources_oof["omni"], 0.2, (1, 2, 3))
        tst_p = softadd(sota_4src_t, sources_test["omni"], 0.2, (1, 2, 3))
        add("D3_sota4src(+w2v2ms)+omni020", "SOTA-4src (+w2v2_ms) + Omni 0.2", "D_replace_base", oof_p, tst_p)

    # === E. cols 范围 (Omni 0.2 但限 cols) ===
    for cols, label in [((1, 3), "TI_only"), ((2,), "BC_only"), ((1, 2, 3, 4), "TBCINA")]:
        oof_p = softadd(sota_oof, sources_oof["omni"], 0.2, cols)
        tst_p = softadd(sota_test, sources_test["omni"], 0.2, cols)
        add(f"E_omni020_{label}", f"SOTA + Omni 0.2 on {label}", "E_cols_range", oof_p, tst_p)

    # === F. 同源 single + ms 双重叠加 (探索 ms 信号是否跟 single 互补) ===
    for src_base, src_ms in [("hubert", "hubert_ms"), ("w2v2", "w2v2_ms"), ("e2v", "e2v_ms")]:
        if src_ms not in sources_oof:
            continue
        oof_p = softadd(sota_oof, sources_oof[src_base], 0.1, (1, 2, 3))
        oof_p = softadd(oof_p, sources_oof[src_ms], 0.1, (1, 2, 3))
        tst_p = softadd(sota_test, sources_test[src_base], 0.1, (1, 2, 3))
        tst_p = softadd(tst_p, sources_test[src_ms], 0.1, (1, 2, 3))
        add(f"F_{src_base}010+{src_ms}010", f"SOTA + {src_base} 0.1 + {src_ms} 0.1", "F_single+ms", oof_p, tst_p)

    # === G. ms 替换 single 在 SOTA-3src base 内 ===
    if "hubert_ms" in sources_oof:
        sota_hms_oof = make_sota_3src(ctx_oof, sources_oof["whisper"], sources_oof["hubert_ms"])
        sota_hms_t = make_sota_3src(ctx_test, sources_test["whisper"], sources_test["hubert_ms"])
        # 与 Omni 0.2 软加
        oof_p = softadd(sota_hms_oof, sources_oof["omni"], 0.2, (1, 2, 3))
        tst_p = softadd(sota_hms_t, sources_test["omni"], 0.2, (1, 2, 3))
        add("G_sota3src(hubert→hubert_ms)+omni020", "SOTA-3src 用 hubert_ms 替 hubert + Omni 0.2", "G_ms_replace", oof_p, tst_p)

    # === 排序输出 ===
    candidates.sort(key=lambda c: -c["oof_cap1_macro"])
    print(f"\n=== {len(candidates)} 候选生成 (按 OOF cap1 排序) ===\n")
    print(f"{'name':<40} {'OOF':>7} {'Δ':>8} | {'C':>5} {'T':>5} {'BC':>5} {'I':>5} {'NA':>5} | dim")
    print("-" * 110)
    for c in candidates:
        pc = c["oof_cap1_per_class"]
        d = c["delta_oof_vs_sota"]
        sign = "+" if d >= 0 else ""
        print(f"{c['name']:<40} {c['oof_cap1_macro']:.4f}  {sign}{d:.4f} | "
              f"{pc['C']:.3f} {pc['T']:.3f} {pc['BC']:.3f} {pc['I']:.3f} {pc['NA']:.3f} | {c['dimension']}")

    (RUN_DIR / "MANIFEST.json").write_text(json.dumps({
        "_note": "v2 扩展候选 + 全量 OOF cap1 + per-class. D-22 后明日 6/3 5 push 备选池.",
        "sota_3src_oof_macro": sota_oof_macro,
        "sota_3src_oof_per_class": {LAB[k]: sota_oof_per[k] for k in range(5)},
        "sources_loaded": sorted(sources_oof.keys()),
        "candidates": candidates,
        "yesterday_real_results": {
            "cand2_sota+omni020": 0.72852,
            "cand1_sota+omni050": 0.69094,
            "cand3_sota+w2v2_TI": 0.71452,
            "cand4_omni_only": 0.61305,
            "cand5_4bcaug_eq": 0.60734,
        },
    }, ensure_ascii=False, indent=2))
    print(f"\nMANIFEST: {RUN_DIR}/MANIFEST.json")


if __name__ == "__main__":
    main()
