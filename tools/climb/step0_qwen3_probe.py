"""Step 0 — D-3 同款风险预验证 (N1+ 前置门).

目的: 用已有 110 通 Qwen3-0.6B 文本 cache (1024d) + ctx 46d, 测 head 在 cap1 切片
vs 非 cap1 切片的 F1/AUC gap. gap <0.03 → N1+ 通过 (LLM 文本特征泛化稳)
否则 → D-3 同款风险, SKIP N1+.

使用前 30 通 (0000-0029) stride40 滑窗, 5fold conv-level GroupKFold.
每窗 = (text emb 1024d @ nvis(window_end_ms) + ctx 46d) → MLP head → 5 类 sigmoid.

cap1 定义: order=0 首窗 (每通 1 个, 共 30 个 cap1 窗)
non-cap1: order >= 1 的窗 (共 17000+ 个)

判读:
  per-class F1 (cap1) vs per-class F1 (non-cap1) gap
  macro 平均 gap < 0.03 → 通过
  AUC 是辅助指标 (阈值无关, 更稳)

Usage: OMP_NUM_THREADS=4 python tools/climb/step0_qwen3_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctxfeat

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM = 5
CTX, TGT, STRIDE, CHUNK_MS = 375, 25, 40, 80
N_CONVS = 30
CACHE_DIR = Path("data/cache/qwen_text")
TEXT_JSON_DIR = Path("data/train/text")
LABEL_DIR = Path("data/train/labels")
SEED = 42


def utt_ends_ms(text_json_path):
    """Return sorted list of end_ms for each utterance in this conv."""
    with open(text_json_path) as f:
        d = json.load(f)
    return sorted(int(u["end_ms"]) for u in d.get("utterances", []))


def nvis_at_ms(ends, end_ms):
    """Count utterances with end_ms <= given end_ms (= nvis cache key)."""
    return sum(1 for e in ends if e <= end_ms)


def build_features_for_conv(cid: str):
    """For one conv, build stride40 sliding windows:
    each window → (text_emb 1024d, ctx_feat 46d, target 5d, is_cap1 0/1, end_ms).
    Returns dict of arrays or None if conv has no cache.
    """
    cache_path = CACHE_DIR / f"{cid}.npz"
    if not cache_path.exists():
        return None
    # allow_pickle=True is safe here: cache files are produced locally by
    # tools/climb/extract_text_feats.py (5/27 H-T3 run), not from external sources.
    # Path is constrained to data/cache/qwen_text/ which is project-internal.
    z = np.load(cache_path, allow_pickle=True)
    cache_keys = sorted(int(k) for k in z.keys())
    if not cache_keys:
        return None
    max_nvis = max(cache_keys)

    label_path = LABEL_DIR / f"{cid}.npy"
    text_path = TEXT_JSON_DIR / f"{cid}.json"
    if not label_path.exists() or not text_path.exists():
        return None

    a = np.load(label_path).astype(int)
    if a.shape[0] < CTX + TGT:
        return None
    ends_ms = utt_ends_ms(text_path)
    if not ends_ms:
        return None

    text_embs, ctxs, targets, is_cap1, end_chunks = [], [], [], [], []
    for order, e in enumerate(range(CTX, a.shape[0] - TGT + 1, STRIDE)):
        end_ms = e * CHUNK_MS
        nvis = nvis_at_ms(ends_ms, end_ms)
        # find closest cache key <= nvis (cache key is utterance count)
        valid_keys = [k for k in cache_keys if k <= nvis]
        if not valid_keys:
            continue  # 窗位于第一句话之前, 无文本可用
        chosen_key = max(valid_keys)  # 取 ≤ nvis 的最大 key (= 此时已知最多 utts)
        if chosen_key > max_nvis:
            continue
        emb = z[str(chosen_key)]  # 1024d
        if emb.shape[0] != 1024:
            continue

        ctx = ctxfeat(a[e - CTX:e])
        fut = set(int(x) for x in a[e:e + TGT])
        target = np.array([1 if k in fut else 0 for k in range(NUM)], dtype=np.int8)

        text_embs.append(emb)
        ctxs.append(ctx)
        targets.append(target)
        is_cap1.append(1 if order == 0 else 0)
        end_chunks.append(e)

    if not text_embs:
        return None

    return {
        "text": np.array(text_embs, dtype=np.float32),
        "ctx": np.array(ctxs, dtype=np.float32),
        "y": np.array(targets, dtype=np.int8),
        "is_cap1": np.array(is_cap1, dtype=np.int8),
        "end": np.array(end_chunks, dtype=np.int32),
    }


def main():
    print(f"[step0] N_CONVS={N_CONVS} stride40 5fold conv-GroupKFold", file=sys.stderr)
    convs = [f"{i:04d}" for i in range(N_CONVS)]

    all_text, all_ctx, all_y, all_cap1, all_G = [], [], [], [], []
    for gi, cid in enumerate(convs):
        d = build_features_for_conv(cid)
        if d is None:
            print(f"  [skip] {cid} (no cache or invalid)", file=sys.stderr)
            continue
        n = d["text"].shape[0]
        all_text.append(d["text"])
        all_ctx.append(d["ctx"])
        all_y.append(d["y"])
        all_cap1.append(d["is_cap1"])
        all_G.append(np.full(n, gi, dtype=np.int16))
        n_cap1 = int(d["is_cap1"].sum())
        print(f"  [{cid}] {n} windows ({n_cap1} cap1)", file=sys.stderr)

    X_text = np.concatenate(all_text)
    X_ctx = np.concatenate(all_ctx)
    Y = np.concatenate(all_y)
    is_cap1 = np.concatenate(all_cap1)
    G = np.concatenate(all_G)
    print(f"\n[step0] total: X_text={X_text.shape}, X_ctx={X_ctx.shape}, Y={Y.shape}", file=sys.stderr)
    print(f"[step0] cap1 windows: {int(is_cap1.sum())} ({100*is_cap1.mean():.1f}%)", file=sys.stderr)
    print(f"[step0] per-class total positives: {Y.sum(axis=0).tolist()}", file=sys.stderr)
    print(f"[step0] per-class cap1 positives:  {Y[is_cap1==1].sum(axis=0).tolist()}", file=sys.stderr)

    # Concat text+ctx for full features. Also test text-only to isolate text contribution.
    X_full = np.concatenate([X_ctx, X_text], axis=1)  # 46 + 1024
    print(f"[step0] X_full={X_full.shape}", file=sys.stderr)

    # 5-fold conv-level GroupKFold
    rng = np.random.default_rng(SEED)
    conv_order = rng.permutation(N_CONVS)

    oof_full = np.full((len(Y), NUM), np.nan, dtype=np.float32)
    oof_text = np.full((len(Y), NUM), np.nan, dtype=np.float32)
    oof_ctx_only = np.full((len(Y), NUM), np.nan, dtype=np.float32)

    for fi in range(5):
        val_convs = {conv_order[i] for i in range(N_CONVS) if i % 5 == fi}
        tr_mask = np.array([g not in val_convs for g in G])
        va_mask = ~tr_mask
        if va_mask.sum() == 0:
            continue
        print(f"  fold {fi+1}/5: train={int(tr_mask.sum())}, val={int(va_mask.sum())}", file=sys.stderr)

        # 标准化 (在 train 拟合)
        sc_full = StandardScaler().fit(X_full[tr_mask])
        sc_text = StandardScaler().fit(X_text[tr_mask])
        sc_ctx = StandardScaler().fit(X_ctx[tr_mask])
        Xtr_full = sc_full.transform(X_full[tr_mask])
        Xva_full = sc_full.transform(X_full[va_mask])
        Xtr_text = sc_text.transform(X_text[tr_mask])
        Xva_text = sc_text.transform(X_text[va_mask])
        Xtr_ctx = sc_ctx.transform(X_ctx[tr_mask])
        Xva_ctx = sc_ctx.transform(X_ctx[va_mask])

        for k in range(NUM):
            y_tr = Y[tr_mask, k]
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                # 全 0 或全 1, 用 prior
                oof_full[va_mask, k] = y_tr.mean()
                oof_text[va_mask, k] = y_tr.mean()
                oof_ctx_only[va_mask, k] = y_tr.mean()
                continue
            for X_tr, X_va, oof in [
                (Xtr_full, Xva_full, oof_full),
                (Xtr_text, Xva_text, oof_text),
                (Xtr_ctx, Xva_ctx, oof_ctx_only),
            ]:
                clf = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0, n_jobs=1)
                clf.fit(X_tr, y_tr)
                oof[va_mask, k] = clf.predict_proba(X_va)[:, 1]

    # 评估: cap1 vs non-cap1 各类 F1 (threshold 0.5) + AUC
    print(f"\n=== Step 0 评估 (head probe per-class F1 @ thr=0.5, AUC) ===")
    print(f"{'class':<6} | {'full cap1':<10} {'full noncap1':<12} {'full gap':<10} | "
          f"{'text cap1':<10} {'text noncap1':<12} {'text gap':<10} | "
          f"{'ctx cap1':<10} {'ctx noncap1':<12}")

    def metrics(oof, mask, k):
        y_k = Y[mask, k]
        p_k = oof[mask, k]
        if y_k.sum() == 0 or y_k.sum() == len(y_k):
            return None, None
        f1 = f1_score(y_k, (p_k >= 0.5).astype(int), zero_division=0)
        try:
            auc = roc_auc_score(y_k, p_k)
        except Exception:
            auc = float("nan")
        return float(f1), float(auc)

    cap1_mask = is_cap1 == 1
    non_cap1_mask = is_cap1 == 0

    results = {"per_class": {}, "macro": {}}
    for k in range(NUM):
        f1_full_c, auc_full_c = metrics(oof_full, cap1_mask, k)
        f1_full_nc, auc_full_nc = metrics(oof_full, non_cap1_mask, k)
        f1_text_c, auc_text_c = metrics(oof_text, cap1_mask, k)
        f1_text_nc, auc_text_nc = metrics(oof_text, non_cap1_mask, k)
        f1_ctx_c, auc_ctx_c = metrics(oof_ctx_only, cap1_mask, k)
        f1_ctx_nc, auc_ctx_nc = metrics(oof_ctx_only, non_cap1_mask, k)
        gap_full = (f1_full_c - f1_full_nc) if (f1_full_c is not None and f1_full_nc is not None) else None
        gap_text = (f1_text_c - f1_text_nc) if (f1_text_c is not None and f1_text_nc is not None) else None

        def fmt(v):
            return f"{v:.3f}" if v is not None else "n/a"

        print(f"  {LAB[k]:<5} | {fmt(f1_full_c):<10} {fmt(f1_full_nc):<12} {fmt(gap_full):<10} | "
              f"{fmt(f1_text_c):<10} {fmt(f1_text_nc):<12} {fmt(gap_text):<10} | "
              f"{fmt(f1_ctx_c):<10} {fmt(f1_ctx_nc):<12}")
        results["per_class"][LAB[k]] = {
            "full": {"cap1_f1": f1_full_c, "noncap1_f1": f1_full_nc, "gap": gap_full,
                     "cap1_auc": auc_full_c, "noncap1_auc": auc_full_nc},
            "text": {"cap1_f1": f1_text_c, "noncap1_f1": f1_text_nc, "gap": gap_text,
                     "cap1_auc": auc_text_c, "noncap1_auc": auc_text_nc},
            "ctx": {"cap1_f1": f1_ctx_c, "noncap1_f1": f1_ctx_nc,
                    "cap1_auc": auc_ctx_c, "noncap1_auc": auc_ctx_nc},
        }

    # Macro: 平均 5 类 (跳过 None)
    def macro_metric(group, mask_name, metric):
        vals = [results["per_class"][LAB[k]][group][f"{mask_name}_{metric}"] for k in range(NUM)
                if results["per_class"][LAB[k]][group][f"{mask_name}_{metric}"] is not None]
        return sum(vals) / len(vals) if vals else None

    for g in ("full", "text", "ctx"):
        results["macro"][g] = {
            "cap1_f1": macro_metric(g, "cap1", "f1"),
            "noncap1_f1": macro_metric(g, "noncap1", "f1"),
            "cap1_auc": macro_metric(g, "cap1", "auc"),
            "noncap1_auc": macro_metric(g, "noncap1", "auc"),
        }
        m_c = results["macro"][g]["cap1_f1"]
        m_nc = results["macro"][g]["noncap1_f1"]
        if m_c is not None and m_nc is not None:
            results["macro"][g]["gap"] = m_c - m_nc

    print(f"\n=== Macro F1 ===")
    for g in ("full", "text", "ctx"):
        m = results["macro"][g]
        gap_str = f"{m.get('gap', 0):.4f}" if m.get('gap') is not None else "n/a"
        print(f"  {g:<6}: cap1={m['cap1_f1']:.4f}  non-cap1={m['noncap1_f1']:.4f}  gap={gap_str}")

    # 决策门
    gap_full_macro = results["macro"]["full"].get("gap")
    gap_text_macro = results["macro"]["text"].get("gap")
    print(f"\n=== 决策门 ===")
    print(f"  text-only macro gap = {gap_text_macro:.4f}" if gap_text_macro is not None else "  text-only n/a")
    print(f"  full   macro gap = {gap_full_macro:.4f}" if gap_full_macro is not None else "  full n/a")

    verdict = "UNKNOWN"
    if gap_text_macro is not None:
        if abs(gap_text_macro) < 0.03:
            verdict = "PASS — N1+ text 路线泛化稳, 可继续"
        else:
            verdict = f"WARN — text gap={gap_text_macro:.3f} 超过 0.03, D-3 同款风险, 但 LoRA 微调可能不同"
    print(f"  verdict: {verdict}")

    # 落盘
    out = {
        "convs_used": N_CONVS,
        "total_windows": int(len(Y)),
        "cap1_windows": int(cap1_mask.sum()),
        "non_cap1_windows": int(non_cap1_mask.sum()),
        "results": results,
        "verdict": verdict,
    }
    out_path = Path("tools/runs/climb/step0-qwen3-probe")
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"\n[step0] saved {out_path}/metrics.json")


if __name__ == "__main__":
    main()
