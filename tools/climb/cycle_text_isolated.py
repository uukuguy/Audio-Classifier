"""climb cycle — paradigm=text-lexical-fusion-ISOLATED (动作1: 文本特征按类隔离).

诊断 workflow 定论: cycle_text_fusion.py 是"每类独立训练但5类全喂 ctx+text",
导致 text 污染 BC(0.217→0.201)却帮 T/I(T 0.54→0.58, I 0.44→0.49)。

修复 = 特征按类掩码:
  - T(1), I(3): 用 concat(ctx_feat[80], text_feat[21]) = 101 维
  - C(0), NA(4), BC(2): 只用 ctx_feat[80]（切掉 text，BC 回到纯 ctx 0.217 不被污染）

阈值 per-class-aware（阈值铁律: C/NA 94%恒正安全于低阈值，floor[0.35,0.65]会砸崩C）:
  - C(0), NA(4): 低阈值搜索 [0.05, 0.40]（保近全正）
  - T(1), I(3), BC(2): 温和搜索 [0.35, 0.65]

对比基线: 纯 ctx-only (cycle_context.py 的 0.5908) 和 text-allclass (cycle_text_fusion.py)。

Usage: python tools/climb/cycle_text_isolated.py <run_dir> [n_folds]
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from cycle_context_v2 import featurize as ctx_featurize  # noqa: E402
from cycle_text_fusion import text_feats  # noqa: E402  (reuse text feature extractor)

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL_TO_LABELID = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
NUM, CTX, TGT, STRIDE, CHUNK_MS = 5, 375, 25, 5, 80
SEED = 42

CTX_DIM = 80  # ctx_featurize output dim (split point: cols [:80]=ctx, [80:]=text)
TEXT_CLASSES = {1, 3}  # T, I use ctx+text; C/NA/BC use ctx only
LOW_THR_CLASSES = {0, 4}  # C, NA: near-always-positive → low threshold (阈值铁律)


def build_train(conv_ids, label_files, text_dir):
    X, Y, G = [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(label_files[cid])
        utts = json.load(open(f"{text_dir}/{cid}.json")).get("utterances", [])
        for e in range(CTX, a.shape[0] - TGT + 1, STRIDE):
            ctx = a[e - CTX:e].astype(int)
            end_ms = e * CHUNK_MS
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(np.concatenate([ctx_featurize(ctx), text_feats(utts, end_ms)]))
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=int), np.array(G)


def feat_cols_for_class(k: int, X: np.ndarray) -> np.ndarray:
    """T/I get full ctx+text; others get ctx-only (text masked out)."""
    return X if k in TEXT_CLASSES else X[:, :CTX_DIM]


def mk_lgbm(spw):
    return LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=48,
                          scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED)


def tune_threshold(y, p, k: int):
    """Per-class-aware threshold (阈值铁律):
      C/NA (near-always-positive): search [0.05, 0.40]
      T/I/BC (mid/low freq): search [0.35, 0.65]
    """
    if k in LOW_THR_CLASSES:
        grid = np.linspace(0.05, 0.40, 15)
    else:
        grid = np.linspace(0.35, 0.65, 13)
    bt, bf = 0.5, -1.0
    for t in grid:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, float(t)
    return bt, bf


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/runs/climb/_text_isolated")
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_dir.mkdir(parents=True, exist_ok=True)

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    print(f"[text-iso] building features ({len(conv_ids)} convs)...", file=sys.stderr)
    X, Y, G = build_train(conv_ids, label_files, "data/train/text")
    print(f"[text-iso] {len(X)} windows feat_dim={X.shape[1]} (ctx={CTX_DIM} text={X.shape[1]-CTX_DIM})",
          file=sys.stderr)
    print(f"[text-iso] TEXT_CLASSES={[LABELS[k] for k in TEXT_CLASSES]} use ctx+text; "
          f"rest use ctx-only", file=sys.stderr)

    gkf = GroupKFold(n_splits=n_folds)
    oof = {k: np.zeros(len(X)) for k in range(NUM)}
    for fold, (tr, va) in enumerate(gkf.split(X, Y[:, 0], groups=G)):
        for k in range(NUM):
            Xtr_k = feat_cols_for_class(k, X[tr])
            Xva_k = feat_cols_for_class(k, X[va])
            ytr = Y[tr, k]
            spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
            oof[k][va] = mk_lgbm(spw).fit(Xtr_k, ytr).predict_proba(Xva_k)[:, 1]
        print(f"[text-iso] fold {fold+1}/{n_folds}", file=sys.stderr)

    thr, f1s = {}, {}
    for k in range(NUM):
        t, f = tune_threshold(Y[:, k], oof[k], k)
        thr[k], f1s[k] = t, f
        tag = "ctx+text" if k in TEXT_CLASSES else "ctx-only"
        print(f"[text-iso] {LABELS[k]:3s} thr={t:.2f} F1={f:.3f} [{tag}]", file=sys.stderr)
    macro = float(np.mean(list(f1s.values())))
    print(f"[text-iso] OOF Macro-F1 = {macro:.4f}  (baseline ctx-only=0.5908, text-allclass≈0.5945)",
          file=sys.stderr)

    # test
    test_ctx = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_ctx]
    Xte = []
    for p in test_ctx:
        ctx = np.load(p).astype(int)
        tj = json.load(open(f"data/test/text/{Path(p).stem}.json"))
        end_ms = int(tj.get("end_ms", 30000))
        Xte.append(np.concatenate([ctx_featurize(ctx), text_feats(tj.get("utterances", []), end_ms)]))
    Xte = np.array(Xte, dtype=np.float32)
    preds = {}
    for k in range(NUM):
        Xk = feat_cols_for_class(k, X)
        Xte_k = feat_cols_for_class(k, Xte)
        spw = (len(Y) - Y[:, k].sum()) / max(1, Y[:, k].sum())
        preds[k] = (mk_lgbm(spw).fit(Xk, Y[:, k]).predict_proba(Xte_k)[:, 1] >= thr[k]).astype(int)

    csv_path = run_dir / "pred_test1.csv"
    with open(csv_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT_COLS) + "\n")
        for i, sid in enumerate(seg_ids):
            f.write(",".join([sid] + [str(int(preds[COL_TO_LABELID[c]][i])) for c in SUBMIT_COLS]) + "\n")

    per_sub = {c: round(f1s[COL_TO_LABELID[c]], 4) for c in ["c", "na", "t", "i", "bc"]}
    (run_dir / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "text-lexical-fusion-isolated", "hypothesis_id": "H-T-iso",
        "cv_macro_f1": round(macro, 4),
        "per_sub_f1": per_sub, "thresholds": {LABELS[k]: round(thr[k], 2) for k in range(NUM)},
        "method": f"{n_folds}-fold OOF, T/I=ctx+text C/NA/BC=ctx-only, per-class-aware threshold",
    }, ensure_ascii=False, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps({
        "cycle": 11, "hypothesis_id": "H-T-iso", "paradigm": "text-lexical-fusion-isolated",
        "start": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"score": round(macro, 4), "per_sub": per_sub}))


if __name__ == "__main__":
    main()
