"""climb cycle — paradigm=text-lexical-fusion v2 (H-T2, 测 ASR 词汇能否救 BC).

负结果链：context-only BC 0.217 → +廉价声学 0.219(无效)。
research(Amazon) 说 BC 更 related 句法语义，文本单模态对 BC 最强。
EDA：短发声 20% 是 backchannel 词（嗯/哦/啊/对/嗯嗯...）。

假设：最近窗内 backchannel-marker 词的出现/时机/说话人，能预测未来 2s 是否 BC。
特征 = context-v2 特征 + ASR 词汇统计（BC词频/最近BC词距/各通道短发声率/疑问词等）。
零神经网络、零新依赖（用 train/text json + 词表规则）。

Usage: python tools/climb/cycle_text_fusion.py <run_dir> [n_folds]
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

LABELS = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL_TO_LABELID = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
NUM, CTX, TGT, STRIDE, CHUNK_MS = 5, 375, 25, 5, 80
SEED = 42

# backchannel / 话语标记词表（EDA 实测高频短发声）
BC_WORDS = {"嗯", "嗯嗯", "嗯嗯嗯", "哦", "哦哦", "啊", "哎", "对", "对呀", "对对",
            "是", "是呀", "是的", "嗯哼", "哼", "哈哈", "呵呵", "好", "好的", "哦哦哦", "哎呀"}
Q_MARK = {"？", "?", "啊？", "吗", "呢"}


def text_feats(utts: list[dict], end_ms: int) -> list[float]:
    """预测点 end_ms 之前的 ASR 词汇特征（因果：只看 end_ms 之前）。"""
    # 收集 end_ms 前的发声（按结束时间）
    past = [u for u in utts if int(u.get("end_ms", 0)) <= end_ms]
    f = []
    for win_ms in (2000, 5000, 10000):
        lo = end_ms - win_ms
        w = [u for u in past if int(u.get("end_ms", 0)) > lo]
        n = len(w)
        n_bc = sum(1 for u in w if str(u.get("text", "")).strip().rstrip("。") in BC_WORDS)
        n_short = sum(1 for u in w if len(str(u.get("text", "")).strip()) <= 3)
        n_q = sum(1 for u in w if any(q in str(u.get("text", "")) for q in Q_MARK))
        # 双声道：各通道发声数
        ch = [0, 0]
        for u in w:
            c = int(u.get("channel_id", 1))
            ch[0 if c == 1 else 1] += 1
        f += [n, n_bc, n_short, n_q, ch[0], ch[1]]
    # 距最近 BC 词的归一化距离 + 最近发声是不是 BC 词 + 最近发声所属通道
    last_bc_dist = 1.0
    last_is_bc = 0.0
    last_ch = 0.0
    if past:
        for u in reversed(past):
            t = str(u.get("text", "")).strip().rstrip("。")
            if t in BC_WORDS:
                last_bc_dist = min(1.0, (end_ms - int(u.get("end_ms", 0))) / 10000.0)
                break
        lastu = past[-1]
        last_is_bc = 1.0 if str(lastu.get("text", "")).strip().rstrip("。") in BC_WORDS else 0.0
        last_ch = 1.0 if int(lastu.get("channel_id", 1)) == 1 else 0.0
    f += [last_bc_dist, last_is_bc, last_ch]
    return [float(x) for x in f]


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


def mk_lgbm(spw):
    return LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=48,
                          scale_pos_weight=spw, n_jobs=-1, verbose=-1, random_state=SEED)


def tune_threshold(y, p, label_id=None):
    """per-class-aware 阈值（铁律：C 安全于低阈值，其余 0.5 附近窄搜防错配过拟合）。
    C(id 0) 恒正→低阈值 [0.03,0.22]；T/I/BC/NA→窄带 [0.40,0.62]。"""
    if label_id == 0:  # C: 94% 恒正，低阈值安全
        lo, hi, n = 0.03, 0.22, 20
    else:
        lo, hi, n = 0.40, 0.62, 12
    bt, bf = 0.5, -1.0
    for t in np.linspace(lo, hi, n):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, float(t)
    return bt, bf


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/runs/climb/_adhoc_text")
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run_dir.mkdir(parents=True, exist_ok=True)

    label_files = {Path(p).stem: p for p in sorted(glob.glob("data/train/labels/*.npy"))}
    conv_ids = sorted(label_files)
    print(f"[text-fuse] building features ({len(conv_ids)} convs)...", file=sys.stderr)
    X, Y, G = build_train(conv_ids, label_files, "data/train/text")
    print(f"[text-fuse] {len(X)} windows feat_dim={X.shape[1]}", file=sys.stderr)

    gkf = GroupKFold(n_splits=n_folds)
    oof = {k: np.zeros(len(X)) for k in range(NUM)}
    for fold, (tr, va) in enumerate(gkf.split(X, Y[:, 0], groups=G)):
        for k in range(NUM):
            ytr = Y[tr, k]
            spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
            oof[k][va] = mk_lgbm(spw).fit(X[tr], ytr).predict_proba(X[va])[:, 1]
        print(f"[text-fuse] fold {fold+1}/{n_folds}", file=sys.stderr)

    thr, f1s = {}, {}
    for k in range(NUM):
        t, f = tune_threshold(Y[:, k], oof[k], label_id=k)
        thr[k], f1s[k] = t, f
        print(f"[text-fuse] {LABELS[k]:3s} thr={t:.2f} F1={f:.3f}", file=sys.stderr)
    macro = float(np.mean(list(f1s.values())))
    print(f"[text-fuse] OOF Macro-F1 = {macro:.4f}  (BC: ctx=0.217 → text=?)", file=sys.stderr)

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
        spw = (len(Y) - Y[:, k].sum()) / max(1, Y[:, k].sum())
        preds[k] = (mk_lgbm(spw).fit(X, Y[:, k]).predict_proba(Xte)[:, 1] >= thr[k]).astype(int)

    csv_path = run_dir / "pred_test1.csv"
    with open(csv_path, "w") as f:
        f.write("segment_id," + ",".join(SUBMIT_COLS) + "\n")
        for i, sid in enumerate(seg_ids):
            f.write(",".join([sid] + [str(int(preds[COL_TO_LABELID[c]][i])) for c in SUBMIT_COLS]) + "\n")

    per_sub = {c: round(f1s[COL_TO_LABELID[c]], 4) for c in ["c", "na", "t", "i", "bc"]}
    (run_dir / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "text-lexical-fusion", "hypothesis_id": "H-T2", "cv_macro_f1": round(macro, 4),
        "per_sub_f1": per_sub, "thresholds": {LABELS[k]: round(thr[k], 2) for k in range(NUM)},
        "method": f"{n_folds}-fold OOF, ctx + ASR 词汇特征(BC词频/距/通道), 温和阈值[0.35,0.65]",
    }, ensure_ascii=False, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps({
        "cycle": 5, "hypothesis_id": "H-T2", "paradigm": "text-lexical-fusion",
        "start": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"score": round(macro, 4), "per_sub": per_sub}))


if __name__ == "__main__":
    main()
