"""climb cycle H-T4 — 攻 T/I: baseline v1 context + 文本语义/时序特征 (干净对照).

战略 (2026-05-30): BC 单源撞墙(信息上限), 转攻 T(0.542)/I(0.434) — 未饱和且
negative cache 有证据文本帮过 T/I. 但旧实验用 v2(falsified 0.683)做底, 增益被污染.
本实验用 baseline v1 context 做干净底, 隔离文本对 T/I 的真实增量.

T = turn-shift: 强相关"对方句子完成度/句末/疑问→该换人说"
I = interruption: 强相关"双声道重叠/打断/抢话模式"

特征对照:
  A. baseline v1 context (46维) — 复现 T 0.542 / I 0.434 锚点
  B. + 文本词汇特征 (旧 BC词/疑问/短发声/通道, 复用 cycle_text_fusion.text_feats)
  C. + T/I 专属增强 (句末完成度/双声道重叠率/最近换人/疑问对齐预测点)

评估: 全切片高分辨率 OOF (相对比较安全) + cap1 参考. 只看 T/I, 监控 C/NA 不退.

Usage: python tools/climb/cycle_ti_text.py [--folds 5] [--stride 40]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, "tools/climb")
from cycle_context import featurize as ctx_v1          # baseline v1 (干净底)
from cycle_text_fusion import text_feats, BC_WORDS, Q_MARK  # 旧文本词汇特征

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
NUM, CTX, TGT, CHUNK_MS, SEED = 5, 375, 25, 80, 42


def robust_text_feats(utts: list[dict], end_ms: int) -> list[float]:
    """train/test 鲁棒文本特征 (修 H-T6 发现的 bug: 去标点依赖 + 去绝对密度).
    只用: ①比例/率(非绝对计数, 对 test 高密度鲁棒) ②train/test 都有的字词
    (BC词/吗/呢是字, 非标点 ？。). 验证: BC词率 test11.4% vs train14.2% 接近."""
    past = [u for u in utts if int(u.get("end_ms", 0)) <= end_ms]
    f = []
    for win_ms in (2000, 5000, 10000):
        lo = end_ms - win_ms
        w = [u for u in past if int(u.get("end_ms", 0)) > lo]
        n = max(1, len(w))
        n_bc = sum(1 for u in w if str(u.get("text", "")).strip().rstrip("。") in BC_WORDS)
        n_short = sum(1 for u in w if len(str(u.get("text", "")).strip().rstrip("。")) <= 3)
        n_q = sum(1 for u in w if ("吗" in str(u.get("text", "")) or "呢" in str(u.get("text", ""))))
        ch = [0, 0]
        for u in w:
            ch[0 if int(u.get("channel_id", 1)) == 1 else 1] += 1
        # 全部用比例 (率), 不用绝对数
        f.append(n_bc / n)                       # BC词占比
        f.append(n_short / n)                    # 短发声占比
        f.append(n_q / n)                        # 疑问(吗呢)占比
        f.append(abs(ch[0] - ch[1]) / n)         # 双声道不平衡率
        f.append(min(ch) / n)                    # 双声道并发率(重叠代理→I)
    # 最近发声: BC词? 通道? (不用标点) + 距最近BC词(归一化)
    if past:
        lastu = past[-1]
        lt = str(lastu.get("text", "")).strip().rstrip("。")
        f.append(1.0 if lt in BC_WORDS else 0.0)
        f.append(1.0 if ("吗" in lt or "呢" in lt) else 0.0)        # 末句疑问字(无标点)
        f.append(min(1.0, len(lt) / 15.0))                          # 末句长度(完整turn代理, 无标点)
        last_bc_dist = 1.0
        for u in reversed(past):
            if str(u.get("text", "")).strip().rstrip("。") in BC_WORDS:
                last_bc_dist = min(1.0, (end_ms - int(u.get("end_ms", 0))) / 10000.0); break
        f.append(last_bc_dist)
    else:
        f += [0.0, 0.0, 0.0, 1.0]
    return [float(x) for x in f]


def ti_enhanced_feats(utts: list[dict], end_ms: int) -> list[float]:
    """T/I 专属增强: 句末完成度 / 双声道重叠 / 换人模式 / 疑问对齐预测点."""
    past = [u for u in utts if int(u.get("end_ms", 0)) <= end_ms]
    f = []
    # 1. 最近发声的"完成度"信号 (T: 完整句末→该换人; 疑问→对方该答=shift)
    if past:
        lastu = past[-1]
        txt = str(lastu.get("text", "")).strip()
        f.append(1.0 if txt and txt[-1] in "。.!！" else 0.0)        # 陈述完成
        f.append(1.0 if any(q in txt for q in Q_MARK) else 0.0)       # 疑问(强shift信号)
        f.append(min(1.0, len(txt) / 20.0))                           # 末句长度(长=完整turn)
        f.append(min(1.0, (end_ms - int(lastu.get("end_ms", 0))) / 2000.0))  # 距末句结束(停顿=shift机会)
    else:
        f += [0.0, 0.0, 0.0, 1.0]
    # 2. 双声道重叠率 (I: interruption = 两声道同时活动)
    for win_ms in (2000, 5000):
        lo = end_ms - win_ms
        w = [u for u in past if int(u.get("end_ms", 0)) > lo]
        # 估计重叠: 按时间区间, 两通道同时有发声的比例
        ch1 = [(int(u["start_ms"]), int(u["end_ms"])) for u in w if int(u.get("channel_id", 1)) == 1]
        ch2 = [(int(u["start_ms"]), int(u["end_ms"])) for u in w if int(u.get("channel_id", 1)) != 1]
        overlap = 0
        for s1, e1 in ch1:
            for s2, e2 in ch2:
                overlap += max(0, min(e1, e2) - max(s1, s2))
        f.append(min(1.0, overlap / max(1, win_ms)))                  # 重叠时长占比
        # 换人次数 (turn alternation density)
        seq = sorted(w, key=lambda u: int(u.get("end_ms", 0)))
        switches = sum(1 for a, b in zip(seq, seq[1:])
                       if int(a.get("channel_id", 1)) != int(b.get("channel_id", 1)))
        f.append(float(switches))
    # 3. 最近谁在说 + 连续单声道时长 (长单声道→对方该插话=shift/interrupt)
    if past:
        last_ch = int(past[-1].get("channel_id", 1))
        run_ms = 0
        for u in reversed(past):
            if int(u.get("channel_id", 1)) == last_ch:
                run_ms = end_ms - int(u.get("start_ms", end_ms))
            else:
                break
        f.append(min(1.0, run_ms / 10000.0))                          # 当前说话人连续时长
    else:
        f.append(0.0)
    return [float(x) for x in f]


def build(conv_ids, mode: str):
    """mode: A=ctx_v1 / B=+text_feats / C=+text_feats+ti_enhanced."""
    X, Y, G = [], [], []
    for gi, cid in enumerate(conv_ids):
        a = np.load(f"data/train/labels/{cid}.npy").astype(int)
        utts = json.load(open(f"data/train/text/{cid}.json")).get("utterances", [])
        for e in range(CTX, len(a) - TGT + 1, args.stride):
            ctx = a[e - CTX:e]
            base = ctx_v1(ctx)
            end_ms = int(e * CHUNK_MS)
            if mode == "A":
                feat = base
            elif mode == "B":
                feat = np.concatenate([base, text_feats(utts, end_ms)])
            elif mode == "C":
                feat = np.concatenate([base, text_feats(utts, end_ms), ti_enhanced_feats(utts, end_ms)])
            else:  # R = robust (train/test 鲁棒, 修标点/密度 bug)
                feat = np.concatenate([base, robust_text_feats(utts, end_ms)])
            fut = set(int(x) for x in a[e:e + TGT])
            X.append(feat)
            Y.append([1 if k in fut else 0 for k in range(NUM)])
            G.append(gi)
    return np.array(X, dtype=np.float32), np.array(Y), np.array(G)


def oof_all(X, Y, G, conv_ids, folds):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(conv_ids))
    oof = np.zeros((len(X), NUM))
    for fi in range(folds):
        val = {perm[i] for i in range(len(conv_ids)) if i % folds == fi}
        tr = [i for i in range(len(X)) if G[i] not in val]
        va = [i for i in range(len(X)) if G[i] in val]
        for k in range(NUM):
            y = Y[tr, k]
            spw = (len(y) - y.sum()) / max(1, y.sum())
            c = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=48,
                               scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=SEED)
            c.fit(X[tr], y)
            oof[va, k] = c.predict_proba(X[va])[:, 1]
    return oof


def best_f1(oof_k, yt):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f = f1_score(yt, (oof_k >= t).astype(int), zero_division=0)
        if f > bf:
            bf, bt = f, t
    return bf, bt


# 变体F SOTA 固定阈值 (线上0.7124验证, 产出 t504/i65/na949 好分布)
# 铁律: 只换特征不调阈值, 避免 cap1/滑窗 CV 调激进阈值搬 test 系统性变差
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}  # C,T,BC,I,NA
THR_CYCLE1 = THR_VARF  # 别名兼容
SUBMIT_COLS = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
# SOTA 变体F 提交分布锚点 (test 1000 段) — 偏移检查用
SOTA_POS = {"c": 974, "na": 949, "i": 65, "bc": 30, "t": 504}


def cap1_macro(oof, Y, G, conv_ids, thr_map):
    """cap1 可信 CV (每通首切片, 模拟 test 独立切片): 用给定阈值算 macro."""
    seen, cap1 = set(), []
    for i in range(len(G)):
        if G[i] not in seen:
            cap1.append(i); seen.add(G[i])
    cap1 = np.array(cap1)
    per = {}
    for k in range(NUM):
        per[k] = f1_score(Y[cap1, k], (oof[cap1, k] >= thr_map[k]).astype(int), zero_division=0)
    return float(np.mean(list(per.values()))), per, len(cap1)


def do_submit():
    """生成 mode C 的 test CSV + cap1 可信验证 + 分布偏移检查.
    阈值策略 (铁律): C/NA/BC 用 cycle1 固定; T/I 用 cap1-OOF 调但限温和 [0.4,0.65]."""
    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    print(f"[ti-submit] building train (mode C)...", file=sys.stderr)
    X, Y, G = build(conv_ids, args.mode)
    oof = oof_all(X, Y, G, conv_ids, args.folds)

    # 阈值: 全用变体F SOTA 固定阈值 (铁律: 只换特征不调阈值, 保 test 分布)
    thr = dict(THR_VARF)
    macro_c, per_c, ncap = cap1_macro(oof, Y, G, conv_ids, thr)
    print(f"[ti-submit] cap1 可信 CV macro={macro_c:.4f} thr={ {LAB[k]:thr[k] for k in range(NUM)} }", file=sys.stderr)
    print(f"[ti-submit] cap1 per-class: " + " ".join(f"{LAB[k]}={per_c[k]:.3f}" for k in range(NUM)), file=sys.stderr)

    # 全量 refit → test. ★5seed 概率平均集成 (变体F 关键: 单模型对文本特征过拟合
    # 出假T正例, 5seed概率平均降方差把不稳定正例压回 — 修 t=709 暴涨根因)
    Xte, seg_ids = build_test(args.mode)
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    NSEED = 5
    preds = {}
    for k in range(NUM):
        y = Y[:, k]; spw = (len(y) - y.sum()) / max(1, y.sum())
        seed_p = []
        for s in range(NSEED):
            c = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=48,
                               scale_pos_weight=spw, n_jobs=4, verbose=-1, random_state=42 + s)
            c.fit(X, y)
            seed_p.append(c.predict_proba(Xte)[:, 1])
        pte = np.mean(seed_p, axis=0)  # 概率平均 (保稀有类绝对置信, 不被负例稀释)
        preds[k] = (pte >= thr[k]).astype(int)

    pos = {col: int(preds[COL2K[col]].sum()) for col in SUBMIT_COLS}
    print(f"\n[ti-submit] test 正例分布 vs SOTA锚点:", file=sys.stderr)
    flag = False
    for col in SUBMIT_COLS:
        d = pos[col] - SOTA_POS[col]
        warn = ""
        if col in ("c", "na") and abs(d) > 80:
            warn = "⚠️饱和类偏移大!"; flag = True
        elif col in ("t", "i") and abs(d) > 250:
            warn = "⚠️偏移过大(旧H-T2教训)"; flag = True
        print(f"    {col}: {pos[col]} (SOTA {SOTA_POS[col]}, {d:+d}) {warn}", file=sys.stderr)

    with open(run / "pred_test1.csv", "w") as f:
        f.write("segment_id," + ",".join(SUBMIT_COLS) + "\n")
        for i, sid in enumerate(seg_ids):
            f.write(",".join([sid] + [str(int(preds[COL2K[c]][i])) for c in SUBMIT_COLS]) + "\n")
    (run / "cv_metrics.json").write_text(json.dumps({
        "paradigm": "ti-text-fusion", "hypothesis_id": "H-T4",
        "cap1_macro_f1": round(macro_c, 4),
        "per_sub_cap1": {LAB[k]: round(per_c[k], 4) for k in range(NUM)},
        "thresholds": {LAB[k]: thr[k] for k in range(NUM)},
        "test_pos": pos, "dist_flag": flag,
    }, ensure_ascii=False, indent=2))
    print(f"\n[ti-submit] wrote {run}/pred_test1.csv {'⚠️分布有偏移警报,谨慎提交' if flag else '✓分布合理'}")
    print(json.dumps({"cycle": "H-T4-submit", "cap1_macro": round(macro_c, 4),
                      "per_sub": {LAB[k]: round(per_c[k], 4) for k in range(NUM)},
                      "test_pos": pos, "dist_flag": flag}))


def build_test(mode: str):
    """test 切片: 预测点=切片末(end_ms). C/NA/BC/T/I 全预测, 但只 T/I 用新文本特征受益."""
    test_ctx = sorted(glob.glob("data/test/context/*.npy"))
    seg_ids = [Path(p).stem for p in test_ctx]
    X = []
    for p in test_ctx:
        ctx = np.load(p).astype(int)
        tj = json.load(open(f"data/test/text/{Path(p).stem}.json"))
        utts = tj.get("utterances", [])
        end_ms = int(tj.get("end_ms", 30000))
        base = ctx_v1(ctx)
        if mode == "A":
            feat = base
        elif mode == "B":
            feat = np.concatenate([base, text_feats(utts, end_ms)])
        elif mode == "C":
            feat = np.concatenate([base, text_feats(utts, end_ms), ti_enhanced_feats(utts, end_ms)])
        else:  # R
            feat = np.concatenate([base, robust_text_feats(utts, end_ms)])
        X.append(feat)
    return np.array(X, dtype=np.float32), seg_ids


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--submit", action="store_true", help="生成 test CSV (mode C) + 分布检查")
    ap.add_argument("--run-dir", default="tools/runs/climb/ti-text")
    ap.add_argument("--mode", default="R", choices=["B","C","R"], help="提交特征模式(R=鲁棒)")
    args = ap.parse_args()

    if args.submit:
        return do_submit()

    conv_ids = sorted(Path(p).stem for p in glob.glob("data/train/labels/*.npy"))
    print(f"[ti-text] {len(conv_ids)} convs stride={args.stride} folds={args.folds}", file=sys.stderr)

    res = {}
    for mode, label in [("A", "ctx_v1"), ("B", "+text_lex"), ("C", "+text_lex+ti_enh"), ("R", "+robust")]:
        X, Y, G = build(conv_ids, mode)
        oof = oof_all(X, Y, G, conv_ids, args.folds)
        per = {}
        for k in range(NUM):
            bf, bt = best_f1(oof[:, k], Y[:, k])
            per[LAB[k]] = (round(bf, 4), round(bt, 2))
        macro = float(np.mean([per[LAB[k]][0] for k in range(NUM)]))
        res[label] = {"dim": X.shape[1], "macro": round(macro, 4),
                      "per": {k: v[0] for k, v in per.items()},
                      "thr": {k: v[1] for k, v in per.items()}}
        print(f"[{label:<18}] dim={X.shape[1]} macro={macro:.4f} | " +
              " ".join(f"{k}={per[k][0]:.3f}" for k in ["C", "T", "BC", "I", "NA"]), file=sys.stderr)

    print("\n=== T/I 文本增益对照 (全切片OOF相对比较; T锚0.542 I锚0.434) ===")
    a = res["ctx_v1"]
    for label, r in res.items():
        dT = r["per"]["T"] - a["per"]["T"]
        dI = r["per"]["I"] - a["per"]["I"]
        dC = r["per"]["C"] - a["per"]["C"]
        dNA = r["per"]["NA"] - a["per"]["NA"]
        dmacro = r["macro"] - a["macro"]
        print(f"  {label:<18} T={r['per']['T']:.3f}({dT:+.3f}) I={r['per']['I']:.3f}({dI:+.3f}) "
              f"| C={dC:+.3f} NA={dNA:+.3f} | macro={r['macro']:.4f}({dmacro:+.4f})")
    print(json.dumps({"cycle": "H-T4", "results": res}))


if __name__ == "__main__":
    main()
