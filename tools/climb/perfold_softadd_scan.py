"""per-fold / per-ckpt 软加扫 (H-D22-2 + H-D22-11).

Omni 5 fold + 4 heads × 15 ckpt = 65 独立信号源.

策略:
  1. 各 fold/ckpt 单独跑 cap1 + per-class, 找最强 fold (而非 mean)
  2. 强 fold/ckpt 替 mean 加进 SOTA 软加
  3. 多 fold 选择性融合 (vs naive mean)

注意: per_ckpt_test 只有 test (1000, 5), 无 OOF.
所以这里只算 test predict 输出, 真分校准要靠 push.

Usage:
  python3 tools/climb/perfold_softadd_scan.py [--top 20]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LAB = {0: "C", 1: "T", 2: "BC", 3: "I", 4: "NA"}
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
SUBMIT = ["c", "na", "i", "bc", "t"]
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}


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


def make_sota_3src_test(ctx_t, wsp_t, hub_t):
    p = np.zeros_like(ctx_t)
    p[:, 0] = ctx_t[:, 0]
    p[:, 1] = 0.7 * wsp_t[:, 1] + 0.3 * hub_t[:, 1]
    p[:, 2] = ctx_t[:, 2]
    p[:, 3] = (ctx_t[:, 3] + wsp_t[:, 3] + hub_t[:, 3]) / 3
    p[:, 4] = ctx_t[:, 4]
    return p


def softadd(base, extra, w, cols):
    p = base.copy()
    for k in cols:
        p[:, k] = (1 - w) * base[:, k] + w * extra[:, k]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(f"tools/runs/climb/perfold-scan-{time.strftime('%Y%m%d-%H%M')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 SOTA test (用 single-seed 版同 build_softadd_candidates_v2)
    zc = np.load("tools/runs/climb/_stack_cache_s40.npz")
    ctx_t = zc["te_lgbm_v1"].astype(np.float32)
    wsp_t = np.load("tools/runs/climb/whisper-fusion-20260531-0143/probs.npz")["test"].astype(np.float32)
    hub_t = np.load("tools/runs/climb/hubert-bcaug-head-20260601-1533/probs.npz")["test"].astype(np.float32)
    sota_test = make_sota_3src_test(ctx_t, wsp_t, hub_t)

    test_ids = np.load("tools/runs/climb/whisper-fusion-20260531-0143/probs.npz")["test_ids"]

    print("=== per-fold / per-ckpt 信号源 ===\n")

    # 2. Omni 7B per-fold (5 fold)
    omni_pf = np.load("tools/runs/climb/omni-lora-20260602-1002/probs_perfold.npz")["per_fold"]
    print(f"Omni-7B per-fold: shape={omni_pf.shape}")
    # 看每 fold 跟 mean 的差距 (用 test 自身比, 无 ground truth, 只看分布)
    omni_mean = omni_pf.mean(axis=0)
    for f in range(5):
        diff_norm = np.linalg.norm(omni_pf[f] - omni_mean, axis=0)
        print(f"  fold {f}: norm(fold - mean) per class = {[f'{d:.2f}' for d in diff_norm]}")

    # 3. 4 heads multiseed per_ckpt (15 ckpt each)
    multi_sources = {}
    for name in ["hubert", "w2v2", "e2v", "whisper"]:
        path = sorted(Path("tools/runs/climb").glob(f"{name}-bcaug-multiseed-*/per_ckpt_test.npz"))
        if not path:
            print(f"  {name} multiseed: SKIP")
            continue
        z = np.load(path[-1])
        multi_sources[name] = z["per_ckpt"]  # (15, 1000, 5)
        print(f"{name} multi-seed per_ckpt: shape={multi_sources[name].shape}")

    print()

    # 4. 候选生成 — 关键 push 候选 (用 OOF 不可算, 只算 test pos)
    candidates = []

    def make_csv(probs_test, name, desc):
        sub_dir = out_dir / name
        sub_dir.mkdir(exist_ok=True)
        pos = {c: 0 for c in SUBMIT}
        with open(sub_dir / "pred_test1.csv", "w") as f:
            f.write("segment_id," + ",".join(SUBMIT) + "\n")
            for i, sid in enumerate(test_ids):
                vals = {c: int(probs_test[i, COL2K[c]] >= THR_VARF[COL2K[c]]) for c in SUBMIT}
                for c, v in vals.items():
                    pos[c] += v
                f.write(",".join([str(sid)] + [str(vals[c]) for c in SUBMIT]) + "\n")
        return pos

    # A. Omni 5 个 fold 独立各软加 0.2 (5 候选)
    for f in range(5):
        p = softadd(sota_test, omni_pf[f], 0.2, (1, 2, 3))
        name = f"PF_omni_fold{f}_w020"
        pos = make_csv(p, name, f"SOTA + Omni fold{f} 0.2 T/BC/I")
        candidates.append({"name": name, "desc": f"Omni fold {f} 0.2 softadd", "test_pos": pos, "source": "omni_per_fold"})

    # B. Omni 5 fold 中位数 (replace mean with median for robustness)
    omni_median = np.median(omni_pf, axis=0)
    p = softadd(sota_test, omni_median, 0.2, (1, 2, 3))
    pos = make_csv(p, "PF_omni_median_w020", "SOTA + Omni 5fold median 0.2")
    candidates.append({"name": "PF_omni_median_w020", "desc": "Omni median 5fold 0.2", "test_pos": pos, "source": "omni_median"})

    # C. Omni 5 fold std-weighted (高 std 的 fold 权重高 — 多样性导向)
    omni_std = omni_pf.std(axis=0)  # (1000, 5)
    # 每个 sample 用 std 加权融合 5 fold (近似 hi-info ckpt)
    # 简化: 直接用 max-per-sample (实际是 5 fold confident vote)
    omni_max = omni_pf.max(axis=0)  # (1000, 5)
    p = softadd(sota_test, omni_max, 0.2, (1, 2, 3))
    pos = make_csv(p, "PF_omni_max_w020", "SOTA + Omni 5fold max 0.2 (confident)")
    candidates.append({"name": "PF_omni_max_w020", "desc": "Omni max (per-sample) 5fold 0.2", "test_pos": pos, "source": "omni_max"})

    # D. 每个 head 15 ckpt mean vs 单 ckpt 最优
    for hname, perck in multi_sources.items():
        # mean of 15 ckpt
        ckpt_mean = perck.mean(axis=0)
        p = softadd(sota_test, ckpt_mean, 0.2, (1, 2, 3))
        pos = make_csv(p, f"PF_{hname}_ms_mean_w020", f"SOTA + {hname}_ms 15-ckpt mean 0.2")
        candidates.append({"name": f"PF_{hname}_ms_mean_w020", "desc": f"{hname} 15-ckpt mean 0.2",
                          "test_pos": pos, "source": f"{hname}_ckpt_mean"})

    # E. 最佳 ckpt 选择 (15 ckpt 中找 cap1 最高的) — 但无 OOF, 暂用 pos count 接近 SOTA pos 的
    # 已知 SOTA pos: c=975 na=947 i=81 bc=27 t=522
    # 选 pos 跟 SOTA 最近的 1 ckpt (避免极端)
    sota_pos_target = np.array([975, 947, 81, 27, 522])  # c na i bc t order
    for hname, perck in multi_sources.items():
        ckpt_distances = []
        for c_idx in range(15):
            ckpt_p = perck[c_idx]
            pred_pos = np.array([
                (ckpt_p[:, COL2K[c]] >= THR_VARF[COL2K[c]]).sum() for c in SUBMIT
            ])
            dist = np.abs(pred_pos - sota_pos_target).sum()
            ckpt_distances.append(dist)
        best_ckpt = int(np.argmin(ckpt_distances))
        p = softadd(sota_test, perck[best_ckpt], 0.2, (1, 2, 3))
        name = f"PF_{hname}_bestckpt{best_ckpt:02d}_w020"
        pos = make_csv(p, name, f"SOTA + {hname} ckpt {best_ckpt} (pos-closest) 0.2")
        candidates.append({"name": name, "desc": f"{hname} ckpt {best_ckpt} 0.2",
                          "test_pos": pos, "source": f"{hname}_best_ckpt", "ckpt_idx": best_ckpt})

    # 输出
    print(f"\n=== {len(candidates)} per-fold/per-ckpt 候选生成 ===\n")
    print(f"{'name':<40} | c   na  i   bc  t  ")
    print("-" * 80)
    for c in candidates:
        p = c["test_pos"]
        print(f"{c['name']:<40} | {p['c']:>3} {p['na']:>3} {p['i']:>3} {p['bc']:>3} {p['t']:>3}")

    print(f"\nSOTA-3src test pos 参考: c=975 na=947 i=81 bc=27 t=522")

    # MANIFEST
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "_note": "per-fold + per-ckpt 软加扫. 无 OOF (per_ckpt_test 仅 test), 真分校准必靠 push.",
        "sota_pos_target": {"c": 975, "na": 947, "i": 81, "bc": 27, "t": 522},
        "candidates": candidates,
    }, ensure_ascii=False, indent=2))
    print(f"\n落盘: {out_dir}/MANIFEST.json")


if __name__ == "__main__":
    main()
