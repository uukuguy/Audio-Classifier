"""比较 multi-seed 各 seed 的 test pos (找最强 seed).

per_ckpt_test (15, 1000, 5) 含 meta (15, 2) 标 seed/fold.
按 seed 聚合: 3 seed × (5 fold mean) → 看哪个 seed 跟 SOTA test pos 最接近.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
COL2K = {"c": 0, "na": 4, "i": 3, "bc": 2, "t": 1}
SUBMIT = ["c", "na", "i", "bc", "t"]


def main():
    # SOTA-3src test pos target (已知)
    sota_pos = {"c": 975, "na": 947, "i": 81, "bc": 27, "t": 522}

    print("SOTA-3src test pos target: " + " ".join(f"{c}={sota_pos[c]}" for c in SUBMIT))
    print()

    for name in ["hubert", "w2v2", "e2v", "whisper"]:
        path_glob = list(Path("tools/runs/climb").glob(f"{name}-bcaug-multiseed-*/per_ckpt_test.npz"))
        if not path_glob:
            print(f"{name}: SKIP\n"); continue
        z = np.load(path_glob[-1])
        per_ckpt = z["per_ckpt"]  # (15, 1000, 5)
        meta = z["meta"]          # (15, 2) — seed, fold
        seeds_unique = sorted(set(int(meta[i, 0]) for i in range(len(meta))))

        print(f"=== {name} (seeds={seeds_unique}) ===")
        for sd in seeds_unique:
            sd_mask = meta[:, 0] == sd
            seed_probs = per_ckpt[sd_mask]  # (5, 1000, 5) for 5 folds
            seed_mean = seed_probs.mean(axis=0)
            pos = {c: int((seed_mean[:, COL2K[c]] >= THR_VARF[COL2K[c]]).sum()) for c in SUBMIT}
            dist = sum(abs(pos[c] - sota_pos[c]) for c in SUBMIT)
            print(f"  seed={sd}: pos = " + " ".join(f"{c}={pos[c]:>3}" for c in SUBMIT)
                  + f"  |  dist={dist}")

        # 全 15 ckpt mean (跟现有 mean 路径一致)
        all_mean = per_ckpt.mean(axis=0)
        pos = {c: int((all_mean[:, COL2K[c]] >= THR_VARF[COL2K[c]]).sum()) for c in SUBMIT}
        dist = sum(abs(pos[c] - sota_pos[c]) for c in SUBMIT)
        print(f"  ALL mean: pos = " + " ".join(f"{c}={pos[c]:>3}" for c in SUBMIT) + f"  |  dist={dist}")
        print()


if __name__ == "__main__":
    main()
