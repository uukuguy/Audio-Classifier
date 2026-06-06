"""V1/V2 dual-model 公榜验证 csv 生成 (D-28 task E).

生成两个公榜验证候选:

  V1 (sanity): 全 30s ctx 喂 dual-ckpt
       - 期望: 全走 long route, 真分 = baseline 0.7458 ± 0.001
       - 用途: 验路由不破 SOTA, dual-ckpt mode 加载/写 csv 不出错

  V2 (real): 测试集 1 一半段 ctx 截短到 10s (125 chunk), 另一半保 30s, 混合喂 dual-ckpt
       - 期望: 一半走 long (30s baseline ckpt), 一半走 short (10s mask050 ckpt)
       - 用途: 验 dual-model 路由实际改善混合分布场景的真分
       - 估真分 (D-28 设计): 0.730-0.745 区间, 比单 ckpt 强 +0.005-0.009

入口:
  python tools/climb/build_dual_model_validation.py \\
      --ckpt_dir models/ctx_only \\
      --ckpt_dir_short models/ctx_only_mask050 \\
      --ctx_route_threshold_chunks 250 \\
      --test_root data/test \\
      --out_dir submission/dual-model-validation-20260606-XXXX/

输出:
  out_dir/V1_full_30s_sanity/pred_test1.csv + MANIFEST.json
  out_dir/V2_half_truncated_to_10s/{pred_test1.csv, MANIFEST.json}
  out_dir/MANIFEST.json (顶层, 含 V1/V2 metadata)

注意:
  - 短 ctx ckpt 没训完时, 可用 --ckpt_dir_short_fallback_to_long 让 V1/V2 都用 baseline ckpt
    跑 sanity (V1 应等于 baseline 0.7458)
  - V2 截短策略: 按 segment_id mod 2 == 0 截短到 125 chunk, == 1 保 375 (确定性, 不随机)
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1/V2 dual-model 公榜验证 csv 生成")
    p.add_argument("--ckpt_dir", default="models/ctx_only",
                   help="baseline ckpt (long ctx). 默认 models/ctx_only")
    p.add_argument("--ckpt_dir_short", default=None,
                   help="mask050 ckpt (short ctx). 不设 + --ckpt_dir_short_fallback_to_long → "
                        "短 ctx 也用 baseline (V1 sanity 用)")
    p.add_argument("--ckpt_dir_short_fallback_to_long", action="store_true",
                   help="mask050 ckpt 没训完时, 让 short = long, 跑 V1 sanity")
    p.add_argument("--ctx_route_threshold_chunks", type=int, default=250,
                   help="dual-model 路由阈值 (默认 250 = 20s, D-28 策略 A)")
    p.add_argument("--test_root", default="data/test", help="测试集 1 根目录")
    p.add_argument("--out_dir", required=True,
                   help="输出目录, V1/V2 各一个子目录")
    p.add_argument("--variants", default="V1,V2",
                   help="逗号分隔, 选 V1 或 V2 或 V1,V2 (默认全跑)")
    p.add_argument("--max_segments", type=int, default=None,
                   help="冒烟用, 仅推前 N 段")
    return p.parse_args()


def materialize_v2_test(orig_test_root: Path, work_root: Path, keep_chunks: int = 125) -> Path:
    """V2: 把测试集 ctx 一半截短到 keep_chunks, 一半保 375.

    确定性策略: segment_id (int) mod 2 == 0 → 截短, 否则保留. 复现稳定.

    Args:
        orig_test_root: 原 data/test 根
        work_root: 临时工作根, 会建 work_root/{context, audio, text}/ 链接 + 改写 context
        keep_chunks: 截短到多少 chunk (默认 125 = 10s)

    Returns:
        work_root (作为新 test_root)
    """
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "context").mkdir(exist_ok=True)
    # audio / text 不动: 直接 symlink (本骨架不读, 但 src/infer.py 不查 audio/text)

    ctx_files = sorted(glob.glob(str(orig_test_root / "context" / "*.npy")))
    n_trunc = 0
    n_keep = 0
    for ctx_file in ctx_files:
        seg_id = Path(ctx_file).stem
        ctx = np.load(ctx_file)
        if int(seg_id) % 2 == 0:
            ctx = ctx[-keep_chunks:]  # 截末 keep_chunks (因果)
            n_trunc += 1
        else:
            n_keep += 1
        out = work_root / "context" / f"{seg_id}.npy"
        np.save(out, ctx)

    print(f"[V2 materialize] truncated {n_trunc} segs → {keep_chunks} chunks "
          f"({keep_chunks * 80 / 1000:.1f}s), kept {n_keep} segs at 375 chunks", file=sys.stderr)
    return work_root


def run_infer(
    ckpt_dir: Path, ckpt_dir_short: Path | None,
    test_root: Path, output_csv: Path,
    threshold_chunks: int, max_segments: int | None,
) -> dict:
    """调 src.infer 子进程跑推理, 返回 pos dist + route dist 解析后的 stats."""
    cmd = [
        sys.executable, "-m", "src.infer",
        "--ckpt_dir", str(ckpt_dir),
        "--test_root", str(test_root),
        "--output_csv", str(output_csv),
        "--ctx_route_threshold_chunks", str(threshold_chunks),
    ]
    if ckpt_dir_short is not None:
        cmd.extend(["--ckpt_dir_short", str(ckpt_dir_short)])
    if max_segments is not None:
        cmd.extend(["--max_segments", str(max_segments)])

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"[run_infer] FAILED:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"src.infer 失败, returncode={result.returncode}")

    # 解析 stderr 拿 pos / route dist
    stats = {"pos": {}, "route": {}}
    for line in result.stderr.splitlines():
        if "[infer] pos dist:" in line:
            # ast.literal_eval 只解析字面量 dict/list/数字, 不执行任意代码
            # (相比 eval, 即使 src/infer.py 被污染输出含表达式也安全)
            try:
                stats["pos"] = ast.literal_eval(line.split("pos dist:", 1)[1].strip())
            except (ValueError, SyntaxError):
                pass
        elif "[infer] route dist:" in line:
            # 例: "[infer] route dist: long=500 (50.0%) short=500 (50.0%)"
            parts = line.split("route dist:", 1)[1].strip()
            for tok in parts.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        stats["route"][k] = int(v)
                    except ValueError:
                        pass
    return stats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"baseline ckpt_dir 不存在: {ckpt_dir}")

    if args.ckpt_dir_short:
        ckpt_dir_short = Path(args.ckpt_dir_short)
        if not ckpt_dir_short.exists():
            raise FileNotFoundError(f"short ckpt 不存在: {ckpt_dir_short}")
    elif args.ckpt_dir_short_fallback_to_long:
        ckpt_dir_short = ckpt_dir  # fallback: 短 = 长 (V1 sanity)
        print(f"[main] ⚠ fallback: short ckpt = long ckpt ({ckpt_dir}). "
              f"V1 应等于 baseline, V2 不可用 (本质单 ckpt)", file=sys.stderr)
    else:
        raise ValueError("必须指定 --ckpt_dir_short 或 --ckpt_dir_short_fallback_to_long")

    variants = args.variants.split(",")
    top_manifest: dict = {
        "_note": "D-28 dual-model 公榜验证. V1 sanity (全 30s), V2 real (一半截到 10s).",
        "_created": datetime.now().isoformat(timespec="seconds"),
        "ckpt_dir": str(ckpt_dir),
        "ckpt_dir_short": str(ckpt_dir_short),
        "ctx_route_threshold_chunks": args.ctx_route_threshold_chunks,
        "fallback_to_long": args.ckpt_dir_short_fallback_to_long,
        "variants": {},
    }

    test_root_orig = Path(args.test_root)

    # V1: 全 30s ctx (用原 test_root)
    if "V1" in variants:
        v1_dir = out_dir / "V1_full_30s_sanity"
        v1_dir.mkdir(exist_ok=True)
        v1_csv = v1_dir / "pred_test1.csv"
        print(f"\n[V1] sanity: 全 30s ctx 喂 dual-ckpt", file=sys.stderr)
        stats = run_infer(
            ckpt_dir, ckpt_dir_short, test_root_orig, v1_csv,
            args.ctx_route_threshold_chunks, args.max_segments,
        )
        v1_manifest = {
            "_role": "V1 sanity: 全 30s ctx 喂 dual-ckpt, 应全走 long route, 真分 = baseline 0.7458",
            "_expected_route": {"long": 1000, "short": 0} if args.max_segments is None else None,
            "_expected_score": "0.7458 ± 0.001 (= baseline single ckpt)",
            "stats": stats,
        }
        (v1_dir / "MANIFEST.json").write_text(json.dumps(v1_manifest, indent=2, ensure_ascii=False))
        top_manifest["variants"]["V1"] = v1_manifest
        print(f"[V1] done → {v1_csv} | pos={stats['pos']} route={stats['route']}", file=sys.stderr)

    # V2: 一半截到 10s, 一半保 30s
    if "V2" in variants:
        if args.ckpt_dir_short_fallback_to_long:
            print(f"\n[V2] skip: fallback 模式下 V2 跟 V1 等价, 跳过", file=sys.stderr)
        else:
            v2_dir = out_dir / "V2_half_truncated_to_10s"
            v2_dir.mkdir(exist_ok=True)
            v2_csv = v2_dir / "pred_test1.csv"
            print(f"\n[V2] real: 一半段 ctx 截 10s, 一半保 30s, 喂 dual-ckpt", file=sys.stderr)

            with tempfile.TemporaryDirectory(prefix="v2_test_") as tmpdir:
                tmp_test_root = materialize_v2_test(
                    test_root_orig, Path(tmpdir) / "test", keep_chunks=125,
                )
                stats = run_infer(
                    ckpt_dir, ckpt_dir_short, tmp_test_root, v2_csv,
                    args.ctx_route_threshold_chunks, args.max_segments,
                )
            v2_manifest = {
                "_role": "V2 real: 一半截 10s 一半保 30s, 模拟测试集 2 混合分布",
                "_truncate_strategy": "seg_id mod 2 == 0 → 截末 125 chunk (10s), else 保 375",
                "_expected_route": "long=500 (≥250) + short=500 (<250) at θ=250",
                "_expected_score": "0.730-0.745 (设计估算), 跟单 ckpt 对照看路由实际改善",
                "stats": stats,
            }
            (v2_dir / "MANIFEST.json").write_text(json.dumps(v2_manifest, indent=2, ensure_ascii=False))
            top_manifest["variants"]["V2"] = v2_manifest
            print(f"[V2] done → {v2_csv} | pos={stats['pos']} route={stats['route']}", file=sys.stderr)

    # 顶层 manifest
    (out_dir / "MANIFEST.json").write_text(json.dumps(top_manifest, indent=2, ensure_ascii=False))
    print(f"\n[main] all done → {out_dir}/", file=sys.stderr)
    print(f"[main] top manifest: {out_dir}/MANIFEST.json", file=sys.stderr)


if __name__ == "__main__":
    main()
