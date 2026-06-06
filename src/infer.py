"""FinVCup 2026 复赛推理单入口 — ctx-only 骨架 (T4 step1 + D-28 dual-model fallback).

复赛约束 (D-26):
  - 测试集 2 上下文动态时长 (0, 30]s 任意 (vs 测试集 1 固定 30s = 375 chunk)
  - 预测窗仍 2s = 25 chunk, chunk = 80ms
  - 输出 pred_test1.csv (segment_id, c, na, i, bc, t)

D-28 dual-model fallback:
  - 长 ctx (默认 ≥ 20s = 250 chunk) 用 baseline ckpt (mask=0 训, 长 ctx 强)
  - 短 ctx (< 20s) 用 mask050 ckpt (CTX_MASK_PROB=0.5 训, 短 ctx 救场 +0.016)
  - 路由按 segment 原始长度 (归一化前) 决定, 整段同一 ckpt
  - 估真分 +0.009 vs 单 ckpt 均匀分布

入口:
  # 单 ckpt 模式 (向后兼容):
  python -m src.infer --ckpt_dir /app/models/ctx_only \\
                     --test_root /data/test \\
                     --output_csv /output/pred_test1.csv

  # Dual-model 模式 (D-28 推荐):
  python -m src.infer --ckpt_dir /app/models/ctx_only \\
                     --ckpt_dir_short /app/models/ctx_only_mask050 \\
                     --ctx_route_threshold_chunks 250 \\
                     --test_root /data/test \\
                     --output_csv /output/pred_test1.csv

约定:
  ckpt_dir/        ← cycle_context.py (CTX_MASK_PROB=0) dump 的 baseline ckpt
  ckpt_dir_short/  ← cycle_context.py (CTX_MASK_PROB=0.5) dump 的 mask050 ckpt (可选)
  test_root/
    context/<segment_id>.npy   ← 任意长度 label 序列 N ∈ [1, 375]
    audio/<segment_id>.wav     (本骨架未使用; R4 全栈升级时启用)
    text/<segment_id>.json     (本骨架未使用)

骨架版只用 ctx 信号 (公榜 0.71 cycle1 真分基准). R4 全栈 SSL/Omni 升级在后续 step.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import joblib
import numpy as np

# 复用 cycle_context.featurize 和 dynamic_ctx_utils — 确保 train/infer 特征一致
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.climb.cycle_context import featurize  # noqa: E402
from tools.climb.dynamic_ctx_utils import normalize_ctx_to_375  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FinVCup ctx-only 推理 (复赛骨架 + D-28 dual-model)")
    p.add_argument("--ckpt_dir", required=True,
                   help="baseline ckpt 目录 (cycle_context.py CTX_MASK_PROB=0 dump 出)")
    p.add_argument("--ckpt_dir_short", default=None,
                   help="短 ctx ckpt 目录 (cycle_context.py CTX_MASK_PROB=0.5 dump 出). "
                        "不设 = 单 ckpt 模式向后兼容")
    p.add_argument("--ctx_route_threshold_chunks", type=int, default=250,
                   help="dual-model 路由阈值: ctx 原始长度 < 此值用 ckpt_dir_short, ≥ 用 ckpt_dir. "
                        "默认 250 = 20s (D-28 策略 A 保守推荐). 仅当 --ckpt_dir_short 设了才生效")
    p.add_argument("--test_root", required=True,
                   help="测试根目录, 含 context/<id>.npy (audio/text 本骨架不用)")
    p.add_argument("--output_csv", required=True, help="输出 pred_test1.csv 路径")
    p.add_argument("--ctx_mode", default="pad_na_left",
                   choices=["pad_na_left", "pad_loop", "truncate_only"],
                   help="变长 context 归一化模式 (默认 pad_na_left 假设早期未观测)")
    p.add_argument("--max_segments", type=int, default=None,
                   help="仅推前 N 段 (冒烟测试用)")
    return p.parse_args()


def load_ckpt(ckpt_dir: Path) -> tuple[dict, dict, dict]:
    """加载 5 个 LGBM + thresholds + spec.

    Security: joblib.load 用 pickle, 仅信任 cycle_context.py 当场 dump 的本地 ckpt
    (镜像构建期 COPY 进来). 不接受外部上传的 ckpt 文件.
    """
    spec_path = ckpt_dir / "feature_spec.json"
    thr_path = ckpt_dir / "thresholds.json"
    if not spec_path.exists() or not thr_path.exists():
        raise FileNotFoundError(
            f"ckpt_dir {ckpt_dir} 缺 feature_spec.json 或 thresholds.json; "
            f"请先跑 tools/climb/cycle_context.py 生成"
        )
    spec = json.loads(spec_path.read_text())
    thresholds = json.loads(thr_path.read_text())

    submit_cols = spec["submit_cols"]
    models = {}
    for col in submit_cols:
        ckpt_file = ckpt_dir / f"lgbm_{col}.joblib"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"缺 {ckpt_file}")
        models[col] = joblib.load(ckpt_file)
    print(f"[infer] loaded {len(models)} LGBM models from {ckpt_dir}", file=sys.stderr)
    print(f"[infer] thresholds: {thresholds}", file=sys.stderr)
    return models, thresholds, spec


def infer_one_segment(
    ctx: np.ndarray,
    models_long: dict, thresholds_long: dict,
    models_short: dict | None, thresholds_short: dict | None,
    ctx_threshold: int,
    submit_cols: list[str], ctx_mode: str,
) -> tuple[dict[str, int], str]:
    """单段推理: 路由选 ckpt → 任意长度 ctx 适配 375 → featurize → 5 LGBM → 阈值化.

    D-28 dual-model fallback:
      - 路由决策按 ctx **原始长度** (归一化前), 不是归一化后的 375
      - n_orig >= ctx_threshold → models_long (baseline, 长 ctx 强)
      - n_orig <  ctx_threshold → models_short (mask050, 短 ctx 救场)
      - models_short=None → 全走 models_long (单 ckpt 向后兼容)

    Returns:
        (pred dict, route 标签 "long"/"short")
    """
    n_orig = len(ctx)
    if models_short is None or n_orig >= ctx_threshold:
        models, thresholds, route = models_long, thresholds_long, "long"
    else:
        # models_short is not None ⇒ thresholds_short 也必非 None (load_ckpt 同生同灭)
        assert thresholds_short is not None
        models, thresholds, route = models_short, thresholds_short, "short"

    ctx_375 = normalize_ctx_to_375(ctx.astype(np.int32), mode=ctx_mode)
    feat = featurize(ctx_375).reshape(1, -1)
    pred = {}
    for col in submit_cols:
        prob = models[col].predict_proba(feat)[0, 1]
        pred[col] = int(prob >= thresholds[col])
    return pred, route


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    test_root = Path(args.test_root)
    output_csv = Path(args.output_csv)

    models_long, thresholds_long, spec = load_ckpt(ckpt_dir)
    submit_cols = spec["submit_cols"]

    # D-28 dual-model: 可选加载短 ctx ckpt
    if args.ckpt_dir_short:
        ckpt_dir_short = Path(args.ckpt_dir_short)
        models_short, thresholds_short, spec_short = load_ckpt(ckpt_dir_short)
        # sanity: 两 ckpt 的 submit_cols 必须一致
        if spec_short["submit_cols"] != submit_cols:
            raise ValueError(
                f"ckpt_dir submit_cols {submit_cols} != "
                f"ckpt_dir_short submit_cols {spec_short['submit_cols']}"
            )
        print(f"[infer] dual-model mode: long ckpt + short ckpt (route θ={args.ctx_route_threshold_chunks} chunks "
              f"= {args.ctx_route_threshold_chunks * 80 / 1000:.1f}s)", file=sys.stderr)
    else:
        models_short, thresholds_short = None, None
        print(f"[infer] single-ckpt mode (long ckpt only)", file=sys.stderr)

    ctx_dir = test_root / "context"
    ctx_files = sorted(glob.glob(str(ctx_dir / "*.npy")))
    if args.max_segments:
        ctx_files = ctx_files[: args.max_segments]
    if not ctx_files:
        raise FileNotFoundError(f"no .npy under {ctx_dir}/")
    print(f"[infer] {len(ctx_files)} segments under {ctx_dir}/", file=sys.stderr)

    # 长度分布统计 (诊断变长输入)
    lens = [len(np.load(p)) for p in ctx_files[: min(50, len(ctx_files))]]
    print(f"[infer] ctx-len sample (first {len(lens)}): min={min(lens)} max={max(lens)} "
          f"mean={np.mean(lens):.0f}", file=sys.stderr)

    rows = []
    route_counts = {"long": 0, "short": 0}
    for ctx_file in ctx_files:
        seg_id = Path(ctx_file).stem
        ctx = np.load(ctx_file)
        pred, route = infer_one_segment(
            ctx,
            models_long, thresholds_long,
            models_short, thresholds_short,
            args.ctx_route_threshold_chunks,
            submit_cols, args.ctx_mode,
        )
        route_counts[route] += 1
        row = {"segment_id": seg_id, **pred}
        rows.append(row)

    # 写 csv (segment_id 升序, 列序按 spec.submit_cols)
    rows.sort(key=lambda r: r["segment_id"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        # lineterminator='\n' 强制 LF 行尾, 跟 cycle_context.py 等 csv 输出一致 (避免 CRLF/LF 提交差异)
        w = csv.DictWriter(f, fieldnames=["segment_id"] + submit_cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # 统计 pos 分布 + 路由分布
    pos = {c: sum(r[c] for r in rows) for c in submit_cols}
    print(f"[infer] wrote {len(rows)} rows → {output_csv}", file=sys.stderr)
    print(f"[infer] pos dist: {pos}", file=sys.stderr)
    if args.ckpt_dir_short:
        n_total = route_counts["long"] + route_counts["short"]
        print(f"[infer] route dist: long={route_counts['long']} ({100*route_counts['long']/n_total:.1f}%) "
              f"short={route_counts['short']} ({100*route_counts['short']/n_total:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
