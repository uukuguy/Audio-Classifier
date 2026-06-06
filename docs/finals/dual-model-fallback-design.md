# Dual-Model Fallback 设计 (D-28, 6/6)

> 起点: D-28 mask sweep 教训 (单一 mask 任何 prob 公榜均匀都比 baseline 差 -0.001~-0.005, dual-model 估真分 0.7417 = +0.009).
>
> 现状: src/infer.py docker 骨架 ctx-only 已验通 (5.3MB ckpt, 1000 段 5s, 三重等价). 本设计扩展到 dual-ckpt 路由。

## 一句话

**根据复赛输入 ctx 的原始长度 N (chunk 单位), 路由到不同 ckpt: N ≥ θ 用 baseline (长 ctx 强), N < θ 用 mask050 (短 ctx 救场)**, 阈值 θ 由公榜实测确定。

## 已知公榜数据 (6/6 D-28 落定)

| 配置 | 真分 | 备注 |
|---|---|---|
| R4 baseline 30s | **0.745798** | SOTA-base, 长 ctx 强 |
| R4 baseline 10s | 0.721787 | 跌 -0.024 |
| R4 baseline 5s | 0.707016 | 跌 -0.039 |
| R4 mask050 30s | 0.727898 | 跌 -0.018 vs baseline (mask 伤长 ctx) |
| R4 mask050 10s | 0.737580 | 救 +0.016 vs baseline (mask 救短 ctx) |
| R4 mask040 30s | 0.724527 | 跌 -0.021 (sweep 反向) |
| R4 mask040 10s | 0.732465 | 救 +0.011 |

**交叉点 θ_opt** (baseline 跟 mask050 真分相等):
- baseline(30s)=0.7458, mask050(30s)=0.7279 → baseline 强 +0.018
- baseline(10s)=0.7218, mask050(10s)=0.7376 → mask050 强 +0.016
- **15s-20s 之间某点**两者相等 (线性插值估 ≈ 18-19s 左右)

## 路由策略 (按推荐度排)

### 策略 A (推荐, 保守): θ = 20s (= 250 chunk)

```python
if N >= 250:  # ≥ 20s
    ckpt = baseline
else:
    ckpt = mask050
```

**估真分** (假设复赛分布 (0, 30]s 均匀):
- N ∈ [20, 30]s (33% 段): 用 baseline, 真分 ≈ 0.745 (强区)
- N ∈ (0, 20]s (67% 段): 用 mask050, 真分 ≈ 0.730-0.738
- 加权均: ≈ 0.737

**风险**: baseline 在 20-25s 已开始退化 (估真分 0.736-0.745), mask050 在 18-20s 已优于 baseline → 阈值偏保守, 漏掉 mask050 救场窗口

### 策略 B (激进): θ = 15s (= 188 chunk)

```python
if N >= 188:  # ≥ 15s
    ckpt = baseline
else:
    ckpt = mask050
```

**估真分** (复赛分布 (0, 30]s 均匀):
- N ∈ [15, 30]s (50% 段): baseline, 真分 ≈ 0.735-0.745
- N ∈ (0, 15]s (50% 段): mask050, 真分 ≈ 0.726-0.738
- 加权均: ≈ 0.738

**风险**: mask050 在 18s 处可能还不强于 baseline → 15-18s 区间用错 ckpt 伤 -0.005~-0.010

### 策略 C (最优, 需公榜 calibration): 三段路由 θ_1 / θ_2

```python
if N >= 250:    # ≥ 20s
    ckpt = baseline
elif N >= 188:  # 15-20s
    ckpt = average(baseline, mask050)  # 软融合
else:           # < 15s
    ckpt = mask050
```

需要额外 1 push 验中段 15-20s 真分 → 6/7-6/8 投。

## 推荐: 先 A (θ=20s), 拿到真分后微调到 B 或 C

理由: D-28 教训"评估错配"刚被狠狠教训, 不再相信本机线性插值。**先发简单策略 A 拿公榜真分, 再调**。

## 实现改动 (3 步)

### 步骤 1: 训 mask050 ckpt (5-8h 云端 GPU 或本机 4 线程)

```bash
# 本机 (3-4h, OMP 限线程)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  CTX_MASK_PROB=0.5 \
  python3 tools/climb/cycle_context.py \
    --output_dir tools/runs/climb/ctx_mask050_$(date +%Y%m%d-%H%M)

# 出物搬到 models/ctx_only_mask050/
cp tools/runs/climb/ctx_mask050_*/lgbm_*.joblib models/ctx_only_mask050/
cp tools/runs/climb/ctx_mask050_*/thresholds.json models/ctx_only_mask050/
cp tools/runs/climb/ctx_mask050_*/feature_spec.json models/ctx_only_mask050/
```

**注**: cycle_context.py 已支持 CTX_MASK_PROB env var (D-28 commit 75799ad)。

### 步骤 2: 改 src/infer.py 加路由

```python
# parse_args 加:
p.add_argument("--ckpt_dir_short", default=None,
               help="短 ctx ckpt (mask050 训出). 不设则单 ckpt 模式 (向后兼容).")
p.add_argument("--ctx_route_threshold_chunks", type=int, default=250,
               help="ctx 路由阈值, < 此值用 ckpt_dir_short. 默认 250 = 20s.")

# main 里改 load_ckpt:
models_long, thresholds_long, spec = load_ckpt(Path(args.ckpt_dir))
if args.ckpt_dir_short:
    models_short, thresholds_short, _ = load_ckpt(Path(args.ckpt_dir_short))

# infer_one_segment 改路由:
def infer_one_segment(ctx, models_long, thresholds_long,
                      models_short, thresholds_short,
                      ctx_threshold, submit_cols, ctx_mode):
    n_orig = len(ctx)  # 原始长度 (归一化前)
    ctx_375 = normalize_ctx_to_375(ctx.astype(np.int32), mode=ctx_mode)
    feat = featurize(ctx_375).reshape(1, -1)

    # 路由
    if models_short is None or n_orig >= ctx_threshold:
        models, thresholds = models_long, thresholds_long
        route = "long"
    else:
        models, thresholds = models_short, thresholds_short
        route = "short"

    pred = {}
    for col in submit_cols:
        prob = models[col].predict_proba(feat)[0, 1]
        pred[col] = int(prob >= thresholds[col])
    return pred, route
```

### 步骤 3: Docker copy 双 ckpt

```dockerfile
COPY models/ctx_only/ /app/models/ctx_only/                    # baseline (long)
COPY models/ctx_only_mask050/ /app/models/ctx_only_mask050/    # mask050 (short)
ENTRYPOINT ["python", "-m", "src.infer", \
           "--ckpt_dir", "/app/models/ctx_only", \
           "--ckpt_dir_short", "/app/models/ctx_only_mask050", \
           "--ctx_route_threshold_chunks", "250"]
```

镜像体积估 +5.3MB (mask050 ckpt 跟 baseline 同体量) = 总 395MB 仍轻。

## 验证 plan

1. **本机 dry-run 1000 段** (测试集 1 全 30s ctx, 应全走 long route, 真分必须等于 0.745798)
2. **本机变长 dry-run** (用 simulate_truncated_context 模拟 5s/10s/15s/20s, 看路由是否按 θ 切换)
3. **公榜验证 1 push**: 用变长模拟 csv (混合 5s/10s/20s/30s 段) 实测真分, 跟单 ckpt 对照

## 公榜验证 push 设计 (1-2 push)

| push | 内容 | 期望 |
|---|---|---|
| V1 dual-model θ=250 | 全 30s ctx (测试集 1 不变), 验路由不破 SOTA | 真分 = 0.745798 ± 0.001 (全走 long) |
| V2 dual-model + 截短模拟 | 把测试集 1 段 50% 截短到 10s, 喂入 dual-ckpt | 估真分 0.73-0.74 (验 mask050 救场不破 baseline) |

V1 是 sanity check, V2 是真验证。

## 工程复杂度估

- 步骤 1 训练: **5-8h 单次** (云端 4090 或本机 4 线程, 跑通后顺手)
- 步骤 2 代码改: **~30 行**, 1h 完成 + 单元测试
- 步骤 3 docker: **~5 行 Dockerfile**, 10 min
- 验证: **~30 min** 本机 dry-run + 1-2 push 公榜

**总工作量: 半天 + 1-2 push**。

## 注意事项

- **mask050 ckpt 训练用的还是 cap5 数据集 (5 fold OOF)** — 跟 baseline ckpt 同 fold split. 阈值在 OOF 上找, 不要 sweep。
- **路由是 segment 级, 不是 chunk 级** — 整段同一 ckpt, 不混。
- **mask050 在 10s 已被公榜验过 (0.7376 +0.016)**, 短 ctx 救场可信; 但 mask050 在 < 5s 时未实测, 假设外推合理 (跟 10s 类似)。
- **答辩素材**: dual-model 是"评估错配教训 → 多 ckpt 策略适配"的金料, 落 finals/deep-dives/DD-8 动态时长适配。
