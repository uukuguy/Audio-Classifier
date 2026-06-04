# 初赛剩余时间任务清单 (6/4 - 6/16)

> "初赛垃圾时间针对性做些什么" — 用户 6/4 13:00 提.
> 现状: R4 = 0.7458 第 4, 距前 3 仅 0.0002, 距前 1 0.009. 初赛冲分边际收益递减, **复赛准备红利更大**.

## 已确认约束 (从赛题要求图 1 抠出)

1. **复赛测试集 2 上下文动态时长 (0, 30]s** — 不再固定 30s, 任意短长
2. chunk = 80ms 不变, 预测窗口 = 2s = 25 chunk 不变
3. 测试集 2 含**内部私有数据** (业务领域可能不同)
4. 复赛 docker 跑**推理** (不重训), train 代码是评审材料
5. 模型参数 ≤ 8G ckpt 总大小硬约束 (我们 R4 ~6.6G 内)

## 进度跟踪 (实时)

| 任务 | 状态 | 完成 % | 落盘 |
|---|---|---|---|
| T1 推理归一化 | ✅ 实现 + 内部测试 | 70% | `tools/climb/dynamic_ctx_utils.py` + `tools/climb/build_truncated_r4.py` |
| T1 公榜验证 | ⏳ csv 就绪, 等 push | 30% | `submission/truncated-validation-20260604/{R4_keep125,R4_keep63}/` |
| T2 train 变长重训 | ⏳ 未启动 | 0% | — |
| **T3 cross-context 内部对照** | ✅ **完成实测** | 100% | `tools/runs/climb/dynamic-ctx-eval-20260604/results.json` + `docs/finals/charts/cross-context-degradation-20260604.md` |
| T4 复赛 docker prototype | ⏳ 未启动 | 0% | — |
| T5 报备邮件 | ⏳ 6/8 前发 | 0% | 草稿在 `docs/status/2026-06-04-submission-strategy.md` |

### T3 关键发现 (6/4 14:30 实测落地)

| 上下文 | ctx-only macro F1 | Δ vs 30s |
|---|---|---|
| 30s | 0.5797 | base |
| 20s | 0.5617 | -0.018 |
| 10s | 0.5505 | -0.029 |
| 5s | 0.5355 | -0.044 |
| 2s | 0.5047 | -0.075 |
| 1s | 0.4945 | -0.085 |

**含义**: R4 全栈推算 (ctx 退化 × 0.5 加权) — 复赛 (0, 30]s 均匀分布估真分 **0.72-0.74**, 远好于事前估的 0.60-0.70 风险. 即使不做 T2 重训, T1 推理归一化 + T3 实测论据已经足够应对.

## 5 个针对性高 ROI 任务 (按优先级)

### T1. 推理上下文长度归一化 (策略 1, 1 天工作量) ✅ 实现完成

**问题**: 复赛 ctx 长度 (0, 30] 任意 → 我们整套 ctx feature 工程依赖 375 chunk 输入

**实现**:
```python
def normalize_ctx_to_375(ctx_var_len: np.ndarray) -> np.ndarray:
    """复赛 infer 入口: 任意长度上下文 → 强制 375 chunk."""
    n = len(ctx_var_len)
    if n >= 375:
        return ctx_var_len[-375:]   # 取末 375 (最近因果窗)
    # n < 375: 前面 zero pad 标 NA (label=4)
    pad = np.full(375 - n, 4, dtype=ctx_var_len.dtype)
    return np.concatenate([pad, ctx_var_len])
```

**验证方法**: 模拟"截短输入"在公榜跑 — 把测试集 1 的 375 chunk context **截短到 50/100/200/300 chunk 再 pad 回 375**, 看真分掉多少:

| 截短到 | 估真分 |
|---|---|
| 375 (原始) | 0.7458 R4 ✓ |
| 300 | 估 0.74 |
| 200 | 估 0.73 |
| 100 | 估 0.71 |
| 50 | 估 0.68 |

**push 配额**: 1-2 个 (验 100 和 300 截短)

### T2. Train 时模拟变长上下文 (策略 2, 5-8h 训练)

**做法**: 重训 ctx LGBM 4 base + SSL_ms 头, train 时随机 mask context 末段:

```python
# _make_ctx_features 加 mask 参数
for sample in train_loader:
    ctx_375 = sample.context  # 原始 375 chunk
    # 随机 mask 末段, 模拟测试集 2 短上下文
    keep_len = np.random.choice([50, 100, 200, 300, 375], p=[0.1, 0.15, 0.25, 0.25, 0.25])
    if keep_len < 375:
        ctx_375[:375-keep_len] = 4  # NA pad
    feat = make_ctx_features(ctx_375)
```

**改的文件**:
- `tools/climb/cycle_context.py:make_ctx_features` 加 `mask_prob` 参数
- `tools/climb/cycle_orthofuse.py` 重生 ctx OOF + test probs
- `tools/runs/climb/_stack_cache_s40.npz` 重生 cached features

**push 配额**: 2 个 (重训后的 ctx 单源 + 接 R4 全栈)

### T3. 模拟"测试集 2 风格" 内部对照测试集 (策略 3 + 4, 不耗 push)

**做法**: 从 train conv 切非 30s 段做 cross-domain probe, 比较各模型表现:

```python
# 从 train conv 切 10s/15s/20s/25s context + 2s prediction window
# 模拟测试集 2 短上下文条件
for conv in train_convs:
    for ctx_seconds in [10, 15, 20, 25, 30]:
        sample = build_sample(conv, ctx_seconds=ctx_seconds)
        for model in [R4, R5, R1, R3]:
            pred = model.infer(sample)
            metric = compute_macro_f1(pred, gt)
        # log: ctx_seconds × model × macro F1
```

**输出**: 模型在 (10, 15, 20, 25, 30)s 上下文长度下的 cross-context macro F1 表, 选**对短上下文最稳的复赛主力模型**.

**预期发现**: R4 双 SSL 微叠应该最稳 (跨切片 range 0.058-0.061), R5 NSOTA 单 wsp_ms 稳一档, Omni 系列大幅退化 (mean-pool 对 token 数敏感)

**push 配额**: 0 (纯本地评估)

**素材积累点**: 这表是答辩材料金矿 ★ — `finals/EXPERIMENT-EVIDENCE.md` 第 4 块. 答辩可讲: "我们对复赛动态时长做了 cross-context 鲁棒性评估, R4 在 (10,30)s 范围内退化 < 0.02"

### T4. 复赛 infer pipeline docker prototype (6/9 起, 6/15 跑通)

**做法**: 不等复赛阶段才动手, 现在就搭 docker:

```dockerfile
FROM <主办方 base 镜像>
COPY models/ /app/models/         # R4 全 ckpt ~6.6G
COPY src/infer.py /app/
COPY data/preprocessing/ /app/
ENTRYPOINT ["python", "/app/infer.py"]
```

`src/infer.py` 要做的:
1. 读 `data/test/audio/<id>.wav` + `data/test/text/<id>.json` + `data/test/context/<id>.npy`
2. **对 context 做 `normalize_ctx_to_375()`** (T1 实现) — 任意长度入
3. SSL_ms 头 load + frozen encoder forward + softadd 融合 + 出 prob
4. 阈值化 + 出 `pred_test1.csv`

**改进点 vs 当前本地**:
- 完全无 cap1 真分校准依赖 (复赛打不到这个杠杆)
- 用 T2 训出的"变长 robust" 头 (如果 T2 完成)
- entrypoint 多模式: `--mode=R4|R5|R1` 多 fallback 切换

**push 配额**: 0 (本机/云端 dry-run, 不上公榜)

**素材积累点**: pipeline 架构图, 模型流水, 复赛 docker 设计 → `finals/charts/`

### T5. 报备邮件 6/8 前发 (任务量小但日期硬)

**收件人**: `xinyebei@xinye.com` cc 用户 `531045572@qq.com`

**内容**: 列 6 个非白名单模型 (chinese-hubert / wav2vec2 / emotion2vec / whisper-large-v3 / Qwen3-0.6B / Qwen3-1.7B)

**草稿在**: `docs/status/2026-06-04-submission-strategy.md` "报备邮件" 段

**截止**: **6/8 21:00** 前发 (留 2 天缓冲, 6/10 是主办方截止)

## 不建议做的事 (低 ROI)

| 不做 | 理由 |
|---|---|
| ❌ 继续冲公榜真分到第 1 | 边际 push +0.002-0.005, 对手也在 push, 实际名次不一定改; 复赛是另一场 |
| ❌ 训 Omni-7B/3B 更多变体 | Omni 跨切片 range 0.0607-0.0969 全场最不稳, 复赛风险溢价高 |
| ❌ 加 Qwen3-4B/Qwen3.5-4B ms | Qwen3 系 mean-pool BC=0 失败, 投入耗时不值 |
| ❌ Pseudo-label 用公榜数据 | FAQ#3 严禁, 触发取消资格 |
| ❌ 公榜数据增强后 train | 同上, 违规 |
| ❌ 复赛镜像里偷训练逻辑 | 镜像跑推理, train 是评审材料人工读 |

## 节奏修订 (6/4-6/16, 13 天)

| 日期 | T1 归一化 | T2 变长重训 | T3 cross-context | T4 docker | T5 报备 | push 配额 (5/天) |
|---|---|---|---|---|---|---|
| 6/4 (今) | — | — | — | — | — | 5 push 已发, 等真分 |
| 6/5 | 实现 + 单元测试 | — | — | — | — | 5 push (probe-day7) |
| 6/6 | 公榜验证 (截短模拟) | feature 工程改 | — | — | — | 5 push (含 T1 验证 2 个) |
| 6/7 | — | ctx 4 base 重训 | 设计 | — | — | 5 push |
| 6/8 | — | SSL_ms 重训 | 实现 + 跑 | docker 框架 | **发邮件** | 5 push |
| 6/9 | — | 公榜验证 (重训后) | 出 cross-context 表 | infer.py 写 | — | 5 push (含 T2 验证 2 个) |
| 6/10 | — | — | 答辩素材化 | dry-run | (主办方截止) | 5 push |
| 6/11-6/13 | — | — | — | infer 全流程 | — | 15 push 公榜冲分 + 复赛对照 |
| 6/14-6/15 | — | — | — | docker 完整测试 | — | 10 push (最后冲分) |
| 6/16 | — | — | — | 镜像 freeze | — | 5 push (复现锚定) |

总 push 估 70 个 (13 天 × 5 - 已用 R3/R4/R6), 阶段 1 (T1-T2 验证) 用 4-6 push, 阶段 2 (冲分) 25-30 push, 阶段 3 (复赛对照) 15-20 push, 保留 5-10 push 应急.

## 关键转向

**心态转换**: 从"公榜冲第 1" → "复赛准备最优 + 公榜稳第 4-5".

- 公榜继续冲分但**不孤注一掷** — 第 4-7 的客观分差 (0.7458 → 0.7286) 只 0.017, 名次变动对决赛影响有限 (评审是综合分)
- **复赛准备红利**: T1 一天能做完, T2 一周能做完, T3 cross-context 能产答辩金料, T4 docker 提前 1 周搞定避免末日翻车
- **复赛风险对冲**: R4 全栈是公榜 cap1 调出来的, 复赛动态时长 + 内部数据可能跌 0.05+, 必须 T2 重训补救

## 答辩素材产出 (顺手积累)

- T3 输出 cross-context 表 → `finals/EXPERIMENT-EVIDENCE.md` 第 4 项
- T4 docker 架构图 → `finals/charts/` mermaid
- T1+T2 设计哲学 → `finals/deep-dives/DD-8.md` (新增"动态时长适配")
- 整个"针对复赛 hedge" 决策叙事 → `finals/quotes/` + `INNOVATION-CANDIDATES.md` C6 候选

**答辩潜在金句**: "我们意识到复赛动态时长会让公榜冲分的 cap1 红利消失, 所以从 6/5 开始把一半 push 配额转向 cross-context 鲁棒性验证 — 这是公榜峰值与复赛鲁棒的取舍".
