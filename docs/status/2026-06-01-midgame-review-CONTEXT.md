# 中场复盘 — 评审输入包 (CONTEXT for multi-AI review)

> **目的**: 给 3 路独立 AI (Gemini / Opencode / Claude self) 同一份输入做"SOTA 起点努力"中场评审。半天预算 deep research 后落各自报告, 主 Claude 汇总成 SYNTHESIS.md → D-15 决策建议。
> **生成时间**: 2026-06-01 11:30
> **不要**直接修改本文件 (它是 AI 的稳定输入). 修改 → 各路 AI 输出会失去可比性。
>
> ## ⚠️ 评审者须知 — 必读, 决定本评审的质量
>
> **本文档只列了 SOTA + 5/31 起 push 摘要 + 决策摘要 + 工程铁律。** 这是为压缩 context 做的剪裁, 但**有可能因此漏掉关键证据**导致评审误判。
>
> **遇到任何疑问/不确定时, 必须主动读以下"权威源文件"** (路径相对项目根):
> - `docs/status/DECISIONS.md` — D-1~D-14 完整决策链含 rationale (368 行, 不是本文摘要的 30 行能替代)
> - `docs/status/JOURNAL.md` — append-only 事件日志 (319 行, 含每个 push 的实时反应/失误/纠正)
> - `docs/status/2026-06-01-experiment-inventory.md` — 15 push 全账本 + HOT 产物路径 + 遗留任务 (168 行)
> - `docs/status/2026-06-01-knowledge-layer-findings.md` — B4 Knowledge Layer 报告 (今早做的)
> - `docs/status/2026-06-01-top20-attack-plan.md` — D-13 三轨作战图
> - `docs/status/2026-06-01-b1-eda-v3-features.json` — B1 EDA 47 候选特征
> - `tools/climb/cycle_orthofuse.py` — SOTA 主程 168 行 (评审 SOTA 实现细节必读)
> - `tools/climb/gen_variants.py` / `cycle_stack_fusion.py` — 前 SOTA 变体F + ctx 基座生成
> - `tools/runs/climb/{orthofuse-20260531-0319,whisper-fusion-20260531-0143,_stack_cache_s40.npz}` — SOTA 关键产物
> - `cloud/train_head_n1.py` — N1' 云上方案的代码 (252 行, 评审 N1' 必读)
> - `CLAUDE.md` — 项目铁律 / 阈值铁律 / 模型边界 / 模型下载 workaround
> - `MEMORY.md` — 跨会话记忆索引含 negative_cache / mps_hardware_limits / threshold_law
> - `docs/赛题要求.md` — 赛题原文 (排行榜 / 复赛要求 / 数据格式)
>
> **读源文件的方式**: 直接读路径, 别相信摘要。本评审包目的是省 token 不是限制视野。**如果你要回答的问题需要更多细节, 主动展开。**

---

## 0. 任务 & 形势

**任务**: FinVCup 2026 turn-taking — 30s 对话片段 (音频+ASR+历史标签) → 预测未来 2s 内 5 类事件 (C/T/BC/I/NA) 是否出现。Event-level 多标签 sigmoid+BCE, 提交是硬 0/1, Macro-F1 等权平均。

**关键约束**:
- 测试集 1000 段, 模型 ≤8B, 复赛镜像 ≤20GB
- 非 Qwen 公开模型需 2026-06-10 前向 `xinyebei@xinye.com` 报备
- 数据: 369 通训练全长对话 (中位 ~25min), 1000 段测试 30s 切片
- 类分布 (chunk 级): C=64.2% / NA=30.1% / I=4.0% / T=1.2% / BC=0.5% (BC/T/I 稀有, 是 macro 杠杆)

**现状**:
- SOTA = `orthofuse-20260531-0319` 真分 **0.71529**
- 排行榜 **第 37 名** (前 40 进复赛, buffer 3 名危险)
- **目标**: 前 20 真门槛 0.7243, 缺口 **+0.009**
- 剩余时间: 15 天 (到 6/16 初赛阶段一结束)
- 提交配额: 5/天, 剩余 ~70 次极充裕

---

## 1. SOTA 实现主路径 (orthofuse-20260531-0319 = 0.71529)

### 1.1 数据流

```
原始数据 (369 通 × ~25min × 双声道 8kHz)
  │
  ├─ ASR JSON (utterances[{channel, start_ms, end_ms, text}])
  └─ 历史标签 (.npy, 0-4=C/T/BC/I/NA per chunk)
  │
  ▼
46d 上下文标签滑窗特征 (stride40, ~36104 窗 / 通)
  │  (cycle_context.py / cycle_context_v2.py)
  │  包含: 各类计数/占比/位置/转换/末段统计
  │
  ▼  ←──── 同时, whisper-large-v3 encoder (云端 CUDA)
  │        │
  │        │ 提取 stride5 ~17.99 万窗 × 1280d 帧特征 (云端 autodl-fs 64GB)
  │        │   (cloud/extract_whisper_cuda.py)
  │        │
  │        ▼
  │        whisper neural head (变体 N1 训练 5fold 15ep, MLP 头)
  │        │   (cloud/train_head_cuda.py)
  │        │
  │        ▼
  │        whisper OOF + test 概率 [N,5] (probs.npz)
  │
  ▼  + whisper probs
ctx_lgbm_v1 OOF + test (5fold LGBM, _stack_cache_s40.npz)
  │  (cycle_stack_fusion.py 生成 4 ctx base 缓存)
  │
  ▼  + whisper OOF+test 对齐 (天然对齐 cap1, order==0 首窗)
cycle_orthofuse.py: per-class 固定权重凸组合候选
  │  STRATS = {ctx, whisper, eq(0.5/0.5), w70(0.7ctx/0.3whisper), w30(0.3ctx/0.7whisper)}
  │  per-class 选 cap1 全集 F1 最大的 STRAT (无权重网格搜索)
  │  保守门: best > ctx + 0.003 才采纳, 否则守 ctx
  │
  ▼  per-class best: T=w70, I=whisper, C/NA/BC=ctx
test_prob 5 列 × 1000 段 → 阈值 THR_VARF={C:0.05, T:0.50, BC:0.75, I:0.65, NA:0.25}
  │
  ▼
pred_test1.csv (segment_id, c, na, i, bc, t) = SOTA 0.71529 提交件
```

### 1.2 核心代码片段 (cycle_orthofuse.py 三个关键)

**候选策略 (固定权重, 无 cap1 调参)**:
```python
STRATS = {
    "ctx": lambda c, w: c,
    "whisper": lambda c, w: w,
    "eq": lambda c, w: 0.5 * c + 0.5 * w,
    "w70": lambda c, w: 0.7 * c + 0.3 * w,
    "w30": lambda c, w: 0.3 * c + 0.7 * w,
}
```

**per-class 选最优 + 保守门**:
```python
for k in range(NUM):  # 5 类
    scores = {name: strat_cap1_f1(...) for name, fn in STRATS.items()}
    best = max(scores, key=lambda n: scores[n])
    # 保守: 只有 best 比 ctx 高 +0.003 才采纳 (防 369 cap1 样本上策略选择噪声)
    if scores[best] < scores["ctx"] + 0.003:
        best = "ctx"
```

**阈值 (来自变体F = cycle1 + 5seed 概率平均的固定阈值, 不在切片CV上重调)**:
```python
THR_VARF = {0: 0.05, 1: 0.50, 2: 0.75, 3: 0.65, 4: 0.25}
# C=0.05 (94% 恒正不能砍正例), T=0.50, BC=0.75, I=0.65, NA=0.25
```

### 1.3 SOTA 实现的关键设计选择

| 决策 | 选项 | 理由 |
|---|---|---|
| 融合策略空间 | **5 个固定权重 strat** vs grid 权重搜索 | grid 在 369 cap1 上 nested-CV 蒸发 (stack-fusion 真分 0.679, BC 0.364→0.200 实证, D-6) |
| per-class 选 strat | **cap1 全集 F1 max** vs 切片CV / nested | cap1 369 上 STRATS 是固定权重无拟合, 直接 F1 即泛化估计; 切片CV 系统性低估 BC, nested 不必要 (策略无参数) |
| 保守门 | **+0.003** vs +0.005 | D-9 实测 cap1 vs 线上 noise floor ≈0.003 |
| ctx 基座 | **lgbm_v1** vs xgb_v1 / lgbm_v2 / mlp_v1 | cycle 18 实测 mlp BC cap1 +0.108 cherry-pick → 真分 -0.022 (D-11); xgb/lgbm_v2 fused 0.6529 略低 lgbm_v1 0.6540 |
| whisper 来源 | **cloud-whisper-large-v3 encoder 冻结 + MLP 头 5fold** | 全量 LoRA 30-63h 不可行 + 冻结 whisper 单源虽弱 (0.671<0.712) 但逐类 T/I 强 (T=0.667/I=0.555 > ctx 0.625/0.539) |
| 阈值 | **cycle1 固定阈值 THR_VARF** | 阈值铁律: 不在滑窗 CV 上做激进逐类阈值搜索 (cycle1b 真分 -0.027 实证) |

### 1.4 SOTA 关键产物 (HOT, 复赛镜像必带)

| 路径 | 大小 | 内容 |
|---|---|---|
| `tools/runs/climb/orthofuse-20260531-0319/pred_test1.csv` | <1KB | **真 SOTA 提交件** |
| `tools/runs/climb/orthofuse-20260531-0319/fused_probs.npz` | 76K | test 1000×5 融合后概率 + ctx_te + wsp_te |
| `tools/runs/climb/whisper-fusion-20260531-0143/probs.npz` | 3.2M | **whisper OOF 179867×5 + test 1000×5** |
| `tools/runs/climb/_stack_cache_s40.npz` | 36M | **4 ctx base OOF+test 缓存** (lgbm_v1/xgb_v1/lgbm_v2/mlp + Y + G) |
| `tools/runs/climb/variant-F-20260528-0559/` | 20K | 前 SOTA 5seed 集成 (0.71242) |
| 云端 `/root/autodl-fs/backups/whisper_cache_full/` | 64GB | whisper stride5 帧特征备份 |

---

## 2. SOTA 之后所有 push (5/31 起完整, 共 8 次, 全是 0 增益或负增益)

### 2.1 cycle 16 (5/31 早上) — 三源融合

| run | cap1 | 线上 | gap | 备注 |
|---|---|---|---|---|
| orthofuse (双源 stride40 弱基座) | 0.6410 | **0.71529** | +0.0743 | 🏆 SOTA |
| orthofuse-s5 (双源 stride5 强基座) | 0.6455 | 0.71233 | +0.0668 | -0.003 强基座反而线上低 |
| **orthofuse-3src (ctx+whisper+hubert)** | **0.6540** | **0.71523** | +0.0612 | -0.00006 = SOTA |

→ **D-8 写**: 跨源融合范式上限锁 0.715。
→ **D-9 写**: 不是分布差, 是 noise floor ≈0.003 淹没 < 0.003 真信号; cap1 +0.013 → 线上 +0.0001 = noise。

### 2.2 cycle 17 (5/31 上午) — 加 w2v2/e2v 4-5 源

| 组合 | cap1 macro | 线上 | margin vs (N-1) |
|---|---|---|---|
| 3 源 (ctx+whisper+hubert) | 0.6540 | 0.71523 (base) | +0.0131 |
| 4 源 (+w2v2) | 0.6540 | 必然=0.71523 | **+0.0000** (w2v2 无类被选) |
| 5 源 (+e2v) | 0.6540 | 必然=0.71523 | **+0.0000** (e2v 无类被选) |

→ 不提交 (cap1 同 + per-class strat 同 → csv 必然同)
→ **D-10 写**: 5 源融合 cap1 实测上限 = 0.6540 (3 源即顶)。加任何第 N+1 源 cap1 都不会突破。

### 2.3 cycle 18 (5/31 下午) — ctx 基座 mlp BC

| 提交 | BC pos | 真分 | Δ vs SOTA |
|---|---|---|---|
| orthofuse-3src (SOTA) | 27 | 0.71523 | base |
| **cycle18 (BC 改 mlp+whisper_70)** | **17** | **0.69358** | **-0.022** |

→ cap1 BC 9 正例上 mlp 1 TP 跳到 F1=0.308, 实际 test 1000 段 P=0.5 → FP 大爆发
→ **D-11 写**: cap1 369 上稀有类 (BC 9 正例) 的 strat 选择本质是过拟合验证集。守"BC 用 ctx-only" 死规则。

### 2.4 cycle 19 (5/31 晚) — 全 ctx-内方向证伪

- **19c (T/I 用 mlp 子策略)**: mlp 在 T (150 正例) F1=0.635 vs SOTA 0.676, I (60 正例) F1=0.475 vs SOTA 0.557 → mlp 全类系统性弱
- **19b (LGBM 超参 sweep)**: baseline (300/0.05/31/1.0) OOF full=0.5909 = 最高, 其它 -0.002~-0.005

→ **D-12 写**: 接受 SOTA 0.71529 作初赛终态, 转复赛镜像准备。

### 2.5 D-13 三轨 (6/1) — 第一日全 SKIP, 0 提交配额损耗

**触发**: 用户提供"排行榜第 37 名, 前 40 buffer 3, 前 20 门槛 0.7243"信息 → D-12 接受论失效, 用前 20 门槛 0.0091 缺口重新评估 D-12 自己列的"真正未试 4 条"。

| 轨道 | 实测结果 | 决策 | 教训 |
|---|---|---|---|
| **B4 Knowledge Layer** | gemini consult + 9 路 WebSearch + arxiv → 锁 N1 = DB-Loss+SupCon 校准头, B2 取消 (16天来不及) | ✅ 出方向 | 报告 `2026-06-01-knowledge-layer-findings.md` |
| **B3d (DB-Loss+SupCon 校准头 on [ctx_lgbm_v1, whisper] OOF)** | OOF macro 0.5701→0.6012 (+0.031 真训练增益) 但 cap1 = SOTA 0.6410 (Δ=0) | ❌ SKIP | **D-14 写**: 校准头无新源就不涨 cap1。chain-first 救命 bug = 错的 cap1 定义(末窗 vs 首窗)+错的阈值(BC 0.5 vs 0.75) 给假高数字 +0.041。修正后真增益 = 0 |
| **B1 v3 (47 个 info>0.15 EDA 强特征加进 46d→93d)** | OOF Δ=+0.0006, cap1 Δ=-0.004 | ❌ SKIP | D-12 "46d 榨干"假设双重实证。47 个 EDA 强候选特征对 LGBM 几乎无 marginal contribution |
| 原 N1 (本机重训 whisper head) | chain-first 否决 — whisper frames 64G 在云端, 本机不可行 | ❌ pivot | 改 N1' = 云上做 |

### 2.6 N1' 云上 (待启动, 用户决策中)

**方案**: `cloud/train_head_n1.py` (派生 `train_head_hubert.py`, 替换 BCE 为 DB-Loss + α·SupCon)
**输入**: 云端 64GB whisper stride5 帧 + ctx 46d
**期望**: 单源 cap1 ≥ 0.6289 (= hubert head baseline 0.6239 + 0.005), orthofuse 后 cap1 ≥ 0.6460 (= SOTA + 0.005)
**ETA**: 20-30min GPU + 30min rsync/setup, ~1h GPU 钱
**风险**: 若 cap1 <0.6289 SKIP, D-13 失效回 D-12 接受 0.71529

---

## 3. D-1~D-14 决策摘要

详 `docs/status/DECISIONS.md` (368 行)。这里只列结论, 评审时**必读源文件**理解 rationale。

| ID | 决策 (一句话) | 类型 |
|---|---|---|
| **D-1** | VAP/CPC 音频路线整条证伪 (BC \|r\|<0.04) | 范式否决 |
| **D-2** | BC 攻击战略从"硬攻 BC"转"攻 T/I" | 战略转向 |
| **D-3** | T/I 文本路线证伪 (cap1 虚高不泛化, gap 仅 +0.003) | 范式否决 |
| **D-4** | BC 冻结路线信息论上限 ~0.22 (所有信号源 r≈0.13 无强信号) | 范式天花板 |
| **D-5** | 0.712 卡点根因 = 单一有效信号源, 融合救不了, 缺第二个强且正交模型 | 诊断 |
| **D-6** | context 内融合证伪 (4 成员不正交 nested 蒸发), 跨源 whisper 正交是真路径 (T/I 强) | 真路径发现 |
| **D-7** | BC 可学 encoder 上限 0.267 (成本 30-63h 不可行), reconcile D-4 | 修正 D-4 |
| **D-8** | 跨源融合范式锁 0.715, 加 hubert 第三源无线上增益 | 范式天花板 |
| **D-9** | 不是 train/test 分布差, 是 noise floor ≈0.003 淹没真信号 (撤 D-8 "分布差"诊断) | chain-first 第二诊断 |
| **D-10** | 5 源融合 cap1 实测上限 0.6540, 3 源即顶 (w2v2/e2v 无类被选) | 实测上限 |
| **D-11** | BC cap1 +0.108 = 9 样本 +1 TP 跳 F1 = cherry-pick (真分 -0.022) | cap1 陷阱 |
| **D-12** | cycle 19 所有 ctx-内方向全证伪, 初赛个人天花板 0.71529 | 收官诊断 |
| **D-13** | 撤 D-12 接受论, 激活前 20 攻坚 (目标 0.7243+, 三轨并行) | 战略反转 |
| **D-14** | B3d OOF+0.031 但 cap1=SOTA → SKIP. 校准头无新源不涨 cap1 | 范式否决 |

### 3.1 累积红旗 (D-1~D-14 已永久关闭)

- ❌ **不再"加第 N 个音频源"** (D-1/D-8/D-10) — w2v2/e2v/任何 SSL 不动
- ❌ **不再"在 cap1 369 上选 strat"** (D-3/D-9/D-11) — 阈值搜索 / per-class grid / BC 单类替换全禁
- ❌ **不再"context 内同源算法集成"** (D-5) — 4 成员不正交
- ❌ **不再"OOF 校准头无新源"** (D-14) — DB-Loss/SupCon 必须叠新源才有意义
- ✅ 唯一允许 cap1→线上转化路径 = **多源融合在 T (150 正例) / I (60 正例) 中等样本类的真实信号叠加**

### 3.2 工程铁律 (CLAUDE.md / MEMORY 已沉淀)

1. **阈值铁律** — 滑窗 CV 调激进逐类阈值线上更差 (cycle1b -0.027 实证, 用切片化 CV 调或保守不调)
2. **稠密 embedding 不喂 LGBM** (Qwen3-0.6B 1024d pooled → LGBM macro -0.008)
3. **cap1 369 样本 BC 增益永远不可信** (D-3/D-11)
4. **本机训练必须限线程** (OMP/MKL/VECLIB/OPENBLAS=4)
5. **whisper 类大编码器本机 MPS 不可行** (45h+), 必须云 GPU
6. **实验值永不写 default** (污染所有后续默认评估)
7. **代码本机写然后 rsync 上云**, 不在云上直接编辑 (source of truth 在本机 git)

### 3.3 关键探针/诊断结论 (negative cache)

- VAP 预训练 head 原生信号对 BC **\|r\|<0.04**
- BC 所有信号源 r≈0.13, 叠加 (context+F0) 仅 +0.005
- whisper 逐类 cap1: **T=0.667 / I=0.555 强于 context** — 真正交杠杆
- mlp ctx base 在 T/I 系统性弱 0.04-0.08 (不只是 BC 噪声)
- w2v2/hubert/e2v 单源 cap1 0.62-0.64 不弱, 但融合 0 贡献 (同范式 SSL 撞墙)
- B3d DB-Loss+SupCon OOF +0.031 真训练增益, cap1 = SOTA (校准头无新源不涨 cap1)

---

## 4. 评审问题 (必答 4 题, 顺序作答)

> **每题要求**:
> 1. **明确结论** (YES/NO/PIVOT) + 一句话总结
> 2. **理由** (引用本文/源文件/外部知识) — 不能只凭直觉
> 3. **如果答案是"有遗漏/有疑问"**, 指出具体证据和补充验证步骤

### Q1: SOTA 路径本身还能榨出什么? (同套路子优化)

**评审范围**: §1 描述的 SOTA 实现 (ctx_lgbm_v1 × whisper-frozen-head per-class orthofuse + 变体F 固定阈值 + 5 STRATS 固定权重凸组合)。

**子问题**:
- a) **STRATS 设计**: 5 个固定权重 strat {ctx, whisper, eq, w70, w30} 是否过窄? 是否漏掉了诸如 max(ctx, whisper) / per-class 不同 alpha / isotonic 校准后凸组合 / log-pool 几何平均等?
- b) **保守门 +0.003**: 是否 D-9 的 noise floor 估计 (4 push n=4) 本身就是噪声? 用更大量级测会不会发现真的 strat 选择门 < 0.003?
- c) **阈值 THR_VARF**: 5/27 cycle1 调的固定阈值 (C=0.05/T=0.50/BC=0.75/I=0.65/NA=0.25), 在 D-6 跨源融合后是否还是最优? 阈值微调 (±0.05) 配合 strat 重选会不会破 cap1 0.6410?
- d) **whisper head 本身**: cloud-whisper 用 5fold MLP 头, 是否可以**保留 T/I per-class 优势但替换 head 架构** (e.g. transformer with positional encoding for 50Hz 帧序列, 不 pool 直接 attention over 30s 序列), 从而让 BC 也有增量? (D-1 否的是 frozen VAP 整体 BC 弱, 没否过"用 whisper frame 序列训 transformer head")

### Q2: D-13 三轨 (B3d / B1 v3 / N1) 全 SKIP 是否有误判?

**评审范围**: §2.5 D-13 6/1 第一日的三轨实验。

**子问题**:
- a) **B3d 校准头**: D-14 结论"校准头无新源不涨 cap1"是否过于绝对? OOF +0.031 是否说明 calibration 本身有信息没在 cap1 体现? 若 push 真分会不会涨 (cap1=SOTA 但 noise floor ±0.003 内可能 +0.001)?
- b) **B1 v3 47 个 EDA 强特征**: 47 个 info>0.15 的 v3 特征 (runlen_*, burst_*, trans_*, diff1/2_*) 加进 46d → 93d 后 OOF Δ+0.0006 cap1 Δ-0.004, 这是否说明 LGBM 对新特征**不会用** (需重做特征 normalize / 加 feature_selection / 改 boost 算法 catboost) 而非"46d 榨干"?
- c) **原 N1 本机**: chain-first 否决 (frames 在云端) — 是否考虑过本机只重训 head 用云端 OOF 输出, 不重训 whisper encoder? (即 train_head_n1.py 输入 = whisper OOF 概率 [N,5] + ctx 46d 而非 whisper 1280d 帧)

### Q3: N1' 云上 DB-Loss+SupCon 是否抓对了关键?

**评审范围**: §2.6 + `cloud/train_head_n1.py` (252 行, 必读)。

**子问题**:
- a) **方案匹配度**: B3d 已证明 DB-Loss+SupCon 在已有 [ctx, whisper] 上 cap1 不涨。N1' 换成单源 whisper 1280d + ctx 训练, 改 loss 后期望 cap1 涨 0.013, 这个期望的依据是什么? **DB-Loss/SupCon 对 single-modality 训练的增益是否有文献证据**?
- b) **决策门设定**: 单源 cap1 ≥ 0.6289 = hubert head 0.6239 + 0.005 是否合理? hubert head 用 stride40 (不 dense), N1' 用 stride5 dense, 比较基准不一致 (D-10 已证 hubert+whisper+ctx 三源融合 cap1 = 0.6540, 单 whisper-stride5-head 应该已经在这个组合里, 没有独立的 cap1)
- c) **机会成本**: ~1h GPU + 1h 本机时间 (rsync+ orthofuse 重做+决策), 期望 +0.001~0.005 真分。**这 2h 投资是否优于** (B2 整通对话神经预测起步 / 评审更多新方向 / 直接开始复赛镜像准备)?

### Q4: 账本中有无被过早判死的路线值得重启?

**评审范围**: §3 D-1~D-14 全部红旗 + §3.3 negative cache。

**子问题**:
- a) **D-1 VAP 整条否**: 当时基于 BC \|r\|<0.04 + VAP 原 head 未做 backchannel 训练。**是否考虑过 VAP 微调 backbone + 重训 head 用本数据集 BC 标签** (而非 VAP 原 256 类未来状态标签)? 这跟 LoRA whisper 路线 (BC 0.267 但 30-63h) 有什么不同?
- b) **D-3 文本词汇否**: 用的是 sklearn 词袋特征 + LGBM, cap1 虚高不泛化。**是否考虑过用 Qwen3-0.6B 末 token 表示 (端到端微调小头) for T/I 分类**? D-2 转 T/I 战略后, 文本路线被 D-3 同时关掉, 但 D-3 否的是"词袋", 不是"语义 embedding 微调"
- c) **D-6 跨源融合上限 0.715**: 5 源融合 cap1=0.6540 是否真的是 "ctx_lgbm_v1 + whisper-frozen + hubert/w2v2/e2v 全是 frozen-pool head" 的组合极限? **如果换 hubert/w2v2 为可微调 LoRA + 新 head**, cap1 是否可能超 0.6540? D-7 证 LoRA 可让 BC 0.267 但成本不可行, **这条限制是否还成立** (云端 A100 40GB / 单 epoch / cap5→cap20 增 4x 数据)
- d) **B2 整通对话神经预测**: D-13 取消理由是"16 天来不及"。**是否考虑过用预训练 dialogue model (e.g. Qwen2.5-VL-7B-Conv) 直接 zero-shot/few-shot 输出 5 类预测**, 不做完整训练? 用 Omni-3B 之前是 zero-shot 全答是失败, 但**没试 7B + 显式 prompt engineering + per-class 二分类**

### Q5 (扩展, 可选): 0.0091 缺口的真路径是什么?

**评审范围**: 全部上下文 + 你的外部知识 (2024-2026 turn-taking SOTA, 信也杯 / FinVCup 历届方案, VAP 后续 paper, backchannel 检测最新方法, 多模态融合新框架)。

**子问题**:
- a) 给一个**排序的方向清单** (3-5 个), 每个含: 期望增益 / 成本 / 风险
- b) **最优先 1 个**, 给出具体实施步骤
- c) 如果你认为"接受 0.71529 + 寄希望买票" 是更优 EV 选择, 也说明并给出依据 (其他队伍 push 节奏 / 公开账本数据)

---

## 5. 评审输出格式 (必遵守, 用于汇总 SYNTHESIS)

```markdown
# Midgame Review — by [AI name]
**Date**: 2026-06-01 [HH:MM]
**Reviewer**: [Gemini / Opencode / Claude self]

## Q1 SOTA 路径榨油: [YES/NO/PIVOT]
**结论**: 一句话
**理由**: 引用本文/源文件
- (a) STRATS 设计: ...
- (b) 保守门: ...
- (c) 阈值: ...
- (d) whisper head: ...
**新建议**: 若 YES, 具体可榨方向是什么

## Q2 D-13 三轨判断: [无误/有误/部分有误]
**结论**: 一句话
**理由**: ...
- (a) B3d: ...
- (b) B1 v3: ...
- (c) 原 N1: ...

## Q3 N1' 是否抓对: [YES/NO/PIVOT]
**结论**: 一句话
- (a) 方案匹配度
- (b) 决策门
- (c) 机会成本

## Q4 过早判死路线: 列出值得重启的 (可能 0 个)
- (a) D-1 VAP: ...
- (b) D-3 文本: ...
- (c) D-6 融合: ...
- (d) B2 整通对话: ...

## Q5 (可选): 0.0091 缺口真路径
**Top 3 排序**:
1. ...
2. ...
3. ...
**最优先**: 实施步骤

## Final 综合建议
**D-15 决策建议**: [开 N1'/不开 N1'/PIVOT 新方向]
**第一步行动**: 一句话
```

---

## 6. 评审者注意事项

1. **不要装样子说"再读 README"** — 用户希望评审者真的读源文件再答, 不是引用文档标题。
2. **不要重复"经过详细分析"等套话** — 直接给结论 + 证据。
3. **承认不确定性** — 不知道就写"需补充实验验证", 不要硬给答案。
4. **避免 group think** — 不要事先看其他 AI 的输出 (你不会看到), 但要预判主流答案的盲区。
5. **质疑前提** — 如果发现本文档本身有逻辑漏洞 / 数字不一致 / 决策链断点, 直接指出。
6. **chain-first** — 给建议前自己先在脑子里跑一遍数据流, 看新建议会撞哪个红旗。

---

**EOF — 评审包 v1.0, 6 节, 4 道题 (Q5 可选)**
