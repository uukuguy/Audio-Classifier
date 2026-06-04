# 实验对照数据 (桶 3)

> 真分账本 + 跨切片实验 + calibration 表 + 重要负结果数据.
> 完整账本在 `docs/status/2026-06-04-submission-strategy.md`. 这里只摘**答辩可视化**的子集.

## 已有可视化材料源

### 25 push 真分账本
- 路径: `docs/status/2026-06-04-submission-strategy.md` "真分梯队 (25 push)" 段
- 可做: 真分时间线柱状图, color 标 confirmed/falsified, x 轴时间
- 答辩用: 展示渐进上升路径, 标关键 milestone (D-22 软加 / D-25 双 SSL 协同)

### 跨切片稳定性 cap0-cap4
- 路径: `docs/status/2026-06-04-submission-strategy.md` "跨切片稳定性" 段
- 数据: 12 个源在 cap0-cap4 macro F1 + range
- 可做: 折线图 (每源一条线), 横轴 cap_i, 纵轴 macro F1
- 答辩用: 展示 SSL ms 系跨切片最稳 (range 0.058-0.061), 论证 R4 复赛鲁棒性

### OOF vs 真分散点
- 数据: 25 push 每个的 OOF cap1 (369 段) vs 真分 (1000 段)
- 可做: 散点图 + 等比例对角线 (理想是 y=x), 标"D-22 反范式 push" 红色
- 答辩用: 视觉证据 — OOF 跟真分顺序量级都失真, 不能用 OOF 选源

### 阈值铁律 (D-18)
- 数据: w2v2low BC 阈值 0.10 → cap1 0.6667 (OOF 高), 真分 0.6695 (-0.048 vs SOTA)
- 可做: BC 阈值扫描 (0.10/0.30/0.50/0.75) 真分曲线, 标 OOF vs 真分背离
- 答辩用: 举例 1 个负结果数据, 说明"不能用 OOF 选 BC 阈值"

### 双 SSL_ms 协同效应 (D-25)
- 数据: R5 (0.7389) → R6 (+e2v_ms 0.03 = 0.7374 -0.0015) → R4 (+hub_ms 0.03 = 0.7458 +0.0084)
- 可做: 决策树/桑基图 — 起点 R5, 单加 e2v 反降, 双加突涨
- 答辩用: 视觉证明协同效应非加法

## 待积累 (后续 push 真分回来时往这塞)

### 6/5 5 push 真分 (待回)
- 候选: probe-day7-20260604-1005/{S1-S6}
- 期望验证: 三 SSL_ms 微叠是否再突破 / wsp_ms 0.10 / Omni-3B + R4 合规

### T3 cross-context 内部对照表 (D-26 应对, 6/4 14:30 实测)

详细: `charts/cross-context-degradation-20260604.md`

- N=617 cap5 windows, ctx-only LGBM v1, cap0-4 多窗口起点
- 截短 1s/2s/5s/10s/20s/30s 6 挡, base = 30s macro F1 = 0.5797
- 退化最大 = 截到 1s, macro Δ = -0.085 (-14.7%)
- 退化最小 = 截到 20s, macro Δ = -0.018 (-3.1%)
- C/T 类敏感度最高, NA 几乎不敏感 (NA F1 全程 0.831)

### T1 R4 截短公榜验证 csv (待 push)

- `submission/truncated-validation-20260604/R4_keep125_ctx10s/pred_test1.csv` (R4 用 10s 上下文)
- `submission/truncated-validation-20260604/R4_keep63_ctx5s/pred_test1.csv` (R4 用 5s 上下文)
- pos 变化: NA 飙升 / C 急跌 / BC/I 砍半 / T 几乎不变
- 期望真分: 0.72-0.73 (基于 T3 推算)
- 答辩用法: 实证 R4 在短上下文下"真实跌幅" — 1 个 push 换决赛答辩 1 张关键 slide

### v2 ms2 三新源真分 (待 push, 估 6/6-6/8)
- omni_ms2 / omni3b_ms2 / qwen17b_ms2
- 答辩价值: 证明 multi-seed 自身就提分 (跟软加正交)

### 复赛镜像在公榜对照 (估 6/9-6/14)
- 复赛 friendly 路径 (R1 全 SSL low-w) 在公榜真分
- 答辩价值: 展示我们对复赛风险的提前 hedge

## 表格 / 图表 草稿存放

可视化产物 (matplotlib 图 / mermaid 源 / 截图) 放 `charts/`.

格式约定:
- `charts/<topic>-<date>.png` (图片)
- `charts/<topic>-<date>.mermaid` (mermaid 源)
- `charts/<topic>-<date>.py` (生成脚本, 方便后续调整样式)

## 关键真分锚点 (永远要在 PPT 出现的数字)

- **R4 = 0.7458** (6/4 NEW SOTA, 双 SSL 协同, 第 4 名)
- **R5 NSOTA_07 = 0.7389** (6/4 wsp_ms 权重峰)
- **Q5 = 0.7367** (6/4 早 SOTA, NSOTA_05)
- **6/3 P3 = 0.7327** (Omni-7B 0.15 权重峰, D-22 范式开端实证)
- **cand2 = 0.7285** (6/2 D-22 软加范式首破, 排名 14, 9.4B 超 8B)
- **orthofuse-3src = 0.71755** (5/31 合规 SOTA base, 3 源跨源正交融合)
- **cycle1 ctx-only = 0.71079** (5/27 第一日基线, 纯 context LGBM)
- **官方 baseline = 0.619742** (无任何处理的官方代码线)

我们 6 天从 0.6197 推到 0.7458 = +0.126 / 复赛 friendly R1 = 0.7338 (8B 合规 + 跨切片最稳)
