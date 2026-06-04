# Midgame Review — by Gemini CLI
**Date**: 2026-06-01 13:45
**Reviewer**: Gemini CLI (Autonomous ML Engineer)

## Q1 SOTA 路径榨油: [YES]
**结论**: SOTA 路径仍有约 +0.003~0.005 的精调空间，主要在于"概率分布对齐"与"时序头能力"。
**理由**: 当前 `orthofuse` 采用固定权重凸组合（D-6）虽有效规避了 `cap1` 过拟合，但忽略了不同来源概率分布的非线性差异（Calibration）。
- (a) **STRATS 设计**: 过窄。目前只有 5 个固定 alpha。建议在融合前对 `ctx` 和 `whisper` 的 OOF 概率进行 **Isotonic Regression**（保序回归）校准。校准后的等权平均往往优于校准前的加权平均，且不引入 `cap1` 搜索风险。
- (b) **保守门**: +0.003 门槛在 D-9 noise floor 逻辑下合理。但在 D-13 "前 20 攻坚" 背景下，建议降至 **+0.001** 配合 **3-seed 平均**，避免漏掉微弱但真实的正交信号。
- (c) **阈值**: `THR_VARF` 沿用 5/27 的 `cycle1`（单模态）。跨源融合后的联合概率空间已变，建议基于 `cap1` 重新进行全类阈值微调，特别是 `NA`（0.25→0.35 潜力，见 5/30 03:31 记录）和 `I`。
- (d) **whisper head**: **关键盲点**。目前将 30s 序列 pool 成单向量（Mean-pool），丢弃了所有韵律 onset 信息（对 BC/T 致命）。建议替换 MLP 为 **Transformer/LSTM head**，直接输入 1500 帧（stride5）序列，不微调 encoder 也能显著提升对事件时机的捕捉。

## Q2 D-13 三轨判断: [部分有误]
**结论**: B3d 的 SKIP 理由成立，但 B1 v3 可能被过早判死；N1 本机 SKIP 正确。
**理由**: 
- (a) **B3d**: D-14 认为"校准头无新源就不涨"正确，但忽略了 **Calibration 泛化**。B3d 在 OOF 上的 +0.031 是真实校准增益，可能因 `cap1` 样本分布偏移未体现。
- (b) **B1 v3**: 47 个 EDA 强特征（MI>0.15）在 OOF 上无增益（+0.0006），说明 **LGBM 对高维标签序列特征已饱和**。建议 PIVOT：这些特征不该喂树，而应作为 **Side-information 拼入 N1' 的神经头**，让 Transformer head 显式感知 `runlen/burst` 等统计量。
- (c) **原 N1**: 云端 64GB 帧特征在本机无解，SKIP 决策果断，符合工程铁律。

## Q3 N1' 是否抓对: [PIVOT]
**结论**: 抓对了 Loss（长尾），但漏了"新信号源"输入，有重蹈 B3d 覆辙风险。
- (a) **方案匹配度**: DB-Loss + SupCon 针对 BC（0.5%）和 T（1.2%）是学术界公认方案（B4 报告）。但 `train_head_n1.py` 若只输入 `ctx` + `whisper`，本质仍是在挖掘旧信息。**必须引入 F0/Pitch 原始特征**（D-4 证音频最强 BC 分支 r=0.128）作为第三输入。
- (b) **决策门**: 0.6289（Hubert+0.005）作为单源门槛合理。
- (c) **机会成本**: 极低（2h GPU）。支持 N1'，但需将输入从 `[ctx, whisper]` 扩展为 `[ctx, whisper, hubert, F0]` 以确保"信号多样性"。

## Q4 过早判死路线: 重启建议
- (a) **D-1 VAP**: **值得重启（仅 T/I 分支）**。D-1 判死 VAP 是因为其 BC 弱（r<0.04），但 VAP 是 Turn-taking (T/I) 的原生 SOTA。在 N1' 中可尝试将 VAP-CPC 帧作为特征源，其对 T/I 的捕捉能力可能优于 ASR 导向的 Whisper。
- (b) **D-3 文本**: **值得重启（语义 Embedding 模式）**。D-3 否的是"词袋（BoW）"。Qwen2.5-7B 的文本 Embedding 包含对话状态语义，对 `I`（Interruption）的语义触发比 `ctx` 标签强。建议在 N1' 神经头中作为特征拼入。
- (c) **D-6 融合**: D-6 认为 5 源融合封顶 0.715。这是因为 5 源全是 SSL 冻结特征。**重启方向**：将 5 源特征作为输入，训练一个 **统一多模态 Transformer Head**，而非后处理凸组合。
- (d) **B2 整通对话**: 同意取消。16 天来不及。

## Q5 (可选): 0.0091 缺口真路径
**Top 3 排序**:
1. **N1' Pro (多模态 Fusion Head)**: [ctx + whisper + hubert + F0 + Qwen-text] → Transformer Head + DB-Loss (期望 +0.006)
2. **Probability Calibration**: Isotonic Regression on orthofuse OOF + THR_VARF 微调 (期望 +0.003)
3. **Pseudo-labeling**: 用 SOTA 0.71529 的 test 概率作为弱监督，扩充 train 集重训 (期望 +0.002)

**最优先**: **N1' Pro**。
实施步骤：
1. 修改 `cloud/train_head_n1.py`，支持 Concatenate 所有已有的 `probs.npz` (Whisper, Hubert, W2V2, E2V, ctx) 及 F0 统计特征。
2. 架构改为 2 层 Transformer Encoder 处理 30s 序列，输出多标签预测。
3. 采用 DB-Loss + SupCon(BC) 训练。

## Final 综合建议
**D-15 决策建议**: [开 N1' 且升级为 Multi-source Fusion Head]
**第一步行动**: 在云端合并 Whisper 和 Hubert 的 `stride40` 缓存，运行 N1' 实验验证"信号叠加"是否破 `cap1` 0.654 瓶槛。
