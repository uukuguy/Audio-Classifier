# 候选模型 + 融合方案（VAP 全量后做）

> 2026-05-30。用户问"VAP 之外有没有同档/更强模型做融合 + 诊断 workflow 还提到哪些值得做"。
> Research + 已落盘诊断的待办，备查。当前全量 VAP-CPC 训练中(PID 5193)。

## 一、VAP 之外的候选 encoder（可融合）

| 候选 | 强度 vs VAP-CPC | 工程成本 | 关键点 |
|---|---|---|---|
| **VAP-HuBERT**（仓库自带 `vap/modules/encoder_hubert.py`，16kHz）| 可能更强 | **几乎零**（同 train_vap.py 换 build_vap 的 encoder 类）| HuBERT 表征比 CPC 强 |
| **VAP-MMS**（仓库自带 `encoder_mms.py`，dim 512/1024/1280）| 多语言含中文 | **几乎零** | wav2vec2-MMS 多语言 |
| chinese-hubert / chinese-wav2vec2 | 中文电话域可能最强 | 中（需写 encoder 包装+下载）| 最贴域，需 EDA 验证 |
| LLM 文本侧（Qwen 语义，非词袋）| 互补(文献证) | 中 | 文本词汇已证伪，但 LLM 语义可能不同 |

### ★关键洞察：HuBERT/wav2vec2 对本赛题可用（VAP 作者排除它们不适用于我们）
- VAP 原作者排除 HuBERT/wav2vec2 是因为它们**双向**，不满足 VAP 的**逐帧实时流式**因果约束。
- **但赛题是离线批量预测**：窗口右边界=预测点，窗口内双向 attention 不读未来音频 = 合法。
- 所以 HuBERT/MMS 对我们可用，且仓库自带，**几乎零成本试**。

## 二、最高 ROI 融合方向

**VAP-CPC + VAP-HuBERT 双 encoder 概率平均集成**：
- 两个 VAP 模型（换 encoder），对 BC 列预测概率平均
- 不同 encoder 错误模式不同 → 集成通常 +
- 仓库自带 HuBERT encoder，工程成本几乎为零

文献佐证融合有价值：
- "VAP+LLM 的 LSTM 集成提升 TRP"（Jeon 2024, arxiv 2505.12654）
- 多模态（加视觉）补足纯音频缺陷 → 单模态有天花板，融合是正路
- "Yeah Un Oh"(2410.15929) 微调 VAP 做 BC，F1 显著超 baseline

## 三、诊断 workflow 还提到的待办（独立于 VAP，可并行）

来自 `2026-05-29-diagnosis-zero-lift.md`：

### 动作3 — LGBM 集成鲁棒化（本机 ~30min，成功率 ~60%，+0.003~0.006）
- 变体 F 的"5seed 全量重训"→ **5-fold OOF 每 fold 一模型概率平均**（降方差）
- 温和阈值 C0.05/NA0.35/T0.55/I0.55/BC0.55
- **独立于 VAP，本机可跑，不占云、不等 VAP** → 加固 SOTA 基座

### 最终融合架构（诊断核心建议 = VAP 的归宿）
- C/NA/T/I 用 LGBM SOTA（饱和、已验证 0.7124）
- **只把 BC 列换成 VAP 预测**（BC 强项）
- 拿 BC 杠杆全额（ΔBC/5 进 macro），其他类不冒险
- BC 0.20→0.27 进前10 / →0.30 进前3 / →0.40 冲榜首

## 执行优先级（VAP 全量出结果后）
1. VAP-CPC 全量 BC 好 → 融合(LGBM基座+VAP的BC列)提交
2. 想再榨 → 加 VAP-HuBERT(零成本)，CPC+HuBERT 双encoder 对 BC 集成
3. 并行无依赖 → 本机跑动作3(LGBM集成鲁棒化)加固基座
