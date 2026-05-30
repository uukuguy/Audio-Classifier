# DECISIONS — FinVCup Turn-Taking 架构决策账本

> 带 rationale 的架构/范式决策（"选 X 不选 Y 因为 Z"）。negative cache 细节见 MEMORY reference_negative_cache.md / research-tree falsified 段。

## 2026-05-30

### D-1: VAP/CPC 音频路线整条证伪，停止投入

**决策**: 放弃 VAP（及所有纯音频编码器）攻 BC 的路线。

**Rationale**:
- 全量 VAP unfreeze 微调真分 0.6337（−0.079 vs SOTA 0.7124），BC=0.222 仅打平 LGBM 基座。
- 信号探针（纯前向，800 切片）证 VAP 预训练 head 原生信号（p_now/p_future/256类/vad/熵）对 BC 区分度 **|r|<0.04**——不是"没用对 head"，是 VAP 归纳偏置本身抓不住 BC。VAP 原文 `objective.py:325` backchannel 预测是未完成 TODO，作者自己没解决。
- 与 mel/whisper 三连败同根：纯声学/语音活动表征预测不了 backchannel 时机。

**含义**: 不再试 HuBERT/MMS 换 encoder（同属音频路线，同盲区）。BC 可预测信号在 context 标签时序（r≈0.13），不在音频。

### D-2: BC 攻击战略从"硬攻 BC"转"攻 T/I"

**决策**: 不把算力全压最难的 BC，转攻 T(0.542)/I(0.434)——更确定的 macro 增量。

**Rationale**:
- 全战场诊断：等权 Macro-F1 下每类 +0.05 对 macro 贡献相同（÷5），但难度天差地别。
- BC 单源三方堵死：context 时序已榨干（F1 顶 0.21=信息上限）/ 音频 r<0.04 / 词袋否。提 0.08 是撞信息论上限的硬仗。
- T/I 远未饱和 + negative cache 有证据文本特征帮过 T/I。干净对照（baseline v1 底）确认 T+0.038/I+0.048，C/NA 不动。
- BC 暂搁置为"信息论上限"，非永久放弃——若 T/I 突破后仍需 BC，再回头找语义触发信号。

**未决**: T/I 文本增益（OOF 真实）转化为提交时遇 train/test 位置偏置（train 整通采样 T率0.253 vs test 切片末 T率0.325）→ test T 暴涨。下一步：train 改切片末采样匹配 test 分布。
