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

### D-3: T/I 文本路线证伪（CV 虚高不泛化）

**决策**: 放弃文本词汇/语义特征攻 T/I。

**Rationale**:
- ti-robust（修标点 bug 后）真分 **0.6392**，cap1 CV 0.6358≈SOTA 0.6402，**但 gap 仅 +0.003**（变体F gap +0.072）。同样的 cap1 CV，变体F 线上涨 0.072，文本几乎不涨 → **cap1 增益虚高，不泛化到 test**。
- 文本特征过拟合 cap1（对话开头切片）的局部分布，test 各处切片增益蒸发。
- 印证旧 H-T2 当初"CV+0.004 但 test 预测剧烈偏移可疑→没提交"的判断是对的。
- 教训：被 OOF +0.0217 大数字带偏，没充分尊重阈值铁律"CV 高分≠线上高分"。

### D-4: BC 确认信息论上限 ~0.22，停止单点攻坚

**决策**: 停止 BC 单点攻坚（所有角度已交叉验证到顶）。

**Rationale**（完整证据链，多构型/多数据规模/多信号源交叉验证）:
- context 时序（含导数/突发性/周期）→ 0.21 到顶（baseline 46维已榨干）
- VAP mean-pool 0.222 / attention-pool 0.080（全量单变量对照，pool 非瓶颈）/ 微调 / 冻结 → 全 ≤0.22
- F0/pitch（音频对 BC **最强**分支 |r|0.128）融合 → +0.005（正交但增量太小）
- 文本词汇 → OOF 虚高不泛化（D-3）
- whisper/mel → <0.22
- **所有信号源 r≈0.13，无强信号，叠加（context+F0）仅 +0.005** → BC 在此数据集接近信息论上限，"未来2s会不会backchannel"本身高度难预测。

**含义**: 不再投入 BC 单类。转务实：用已验证最优（变体F 0.7124）+ 零风险稳健化 + 接受 BC≈0.22。

**方法论教训（用户 3 次质疑均成立）**: ①音频证伪过快→修正后 F0 确是最强分支但弱 ②"同分布=拟合test"→揪出 T/I CV 虚高 ③构型/参数→证 mean-pool 反优于 attention-pool。共同收敛：BC 是数据本身难，非方法不对。
